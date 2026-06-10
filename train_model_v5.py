"""
Traffic Demand Prediction - V5
===============================
Strategy: V1 was best (89.2). Don't add complexity - reduce overfitting.

Key insights:
- V1 (simple, 57 features, light reg) = 89.2 ✓
- V3 (upweighted day49) = 88.5 ✗
- V4 (97 features, diverse) = 87.0 ✗
- More features = worse generalization

V5 approach:
1. V1 feature set (proven best) with minimal additions
2. Log1p target transform (right-skewed demand)
3. Timestamp-weighted training (match test time distribution)
4. Multiple seeds for stability
5. Blend with V1 predictions at various ratios
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv('/Users/hiya/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/hiya/Downloads/dataset/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# Feature Engineering (V1-style, minimal)
# ============================================================
print("Feature engineering (V1-style)...")

def parse_timestamp(ts):
    parts = ts.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def engineer_features(df):
    df = df.copy()
    
    # Timestamp
    df['minutes'] = df['timestamp'].apply(parse_timestamp)
    df['hour'] = df['minutes'] // 60
    df['minute_of_hour'] = df['minutes'] % 60
    df['quarter_of_day'] = df['minutes'] // 15
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minutes'] / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * df['minutes'] / 1440)
    
    # Time indicators
    df['is_rush_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_rush_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_rush_morning'] | df['is_rush_evening']).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
    
    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 6),
        (df['hour'] >= 6) & (df['hour'] < 10),
        (df['hour'] >= 10) & (df['hour'] < 14),
        (df['hour'] >= 14) & (df['hour'] < 18),
        (df['hour'] >= 18) & (df['hour'] < 22),
        (df['hour'] >= 22)
    ]
    df['period_of_day'] = np.select(conditions, [0,1,2,3,4,5], default=0)
    
    # Geohash prefixes
    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]
    
    # Categoricals
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc'] = (df['Landmarks'] == 'Yes').astype(int)
    
    road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
    df['RoadType_enc'] = df['RoadType'].map(road_map)
    df['RoadType_missing'] = df['RoadType'].isna().astype(int)
    df['RoadType_enc'] = df['RoadType_enc'].fillna(-1)
    
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['Weather_enc'] = df['Weather'].map(weather_map)
    df['Weather_missing'] = df['Weather'].isna().astype(int)
    df['Weather_enc'] = df['Weather_enc'].fillna(-1)
    
    # Temperature
    df['Temperature_missing'] = df['Temperature'].isna().astype(int)
    df['temp_squared'] = df['Temperature'] ** 2
    df['temp_abs'] = df['Temperature'].abs()
    
    # Interactions (same as V1)
    df['road_lanes'] = df['RoadType_enc'] * 10 + df['NumberofLanes']
    df['road_large_vehicles'] = df['RoadType_enc'] * 10 + df['LargeVehicles_enc']
    df['lanes_large_vehicles'] = df['NumberofLanes'] * 10 + df['LargeVehicles_enc']
    df['hour_road'] = df['hour'] * 10 + df['RoadType_enc']
    df['hour_weather'] = df['hour'] * 10 + df['Weather_enc']
    df['road_weather'] = df['RoadType_enc'] * 10 + df['Weather_enc']
    df['lanes_landmarks'] = df['NumberofLanes'] * 10 + df['Landmarks_enc']
    df['road_capacity'] = df['NumberofLanes'] * (1 + df['LargeVehicles_enc'])
    
    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

temp_median = train['Temperature'].median()
train_fe['Temperature'] = train_fe['Temperature'].fillna(temp_median)
test_fe['Temperature'] = test_fe['Temperature'].fillna(temp_median)
train_fe['temp_squared'] = train_fe['temp_squared'].fillna(temp_median**2)
test_fe['temp_squared'] = test_fe['temp_squared'].fillna(temp_median**2)
train_fe['temp_abs'] = train_fe['temp_abs'].fillna(abs(temp_median))
test_fe['temp_abs'] = test_fe['temp_abs'].fillna(abs(temp_median))

temp_bin = pd.cut(train_fe['Temperature'], bins=10, labels=False, retbins=True)
train_fe['temp_bin'] = pd.cut(train_fe['Temperature'], bins=temp_bin[1], labels=False)
test_fe['temp_bin'] = pd.cut(test_fe['Temperature'], bins=temp_bin[1], labels=False)
train_fe['temp_bin'] = train_fe['temp_bin'].fillna(5)
test_fe['temp_bin'] = test_fe['temp_bin'].fillna(5)

# Label encode
for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5']:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]])
    le.fit(combined)
    train_fe[col + '_le'] = le.transform(train_fe[col])
    test_fe[col + '_le'] = le.transform(test_fe[col])

# ============================================================
# KFold Target Encoding (V1 style - from ALL data)
# ============================================================
print("KFold target encoding...")

def target_encode_kfold(train_df, test_df, col, target_col, n_folds=5):
    train_encoded = np.zeros(len(train_df))
    global_mean = train_df[target_col].mean()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        means = train_df.iloc[tr_idx].groupby(col)[target_col].mean()
        train_encoded[val_idx] = train_df.iloc[val_idx][col].map(means)
    
    train_encoded = np.where(np.isnan(train_encoded), global_mean, train_encoded)
    test_means = train_df.groupby(col)[target_col].mean()
    test_encoded = test_df[col].map(test_means).fillna(global_mean).values
    
    return train_encoded, test_encoded

for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5']:
    tr_enc, te_enc = target_encode_kfold(train_fe, test_fe, col, 'demand')
    train_fe[f'{col}_te'] = tr_enc
    test_fe[f'{col}_te'] = te_enc

for col in ['RoadType_enc', 'Weather_enc', 'hour', 'quarter_of_day']:
    tr_enc, te_enc = target_encode_kfold(
        train_fe.assign(**{str(col): train_fe[col].astype(str)}),
        test_fe.assign(**{str(col): test_fe[col].astype(str)}),
        col, 'demand'
    )
    train_fe[f'{col}_te'] = tr_enc
    test_fe[f'{col}_te'] = te_enc

# ============================================================
# Aggregate features (V1 style - from ALL data)
# ============================================================
print("Aggregate features...")

geo_stats = train.groupby('geohash')['demand'].agg(['mean','std','median','min','max','count'])
geo_stats.columns = ['geo_mean','geo_std','geo_median','geo_min','geo_max','geo_count']
geo_stats['geo_std'] = geo_stats['geo_std'].fillna(0)

train_fe = train_fe.merge(geo_stats, on='geohash', how='left')
test_fe = test_fe.merge(geo_stats, on='geohash', how='left')

for col in geo_stats.columns:
    fill = train['demand'].mean() if 'mean' in col or 'median' in col else 0
    if 'min' in col: fill = train['demand'].min()
    if 'max' in col: fill = train['demand'].max()
    if 'std' in col: fill = train['demand'].std()
    test_fe[col] = test_fe[col].fillna(fill)
    train_fe[col] = train_fe[col].fillna(fill)

# Geo x hour 
geo_hour = train.groupby(['geohash', train_fe['hour']])['demand'].agg(['mean','median']).reset_index()
geo_hour.columns = ['geohash','hour','geo_hour_mean','geo_hour_median']
train_fe = train_fe.merge(geo_hour, on=['geohash','hour'], how='left')
test_fe = test_fe.merge(geo_hour, on=['geohash','hour'], how='left')
train_fe['geo_hour_mean'] = train_fe['geo_hour_mean'].fillna(train['demand'].mean())
test_fe['geo_hour_mean'] = test_fe['geo_hour_mean'].fillna(train['demand'].mean())
train_fe['geo_hour_median'] = train_fe['geo_hour_median'].fillna(train['demand'].mean())
test_fe['geo_hour_median'] = test_fe['geo_hour_median'].fillna(train['demand'].mean())

# Road stats
road_stats = train_fe.groupby('RoadType_enc')['demand'].agg(['mean','std']).reset_index()
road_stats.columns = ['RoadType_enc','road_mean','road_std']
train_fe = train_fe.merge(road_stats, on='RoadType_enc', how='left')
test_fe = test_fe.merge(road_stats, on='RoadType_enc', how='left')
train_fe['road_mean'] = train_fe['road_mean'].fillna(train['demand'].mean())
test_fe['road_mean'] = test_fe['road_mean'].fillna(train['demand'].mean())
train_fe['road_std'] = train_fe['road_std'].fillna(train['demand'].std())
test_fe['road_std'] = test_fe['road_std'].fillna(train['demand'].std())

# ============================================================
# Feature columns (V1-style, ~57 features)
# ============================================================
feature_cols = [
    'day', 'minutes', 'hour', 'minute_of_hour', 'quarter_of_day',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'is_rush_morning', 'is_rush_evening', 'is_rush_hour', 'is_night', 'is_midday',
    'period_of_day',
    'geohash_le', 'geo_prefix_4_le', 'geo_prefix_5_le',
    'RoadType_enc', 'RoadType_missing', 'NumberofLanes',
    'LargeVehicles_enc', 'Landmarks_enc',
    'Weather_enc', 'Weather_missing',
    'Temperature', 'Temperature_missing', 'temp_squared', 'temp_abs', 'temp_bin',
    'road_lanes', 'road_large_vehicles', 'lanes_large_vehicles',
    'hour_road', 'hour_weather', 'road_weather', 'lanes_landmarks', 'road_capacity',
    'geohash_te', 'geo_prefix_4_te', 'geo_prefix_5_te',
    'RoadType_enc_te', 'Weather_enc_te', 'hour_te', 'quarter_of_day_te',
    'geo_mean', 'geo_std', 'geo_median', 'geo_min', 'geo_max', 'geo_count',
    'geo_hour_mean', 'geo_hour_median',
    'road_mean', 'road_std',
]

X_train = train_fe[feature_cols].values.astype(np.float32)
X_test = test_fe[feature_cols].values.astype(np.float32)
y_train = train['demand'].values

print(f"Features: {len(feature_cols)}")

# ============================================================
# Compute sample weights based on test timestamp distribution
# ============================================================
print("Computing sample weights...")

# Test timestamps are 2:15-23:45 (minutes 135-1425)
# Upweight training samples in this time range
train_minutes = train_fe['minutes'].values
test_minutes_set = set(test_fe['minutes'].unique())

# Weight = 1.5 if timestamp matches test range, 1.0 otherwise
weights = np.where(
    np.isin(train_minutes, list(test_minutes_set)), 
    1.5, 1.0
)
weights_norm = weights / weights.mean()

# ============================================================
# Train: RAW target + LOG1P target, blend results
# ============================================================
N_FOLDS = 5
SEEDS = [42, 123, 456, 789]

# Store predictions
raw_preds_all = []
log_preds_all = []

for SEED in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {SEED}")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # ---- RAW target ----
    # LightGBM (V1-style params)
    lgb_params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'learning_rate': 0.05, 'num_leaves': 127, 'max_depth': -1,
        'min_child_samples': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'verbose': -1, 'n_jobs': -1, 'seed': SEED,
    }
    
    lgb_preds = np.zeros(len(X_test))
    lgb_oof = np.zeros(len(X_train))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx], weight=weights_norm[tr_idx])
        dval = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
        model = lgb.train(lgb_params, dtrain, num_boost_round=3000,
                           valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        lgb_oof[val_idx] = model.predict(X_train[val_idx])
        lgb_preds += model.predict(X_test) / N_FOLDS
    print(f"  LGB raw R2: {r2_score(y_train, lgb_oof):.6f}")
    
    # XGBoost
    xgb_params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 10,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'tree_method': 'hist', 'seed': SEED, 'n_jobs': -1,
    }
    
    xgb_preds = np.zeros(len(X_test))
    xgb_oof = np.zeros(len(X_train))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx], weight=weights_norm[tr_idx])
        dval = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx])
        model = xgb.train(xgb_params, dtrain, num_boost_round=3000,
                           evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=0)
        xgb_oof[val_idx] = model.predict(dval)
        xgb_preds += model.predict(xgb.DMatrix(X_test)) / N_FOLDS
    print(f"  XGB raw R2: {r2_score(y_train, xgb_oof):.6f}")
    
    # CatBoost  
    cb_preds = np.zeros(len(X_test))
    cb_oof = np.zeros(len(X_train))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = CatBoostRegressor(
            iterations=3000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
            early_stopping_rounds=50, task_type='CPU',
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                   sample_weight=weights_norm[tr_idx],
                   eval_set=(X_train[val_idx], y_train[val_idx]), verbose=0)
        cb_oof[val_idx] = model.predict(X_train[val_idx])
        cb_preds += model.predict(X_test) / N_FOLDS
    print(f"  CB  raw R2: {r2_score(y_train, cb_oof):.6f}")
    
    # Blend raw (V1-style weights: LGB=0.25, XGB=0.20, CB=0.55)
    raw_blend = 0.25 * lgb_preds + 0.20 * xgb_preds + 0.55 * cb_preds
    raw_preds_all.append(raw_blend)
    
    # ---- LOG1P target ----
    y_log = np.log1p(y_train)
    
    lgb_params_log = lgb_params.copy()
    lgb_preds_log = np.zeros(len(X_test))
    lgb_oof_log = np.zeros(len(X_train))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = lgb.Dataset(X_train[tr_idx], label=y_log[tr_idx], weight=weights_norm[tr_idx])
        dval = lgb.Dataset(X_train[val_idx], label=y_log[val_idx])
        model = lgb.train(lgb_params_log, dtrain, num_boost_round=3000,
                           valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        lgb_oof_log[val_idx] = model.predict(X_train[val_idx])
        lgb_preds_log += model.predict(X_test) / N_FOLDS
    
    xgb_preds_log = np.zeros(len(X_test))
    xgb_oof_log = np.zeros(len(X_train))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_log[tr_idx], weight=weights_norm[tr_idx])
        dval = xgb.DMatrix(X_train[val_idx], label=y_log[val_idx])
        model = xgb.train(xgb_params, dtrain, num_boost_round=3000,
                           evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=0)
        xgb_oof_log[val_idx] = model.predict(dval)
        xgb_preds_log += model.predict(xgb.DMatrix(X_test)) / N_FOLDS
    
    cb_preds_log = np.zeros(len(X_test))
    cb_oof_log = np.zeros(len(X_train))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = CatBoostRegressor(
            iterations=3000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
            early_stopping_rounds=50, task_type='CPU',
        )
        model.fit(X_train[tr_idx], y_log[tr_idx],
                   sample_weight=weights_norm[tr_idx],
                   eval_set=(X_train[val_idx], y_log[val_idx]), verbose=0)
        cb_oof_log[val_idx] = model.predict(X_train[val_idx])
        cb_preds_log += model.predict(X_test) / N_FOLDS
    
    # Convert back from log space
    log_blend = 0.25 * np.expm1(lgb_preds_log) + 0.20 * np.expm1(xgb_preds_log) + 0.55 * np.expm1(cb_preds_log)
    
    oof_log_r2 = r2_score(y_train, 0.25*np.expm1(lgb_oof_log) + 0.20*np.expm1(xgb_oof_log) + 0.55*np.expm1(cb_oof_log))
    print(f"  Log blend OOF R2: {oof_log_r2:.6f}")
    
    log_preds_all.append(log_blend)

# ============================================================
# Also train WITHOUT sample weights (pure V1 replica)
# ============================================================
print(f"\n{'='*60}")
print("Training pure V1 replica (no weights)...")
print(f"{'='*60}")

pure_v1_preds = []
for SEED in SEEDS:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    lgb_params_v1 = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'learning_rate': 0.05, 'num_leaves': 127, 'max_depth': -1,
        'min_child_samples': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'verbose': -1, 'n_jobs': -1, 'seed': SEED,
    }
    
    lgb_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
        dval = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
        model = lgb.train(lgb_params_v1, dtrain, num_boost_round=3000,
                           valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        lgb_p += model.predict(X_test) / N_FOLDS
    
    xgb_params_v1 = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 10,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'tree_method': 'hist', 'seed': SEED, 'n_jobs': -1,
    }
    
    xgb_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx])
        dval = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx])
        model = xgb.train(xgb_params_v1, dtrain, num_boost_round=3000,
                           evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=0)
        xgb_p += model.predict(xgb.DMatrix(X_test)) / N_FOLDS
    
    cb_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = CatBoostRegressor(
            iterations=3000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
            early_stopping_rounds=50, task_type='CPU',
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                   eval_set=(X_train[val_idx], y_train[val_idx]), verbose=0)
        cb_p += model.predict(X_test) / N_FOLDS
    
    v1_blend = 0.25 * lgb_p + 0.20 * xgb_p + 0.55 * cb_p
    pure_v1_preds.append(v1_blend)
    print(f"  Seed {SEED} done")

# ============================================================
# Create ALL submission variants
# ============================================================
print(f"\n{'='*60}")
print("Creating submissions...")
print(f"{'='*60}")

# Load V1 original predictions
v1_original = pd.read_csv('/Users/hiya/Downloads/dataset/submission.csv')['demand'].values

# Average predictions
raw_avg = np.clip(np.mean(raw_preds_all, axis=0), 0, None)
log_avg = np.clip(np.mean(log_preds_all, axis=0), 0, None)
v1_multi = np.clip(np.mean(pure_v1_preds, axis=0), 0, None)

# Raw + Log blend
raw_log_blend = np.clip(0.5 * raw_avg + 0.5 * log_avg, 0, None)

# V1 multi-seed (should be most similar to original V1)
sub = pd.DataFrame({'Index': test['Index'], 'demand': v1_multi})
sub.to_csv('/Users/hiya/Downloads/dataset/submission_v5_v1multi.csv', index=False)
print("Saved v5_v1multi (V1 with 4 seeds)")

# Raw weighted
sub = pd.DataFrame({'Index': test['Index'], 'demand': raw_avg})
sub.to_csv('/Users/hiya/Downloads/dataset/submission_v5_weighted.csv', index=False)
print("Saved v5_weighted (V1 + timestamp weights)")

# Log transform
sub = pd.DataFrame({'Index': test['Index'], 'demand': log_avg})
sub.to_csv('/Users/hiya/Downloads/dataset/submission_v5_log.csv', index=False)
print("Saved v5_log (log1p target)")

# Raw + Log blend
sub = pd.DataFrame({'Index': test['Index'], 'demand': raw_log_blend})
sub.to_csv('/Users/hiya/Downloads/dataset/submission_v5_rawlog.csv', index=False)
print("Saved v5_rawlog (raw+log blend)")

# Blend V1_multi with V1_original
for alpha in [0.7, 0.5, 0.3]:
    blended = alpha * v1_multi + (1 - alpha) * v1_original
    sub = pd.DataFrame({'Index': test['Index'], 'demand': np.clip(blended, 0, None)})
    sub.to_csv(f'/Users/hiya/Downloads/dataset/submission_v5_v1blend_{int(alpha*100)}.csv', index=False)

# Grand blend: V1_original + V1_multi + weighted + log
grand = np.clip(0.3 * v1_original + 0.3 * v1_multi + 0.2 * raw_avg + 0.2 * log_avg, 0, None)
sub = pd.DataFrame({'Index': test['Index'], 'demand': grand})
sub.to_csv('/Users/hiya/Downloads/dataset/submission_v5_grand.csv', index=False)
print("Saved v5_grand (grand blend)")

# V1_original + log blend (different perspective)
for alpha in [0.2, 0.3, 0.4]:
    blended = (1 - alpha) * v1_original + alpha * log_avg
    sub = pd.DataFrame({'Index': test['Index'], 'demand': np.clip(blended, 0, None)})
    sub.to_csv(f'/Users/hiya/Downloads/dataset/submission_v5_v1log_{int(alpha*100)}.csv', index=False)

print(f"\n✅ All done!")
print("\nRecommended try order:")
print("1. submission_v5_v1multi.csv (V1 with 4-seed stability)")
print("2. submission_v5_grand.csv (grand blend)")
print("3. submission_v5_v1log_20.csv (V1 + 20% log)")
print("4. submission_v5_rawlog.csv (raw + log blend)")
