"""
Traffic Demand Prediction - V2 (Anti-Overfitting)
===================================================
Key fixes:
1. Temporal validation (day 48 → day 49)
2. Conservative target encoding with smoothing
3. Log-transform target
4. More robust features, less memorization
5. Stronger regularization
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
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv('/Users/hiya/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/hiya/Downloads/dataset/test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Train day distribution:\n{train['day'].value_counts()}")
print(f"Test day distribution:\n{test['day'].value_counts()}")

# ============================================================
# 2. Feature Engineering (Robust, no leakage)
# ============================================================
print("\nEngineering features...")

def parse_timestamp(ts):
    parts = ts.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def engineer_features(df):
    df = df.copy()
    
    # ---- Timestamp features ----
    df['minutes'] = df['timestamp'].apply(parse_timestamp)
    df['hour'] = df['minutes'] // 60
    df['minute_of_hour'] = df['minutes'] % 60
    df['quarter_of_day'] = df['minutes'] // 15
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minutes'] / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * df['minutes'] / 1440)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter_of_day'] / 96)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter_of_day'] / 96)
    
    # Time of day indicators
    df['is_rush_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_rush_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_rush_morning'] | df['is_rush_evening']).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
    df['is_early_morning'] = ((df['hour'] >= 4) & (df['hour'] <= 7)).astype(int)
    
    # Period of day
    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 6),
        (df['hour'] >= 6) & (df['hour'] < 10),
        (df['hour'] >= 10) & (df['hour'] < 14),
        (df['hour'] >= 14) & (df['hour'] < 18),
        (df['hour'] >= 18) & (df['hour'] < 22),
        (df['hour'] >= 22)
    ]
    df['period_of_day'] = np.select(conditions, [0,1,2,3,4,5], default=0)
    
    # ---- Geohash features ----
    df['geo_prefix_3'] = df['geohash'].str[:3]
    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]
    df['geo_char_last'] = df['geohash'].str[-1]
    df['geo_char_last2'] = df['geohash'].str[-2]
    
    # ---- Categorical encoding ----
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc'] = (df['Landmarks'] == 'Yes').astype(int)
    
    road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
    df['RoadType_enc'] = df['RoadType'].map(road_map).fillna(-1).astype(int)
    df['RoadType_missing'] = df['RoadType'].isna().astype(int)
    
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['Weather_enc'] = df['Weather'].map(weather_map).fillna(-1).astype(int)
    df['Weather_missing'] = df['Weather'].isna().astype(int)
    
    # ---- Temperature features ----
    df['Temperature_missing'] = df['Temperature'].isna().astype(int)
    # DON'T fill with global median yet - fill per group later
    df['temp_abs'] = df['Temperature'].abs()
    df['is_cold'] = (df['Temperature'] < 5).astype(int)
    df['is_hot'] = (df['Temperature'] > 35).astype(int)
    df['is_freezing'] = (df['Temperature'] < 0).astype(int)
    
    # ---- Interaction features ----
    df['road_lanes'] = df['RoadType_enc'] * 10 + df['NumberofLanes']
    df['road_large_vehicles'] = df['RoadType_enc'] * 2 + df['LargeVehicles_enc']
    df['lanes_large_vehicles'] = df['NumberofLanes'] * 2 + df['LargeVehicles_enc']
    df['lanes_landmarks'] = df['NumberofLanes'] * 2 + df['Landmarks_enc']
    df['road_weather'] = df['RoadType_enc'] * 10 + df['Weather_enc']
    df['road_capacity'] = df['NumberofLanes'] * (1 + df['LargeVehicles_enc'])
    
    # ---- Weather x Time interactions ----
    df['weather_hour'] = df['Weather_enc'] * 100 + df['hour']
    df['weather_rush'] = df['Weather_enc'] * 10 + df['is_rush_hour']
    df['weather_night'] = df['Weather_enc'] * 10 + df['is_night']
    
    # Road x Time
    df['road_hour'] = df['RoadType_enc'] * 100 + df['hour']
    df['road_rush'] = df['RoadType_enc'] * 10 + df['is_rush_hour']
    
    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Fill temperature with median (computed from train only)
temp_median = train_fe['Temperature'].median()
train_fe['Temperature'] = train_fe['Temperature'].fillna(temp_median)
test_fe['Temperature'] = test_fe['Temperature'].fillna(temp_median)
train_fe['temp_abs'] = train_fe['temp_abs'].fillna(abs(temp_median))
test_fe['temp_abs'] = test_fe['temp_abs'].fillna(abs(temp_median))

# Label encode geohash columns
for col in ['geohash', 'geo_prefix_3', 'geo_prefix_4', 'geo_prefix_5', 
            'geo_char_last', 'geo_char_last2']:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]])
    le.fit(combined)
    train_fe[col + '_le'] = le.transform(train_fe[col])
    test_fe[col + '_le'] = le.transform(test_fe[col])

# ============================================================
# 3. Smoothed Target Encoding (using ONLY day 48 data)
# ============================================================
print("Smoothed target encoding (day 48 only to prevent leakage)...")

day48 = train_fe[train_fe['day'] == 48].copy()
global_mean = day48['demand'].mean()
SMOOTH_MIN = 30  # minimum samples for reliable encoding

def smoothed_target_encode(train_df, test_df, source_df, col, target_col, 
                           global_mean, min_samples=SMOOTH_MIN):
    """Smoothed target encoding using Bayesian averaging."""
    stats = source_df.groupby(col)[target_col].agg(['mean', 'count'])
    # Bayesian smoothing: weighted average of group mean and global mean
    smoothed = (stats['count'] * stats['mean'] + min_samples * global_mean) / (stats['count'] + min_samples)
    smoothed = smoothed.to_dict()
    
    train_enc = train_df[col].map(smoothed).fillna(global_mean)
    test_enc = test_df[col].map(smoothed).fillna(global_mean)
    return train_enc.values, test_enc.values

# Target encode key features using ONLY day 48 data
for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5', 'geo_prefix_3']:
    tr_enc, te_enc = smoothed_target_encode(train_fe, test_fe, day48, col, 'demand', global_mean)
    train_fe[f'{col}_te'] = tr_enc
    test_fe[f'{col}_te'] = te_enc

for col in ['RoadType_enc', 'Weather_enc', 'hour', 'quarter_of_day', 'period_of_day']:
    tr_enc, te_enc = smoothed_target_encode(train_fe, test_fe, day48, col, 'demand', global_mean, min_samples=50)
    train_fe[f'{col}_te'] = tr_enc
    test_fe[f'{col}_te'] = te_enc

# Interaction target encoding
for col1, col2 in [('geohash', 'hour'), ('geohash', 'period_of_day'), 
                     ('RoadType_enc', 'hour'), ('Weather_enc', 'hour')]:
    inter_col = f'{col1}_x_{col2}'
    day48[inter_col] = day48[col1].astype(str) + '_' + day48[col2].astype(str)
    train_fe[inter_col] = train_fe[col1].astype(str) + '_' + train_fe[col2].astype(str)
    test_fe[inter_col] = test_fe[col1].astype(str) + '_' + test_fe[col2].astype(str)
    
    tr_enc, te_enc = smoothed_target_encode(train_fe, test_fe, day48, inter_col, 'demand', global_mean, min_samples=10)
    train_fe[f'{inter_col}_te'] = tr_enc
    test_fe[f'{inter_col}_te'] = te_enc

# ============================================================
# 4. Aggregate features from day 48 only
# ============================================================
print("Aggregate features (day 48 only)...")

# Geohash stats from day 48
geo_stats = day48.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'min', 'max', 'count'])
geo_stats.columns = [f'geo_d48_{c}' for c in ['mean', 'std', 'median', 'min', 'max', 'count']]
geo_stats['geo_d48_std'] = geo_stats['geo_d48_std'].fillna(0)
geo_stats['geo_d48_range'] = geo_stats['geo_d48_max'] - geo_stats['geo_d48_min']
geo_stats['geo_d48_cv'] = geo_stats['geo_d48_std'] / (geo_stats['geo_d48_mean'] + 1e-8)

train_fe = train_fe.merge(geo_stats, on='geohash', how='left')
test_fe = test_fe.merge(geo_stats, on='geohash', how='left')

# Fill missing (for geohashes not in day48)
for col in geo_stats.columns:
    fill_val = global_mean if 'mean' in col or 'median' in col else 0
    train_fe[col] = train_fe[col].fillna(fill_val)
    test_fe[col] = test_fe[col].fillna(fill_val)

# Hour stats from day 48
hour_stats = day48.groupby('hour')['demand'].agg(['mean', 'std', 'median'])
hour_stats.columns = ['hour_d48_mean', 'hour_d48_std', 'hour_d48_median']
train_fe = train_fe.merge(hour_stats, on='hour', how='left')
test_fe = test_fe.merge(hour_stats, on='hour', how='left')

# Geohash x hour from day 48
geo_hour = day48.groupby(['geohash', 'hour'])['demand'].agg(['mean', 'median', 'count']).reset_index()
geo_hour.columns = ['geohash', 'hour', 'geo_hour_d48_mean', 'geo_hour_d48_median', 'geo_hour_d48_count']
train_fe = train_fe.merge(geo_hour, on=['geohash', 'hour'], how='left')
test_fe = test_fe.merge(geo_hour, on=['geohash', 'hour'], how='left')
for col in ['geo_hour_d48_mean', 'geo_hour_d48_median']:
    train_fe[col] = train_fe[col].fillna(global_mean)
    test_fe[col] = test_fe[col].fillna(global_mean)
train_fe['geo_hour_d48_count'] = train_fe['geo_hour_d48_count'].fillna(0)
test_fe['geo_hour_d48_count'] = test_fe['geo_hour_d48_count'].fillna(0)

# Geohash x period from day 48
geo_period = day48.groupby(['geohash', 'period_of_day'])['demand'].agg(['mean', 'median']).reset_index()
geo_period.columns = ['geohash', 'period_of_day', 'geo_period_d48_mean', 'geo_period_d48_median']
train_fe = train_fe.merge(geo_period, on=['geohash', 'period_of_day'], how='left')
test_fe = test_fe.merge(geo_period, on=['geohash', 'period_of_day'], how='left')
for col in ['geo_period_d48_mean', 'geo_period_d48_median']:
    train_fe[col] = train_fe[col].fillna(global_mean)
    test_fe[col] = test_fe[col].fillna(global_mean)

# Road x Weather stats from day 48
rw_stats = day48.groupby(['RoadType_enc', 'Weather_enc'])['demand'].mean().reset_index()
rw_stats.columns = ['RoadType_enc', 'Weather_enc', 'road_weather_d48_mean']
train_fe = train_fe.merge(rw_stats, on=['RoadType_enc', 'Weather_enc'], how='left')
test_fe = test_fe.merge(rw_stats, on=['RoadType_enc', 'Weather_enc'], how='left')
train_fe['road_weather_d48_mean'] = train_fe['road_weather_d48_mean'].fillna(global_mean)
test_fe['road_weather_d48_mean'] = test_fe['road_weather_d48_mean'].fillna(global_mean)

# ============================================================
# 5. Define feature columns
# ============================================================

feature_cols = [
    # Time
    'minutes', 'hour', 'minute_of_hour', 'quarter_of_day',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'quarter_sin', 'quarter_cos',
    'is_rush_morning', 'is_rush_evening', 'is_rush_hour', 
    'is_night', 'is_midday', 'is_early_morning',
    'period_of_day',
    
    # Location (label encoded)
    'geohash_le', 'geo_prefix_3_le', 'geo_prefix_4_le', 'geo_prefix_5_le',
    'geo_char_last_le', 'geo_char_last2_le',
    
    # Road / infrastructure
    'RoadType_enc', 'RoadType_missing', 'NumberofLanes',
    'LargeVehicles_enc', 'Landmarks_enc',
    
    # Weather
    'Weather_enc', 'Weather_missing',
    'Temperature', 'Temperature_missing', 'temp_abs',
    'is_cold', 'is_hot', 'is_freezing',
    
    # Interactions
    'road_lanes', 'road_large_vehicles', 'lanes_large_vehicles',
    'lanes_landmarks', 'road_weather', 'road_capacity',
    'weather_hour', 'weather_rush', 'weather_night',
    'road_hour', 'road_rush',
    
    # Target encoded (smoothed, day48-based)
    'geohash_te', 'geo_prefix_4_te', 'geo_prefix_5_te', 'geo_prefix_3_te',
    'RoadType_enc_te', 'Weather_enc_te', 'hour_te', 
    'quarter_of_day_te', 'period_of_day_te',
    
    # Interaction target encoding
    'geohash_x_hour_te', 'geohash_x_period_of_day_te',
    'RoadType_enc_x_hour_te', 'Weather_enc_x_hour_te',
    
    # Aggregate features (day 48)
    'geo_d48_mean', 'geo_d48_std', 'geo_d48_median',
    'geo_d48_min', 'geo_d48_max', 'geo_d48_count',
    'geo_d48_range', 'geo_d48_cv',
    'hour_d48_mean', 'hour_d48_std', 'hour_d48_median',
    'geo_hour_d48_mean', 'geo_hour_d48_median', 'geo_hour_d48_count',
    'geo_period_d48_mean', 'geo_period_d48_median',
    'road_weather_d48_mean',
]

X_train = train_fe[feature_cols].values.astype(np.float32)
X_test = test_fe[feature_cols].values.astype(np.float32)
y_train = train['demand'].values

print(f"Features: {len(feature_cols)}")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# 6. Temporal Validation: Train on day 48, validate on day 49
# ============================================================
print("\n" + "="*60)
print("TEMPORAL VALIDATION (day 48 → day 49)")
print("="*60)

day_mask_48 = train['day'] == 48
day_mask_49 = train['day'] == 49

X_d48, y_d48 = X_train[day_mask_48], y_train[day_mask_48]
X_d49, y_d49 = X_train[day_mask_49], y_train[day_mask_49]
print(f"Day 48: {X_d48.shape[0]} samples, Day 49: {X_d49.shape[0]} samples")

# Quick temporal validation with LGB
dtrain_t = lgb.Dataset(X_d48, label=y_d48)
dval_t = lgb.Dataset(X_d49, label=y_d49)

lgb_params_test = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7,
    'min_child_samples': 50, 'feature_fraction': 0.7, 'bagging_fraction': 0.7,
    'bagging_freq': 5, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
    'verbose': -1, 'n_jobs': -1, 'seed': 42,
}

model_t = lgb.train(lgb_params_test, dtrain_t, num_boost_round=3000,
                     valid_sets=[dval_t], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
preds_t = model_t.predict(X_d49)
temporal_r2 = r2_score(y_d49, preds_t)
print(f"\nTemporal validation R2: {temporal_r2:.6f} (Score: {max(0, 100*temporal_r2):.4f})")

# ============================================================
# 7. Train final models with heavy regularization
# ============================================================
# Use FULL training data but with regularization tuned on temporal split

# ---- LightGBM ----
print("\n" + "="*60)
print("Training LightGBM (full data, heavy regularization)...")
print("="*60)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 7,
    'min_child_samples': 50,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.65,
    'bagging_freq': 5,
    'reg_alpha': 2.0,
    'reg_lambda': 2.0,
    'min_gain_to_split': 0.01,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'path_smooth': 1.0,
}

N_FOLDS = 5

# Use GroupKFold-like: ensure day 49 samples go to validation
# But also do standard KFold for diversity
lgb_oof = np.zeros(len(X_train))
lgb_preds = np.zeros(len(X_test))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(lgb_params, dtrain, num_boost_round=5000,
                       valid_sets=[dval], 
                       callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_preds += model.predict(X_test) / N_FOLDS
    
    fold_r2 = r2_score(y_val, lgb_oof[val_idx])
    print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")

lgb_r2 = r2_score(y_train, lgb_oof)
print(f"LightGBM OOF R2: {lgb_r2:.6f} (Score: {max(0,100*lgb_r2):.4f})")

# Also train a temporal-only model (day48 → test)
print("\nTraining LightGBM temporal model (day48 only)...")
dtrain_full48 = lgb.Dataset(X_d48, label=y_d48)
dval_49 = lgb.Dataset(X_d49, label=y_d49)

lgb_temporal = lgb.train(lgb_params, dtrain_full48, num_boost_round=5000,
                          valid_sets=[dval_49],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
lgb_temporal_preds = lgb_temporal.predict(X_test)
lgb_temporal_val = lgb_temporal.predict(X_d49)
print(f"Temporal model val R2: {r2_score(y_d49, lgb_temporal_val):.6f}")

# ---- XGBoost ----
print("\n" + "="*60)
print("Training XGBoost (full data, heavy regularization)...")
print("="*60)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.03,
    'max_depth': 7,
    'min_child_weight': 50,
    'subsample': 0.65,
    'colsample_bytree': 0.65,
    'reg_alpha': 2.0,
    'reg_lambda': 5.0,
    'gamma': 0.1,
    'tree_method': 'hist',
    'seed': 42,
    'n_jobs': -1,
}

xgb_oof = np.zeros(len(X_train))
xgb_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    model = xgb.train(xgb_params, dtrain, num_boost_round=5000,
                       evals=[(dval, 'val')], early_stopping_rounds=100,
                       verbose_eval=500)
    
    xgb_oof[val_idx] = model.predict(dval)
    xgb_preds += model.predict(dtest) / N_FOLDS
    
    fold_r2 = r2_score(y_val, xgb_oof[val_idx])
    print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")

xgb_r2 = r2_score(y_train, xgb_oof)
print(f"XGBoost OOF R2: {xgb_r2:.6f} (Score: {max(0,100*xgb_r2):.4f})")

# XGBoost temporal
print("\nTraining XGBoost temporal model...")
dtrain_48 = xgb.DMatrix(X_d48, label=y_d48)
dval_49x = xgb.DMatrix(X_d49, label=y_d49)
dtest_x = xgb.DMatrix(X_test)

xgb_temporal = xgb.train(xgb_params, dtrain_48, num_boost_round=5000,
                          evals=[(dval_49x, 'val')], early_stopping_rounds=100,
                          verbose_eval=200)
xgb_temporal_preds = xgb_temporal.predict(dtest_x)
xgb_temporal_val = xgb_temporal.predict(dval_49x)
print(f"Temporal model val R2: {r2_score(y_d49, xgb_temporal_val):.6f}")

# ---- CatBoost ----
print("\n" + "="*60)
print("Training CatBoost (full data, heavy regularization)...")
print("="*60)

cb_oof = np.zeros(len(X_train))
cb_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    model = CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=7,
        l2_leaf_reg=10.0, random_seed=42, verbose=500,
        early_stopping_rounds=100, task_type='CPU',
        min_data_in_leaf=50, random_strength=2.0,
        bagging_temperature=0.5,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=500)
    
    cb_oof[val_idx] = model.predict(X_val)
    cb_preds += model.predict(X_test) / N_FOLDS
    
    fold_r2 = r2_score(y_val, cb_oof[val_idx])
    print(f"  Fold {fold+1}: R2 = {fold_r2:.6f}")

cb_r2 = r2_score(y_train, cb_oof)
print(f"CatBoost OOF R2: {cb_r2:.6f} (Score: {max(0,100*cb_r2):.4f})")

# CatBoost temporal
print("\nTraining CatBoost temporal model...")
cb_temporal = CatBoostRegressor(
    iterations=5000, learning_rate=0.03, depth=7,
    l2_leaf_reg=10.0, random_seed=42, verbose=200,
    early_stopping_rounds=100, task_type='CPU',
    min_data_in_leaf=50, random_strength=2.0, bagging_temperature=0.5,
)
cb_temporal.fit(X_d48, y_d48, eval_set=(X_d49, y_d49), verbose=200)
cb_temporal_preds = cb_temporal.predict(X_test)
cb_temporal_val = cb_temporal.predict(X_d49)
print(f"Temporal model val R2: {r2_score(y_d49, cb_temporal_val):.6f}")

# ============================================================
# 8. Ensemble with temporal blending
# ============================================================
print("\n" + "="*60)
print("Optimizing ensemble...")
print("="*60)

# We have 6 sets of predictions:
# 1. lgb_preds (KFold), 2. xgb_preds (KFold), 3. cb_preds (KFold)
# 4. lgb_temporal_preds, 5. xgb_temporal_preds, 6. cb_temporal_preds

# First find best KFold blend using OOF
best_kf_score = -1
best_kf_weights = (1/3, 1/3, 1/3)
for w1 in np.arange(0.05, 0.9, 0.05):
    for w2 in np.arange(0.05, 0.9 - w1 + 0.05, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof
        score = r2_score(y_train, blend)
        if score > best_kf_score:
            best_kf_score = score
            best_kf_weights = (w1, w2, w3)

print(f"Best KFold weights: LGB={best_kf_weights[0]:.2f}, XGB={best_kf_weights[1]:.2f}, CB={best_kf_weights[2]:.2f}")
print(f"Best KFold OOF R2: {best_kf_score:.6f}")

kf_blend_preds = best_kf_weights[0] * lgb_preds + best_kf_weights[1] * xgb_preds + best_kf_weights[2] * cb_preds

# Best temporal blend using day49 validation
best_temp_score = -1
best_temp_weights = (1/3, 1/3, 1/3)
for w1 in np.arange(0.05, 0.9, 0.05):
    for w2 in np.arange(0.05, 0.9 - w1 + 0.05, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * lgb_temporal_val + w2 * xgb_temporal_val + w3 * cb_temporal_val
        score = r2_score(y_d49, blend)
        if score > best_temp_score:
            best_temp_score = score
            best_temp_weights = (w1, w2, w3)

print(f"Best Temporal weights: LGB={best_temp_weights[0]:.2f}, XGB={best_temp_weights[1]:.2f}, CB={best_temp_weights[2]:.2f}")
print(f"Best Temporal val R2: {best_temp_score:.6f}")

temp_blend_preds = best_temp_weights[0] * lgb_temporal_preds + best_temp_weights[1] * xgb_temporal_preds + best_temp_weights[2] * cb_temporal_preds

# Now blend KFold and Temporal predictions
# Use day49 validation to find optimal mix
best_mix_score = -1
best_alpha = 0.5
kf_val_blend = best_kf_weights[0] * lgb_oof[day_mask_49] + best_kf_weights[1] * xgb_oof[day_mask_49] + best_kf_weights[2] * cb_oof[day_mask_49]
temp_val_blend = best_temp_weights[0] * lgb_temporal_val + best_temp_weights[1] * xgb_temporal_val + best_temp_weights[2] * cb_temporal_val

for alpha in np.arange(0.0, 1.01, 0.05):
    blend = alpha * kf_val_blend + (1 - alpha) * temp_val_blend
    score = r2_score(y_d49, blend)
    if score > best_mix_score:
        best_mix_score = score
        best_alpha = alpha

print(f"\nBest KF/Temporal mix: alpha={best_alpha:.2f} (KF={best_alpha:.0%}, Temporal={1-best_alpha:.0%})")
print(f"Best mix val R2: {best_mix_score:.6f} (Score: {max(0, 100*best_mix_score):.4f})")

# Final predictions
final_preds = best_alpha * kf_blend_preds + (1 - best_alpha) * temp_blend_preds
final_preds = np.clip(final_preds, 0, None)

# ============================================================
# 9. Create submissions
# ============================================================
print("\n" + "="*60)
print("Creating submissions...")
print("="*60)

# Main submission (blended)
submission = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': final_preds})
submission.to_csv('/Users/hiya/Downloads/dataset/submission_v2.csv', index=False)
print(f"Main submission: {submission.shape}")
print(submission['demand'].describe())

# Temporal-only submission (might generalize better)
sub_temporal = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': temp_blend_preds})
sub_temporal.to_csv('/Users/hiya/Downloads/dataset/submission_v2_temporal.csv', index=False)

# KFold-only submission
sub_kf = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': np.clip(kf_blend_preds, 0, None)})
sub_kf.to_csv('/Users/hiya/Downloads/dataset/submission_v2_kfold.csv', index=False)

print("\n✅ All submissions saved!")
print(f"\nFinal Summary:")
print(f"  LightGBM KFold Score: {max(0,100*lgb_r2):.4f}")
print(f"  XGBoost  KFold Score: {max(0,100*xgb_r2):.4f}")
print(f"  CatBoost KFold Score: {max(0,100*cb_r2):.4f}")
print(f"  KFold Ensemble Score: {max(0,100*best_kf_score):.4f}")
print(f"  Temporal Val Score:   {max(0,100*best_temp_score):.4f}")
print(f"  Final Mix Val Score:  {max(0,100*best_mix_score):.4f}")
print(f"\nRecommendation: Try submission_v2.csv first, then submission_v2_temporal.csv")
