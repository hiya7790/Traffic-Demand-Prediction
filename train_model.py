"""
Traffic Demand Prediction - Full Pipeline
==========================================
Ensemble of LightGBM, XGBoost, CatBoost with extensive feature engineering.
Metric: max(0, 100 * r2_score(actual, predicted))
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv('/Users/hiya/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/hiya/Downloads/dataset/test.csv')

target = train['demand'].values
train_idx = train['Index'].values
test_idx = test['Index'].values

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. Feature Engineering
# ============================================================
print("Engineering features...")

def parse_timestamp(ts):
    """Convert 'H:M' format to minutes since midnight."""
    parts = ts.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def engineer_features(df):
    """Create features from raw data."""
    df = df.copy()
    
    # ---- Timestamp features ----
    df['minutes'] = df['timestamp'].apply(parse_timestamp)
    df['hour'] = df['minutes'] // 60
    df['minute_of_hour'] = df['minutes'] % 60
    df['quarter_of_day'] = df['minutes'] // 15  # 96 intervals
    
    # Cyclical time encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minutes'] / 1440)
    df['minute_cos'] = np.cos(2 * np.pi * df['minutes'] / 1440)
    
    # Time of day buckets
    df['is_rush_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_rush_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_rush_morning'] | df['is_rush_evening']).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
    
    # Period of day (categorical)
    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 6),
        (df['hour'] >= 6) & (df['hour'] < 10),
        (df['hour'] >= 10) & (df['hour'] < 14),
        (df['hour'] >= 14) & (df['hour'] < 18),
        (df['hour'] >= 18) & (df['hour'] < 22),
        (df['hour'] >= 22)
    ]
    choices = [0, 1, 2, 3, 4, 5]  # night, morning, midday, afternoon, evening, late_night
    df['period_of_day'] = np.select(conditions, choices, default=0)
    
    # ---- Geohash features ----
    # Extract geohash prefixes for hierarchical location
    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]
    df['geohash_len'] = df['geohash'].str.len()
    
    # ---- Categorical encoding ----
    # Binary features
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc'] = (df['Landmarks'] == 'Yes').astype(int)
    
    # RoadType encoding
    road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
    df['RoadType_enc'] = df['RoadType'].map(road_map)
    df['RoadType_missing'] = df['RoadType'].isna().astype(int)
    df['RoadType_enc'] = df['RoadType_enc'].fillna(-1)
    
    # Weather encoding  
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['Weather_enc'] = df['Weather'].map(weather_map)
    df['Weather_missing'] = df['Weather'].isna().astype(int)
    df['Weather_enc'] = df['Weather_enc'].fillna(-1)
    
    # ---- Temperature features ----
    df['Temperature_missing'] = df['Temperature'].isna().astype(int)
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['temp_squared'] = df['Temperature'] ** 2
    df['temp_abs'] = df['Temperature'].abs()
    
    # Temperature bins
    df['temp_bin'] = pd.cut(df['Temperature'], bins=10, labels=False)
    
    # Is extreme temperature
    df['is_cold'] = (df['Temperature'] < 5).astype(int)
    df['is_hot'] = (df['Temperature'] > 35).astype(int)
    
    # ---- Interaction features ----
    df['road_lanes'] = df['RoadType_enc'] * 10 + df['NumberofLanes']
    df['road_large_vehicles'] = df['RoadType_enc'] * 10 + df['LargeVehicles_enc']
    df['lanes_large_vehicles'] = df['NumberofLanes'] * 10 + df['LargeVehicles_enc']
    df['hour_road'] = df['hour'] * 10 + df['RoadType_enc']
    df['hour_weather'] = df['hour'] * 10 + df['Weather_enc']
    df['road_weather'] = df['RoadType_enc'] * 10 + df['Weather_enc']
    df['lanes_landmarks'] = df['NumberofLanes'] * 10 + df['Landmarks_enc']
    
    # Road capacity proxy
    df['road_capacity'] = df['NumberofLanes'] * (1 + df['LargeVehicles_enc'])
    
    return df


# Apply feature engineering
train_fe = engineer_features(train)
test_fe = engineer_features(test)

# ============================================================
# 3. Target encoding for high-cardinality categoricals
# ============================================================
print("Target encoding...")

# Label encode geohash columns for tree models
for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5']:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]], axis=0)
    le.fit(combined)
    train_fe[col + '_le'] = le.transform(train_fe[col])
    test_fe[col + '_le'] = le.transform(test_fe[col])

# Target encoding for geohash (with KFold to avoid leakage)
def target_encode_kfold(train_df, test_df, col, target_col, n_folds=5):
    """KFold target encoding to prevent data leakage."""
    train_encoded = np.zeros(len(train_df))
    global_mean = train_df[target_col].mean()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        means = train_df.iloc[tr_idx].groupby(col)[target_col].mean()
        train_encoded[val_idx] = train_df.iloc[val_idx][col].map(means)
    
    # Fill NaN with global mean
    train_encoded = np.where(np.isnan(train_encoded), global_mean, train_encoded)
    
    # For test, use all train data
    test_means = train_df.groupby(col)[target_col].mean()
    test_encoded = test_df[col].map(test_means).fillna(global_mean).values
    
    return train_encoded, test_encoded

# Target encode geohash and its prefixes
for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5']:
    tr_enc, te_enc = target_encode_kfold(train_fe, test_fe, col, 'demand')
    train_fe[f'{col}_target_enc'] = tr_enc
    test_fe[f'{col}_target_enc'] = te_enc

# Target encode for other categoricals
for col in ['RoadType_enc', 'Weather_enc', 'hour', 'quarter_of_day']:
    tr_enc, te_enc = target_encode_kfold(
        train_fe.assign(**{str(col): train_fe[col].astype(str)}), 
        test_fe.assign(**{str(col): test_fe[col].astype(str)}),
        col, 'demand'
    )
    train_fe[f'{col}_target_enc'] = tr_enc
    test_fe[f'{col}_target_enc'] = te_enc

# ============================================================
# 4. Aggregate / Statistical features per geohash from train
# ============================================================
print("Aggregate features...")

# Geohash-level statistics from training data
geo_stats = train.groupby('geohash')['demand'].agg(['mean', 'std', 'median', 'min', 'max', 'count'])
geo_stats.columns = ['geo_demand_mean', 'geo_demand_std', 'geo_demand_median', 
                       'geo_demand_min', 'geo_demand_max', 'geo_demand_count']
geo_stats['geo_demand_std'] = geo_stats['geo_demand_std'].fillna(0)

train_fe = train_fe.merge(geo_stats, on='geohash', how='left')
test_fe = test_fe.merge(geo_stats, on='geohash', how='left')

# Fill missing for test-only geohashes
for col in geo_stats.columns:
    global_val = train['demand'].agg(col.split('_')[-1]) if col.split('_')[-1] != 'count' else 0
    if col == 'geo_demand_std':
        global_val = train['demand'].std()
    elif col == 'geo_demand_count':
        global_val = 0
    elif col == 'geo_demand_mean':
        global_val = train['demand'].mean()
    elif col == 'geo_demand_median':
        global_val = train['demand'].median()
    elif col == 'geo_demand_min':
        global_val = train['demand'].min()
    elif col == 'geo_demand_max':
        global_val = train['demand'].max()
    test_fe[col] = test_fe[col].fillna(global_val)

# Geohash x hour stats
geo_hour_stats = train.groupby(['geohash', train_fe.loc[train_fe.index.isin(train.index), 'hour']])['demand'].agg(['mean', 'median'])
geo_hour_stats.columns = ['geo_hour_mean', 'geo_hour_median']
geo_hour_stats = geo_hour_stats.reset_index()

train_fe = train_fe.merge(geo_hour_stats, on=['geohash', 'hour'], how='left')
test_fe = test_fe.merge(geo_hour_stats, on=['geohash', 'hour'], how='left')
for col in ['geo_hour_mean', 'geo_hour_median']:
    train_fe[col] = train_fe[col].fillna(train['demand'].mean())
    test_fe[col] = test_fe[col].fillna(train['demand'].mean())

# Road-level statistics
road_stats = train_fe.groupby('RoadType_enc')['demand'].agg(['mean', 'std']).reset_index()
road_stats.columns = ['RoadType_enc', 'road_demand_mean', 'road_demand_std']
train_fe = train_fe.merge(road_stats, on='RoadType_enc', how='left')
test_fe = test_fe.merge(road_stats, on='RoadType_enc', how='left')
train_fe['road_demand_mean'] = train_fe['road_demand_mean'].fillna(train['demand'].mean())
test_fe['road_demand_mean'] = test_fe['road_demand_mean'].fillna(train['demand'].mean())
train_fe['road_demand_std'] = train_fe['road_demand_std'].fillna(train['demand'].std())
test_fe['road_demand_std'] = test_fe['road_demand_std'].fillna(train['demand'].std())

# ============================================================
# 5. Define feature columns
# ============================================================
feature_cols = [
    # Time features
    'day', 'minutes', 'hour', 'minute_of_hour', 'quarter_of_day',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'is_rush_morning', 'is_rush_evening', 'is_rush_hour', 'is_night', 'is_midday',
    'period_of_day',
    
    # Location features
    'geohash_le', 'geo_prefix_4_le', 'geo_prefix_5_le',
    
    # Road features
    'RoadType_enc', 'RoadType_missing', 'NumberofLanes', 
    'LargeVehicles_enc', 'Landmarks_enc',
    
    # Weather features
    'Weather_enc', 'Weather_missing',
    'Temperature', 'Temperature_missing', 'temp_squared', 'temp_abs',
    'temp_bin', 'is_cold', 'is_hot',
    
    # Interaction features
    'road_lanes', 'road_large_vehicles', 'lanes_large_vehicles',
    'hour_road', 'hour_weather', 'road_weather', 'lanes_landmarks',
    'road_capacity',
    
    # Target encoded features
    'geohash_target_enc', 'geo_prefix_4_target_enc', 'geo_prefix_5_target_enc',
    'RoadType_enc_target_enc', 'Weather_enc_target_enc',
    'hour_target_enc', 'quarter_of_day_target_enc',
    
    # Aggregate features
    'geo_demand_mean', 'geo_demand_std', 'geo_demand_median',
    'geo_demand_min', 'geo_demand_max', 'geo_demand_count',
    'geo_hour_mean', 'geo_hour_median',
    'road_demand_mean', 'road_demand_std',
]

X_train = train_fe[feature_cols].values
X_test = test_fe[feature_cols].values
y_train = target

print(f"Features: {len(feature_cols)}")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# 6. Model Training with KFold
# ============================================================
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ---- LightGBM ----
print("\n" + "="*60)
print("Training LightGBM...")
print("="*60)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

lgb_oof = np.zeros(len(X_train))
lgb_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)]
    )
    
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_preds += model.predict(X_test) / N_FOLDS
    
    fold_r2 = r2_score(y_val, lgb_oof[val_idx])
    print(f"  Fold {fold+1}: R2 = {fold_r2:.6f} (Score: {max(0, 100*fold_r2):.4f})")

lgb_r2 = r2_score(y_train, lgb_oof)
print(f"\nLightGBM OOF R2: {lgb_r2:.6f} (Score: {max(0, 100*lgb_r2):.4f})")

# ---- XGBoost ----
print("\n" + "="*60)
print("Training XGBoost...")
print("="*60)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
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
    
    model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=3000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=500
    )
    
    xgb_oof[val_idx] = model.predict(dval)
    xgb_preds += model.predict(dtest) / N_FOLDS
    
    fold_r2 = r2_score(y_val, xgb_oof[val_idx])
    print(f"  Fold {fold+1}: R2 = {fold_r2:.6f} (Score: {max(0, 100*fold_r2):.4f})")

xgb_r2 = r2_score(y_train, xgb_oof)
print(f"\nXGBoost OOF R2: {xgb_r2:.6f} (Score: {max(0, 100*xgb_r2):.4f})")

# ---- CatBoost ----
print("\n" + "="*60)
print("Training CatBoost...")
print("="*60)

cb_oof = np.zeros(len(X_train))
cb_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=500,
        early_stopping_rounds=50,
        task_type='CPU',
    )
    
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=500)
    
    cb_oof[val_idx] = model.predict(X_val)
    cb_preds += model.predict(X_test) / N_FOLDS
    
    fold_r2 = r2_score(y_val, cb_oof[val_idx])
    print(f"  Fold {fold+1}: R2 = {fold_r2:.6f} (Score: {max(0, 100*fold_r2):.4f})")

cb_r2 = r2_score(y_train, cb_oof)
print(f"\nCatBoost OOF R2: {cb_r2:.6f} (Score: {max(0, 100*cb_r2):.4f})")

# ============================================================
# 7. Ensemble - Optimize weights using OOF predictions
# ============================================================
print("\n" + "="*60)
print("Optimizing ensemble weights...")
print("="*60)

best_score = -1
best_weights = (1/3, 1/3, 1/3)

# Grid search over weight combinations
for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.8 - w1 + 0.05, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        oof_blend = w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof
        score = r2_score(y_train, oof_blend)
        if score > best_score:
            best_score = score
            best_weights = (w1, w2, w3)

print(f"Best weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, CB={best_weights[2]:.2f}")
print(f"Best ensemble OOF R2: {best_score:.6f} (Score: {max(0, 100*best_score):.4f})")

# Final predictions
final_preds = best_weights[0] * lgb_preds + best_weights[1] * xgb_preds + best_weights[2] * cb_preds

# Clip predictions to valid range
final_preds = np.clip(final_preds, 0, None)

# ============================================================
# 8. Create Submission
# ============================================================
print("\n" + "="*60)
print("Creating submission...")
print("="*60)

submission = pd.DataFrame({
    'Index': test_idx,
    'demand': final_preds
})

print(f"Submission shape: {submission.shape}")
print(submission.head(10))
print(f"\nPrediction stats:")
print(submission['demand'].describe())

submission.to_csv('/Users/hiya/Downloads/dataset/submission.csv', index=False)
print(f"\nSubmission saved to /Users/hiya/Downloads/dataset/submission.csv")

# Also save individual model predictions
sub_lgb = pd.DataFrame({'Index': test_idx, 'demand': lgb_preds})
sub_lgb.to_csv('/Users/hiya/Downloads/dataset/submission_lgb.csv', index=False)

sub_xgb = pd.DataFrame({'Index': test_idx, 'demand': xgb_preds})
sub_xgb.to_csv('/Users/hiya/Downloads/dataset/submission_xgb.csv', index=False)

sub_cb = pd.DataFrame({'Index': test_idx, 'demand': cb_preds})
sub_cb.to_csv('/Users/hiya/Downloads/dataset/submission_cb.csv', index=False)

print("\nAll done! ✅")
print(f"\nFinal Summary:")
print(f"  LightGBM OOF Score: {max(0, 100*lgb_r2):.4f}")
print(f"  XGBoost  OOF Score: {max(0, 100*xgb_r2):.4f}")
print(f"  CatBoost OOF Score: {max(0, 100*cb_r2):.4f}")
print(f"  Ensemble OOF Score: {max(0, 100*best_score):.4f}")
