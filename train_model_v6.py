"""
Traffic Demand Prediction - V6
===============================
Hypothesis:
Day 48 has full 24-hour coverage.
Day 49 train ONLY has 0:00 - 2:00 (night time).
Test set has 2:15 - 23:45 (day time).

If we train on Day 48 + Day 49, the model sees a HUGE block of night-time data right at the end of the training set. Tree-based models might learn that "day 49 is always low demand" or get confused by the sudden drop in variance for day 49.
What if we just train EXCLUSIVELY on Day 48? And drop Day 49 train completely?

Strategy:
1. Exact same feature engineering as V1 (which scored 89.2).
2. Train ONLY on Day 48 data.
3. Use a strong Multi-seed Ensemble (like V5).
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

# DROP DAY 49 TRAIN!
train_d48 = train[train['day'] == 48].reset_index(drop=True)
print(f"Original Train: {train.shape}")
print(f"Day 48 Train: {train_d48.shape}, Test: {test.shape}")

# ============================================================
# Feature Engineering (V1-style, minimal)
# ============================================================
print("Feature engineering...")

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
    
    # Interactions
    df['road_lanes'] = df['RoadType_enc'] * 10 + df['NumberofLanes']
    df['road_large_vehicles'] = df['RoadType_enc'] * 10 + df['LargeVehicles_enc']
    df['lanes_large_vehicles'] = df['NumberofLanes'] * 10 + df['LargeVehicles_enc']
    df['hour_road'] = df['hour'] * 10 + df['RoadType_enc']
    df['hour_weather'] = df['hour'] * 10 + df['Weather_enc']
    df['road_weather'] = df['RoadType_enc'] * 10 + df['Weather_enc']
    df['lanes_landmarks'] = df['NumberofLanes'] * 10 + df['Landmarks_enc']
    df['road_capacity'] = df['NumberofLanes'] * (1 + df['LargeVehicles_enc'])
    
    return df

train_fe = engineer_features(train_d48) # Train ONLY on D48!
test_fe = engineer_features(test)

temp_median = train_fe['Temperature'].median()
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
# KFold Target Encoding
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
# Aggregate features
# ============================================================
print("Aggregate features...")

geo_stats = train_d48.groupby('geohash')['demand'].agg(['mean','std','median','min','max','count'])
geo_stats.columns = ['geo_mean','geo_std','geo_median','geo_min','geo_max','geo_count']
geo_stats['geo_std'] = geo_stats['geo_std'].fillna(0)

train_fe = train_fe.merge(geo_stats, on='geohash', how='left')
test_fe = test_fe.merge(geo_stats, on='geohash', how='left')

global_d48_mean = train_d48['demand'].mean()
global_d48_std = train_d48['demand'].std()
global_d48_min = train_d48['demand'].min()
global_d48_max = train_d48['demand'].max()

for col in geo_stats.columns:
    fill = global_d48_mean if 'mean' in col or 'median' in col else 0
    if 'min' in col: fill = global_d48_min
    if 'max' in col: fill = global_d48_max
    if 'std' in col: fill = global_d48_std
    test_fe[col] = test_fe[col].fillna(fill)
    train_fe[col] = train_fe[col].fillna(fill)

# Geo x hour 
geo_hour = train_d48.groupby(['geohash', train_fe['hour']])['demand'].agg(['mean','median']).reset_index()
geo_hour.columns = ['geohash','hour','geo_hour_mean','geo_hour_median']
train_fe = train_fe.merge(geo_hour, on=['geohash','hour'], how='left')
test_fe = test_fe.merge(geo_hour, on=['geohash','hour'], how='left')
train_fe['geo_hour_mean'] = train_fe['geo_hour_mean'].fillna(global_d48_mean)
test_fe['geo_hour_mean'] = test_fe['geo_hour_mean'].fillna(global_d48_mean)
train_fe['geo_hour_median'] = train_fe['geo_hour_median'].fillna(global_d48_mean)
test_fe['geo_hour_median'] = test_fe['geo_hour_median'].fillna(global_d48_mean)

# Road stats
road_stats = train_fe.groupby('RoadType_enc')['demand'].agg(['mean','std']).reset_index()
road_stats.columns = ['RoadType_enc','road_mean','road_std']
train_fe = train_fe.merge(road_stats, on='RoadType_enc', how='left')
test_fe = test_fe.merge(road_stats, on='RoadType_enc', how='left')
train_fe['road_mean'] = train_fe['road_mean'].fillna(global_d48_mean)
test_fe['road_mean'] = test_fe['road_mean'].fillna(global_d48_mean)
train_fe['road_std'] = train_fe['road_std'].fillna(global_d48_std)
test_fe['road_std'] = test_fe['road_std'].fillna(global_d48_std)

# ============================================================
# Feature columns
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
y_train = train_d48['demand'].values

print(f"Features: {len(feature_cols)}")

# ============================================================
# Model Training
# ============================================================
N_FOLDS = 5
SEEDS = [42, 123, 456, 789]

preds_all = []

for SEED in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {SEED}")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # LGBM
    lgb_params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'learning_rate': 0.05, 'num_leaves': 127, 'max_depth': -1,
        'min_child_samples': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        'verbose': -1, 'n_jobs': -1, 'seed': SEED,
    }
    
    lgb_preds = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
        dval = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
        model = lgb.train(lgb_params, dtrain, num_boost_round=3000,
                           valid_sets=[dval], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        lgb_preds += model.predict(X_test) / N_FOLDS
        
    # XGB
    xgb_params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'learning_rate': 0.05, 'max_depth': 8, 'min_child_weight': 10,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'tree_method': 'hist', 'seed': SEED, 'n_jobs': -1,
    }
    
    xgb_preds = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx])
        dval = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx])
        model = xgb.train(xgb_params, dtrain, num_boost_round=3000,
                           evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=0)
        xgb_preds += model.predict(xgb.DMatrix(X_test)) / N_FOLDS
        
    # CatBoost
    cb_preds = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = CatBoostRegressor(
            iterations=3000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
            early_stopping_rounds=50, task_type='CPU',
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                   eval_set=(X_train[val_idx], y_train[val_idx]), verbose=0)
        cb_preds += model.predict(X_test) / N_FOLDS
        
    blend = 0.25 * lgb_preds + 0.20 * xgb_preds + 0.55 * cb_preds
    preds_all.append(blend)

# ============================================================
# Create Submissions
# ============================================================
print(f"\n{'='*60}")
print("Creating Submissions...")
print(f"{'='*60}")

final_preds = np.clip(np.mean(preds_all, axis=0), 0, None)
sub = pd.DataFrame({'Index': test['Index'], 'demand': final_preds})
sub.to_csv('/Users/hiya/Downloads/dataset/submission_v6_d48only.csv', index=False)

# Blend with previous best (V5 V1 Log 20%)
v5_best = pd.read_csv('/Users/hiya/Downloads/dataset/submission_v5_v1log_20.csv')['demand'].values

for alpha in [0.3, 0.5, 0.7]:
    blended = alpha * final_preds + (1-alpha) * v5_best
    sub = pd.DataFrame({'Index': test['Index'], 'demand': blended})
    sub.to_csv(f'/Users/hiya/Downloads/dataset/submission_v6_blend_{int(alpha*100)}.csv', index=False)

print("✅ Saved v6_d48only.csv")
print("✅ Saved v6 blends with v5_best")
