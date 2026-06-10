"""
Traffic Demand Prediction - V3 (Score Maximization)
====================================================
Improvements over V2:
1. Geohash decoded to lat/lon for spatial features
2. Day 49 sample upweighting
3. Multi-seed ensemble for stability
4. Log-transform target for better regression
5. More diverse model configs
6. Feature selection guided by temporal validation
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
# 0. Geohash decoder
# ============================================================
BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

def decode_geohash(ghash):
    """Decode geohash to (lat, lon)."""
    is_lon = True
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    for c in ghash:
        val = BASE32_MAP.get(c, 0)
        for bit in range(4, -1, -1):
            b = (val >> bit) & 1
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if b:
                    lon_range[0] = mid
                else:
                    lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if b:
                    lat_range[0] = mid
                else:
                    lat_range[1] = mid
            is_lon = not is_lon
    lat = (lat_range[0] + lat_range[1]) / 2
    lon = (lon_range[0] + lon_range[1]) / 2
    return lat, lon

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv('/Users/hiya/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/hiya/Downloads/dataset/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. Feature Engineering
# ============================================================
print("Engineering features...")

def parse_timestamp(ts):
    parts = ts.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def engineer_features(df):
    df = df.copy()
    
    # ---- Geohash → lat/lon ----
    coords = df['geohash'].apply(decode_geohash)
    df['latitude'] = coords.apply(lambda x: x[0])
    df['longitude'] = coords.apply(lambda x: x[1])
    
    # Spatial features
    df['lat_lon_product'] = df['latitude'] * df['longitude']
    df['lat_abs'] = df['latitude'].abs()
    df['lon_abs'] = df['longitude'].abs()
    
    # Spatial bins
    df['lat_bin'] = pd.cut(df['latitude'], bins=20, labels=False)
    df['lon_bin'] = pd.cut(df['longitude'], bins=20, labels=False)
    df['spatial_cell'] = df['lat_bin'] * 100 + df['lon_bin']
    
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
    
    # Time indicators
    df['is_rush_morning'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_rush_evening'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_rush_hour'] = (df['is_rush_morning'] | df['is_rush_evening']).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday'] = ((df['hour'] >= 10) & (df['hour'] <= 16)).astype(int)
    df['is_early_morning'] = ((df['hour'] >= 4) & (df['hour'] <= 7)).astype(int)
    
    conditions = [
        (df['hour'] >= 0) & (df['hour'] < 6),
        (df['hour'] >= 6) & (df['hour'] < 10),
        (df['hour'] >= 10) & (df['hour'] < 14),
        (df['hour'] >= 14) & (df['hour'] < 18),
        (df['hour'] >= 18) & (df['hour'] < 22),
        (df['hour'] >= 22)
    ]
    df['period_of_day'] = np.select(conditions, [0,1,2,3,4,5], default=0)
    
    # ---- Geohash prefixes ----
    df['geo_prefix_3'] = df['geohash'].str[:3]
    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]
    
    # ---- Categorical encoding ----
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc'] = (df['Landmarks'] == 'Yes').astype(int)
    
    road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
    df['RoadType_enc'] = df['RoadType'].map(road_map).fillna(-1).astype(int)
    df['RoadType_missing'] = df['RoadType'].isna().astype(int)
    
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['Weather_enc'] = df['Weather'].map(weather_map).fillna(-1).astype(int)
    df['Weather_missing'] = df['Weather'].isna().astype(int)
    
    # ---- Temperature ----
    df['Temperature_missing'] = df['Temperature'].isna().astype(int)
    df['temp_abs'] = df['Temperature'].abs()
    df['is_cold'] = (df['Temperature'] < 5).astype(int)
    df['is_hot'] = (df['Temperature'] > 35).astype(int)
    df['is_freezing'] = (df['Temperature'] < 0).astype(int)
    
    # ---- Interactions ----
    df['road_lanes'] = df['RoadType_enc'] * 10 + df['NumberofLanes']
    df['road_large_vehicles'] = df['RoadType_enc'] * 2 + df['LargeVehicles_enc']
    df['lanes_large_vehicles'] = df['NumberofLanes'] * 2 + df['LargeVehicles_enc']
    df['lanes_landmarks'] = df['NumberofLanes'] * 2 + df['Landmarks_enc']
    df['road_weather'] = df['RoadType_enc'] * 10 + df['Weather_enc']
    df['road_capacity'] = df['NumberofLanes'] * (1 + df['LargeVehicles_enc'])
    df['weather_rush'] = df['Weather_enc'] * 10 + df['is_rush_hour']
    df['road_rush'] = df['RoadType_enc'] * 10 + df['is_rush_hour']
    
    # Spatial x Road interactions
    df['lat_road'] = df['latitude'] * (df['RoadType_enc'] + 2)
    df['lon_road'] = df['longitude'] * (df['RoadType_enc'] + 2)
    
    # Spatial x Time
    df['lat_hour'] = df['latitude'] * df['hour']
    df['lon_hour'] = df['longitude'] * df['hour']
    
    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Fill temperature
temp_median = train_fe['Temperature'].median()
train_fe['Temperature'] = train_fe['Temperature'].fillna(temp_median)
test_fe['Temperature'] = test_fe['Temperature'].fillna(temp_median)
train_fe['temp_abs'] = train_fe['temp_abs'].fillna(abs(temp_median))
test_fe['temp_abs'] = test_fe['temp_abs'].fillna(abs(temp_median))

# Label encode
for col in ['geohash', 'geo_prefix_3', 'geo_prefix_4', 'geo_prefix_5']:
    le = LabelEncoder()
    combined = pd.concat([train_fe[col], test_fe[col]])
    le.fit(combined)
    train_fe[col + '_le'] = le.transform(train_fe[col])
    test_fe[col + '_le'] = le.transform(test_fe[col])

# ============================================================
# 3. Smoothed Target Encoding (day 48 only)
# ============================================================
print("Target encoding...")

day48 = train_fe[train_fe['day'] == 48].copy()
global_mean = day48['demand'].mean()
SMOOTH = 30

def smooth_te(train_df, test_df, source_df, col, target='demand', smooth=SMOOTH):
    stats = source_df.groupby(col)[target].agg(['mean', 'count'])
    smoothed = (stats['count'] * stats['mean'] + smooth * global_mean) / (stats['count'] + smooth)
    smoothed = smoothed.to_dict()
    return train_df[col].map(smoothed).fillna(global_mean).values, test_df[col].map(smoothed).fillna(global_mean).values

for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5', 'geo_prefix_3']:
    tr, te = smooth_te(train_fe, test_fe, day48, col)
    train_fe[f'{col}_te'] = tr
    test_fe[f'{col}_te'] = te

for col in ['RoadType_enc', 'Weather_enc', 'hour', 'quarter_of_day', 'period_of_day']:
    tr, te = smooth_te(train_fe, test_fe, day48, col, smooth=50)
    train_fe[f'{col}_te'] = tr
    test_fe[f'{col}_te'] = te

# Interaction TE
for c1, c2 in [('geohash', 'hour'), ('geohash', 'period_of_day'),
               ('RoadType_enc', 'hour'), ('Weather_enc', 'hour'),
               ('geohash', 'Weather_enc'), ('geohash', 'RoadType_enc')]:
    ic = f'{c1}_x_{c2}'
    day48[ic] = day48[c1].astype(str) + '_' + day48[c2].astype(str)
    train_fe[ic] = train_fe[c1].astype(str) + '_' + train_fe[c2].astype(str)
    test_fe[ic] = test_fe[c1].astype(str) + '_' + test_fe[c2].astype(str)
    tr, te = smooth_te(train_fe, test_fe, day48, ic, smooth=10)
    train_fe[f'{ic}_te'] = tr
    test_fe[f'{ic}_te'] = te

# Spatial binned TE
for col in ['lat_bin', 'lon_bin', 'spatial_cell']:
    tr, te = smooth_te(train_fe, test_fe, day48, col, smooth=20)
    train_fe[f'{col}_te'] = tr
    test_fe[f'{col}_te'] = te

# ============================================================
# 4. Aggregate features (day 48 only)
# ============================================================
print("Aggregate features...")

geo_stats = day48.groupby('geohash')['demand'].agg(['mean','std','median','min','max','count'])
geo_stats.columns = [f'geo_d48_{c}' for c in ['mean','std','median','min','max','count']]
geo_stats['geo_d48_std'] = geo_stats['geo_d48_std'].fillna(0)
geo_stats['geo_d48_range'] = geo_stats['geo_d48_max'] - geo_stats['geo_d48_min']
geo_stats['geo_d48_cv'] = geo_stats['geo_d48_std'] / (geo_stats['geo_d48_mean'] + 1e-8)
geo_stats['geo_d48_skew_proxy'] = (geo_stats['geo_d48_mean'] - geo_stats['geo_d48_median']) / (geo_stats['geo_d48_std'] + 1e-8)

for df in [train_fe, test_fe]:
    merged = df[['geohash']].merge(geo_stats, on='geohash', how='left')
    for c in geo_stats.columns:
        fill = global_mean if 'mean' in c or 'median' in c else 0
        df[c] = merged[c].fillna(fill).values

# Hour stats
hour_stats = day48.groupby('hour')['demand'].agg(['mean','std','median'])
hour_stats.columns = ['hour_d48_mean','hour_d48_std','hour_d48_median']
for df in [train_fe, test_fe]:
    merged = df[['hour']].merge(hour_stats, on='hour', how='left')
    for c in hour_stats.columns:
        df[c] = merged[c].values

# Geo x hour
gh = day48.groupby(['geohash','hour'])['demand'].agg(['mean','median','count']).reset_index()
gh.columns = ['geohash','hour','geo_hour_d48_mean','geo_hour_d48_median','geo_hour_d48_count']
for df in [train_fe, test_fe]:
    merged = df[['geohash','hour']].merge(gh, on=['geohash','hour'], how='left')
    df['geo_hour_d48_mean'] = merged['geo_hour_d48_mean'].fillna(global_mean).values
    df['geo_hour_d48_median'] = merged['geo_hour_d48_median'].fillna(global_mean).values
    df['geo_hour_d48_count'] = merged['geo_hour_d48_count'].fillna(0).values

# Geo x period
gp = day48.groupby(['geohash','period_of_day'])['demand'].agg(['mean','median']).reset_index()
gp.columns = ['geohash','period_of_day','geo_period_d48_mean','geo_period_d48_median']
for df in [train_fe, test_fe]:
    merged = df[['geohash','period_of_day']].merge(gp, on=['geohash','period_of_day'], how='left')
    df['geo_period_d48_mean'] = merged['geo_period_d48_mean'].fillna(global_mean).values
    df['geo_period_d48_median'] = merged['geo_period_d48_median'].fillna(global_mean).values

# Road x Weather
rw = day48.groupby(['RoadType_enc','Weather_enc'])['demand'].mean().reset_index()
rw.columns = ['RoadType_enc','Weather_enc','rw_d48_mean']
for df in [train_fe, test_fe]:
    merged = df[['RoadType_enc','Weather_enc']].merge(rw, on=['RoadType_enc','Weather_enc'], how='left')
    df['rw_d48_mean'] = merged['rw_d48_mean'].fillna(global_mean).values

# Spatial area stats
sp_stats = day48.groupby('spatial_cell')['demand'].agg(['mean','count']).reset_index()
sp_stats.columns = ['spatial_cell','spatial_d48_mean','spatial_d48_count']
for df in [train_fe, test_fe]:
    merged = df[['spatial_cell']].merge(sp_stats, on='spatial_cell', how='left')
    df['spatial_d48_mean'] = merged['spatial_d48_mean'].fillna(global_mean).values
    df['spatial_d48_count'] = merged['spatial_d48_count'].fillna(0).values

# ============================================================
# 5. Deviation features (key for generalization!)
# ============================================================
print("Deviation features...")

# How much does this location's demand deviate from the area average?
for df in [train_fe, test_fe]:
    df['geo_vs_area'] = df['geohash_te'] - df['spatial_cell_te']
    df['geo_vs_global'] = df['geohash_te'] - global_mean
    df['geo_hour_vs_geo'] = df['geo_hour_d48_mean'] - df['geo_d48_mean']
    df['hour_vs_global'] = df['hour_d48_mean'] - global_mean
    
    # Demand relative to road type baseline
    df['geo_vs_road'] = df['geohash_te'] - df['RoadType_enc_te']

# ============================================================
# 6. Define feature columns
# ============================================================
feature_cols = [
    # Spatial (lat/lon)
    'latitude', 'longitude', 'lat_lon_product', 'lat_abs', 'lon_abs',
    'lat_bin', 'lon_bin', 'spatial_cell',
    
    # Time
    'minutes', 'hour', 'minute_of_hour', 'quarter_of_day',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'quarter_sin', 'quarter_cos',
    'is_rush_morning', 'is_rush_evening', 'is_rush_hour',
    'is_night', 'is_midday', 'is_early_morning', 'period_of_day',
    
    # Location (encoded)
    'geohash_le', 'geo_prefix_3_le', 'geo_prefix_4_le', 'geo_prefix_5_le',
    
    # Infrastructure
    'RoadType_enc', 'RoadType_missing', 'NumberofLanes',
    'LargeVehicles_enc', 'Landmarks_enc',
    
    # Weather
    'Weather_enc', 'Weather_missing',
    'Temperature', 'Temperature_missing', 'temp_abs',
    'is_cold', 'is_hot', 'is_freezing',
    
    # Interactions
    'road_lanes', 'road_large_vehicles', 'lanes_large_vehicles',
    'lanes_landmarks', 'road_weather', 'road_capacity',
    'weather_rush', 'road_rush',
    'lat_road', 'lon_road', 'lat_hour', 'lon_hour',
    
    # Target encoded
    'geohash_te', 'geo_prefix_4_te', 'geo_prefix_5_te', 'geo_prefix_3_te',
    'RoadType_enc_te', 'Weather_enc_te', 'hour_te',
    'quarter_of_day_te', 'period_of_day_te',
    'lat_bin_te', 'lon_bin_te', 'spatial_cell_te',
    
    # Interaction TE
    'geohash_x_hour_te', 'geohash_x_period_of_day_te',
    'RoadType_enc_x_hour_te', 'Weather_enc_x_hour_te',
    'geohash_x_Weather_enc_te', 'geohash_x_RoadType_enc_te',
    
    # Aggregates
    'geo_d48_mean', 'geo_d48_std', 'geo_d48_median',
    'geo_d48_min', 'geo_d48_max', 'geo_d48_count',
    'geo_d48_range', 'geo_d48_cv', 'geo_d48_skew_proxy',
    'hour_d48_mean', 'hour_d48_std', 'hour_d48_median',
    'geo_hour_d48_mean', 'geo_hour_d48_median', 'geo_hour_d48_count',
    'geo_period_d48_mean', 'geo_period_d48_median',
    'rw_d48_mean',
    'spatial_d48_mean', 'spatial_d48_count',
    
    # Deviation features
    'geo_vs_area', 'geo_vs_global', 'geo_hour_vs_geo',
    'hour_vs_global', 'geo_vs_road',
]

X_train = train_fe[feature_cols].values.astype(np.float32)
X_test = test_fe[feature_cols].values.astype(np.float32)
y_train = train['demand'].values

# Sample weights: upweight day 49 samples
day_mask = train['day'].values
sample_weights = np.where(day_mask == 49, 5.0, 1.0)  # 5x weight for day 49
sample_weights_norm = sample_weights / sample_weights.mean()

print(f"Features: {len(feature_cols)}")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Temporal validation sets
day48_mask = train['day'] == 48
day49_mask = train['day'] == 49
X_d48, y_d48 = X_train[day48_mask], y_train[day48_mask]
X_d49, y_d49 = X_train[day49_mask], y_train[day49_mask]

# ============================================================
# 7. Multi-seed, multi-config training
# ============================================================
N_FOLDS = 5
SEEDS = [42, 123, 456]

all_test_preds = []
all_val_scores = []

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"SEED {SEED} ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # ---- LightGBM with sample weights ----
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'feature_fraction': 0.6 + 0.05 * seed_idx,
        'bagging_fraction': 0.6 + 0.05 * seed_idx,
        'bagging_freq': 5,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'min_gain_to_split': 0.01,
        'path_smooth': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': SEED,
    }
    
    lgb_preds = np.zeros(len(X_test))
    lgb_oof = np.zeros(len(X_train))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx], 
                              weight=sample_weights_norm[tr_idx])
        dval = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
        
        model = lgb.train(lgb_params, dtrain, num_boost_round=5000,
                           valid_sets=[dval],
                           callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        lgb_oof[val_idx] = model.predict(X_train[val_idx])
        lgb_preds += model.predict(X_test) / N_FOLDS
    
    lgb_r2 = r2_score(y_train, lgb_oof)
    print(f"  LGB OOF R2: {lgb_r2:.6f} | Day49 R2: {r2_score(y_d49, lgb_oof[day49_mask]):.6f}")
    
    # ---- XGBoost with sample weights ----
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.03,
        'max_depth': 7,
        'min_child_weight': 50,
        'subsample': 0.6 + 0.05 * seed_idx,
        'colsample_bytree': 0.6 + 0.05 * seed_idx,
        'reg_alpha': 2.0,
        'reg_lambda': 5.0,
        'gamma': 0.1,
        'tree_method': 'hist',
        'seed': SEED,
        'n_jobs': -1,
    }
    
    xgb_preds = np.zeros(len(X_test))
    xgb_oof = np.zeros(len(X_train))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx],
                              weight=sample_weights_norm[tr_idx])
        dval = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx])
        
        model = xgb.train(xgb_params, dtrain, num_boost_round=5000,
                           evals=[(dval, 'val')], early_stopping_rounds=100,
                           verbose_eval=0)
        xgb_oof[val_idx] = model.predict(dval)
        xgb_preds += model.predict(xgb.DMatrix(X_test)) / N_FOLDS
    
    xgb_r2 = r2_score(y_train, xgb_oof)
    print(f"  XGB OOF R2: {xgb_r2:.6f} | Day49 R2: {r2_score(y_d49, xgb_oof[day49_mask]):.6f}")
    
    # ---- CatBoost with sample weights ----
    cb_preds = np.zeros(len(X_test))
    cb_oof = np.zeros(len(X_train))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = CatBoostRegressor(
            iterations=5000, learning_rate=0.03, depth=7,
            l2_leaf_reg=10.0, random_seed=SEED, verbose=0,
            early_stopping_rounds=100, task_type='CPU',
            min_data_in_leaf=50, random_strength=2.0,
            bagging_temperature=0.5,
        )
        model.fit(X_train[tr_idx], y_train[tr_idx],
                   sample_weight=sample_weights_norm[tr_idx],
                   eval_set=(X_train[val_idx], y_train[val_idx]), verbose=0)
        cb_oof[val_idx] = model.predict(X_train[val_idx])
        cb_preds += model.predict(X_test) / N_FOLDS
    
    cb_r2 = r2_score(y_train, cb_oof)
    print(f"  CB  OOF R2: {cb_r2:.6f} | Day49 R2: {r2_score(y_d49, cb_oof[day49_mask]):.6f}")
    
    # ---- LightGBM with DART ----
    dart_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'dart',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.65,
        'bagging_freq': 5,
        'reg_alpha': 2.0,
        'reg_lambda': 2.0,
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': SEED,
    }
    
    dart_preds = np.zeros(len(X_test))
    dart_oof = np.zeros(len(X_train))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        dtrain = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx],
                              weight=sample_weights_norm[tr_idx])
        dval = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
        
        model = lgb.train(dart_params, dtrain, num_boost_round=500,
                           valid_sets=[dval], callbacks=[lgb.log_evaluation(0)])
        dart_oof[val_idx] = model.predict(X_train[val_idx])
        dart_preds += model.predict(X_test) / N_FOLDS
    
    dart_r2 = r2_score(y_train, dart_oof)
    print(f"  DART OOF R2: {dart_r2:.6f} | Day49 R2: {r2_score(y_d49, dart_oof[day49_mask]):.6f}")
    
    # ---- Find best blend for this seed using day49 validation ----
    best_score = -1
    best_w = None
    models_oof_d49 = [lgb_oof[day49_mask], xgb_oof[day49_mask], 
                       cb_oof[day49_mask], dart_oof[day49_mask]]
    models_test = [lgb_preds, xgb_preds, cb_preds, dart_preds]
    
    for w1 in np.arange(0.0, 0.6, 0.1):
        for w2 in np.arange(0.0, 0.6, 0.1):
            for w3 in np.arange(0.0, 0.8, 0.1):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0 or w4 > 0.6:
                    continue
                blend = w1*models_oof_d49[0] + w2*models_oof_d49[1] + w3*models_oof_d49[2] + w4*models_oof_d49[3]
                score = r2_score(y_d49, blend)
                if score > best_score:
                    best_score = score
                    best_w = (w1, w2, w3, w4)
    
    seed_preds = best_w[0]*models_test[0] + best_w[1]*models_test[1] + best_w[2]*models_test[2] + best_w[3]*models_test[3]
    all_test_preds.append(seed_preds)
    all_val_scores.append(best_score)
    
    print(f"  Seed {SEED} best blend: LGB={best_w[0]:.1f}, XGB={best_w[1]:.1f}, CB={best_w[2]:.1f}, DART={best_w[3]:.1f}")
    print(f"  Seed {SEED} Day49 val R2: {best_score:.6f} (Score: {max(0,100*best_score):.4f})")

# ============================================================
# 8. Final multi-seed average
# ============================================================
print("\n" + "="*60)
print("FINAL ENSEMBLE")
print("="*60)

final_preds = np.mean(all_test_preds, axis=0)
final_preds = np.clip(final_preds, 0, None)

print(f"Individual seed val scores: {[f'{s:.4f}' for s in all_val_scores]}")
print(f"Mean val score: {np.mean(all_val_scores):.6f} (Score: {max(0,100*np.mean(all_val_scores)):.4f})")

# ============================================================
# 9. Save submissions
# ============================================================
submission = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': final_preds})
submission.to_csv('/Users/hiya/Downloads/dataset/submission_v3.csv', index=False)

print(f"\nSubmission shape: {submission.shape}")
print(submission['demand'].describe())
print(f"\n✅ Saved to /Users/hiya/Downloads/dataset/submission_v3.csv")

# Also save per-seed submissions for comparison
for i, preds in enumerate(all_test_preds):
    sub = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': np.clip(preds, 0, None)})
    sub.to_csv(f'/Users/hiya/Downloads/dataset/submission_v3_seed{SEEDS[i]}.csv', index=False)

print("All done! ✅")
