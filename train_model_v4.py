"""
Traffic Demand Prediction - V4
===============================
KEY INSIGHT: Day 49 train covers 0:00-2:00, test covers 2:15-23:45.
ZERO timestamp overlap between day49 train and test!

Strategy:
- Day 48 has FULL 24-hour coverage (0:00-23:45) → best source for time patterns
- Day 49 train only has night hours → DON'T upweight (misleading distribution)
- Focus on features that capture time-of-day patterns from day 48
- Use day 48 same-hour data as the primary signal
- Moderate regularization, no upweighting
- Let the model learn time patterns from day 48 and location patterns from both days
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
# Geohash decoder
# ============================================================
BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

def decode_geohash(ghash):
    is_lon = True
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    for c in ghash:
        val = BASE32_MAP.get(c, 0)
        for bit in range(4, -1, -1):
            b = (val >> bit) & 1
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if b: lon_range[0] = mid
                else: lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if b: lat_range[0] = mid
                else: lat_range[1] = mid
            is_lon = not is_lon
    return (lat_range[0]+lat_range[1])/2, (lon_range[0]+lon_range[1])/2

# ============================================================
# Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv('/Users/hiya/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/hiya/Downloads/dataset/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# Feature Engineering
# ============================================================
print("Engineering features...")

def parse_timestamp(ts):
    parts = ts.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def engineer_features(df):
    df = df.copy()
    
    # Lat/Lon
    coords = df['geohash'].apply(decode_geohash)
    df['latitude'] = coords.apply(lambda x: x[0])
    df['longitude'] = coords.apply(lambda x: x[1])
    df['lat_lon_product'] = df['latitude'] * df['longitude']
    
    # Spatial bins
    df['lat_bin'] = pd.cut(df['latitude'], bins=15, labels=False)
    df['lon_bin'] = pd.cut(df['longitude'], bins=15, labels=False)
    df['spatial_cell'] = df['lat_bin'] * 100 + df['lon_bin']
    
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
    
    # Geohash prefixes
    df['geo_prefix_3'] = df['geohash'].str[:3]
    df['geo_prefix_4'] = df['geohash'].str[:4]
    df['geo_prefix_5'] = df['geohash'].str[:5]
    
    # Categoricals
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc'] = (df['Landmarks'] == 'Yes').astype(int)
    
    road_map = {'Residential': 0, 'Street': 1, 'Highway': 2}
    df['RoadType_enc'] = df['RoadType'].map(road_map).fillna(-1).astype(int)
    df['RoadType_missing'] = df['RoadType'].isna().astype(int)
    
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['Weather_enc'] = df['Weather'].map(weather_map).fillna(-1).astype(int)
    df['Weather_missing'] = df['Weather'].isna().astype(int)
    
    # Temperature
    df['Temperature_missing'] = df['Temperature'].isna().astype(int)
    df['temp_abs'] = df['Temperature'].abs()
    df['is_cold'] = (df['Temperature'] < 5).astype(int)
    df['is_hot'] = (df['Temperature'] > 35).astype(int)
    df['is_freezing'] = (df['Temperature'] < 0).astype(int)
    
    # Interactions
    df['road_lanes'] = df['RoadType_enc'] * 10 + df['NumberofLanes']
    df['road_large_vehicles'] = df['RoadType_enc'] * 2 + df['LargeVehicles_enc']
    df['lanes_large_vehicles'] = df['NumberofLanes'] * 2 + df['LargeVehicles_enc']
    df['lanes_landmarks'] = df['NumberofLanes'] * 2 + df['Landmarks_enc']
    df['road_weather'] = df['RoadType_enc'] * 10 + df['Weather_enc']
    df['road_capacity'] = df['NumberofLanes'] * (1 + df['LargeVehicles_enc'])
    df['weather_rush'] = df['Weather_enc'] * 10 + df['is_rush_hour']
    df['road_rush'] = df['RoadType_enc'] * 10 + df['is_rush_hour']
    
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
# Target Encoding - TWO versions
# ============================================================
print("Target encoding...")

day48 = train_fe[train_fe['day'] == 48].copy()
global_mean = train['demand'].mean()
global_mean_d48 = day48['demand'].mean()

def smooth_te(train_df, test_df, source_df, col, target='demand', smooth=30, gm=None):
    if gm is None:
        gm = source_df[target].mean()
    stats = source_df.groupby(col)[target].agg(['mean', 'count'])
    smoothed = (stats['count'] * stats['mean'] + smooth * gm) / (stats['count'] + smooth)
    smoothed = smoothed.to_dict()
    return train_df[col].map(smoothed).fillna(gm).values, test_df[col].map(smoothed).fillna(gm).values

# Version A: Using day 48 only (no temporal leakage, covers all hours)
for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5', 'geo_prefix_3']:
    tr, te = smooth_te(train_fe, test_fe, day48, col, smooth=30, gm=global_mean_d48)
    train_fe[f'{col}_te48'] = tr
    test_fe[f'{col}_te48'] = te

for col in ['RoadType_enc', 'Weather_enc', 'hour', 'quarter_of_day', 'period_of_day']:
    tr, te = smooth_te(train_fe, test_fe, day48, col, smooth=50, gm=global_mean_d48)
    train_fe[f'{col}_te48'] = tr
    test_fe[f'{col}_te48'] = te

# Version B: Using ALL data with KFold (more data, slight leak risk)
def kfold_te(train_df, test_df, col, target_col='demand', n_folds=5, smooth=30):
    train_enc = np.zeros(len(train_df))
    gm = train_df[target_col].mean()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_df):
        src = train_df.iloc[tr_idx]
        stats = src.groupby(col)[target_col].agg(['mean','count'])
        smoothed = (stats['count'] * stats['mean'] + smooth * gm) / (stats['count'] + smooth)
        train_enc[val_idx] = train_df.iloc[val_idx][col].map(smoothed)
    
    train_enc = np.where(np.isnan(train_enc), gm, train_enc)
    
    stats = train_df.groupby(col)[target_col].agg(['mean','count'])
    smoothed = (stats['count'] * stats['mean'] + smooth * gm) / (stats['count'] + smooth)
    test_enc = test_df[col].map(smoothed).fillna(gm).values
    
    return train_enc, test_enc

for col in ['geohash', 'geo_prefix_4', 'geo_prefix_5']:
    tr, te = kfold_te(train_fe, test_fe, col, smooth=30)
    train_fe[f'{col}_te_kf'] = tr
    test_fe[f'{col}_te_kf'] = te

# Interaction TEs (day 48 only - these generalize best)
for c1, c2 in [('geohash','hour'), ('geohash','period_of_day'),
               ('RoadType_enc','hour'), ('Weather_enc','hour'),
               ('geohash','Weather_enc'), ('geohash','RoadType_enc')]:
    ic = f'{c1}_x_{c2}'
    day48[ic] = day48[c1].astype(str) + '_' + day48[c2].astype(str)
    train_fe[ic] = train_fe[c1].astype(str) + '_' + train_fe[c2].astype(str)
    test_fe[ic] = test_fe[c1].astype(str) + '_' + test_fe[c2].astype(str)
    tr, te = smooth_te(train_fe, test_fe, day48, ic, smooth=10, gm=global_mean_d48)
    train_fe[f'{ic}_te'] = tr
    test_fe[f'{ic}_te'] = te

# ============================================================
# Aggregate features (day 48 - full 24h coverage)
# ============================================================
print("Aggregate features...")

# Overall geohash stats
geo_stats = day48.groupby('geohash')['demand'].agg(['mean','std','median','min','max','count'])
geo_stats.columns = [f'geo_d48_{c}' for c in ['mean','std','median','min','max','count']]
geo_stats['geo_d48_std'] = geo_stats['geo_d48_std'].fillna(0)
geo_stats['geo_d48_range'] = geo_stats['geo_d48_max'] - geo_stats['geo_d48_min']
geo_stats['geo_d48_cv'] = geo_stats['geo_d48_std'] / (geo_stats['geo_d48_mean'] + 1e-8)

for df in [train_fe, test_fe]:
    merged = df[['geohash']].merge(geo_stats, on='geohash', how='left')
    for c in geo_stats.columns:
        fill = global_mean_d48 if 'mean' in c or 'median' in c else 0
        df[c] = merged[c].fillna(fill).values

# Hour stats from day 48
hour_stats = day48.groupby('hour')['demand'].agg(['mean','std','median'])
hour_stats.columns = ['hour_d48_mean','hour_d48_std','hour_d48_median']
for df in [train_fe, test_fe]:
    merged = df[['hour']].merge(hour_stats, on='hour', how='left')
    for c in hour_stats.columns:
        df[c] = merged[c].fillna(global_mean_d48).values

# Geo x hour from day 48 (KEY feature - captures location-time patterns)
gh = day48.groupby(['geohash','hour'])['demand'].agg(['mean','median','std','count']).reset_index()
gh.columns = ['geohash','hour','geo_hour_mean','geo_hour_median','geo_hour_std','geo_hour_count']
gh['geo_hour_std'] = gh['geo_hour_std'].fillna(0)
for df in [train_fe, test_fe]:
    merged = df[['geohash','hour']].merge(gh, on=['geohash','hour'], how='left')
    df['geo_hour_mean'] = merged['geo_hour_mean'].fillna(global_mean_d48).values
    df['geo_hour_median'] = merged['geo_hour_median'].fillna(global_mean_d48).values
    df['geo_hour_std'] = merged['geo_hour_std'].fillna(0).values
    df['geo_hour_count'] = merged['geo_hour_count'].fillna(0).values

# Geo x period from day 48
gp = day48.groupby(['geohash','period_of_day'])['demand'].agg(['mean','median']).reset_index()
gp.columns = ['geohash','period_of_day','geo_period_mean','geo_period_median']
for df in [train_fe, test_fe]:
    merged = df[['geohash','period_of_day']].merge(gp, on=['geohash','period_of_day'], how='left')
    df['geo_period_mean'] = merged['geo_period_mean'].fillna(global_mean_d48).values
    df['geo_period_median'] = merged['geo_period_median'].fillna(global_mean_d48).values

# Geo x quarter from day 48 (finest granularity)
gq = day48.groupby(['geohash','quarter_of_day'])['demand'].agg(['mean','count']).reset_index()
gq.columns = ['geohash','quarter_of_day','geo_quarter_mean','geo_quarter_count']
for df in [train_fe, test_fe]:
    merged = df[['geohash','quarter_of_day']].merge(gq, on=['geohash','quarter_of_day'], how='left')
    df['geo_quarter_mean'] = merged['geo_quarter_mean'].fillna(global_mean_d48).values
    df['geo_quarter_count'] = merged['geo_quarter_count'].fillna(0).values

# Road x Weather
rw = day48.groupby(['RoadType_enc','Weather_enc'])['demand'].mean().reset_index()
rw.columns = ['RoadType_enc','Weather_enc','rw_d48_mean']
for df in [train_fe, test_fe]:
    merged = df[['RoadType_enc','Weather_enc']].merge(rw, on=['RoadType_enc','Weather_enc'], how='left')
    df['rw_d48_mean'] = merged['rw_d48_mean'].fillna(global_mean_d48).values

# Road x Hour
rh = day48.groupby(['RoadType_enc','hour'])['demand'].mean().reset_index()
rh.columns = ['RoadType_enc','hour','road_hour_mean']
for df in [train_fe, test_fe]:
    merged = df[['RoadType_enc','hour']].merge(rh, on=['RoadType_enc','hour'], how='left')
    df['road_hour_mean'] = merged['road_hour_mean'].fillna(global_mean_d48).values

# Weather x Hour
wh = day48.groupby(['Weather_enc','hour'])['demand'].mean().reset_index()
wh.columns = ['Weather_enc','hour','weather_hour_mean']
for df in [train_fe, test_fe]:
    merged = df[['Weather_enc','hour']].merge(wh, on=['Weather_enc','hour'], how='left')
    df['weather_hour_mean'] = merged['weather_hour_mean'].fillna(global_mean_d48).values

# ============================================================
# Deviation features
# ============================================================
print("Deviation features...")
for df in [train_fe, test_fe]:
    df['geo_vs_global'] = df['geohash_te48'] - global_mean_d48
    df['geo_hour_vs_geo'] = df['geo_hour_mean'] - df['geo_d48_mean']
    df['hour_vs_global'] = df['hour_d48_mean'] - global_mean_d48
    df['geo_vs_road'] = df['geohash_te48'] - df['RoadType_enc_te48']
    # Ratio: how much more/less demand at this hour vs average for this location
    df['geo_hour_ratio'] = df['geo_hour_mean'] / (df['geo_d48_mean'] + 1e-8)
    df['geo_period_ratio'] = df['geo_period_mean'] / (df['geo_d48_mean'] + 1e-8)

# ============================================================
# Feature columns
# ============================================================
feature_cols = [
    # Spatial
    'latitude', 'longitude', 'lat_lon_product',
    'lat_bin', 'lon_bin', 'spatial_cell',
    
    # Time
    'minutes', 'hour', 'minute_of_hour', 'quarter_of_day',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'quarter_sin', 'quarter_cos',
    'is_rush_morning', 'is_rush_evening', 'is_rush_hour',
    'is_night', 'is_midday', 'is_early_morning', 'period_of_day',
    
    # Location encoded
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
    
    # TE day 48 (clean, full 24h)
    'geohash_te48', 'geo_prefix_4_te48', 'geo_prefix_5_te48', 'geo_prefix_3_te48',
    'RoadType_enc_te48', 'Weather_enc_te48', 'hour_te48',
    'quarter_of_day_te48', 'period_of_day_te48',
    
    # TE KFold (more data)
    'geohash_te_kf', 'geo_prefix_4_te_kf', 'geo_prefix_5_te_kf',
    
    # Interaction TE
    'geohash_x_hour_te', 'geohash_x_period_of_day_te',
    'RoadType_enc_x_hour_te', 'Weather_enc_x_hour_te',
    'geohash_x_Weather_enc_te', 'geohash_x_RoadType_enc_te',
    
    # Aggregates
    'geo_d48_mean', 'geo_d48_std', 'geo_d48_median',
    'geo_d48_min', 'geo_d48_max', 'geo_d48_count',
    'geo_d48_range', 'geo_d48_cv',
    'hour_d48_mean', 'hour_d48_std', 'hour_d48_median',
    'geo_hour_mean', 'geo_hour_median', 'geo_hour_std', 'geo_hour_count',
    'geo_period_mean', 'geo_period_median',
    'geo_quarter_mean', 'geo_quarter_count',
    'rw_d48_mean', 'road_hour_mean', 'weather_hour_mean',
    
    # Deviation features
    'geo_vs_global', 'geo_hour_vs_geo', 'hour_vs_global',
    'geo_vs_road', 'geo_hour_ratio', 'geo_period_ratio',
]

X_train = train_fe[feature_cols].values.astype(np.float32)
X_test = test_fe[feature_cols].values.astype(np.float32)
y_train = train['demand'].values

print(f"Features: {len(feature_cols)}")

# ============================================================
# Model Training - Multi-config, Multi-seed
# ============================================================
N_FOLDS = 5
SEEDS = [42, 123, 456, 789]

all_lgb_preds = []
all_xgb_preds = []
all_cb_preds = []

# Config variations for diversity
lgb_configs = [
    {'num_leaves': 63, 'max_depth': 7, 'min_child_samples': 40, 
     'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 0.5},
    {'num_leaves': 127, 'max_depth': 8, 'min_child_samples': 30,
     'feature_fraction': 0.75, 'bagging_fraction': 0.75, 'reg_alpha': 0.3, 'reg_lambda': 0.3},
    {'num_leaves': 50, 'max_depth': 6, 'min_child_samples': 50,
     'feature_fraction': 0.65, 'bagging_fraction': 0.65, 'reg_alpha': 1.0, 'reg_lambda': 1.0},
]

xgb_configs = [
    {'max_depth': 7, 'min_child_weight': 30, 'subsample': 0.7, 'colsample_bytree': 0.7,
     'reg_alpha': 0.5, 'reg_lambda': 1.0, 'gamma': 0.05},
    {'max_depth': 8, 'min_child_weight': 20, 'subsample': 0.75, 'colsample_bytree': 0.75,
     'reg_alpha': 0.3, 'reg_lambda': 0.5, 'gamma': 0.01},
    {'max_depth': 6, 'min_child_weight': 50, 'subsample': 0.65, 'colsample_bytree': 0.65,
     'reg_alpha': 1.0, 'reg_lambda': 2.0, 'gamma': 0.1},
]

cb_configs = [
    {'depth': 7, 'l2_leaf_reg': 5.0, 'min_data_in_leaf': 30, 'random_strength': 1.0},
    {'depth': 8, 'l2_leaf_reg': 3.0, 'min_data_in_leaf': 20, 'random_strength': 0.5},
    {'depth': 6, 'l2_leaf_reg': 10.0, 'min_data_in_leaf': 50, 'random_strength': 2.0},
]

total_models = len(SEEDS) * (len(lgb_configs) + len(xgb_configs) + len(cb_configs)) * N_FOLDS
print(f"\nTotal models to train: {total_models}")

for seed_idx, SEED in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"SEED {SEED} ({seed_idx+1}/{len(SEEDS)})")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # LightGBM configs
    for cfg_idx, cfg in enumerate(lgb_configs):
        lgb_params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': 0.05, 'bagging_freq': 5,
            'verbose': -1, 'n_jobs': -1, 'seed': SEED,
            **cfg
        }
        
        preds = np.zeros(len(X_test))
        oof = np.zeros(len(X_train))
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            dtrain = lgb.Dataset(X_train[tr_idx], label=y_train[tr_idx])
            dval = lgb.Dataset(X_train[val_idx], label=y_train[val_idx])
            model = lgb.train(lgb_params, dtrain, num_boost_round=3000,
                               valid_sets=[dval],
                               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            oof[val_idx] = model.predict(X_train[val_idx])
            preds += model.predict(X_test) / N_FOLDS
        
        r2 = r2_score(y_train, oof)
        print(f"  LGB cfg{cfg_idx} R2: {r2:.6f} (Score: {max(0,100*r2):.2f})")
        all_lgb_preds.append(preds)
    
    # XGBoost configs
    for cfg_idx, cfg in enumerate(xgb_configs):
        xgb_params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'learning_rate': 0.05, 'tree_method': 'hist',
            'seed': SEED, 'n_jobs': -1,
            **cfg
        }
        
        preds = np.zeros(len(X_test))
        oof = np.zeros(len(X_train))
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx])
            dval = xgb.DMatrix(X_train[val_idx], label=y_train[val_idx])
            model = xgb.train(xgb_params, dtrain, num_boost_round=3000,
                               evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=0)
            oof[val_idx] = model.predict(dval)
            preds += model.predict(xgb.DMatrix(X_test)) / N_FOLDS
        
        r2 = r2_score(y_train, oof)
        print(f"  XGB cfg{cfg_idx} R2: {r2:.6f} (Score: {max(0,100*r2):.2f})")
        all_xgb_preds.append(preds)
    
    # CatBoost configs
    for cfg_idx, cfg in enumerate(cb_configs):
        preds = np.zeros(len(X_test))
        oof = np.zeros(len(X_train))
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            model = CatBoostRegressor(
                iterations=3000, learning_rate=0.05, random_seed=SEED,
                verbose=0, early_stopping_rounds=50, task_type='CPU',
                bagging_temperature=0.5, **cfg
            )
            model.fit(X_train[tr_idx], y_train[tr_idx],
                       eval_set=(X_train[val_idx], y_train[val_idx]), verbose=0)
            oof[val_idx] = model.predict(X_train[val_idx])
            preds += model.predict(X_test) / N_FOLDS
        
        r2 = r2_score(y_train, oof)
        print(f"  CB  cfg{cfg_idx} R2: {r2:.6f} (Score: {max(0,100*r2):.2f})")
        all_cb_preds.append(preds)

# ============================================================
# Final ensemble: simple average of all predictions
# ============================================================
print(f"\n{'='*60}")
print("FINAL ENSEMBLE")
print(f"{'='*60}")

all_preds = all_lgb_preds + all_xgb_preds + all_cb_preds
print(f"Total prediction sets: {len(all_preds)}")

# Simple average (most robust)
final_preds = np.mean(all_preds, axis=0)
final_preds = np.clip(final_preds, 0, None)

# Also try median (more robust to outliers)
final_preds_median = np.median(all_preds, axis=0)
final_preds_median = np.clip(final_preds_median, 0, None)

# Trimmed mean (remove top/bottom 10%)
sorted_preds = np.sort(all_preds, axis=0)
n = len(all_preds)
trim = max(1, n // 10)
final_preds_trimmed = np.mean(sorted_preds[trim:-trim], axis=0)
final_preds_trimmed = np.clip(final_preds_trimmed, 0, None)

# Save all variants
submission = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': final_preds})
submission.to_csv('/Users/hiya/Downloads/dataset/submission_v4_mean.csv', index=False)

submission_med = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': final_preds_median})
submission_med.to_csv('/Users/hiya/Downloads/dataset/submission_v4_median.csv', index=False)

submission_trim = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': final_preds_trimmed})
submission_trim.to_csv('/Users/hiya/Downloads/dataset/submission_v4_trimmed.csv', index=False)

print(f"Submission (mean): {submission.shape}")
print(submission['demand'].describe())

# Also blend V1 and V4 (V1 scored 89.2, use as anchor)
v1_sub = pd.read_csv('/Users/hiya/Downloads/dataset/submission.csv')
for alpha in [0.3, 0.5, 0.7]:
    blended = alpha * final_preds + (1-alpha) * v1_sub['demand'].values
    sub_b = pd.DataFrame({'Index': test_fe['Index'].values, 'demand': blended})
    sub_b.to_csv(f'/Users/hiya/Downloads/dataset/submission_v4_blend_{int(alpha*100)}.csv', index=False)
    print(f"Saved blend alpha={alpha:.1f}")

print("\n✅ All submissions saved!")
print("Files:")
print("  submission_v4_mean.csv - Average of all models")
print("  submission_v4_median.csv - Median of all models")
print("  submission_v4_trimmed.csv - Trimmed mean")
print("  submission_v4_blend_30.csv - 30% V4 + 70% V1")
print("  submission_v4_blend_50.csv - 50% V4 + 50% V1")
print("  submission_v4_blend_70.csv - 70% V4 + 30% V1")
