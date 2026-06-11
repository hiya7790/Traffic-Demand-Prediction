import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import pygeohash as pgh

print("Loading data...")
train = pd.read_csv('/Users/hiya/Downloads/dataset/train.csv')

# --- 1. SYNTHESIZE NEW FEATURES ---
# The dataset doesn't have Population or Holiday natively, so we create heuristics to simulate them.
# We will base "Population" on the density of traffic events in a geohash.
print("Synthesizing features (Population & Holiday)...")
geohash_counts = train['geohash'].value_counts()
max_count = geohash_counts.max()
# Population metric (0 to 1) based on geohash frequency
population_map = (geohash_counts / max_count).to_dict()
train['population'] = train['geohash'].map(population_map)

# Simulate Holiday (Let's say Day 7, 14, 21, 28, 35, 42, 49 are weekends/holidays)
train['is_holiday'] = train['day'].apply(lambda x: 1 if x % 7 == 0 else 0)

# Simulate Weather if missing (fill NaNs)
train['Weather'] = train['Weather'].fillna('Clear')
train['Temperature'] = train['Temperature'].fillna(train['Temperature'].mean())

# --- 2. FEATURE ENGINEERING ---
print("Feature Engineering...")
def parse_timestamp(df):
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_sin'] = np.sin(2 * np.pi * (df['hour'] * 60 + df['minute']) / 1440)
    df['time_cos'] = np.cos(2 * np.pi * (df['hour'] * 60 + df['minute']) / 1440)
    return df

train = parse_timestamp(train)

# Weather mapping
weather_map = {'Clear': 0, 'Rain': 1, 'Snow': 2, 'Fog': 3, 'Cloudy': 4}
train['weather_encoded'] = train['Weather'].map(weather_map).fillna(0)

# Target Encoding for Geohash
geohash_target_mean = train.groupby('geohash')['demand'].mean().to_dict()
train['geohash_encoded'] = train['geohash'].map(geohash_target_mean)

features = ['geohash_encoded', 'time_sin', 'time_cos', 'Temperature', 'weather_encoded', 'population', 'is_holiday']

# --- 3. TRAINING MODEL ---
print("Training Model...")
X = train[features]
y = train['demand']

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
model.fit(X, y)

# --- 4. EXPORT ARTIFACTS ---
print("Exporting Artifacts...")
model.booster_.save_model('/Users/hiya/Downloads/dataset/model.txt')

# Save encoders
encoders = {
    'geohash_target_mean': geohash_target_mean,
    'population_map': population_map,
    'weather_map': weather_map,
    'global_mean_temp': train['Temperature'].mean(),
    'global_mean_demand': train['demand'].mean()
}

with open('/Users/hiya/Downloads/dataset/encoders.json', 'w') as f:
    json.dump(encoders, f)

print("✅ Model and Encoders saved successfully!")
