"""
Traffic Demand Prediction - EDA & Modeling
==========================================
Goal: Predict 'demand' from features.
Metric: max(0, 100 * r2_score(actual, predicted))
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================
train = pd.read_csv('/Users/hiya/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/hiya/Downloads/dataset/test.csv')
sample_sub = pd.read_csv('/Users/hiya/Downloads/dataset/sample_submission.csv')

print("="*60)
print("DATASET SHAPES")
print("="*60)
print(f"Train: {train.shape}")
print(f"Test:  {test.shape}")
print(f"Sample submission: {sample_sub.shape}")

print("\n" + "="*60)
print("TRAIN INFO")
print("="*60)
print(train.dtypes)
print("\nFirst 5 rows:")
print(train.head())

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print("Train missing:")
print(train.isnull().sum())
print(f"\nTrain missing %:")
print((train.isnull().sum() / len(train) * 100).round(2))
print("\nTest missing:")
print(test.isnull().sum())

print("\n" + "="*60)
print("TARGET DISTRIBUTION")
print("="*60)
print(train['demand'].describe())

print("\n" + "="*60)
print("CATEGORICAL COLUMNS - UNIQUE VALUES")
print("="*60)
for col in ['geohash', 'day', 'timestamp', 'RoadType', 'NumberofLanes', 
            'LargeVehicles', 'Landmarks', 'Weather']:
    print(f"\n{col}: {train[col].nunique()} unique values")
    if train[col].nunique() <= 20:
        print(train[col].value_counts())

print("\n" + "="*60)
print("NUMERIC COLUMNS STATS")
print("="*60)
print(train[['Temperature', 'demand']].describe())

print("\n" + "="*60)
print("GEOHASH ANALYSIS")
print("="*60)
print(f"Train unique geohash: {train['geohash'].nunique()}")
print(f"Test unique geohash: {test['geohash'].nunique()}")
train_geo = set(train['geohash'].unique())
test_geo = set(test['geohash'].unique())
print(f"Overlap: {len(train_geo & test_geo)}")
print(f"Test-only geohash: {len(test_geo - train_geo)}")

print("\n" + "="*60)
print("TIMESTAMP ANALYSIS")
print("="*60)
print(f"Unique timestamps: {train['timestamp'].nunique()}")
print(train['timestamp'].value_counts().head(10))

print("\n" + "="*60)
print("DAY ANALYSIS")
print("="*60)
print(f"Train days: {sorted(train['day'].unique())}")
print(f"Test days: {sorted(test['day'].unique())}")
