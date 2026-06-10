# Traffic Demand Prediction

This repository contains the code and methodology for an AI-powered urban traffic demand prediction model. The goal of this project is to accurately forecast traffic demand across various geographic locations (geohashes) at 15-minute intervals, enabling data-driven strategies for alleviating traffic congestion.

## Problem Statement

Cities worldwide are turning to AI-powered solutions to tackle traffic congestion. To address this challenge effectively, it's essential to understand travel demand and patterns comprehensively. 

This project builds a machine learning system to predict `demand` using historical data. The dataset includes temporal features (day, timestamp), spatial features (geohash), infrastructure details (road type, lanes, large vehicle allowances, landmarks), and environmental factors (weather, temperature).

## Repository Structure

The project was developed in multiple iterations to handle complex distribution shifts in the temporal data:

* `explore_and_model.py`: Initial Exploratory Data Analysis (EDA) and baseline modeling.
* `train_model.py` (V1): The foundational ensemble model (LightGBM, XGBoost, CatBoost) with extensive feature engineering. *Achieved strong leaderboard performance (89.2).*
* `train_model_v2.py` - `v4.py`: Iterations exploring temporal cross-validation, DART boosting, and sample weighting to combat temporal leakage.
* `train_model_v5.py`: The final refined architecture focusing on multi-seed stability, log1p target transformations (for right-skewed demand), and optimal model blending. *Achieved the highest score (89.29).*
* `train_model_v6.py`: An experimental model trained exclusively on Day 48 data to combat a severe night-vs-day distribution shift detected in Day 49.

## Feature Engineering Highlights

The models rely heavily on robust feature engineering to extract signal from the raw data:

1. **Spatial Decoding:** Decoding Base32 Geohashes into numerical Latitude and Longitude coordinates to allow the tree models to learn spatial proximity.
2. **Temporal Cyclical Features:** Sine and cosine transformations of the hour and minute to represent the continuous, cyclical nature of time.
3. **Traffic Period Indicators:** Categorizing times into rush hours (morning/evening), midday, and night periods.
4. **Target Encoding:** K-Fold target encoding applied to high-cardinality categorical features (like Geohashes and categorical interactions) to prevent overfitting while capturing historical demand averages.
5. **Interaction Features:** Combining infrastructure and environmental variables (e.g., Road Capacity, Weather during Rush Hour).

## Key Insights & Challenges

**The Temporal Shift Anomaly:**
A deep dive into the dataset revealed a severe distribution shift between the final training day and the test set:
- Day 49 in the *training* set only contained data from `0:00 to 2:00` (Night).
- Day 49 in the *test* set contained data from `2:15 to 23:45` (Daytime).
- Furthermore, baseline demand at night on Day 49 was discovered to be **~55% higher** than the baseline demand at night on Day 48. 

The iterations in this repository track the progression of attempting to calibrate the model against this severe temporal shift using techniques like log-transformations, test-timestamp sample weighting, and selective day-dropping.

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hiya7790/Traffic-Demand-Prediction.git
   cd Traffic-Demand-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: LightGBM on macOS may require `libomp`. You can install it via Homebrew: `brew install libomp` and set the `DYLD_LIBRARY_PATH` accordingly).*

## Execution

Ensure the `train.csv` and `test.csv` dataset files are placed in the root directory.

To run the best performing model pipeline (V5):
```bash
python train_model_v5.py
```
This will output multiple blended submission files (`.csv`) in the directory, utilizing multi-seed averaging and log-target transforms for stability.

## Evaluation Metric

Models are evaluated based on a scaled R-squared metric:
`score = max(0, 100 * r2_score(actual, predicted))`
