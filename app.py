from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np
import json
import uvicorn
import os

app = FastAPI(title="Traffic Demand Predictor API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Encoders (Using relative paths for cloud deployment)
MODEL_PATH = 'model.txt'
ENCODERS_PATH = 'encoders.json'

model = lgb.Booster(model_file=MODEL_PATH)
with open(ENCODERS_PATH, 'r') as f:
    encoders = json.load(f)

class PredictionRequest(BaseModel):
    geohash: str
    time: str  # Format "HH:MM"
    weather: str
    temperature: float
    population: float # 0.0 to 1.0
    is_holiday: int # 0 or 1

@app.post("/predict")
def predict_traffic(request: PredictionRequest):
    # 1. Process Time
    hour, minute = map(int, request.time.split(':'))
    time_sin = np.sin(2 * np.pi * (hour * 60 + minute) / 1440)
    time_cos = np.cos(2 * np.pi * (hour * 60 + minute) / 1440)
    
    # 2. Process Geohash
    geo_encoded = encoders['geohash_target_mean'].get(request.geohash, encoders['global_mean_demand'])
    
    # 3. Process Weather
    weather_encoded = encoders['weather_map'].get(request.weather, 0)
    
    # 4. Construct Feature Array
    # Features must match training order: 
    # ['geohash_encoded', 'time_sin', 'time_cos', 'Temperature', 'weather_encoded', 'population', 'is_holiday']
    features = np.array([[
        geo_encoded,
        time_sin,
        time_cos,
        request.temperature,
        weather_encoded,
        request.population,
        request.is_holiday
    ]])
    
    # 5. Predict
    prediction = model.predict(features)[0]
    
    # Clip negative predictions
    final_prediction = max(0, float(prediction))
    
    return {
        "demand": round(final_prediction, 4),
        "geohash": request.geohash,
        "time": request.time
    }

# Mount the docs directory to serve the frontend
if os.path.exists('docs'):
    app.mount("/", StaticFiles(directory="docs", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
