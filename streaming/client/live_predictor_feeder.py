"""
Feeds latest sensor data into FastAPI, logs GRU and Prophet predictions,
and triggers drift detection.

Requirements:
- FastAPI server with GRU endpoint must be running at http://localhost:8000/predict
- Prophet model must be saved as artifacts/prophet_model.json
"""

import glob, time, requests, pandas as pd
from utils.rca_logger import log_prediction
from utils.prophet_helper import load_prophet_model, forecast_next
from monitoring.drift_detector import detect_and_flag

API = "http://localhost:8000/predict"
WINDOW = 5                 # sequence length
SLEEP  = 5                 # seconds

# Load Prophet model once at start
prophet_model = load_prophet_model()

def latest_parquets(n):
    """Return last n parquet file paths."""
    return sorted(glob.glob("data/processed/part-*.parquet"))[-n:]

while True:
    files = latest_parquets(WINDOW + 1)
    if len(files) < WINDOW + 1:
        time.sleep(1)
        continue

    # Load sequence and target from parquet files
    df = pd.concat([pd.read_parquet(f) for f in files])
    seq    = df["temperature"].iloc[:-1].tolist()
    actual = df["temperature"].iloc[-1]

    # -------- GRU ----------
    try:
        response = requests.post(API, json={"sequence": [[x] for x in seq]})
        response.raise_for_status()
        forecast = response.json()["forecast"]
        log_prediction(seq, forecast, actual, model="GRU")
    except Exception as e:
        print(f"[GRU] Prediction failed: {e}")

    # -------- Prophet -------
    try:
        prophet_fc = forecast_next(df, prophet_model)
        log_prediction(seq, prophet_fc, actual, model="Prophet")
    except Exception as e:
        print(f"[Prophet] Prediction failed: {e}")

    # -------- Drift Detection --------
    detect_and_flag()

    time.sleep(SLEEP)
