"""
Feeds latest sensor data into FastAPI, logs GRU and Prophet predictions,
and triggers drift detection.

Requirements:
- FastAPI server with GRU endpoint running at http://localhost:8000/predict
- Prophet model saved as artifacts/prophet_model.json (saved via model_to_json)
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import glob
import time
import requests
import pandas as pd

from utils.rca_logger import log_prediction
from utils.prophet_helper import load_prophet_model, forecast_next
from monitoring.drift_detector import detect_and_flag

API = "http://localhost:8000/predict"
WINDOW = 5  # sequence length
SLEEP = 5   # seconds

# Load Prophet model once at start
prophet_model = load_prophet_model()

def latest_parquets(n: int):
    """Return last n parquet file paths."""
    return sorted(glob.glob("data/processed/part-*.parquet"))[-n:]

while True:
    files = latest_parquets(WINDOW + 1)
    if len(files) < WINDOW + 1:
        time.sleep(1)
        continue

    # Load sequence and target from parquet files
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    seq = df["temperature"].iloc[:-1].tolist()
    actual = df["temperature"].iloc[-1]
    ts_last = pd.to_datetime(df["ts"].iloc[-1])

    # -------- GRU ----------
    gru_fc = None
    try:
        resp = requests.post(API, json={"sequence": [[x] for x in seq]}, timeout=5)
        resp.raise_for_status()
        gru_fc = resp.json()["forecast"]
        log_prediction(seq, gru_fc, actual, model="GRU")
    except Exception as e:
        print(f"[GRU] Prediction failed: {e}", flush=True)

    # -------- Prophet -------
    prophet_fc = None
    try:
        prophet_fc = forecast_next(df[["ts", "temperature"]], prophet_model)
        log_prediction(seq, prophet_fc, actual, model="Prophet")
    except Exception as e:
        print(f"[Prophet] Prediction failed: {e}", flush=True)

    # -------- Drift Detection --------
    try:
        detect_and_flag()
    except Exception as e:
        print(f"[Drift] Detection failed: {e}", flush=True)

    # ---- to watch the progress ----
    print(f"[Feeder] {ts_last.isoformat()}  GRU={gru_fc}  Prophet={prophet_fc}  actual={actual}", flush=True)

    time.sleep(SLEEP)
