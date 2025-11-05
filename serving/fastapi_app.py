from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Gauge,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import torch
import numpy as np
import time
import os
import glob
import pandas as pd
from datetime import datetime

from model.gru_model import GRUWeatherForecaster
from utils.rca_logger import log_prediction  # keep your logger

# ---- Prom metrics ----
PREDICTIONS = Counter("predictions_total", "Total predictions", ["model_type"])
ERR_GAUGE = Gauge("forecast_error", "Absolute error", ["model_type"])
LAT_SUMMARY = Summary(
    "aerocast_predict_latency_ms",
    "Prediction latency (ms) for AeroCast GRU endpoint",
)
RMSE_GAUGE = Gauge(
    "aerocast_last_rmse",
    "last reported RMSE (client-evaluated)",
)

# in-memory RMSE state
_MSE_SUM = 0.0
_MSE_COUNT = 0

# ---- model load ----
gru = GRUWeatherForecaster(input_dim=3, hidden_dim=16, output_dim=1)
ARTIFACT_PATH = os.path.join("artifacts", "gru_weather_forecaster.pt")
if os.path.exists(ARTIFACT_PATH):
    gru.load_state_dict(torch.load(ARTIFACT_PATH, map_location="cpu"))
gru.eval()

app = FastAPI(title="AeroCast++ API", version="1.0")

# Spark bronze writes here
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET_DIR = os.path.join(ROOT, "data", "processed", "noaa")


class SequenceIn(BaseModel):
    # 5 timesteps, each: [temperature, windspeed, pressure]
    sequence: list[list[float]] = Field(..., min_items=5)


class EvalPayload(BaseModel):
    predicted: float
    actual: float


def _predict(model, seq):
    start = time.perf_counter()
    arr = np.array(seq, dtype=np.float32)  # (5, 3)
    tensor = torch.tensor(arr).unsqueeze(0)  # (1, 5, 3)
    with torch.no_grad():
        out = model(tensor).item()
    dur_ms = (time.perf_counter() - start) * 1000.0
    LAT_SUMMARY.observe(dur_ms)
    return out


def _latest_parquet_df() -> pd.DataFrame:
    if not os.path.isdir(PARQUET_DIR):
        raise FileNotFoundError(f"{PARQUET_DIR} does not exist")

    files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {PARQUET_DIR}")

    newest = max(files, key=os.path.getmtime)
    df = pd.read_parquet(newest)

    # drop kafka/producer extras if present
    for col in ("v", "station"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")

    return df


@app.post("/predict")
def predict_gru(data: SequenceIn):
    seq = data.sequence
    for step in seq:
        if len(step) != 3:
            raise HTTPException(
                status_code=400,
                detail="each timestep must have exactly 3 features: [temperature, windspeed, pressure]",
            )
    try:
        fc = round(_predict(gru, seq), 2)
        try:
            log_prediction({"forecast": fc})
        except Exception:
            # don't kill the API if logging fails
            pass

        ERR_GAUGE.labels("GRU").set(0)
        PREDICTIONS.labels("GRU").inc()
        return {"forecast": fc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/latest")
def latest():
    try:
        df = _latest_parquet_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    row = df.iloc[-1].to_dict()
    return {"row": row}


@app.post("/eval")
def eval_model(payload: EvalPayload):
    global _MSE_SUM, _MSE_COUNT
    err = payload.actual - payload.predicted
    _MSE_SUM += err * err
    _MSE_COUNT += 1
    rmse = (_MSE_SUM / _MSE_COUNT) ** 0.5
    RMSE_GAUGE.set(rmse)
    return {"rmse": rmse, "count": _MSE_COUNT}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()}
