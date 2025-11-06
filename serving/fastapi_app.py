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
from datetime import datetime, timezone

from model.gru_model import GRUWeatherForecaster

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET_DIR = os.path.join(ROOT, "data", "processed", "noaa")
FLAG_FILE = os.path.join(ROOT, "retrain.flag")
LAST_RETRAIN_FILE = os.path.join(ROOT, "last_retrain.txt")

ROUTINE_MAX_AGE_SEC = 24 * 3600  # 1 day

PREDICTIONS = Counter("predictions_total", "Total predictions", ["model_type"])
ERR_GAUGE = Gauge("forecast_error", "Absolute error", ["model_type"])
LAT_SUMMARY = Summary(
    "aerocast_predict_latency_ms",
    "Prediction latency (ms) for AeroCast GRU endpoint",
)
RMSE_GAUGE = Gauge("aerocast_last_rmse", "last reported RMSE (client-evaluated)")

_MSE_SUM = 0.0
_MSE_COUNT = 0

gru = GRUWeatherForecaster(input_dim=3, hidden_dim=16, output_dim=1)
ART_PATH = os.path.join(ROOT, "artifacts", "gru_weather_forecaster.pt")
if os.path.exists(ART_PATH):
    gru.load_state_dict(torch.load(ART_PATH, map_location="cpu"))
gru.eval()

app = FastAPI(title="AeroCast++ API", version="1.0")


class SequenceIn(BaseModel):
    sequence: list[list[float]] = Field(..., min_items=5)


class EvalPayload(BaseModel):
    predicted: float
    actual: float


def _predict(model, seq):
    start = time.perf_counter()
    arr = np.array(seq, dtype=np.float32)
    tensor = torch.tensor(arr).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor).item()
    LAT_SUMMARY.observe((time.perf_counter() - start) * 1000.0)
    return out


def _latest_parquet_df() -> pd.DataFrame:
    if not os.path.isdir(PARQUET_DIR):
        raise FileNotFoundError(f"{PARQUET_DIR} does not exist")

    files = [
        f
        for f in glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
        if os.path.getsize(f) > 0
    ]
    if not files:
        raise FileNotFoundError(f"no parquet files in {PARQUET_DIR}")

    newest = max(files, key=os.path.getmtime)
    df = pd.read_parquet(newest)

    for col in ("v", "station"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.sort_values("ts")

    return df


def _read_last_retrain_ts():
    # 1) try file contents
    if os.path.exists(LAST_RETRAIN_FILE):
        raw = open(LAST_RETRAIN_FILE, "r").read().strip()
        if raw:
            try:
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
        # 2) fallback to mtime
        try:
            mtime = os.path.getmtime(LAST_RETRAIN_FILE)
            return datetime.fromtimestamp(mtime, tz=timezone.utc)
        except Exception:
            return None
    return None


@app.post("/predict")
def predict_gru(data: SequenceIn):
    seq = data.sequence
    for step in seq:
        if len(step) != 3:
            raise HTTPException(
                status_code=400,
                detail="each timestep must have exactly 3 features: [temperature, windspeed, pressure]",
            )
    fc = round(_predict(gru, seq), 2)
    ERR_GAUGE.labels("GRU").set(0)
    PREDICTIONS.labels("GRU").inc()
    return {"forecast": fc}


@app.get("/latest")
def latest():
    try:
        df = _latest_parquet_df()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"row": df.iloc[-1].to_dict()}


@app.post("/eval")
def eval_model(payload: EvalPayload):
    global _MSE_SUM, _MSE_COUNT

    err = payload.actual - payload.predicted
    abs_err = abs(err)

    _MSE_SUM += err * err
    _MSE_COUNT += 1
    rmse = (_MSE_SUM / _MSE_COUNT) ** 0.5
    RMSE_GAUGE.set(rmse)

    drift_flagged = False
    if abs_err > 5.0:
        with open(FLAG_FILE, "w") as f:
            f.write(f"drift {datetime.utcnow().isoformat()}")
        drift_flagged = True

    return {
        "rmse": rmse,
        "count": _MSE_COUNT,
        "abs_error": abs_err,
        "drift_flagged": drift_flagged,
    }


@app.post("/routine-retrain-check")
def routine_retrain_check():
    now = datetime.now(timezone.utc)
    last = _read_last_retrain_ts()

    if last is None:
        with open(FLAG_FILE, "w") as f:
            f.write(f"schedule {now.isoformat()}")
        return {"scheduled": True, "reason": "stale", "age_sec": None}

    age_sec = (now - last).total_seconds()
    if age_sec > ROUTINE_MAX_AGE_SEC:
        with open(FLAG_FILE, "w") as f:
            f.write(f"schedule {now.isoformat()}")
        return {"scheduled": True, "reason": "stale", "age_sec": age_sec}

    return {"scheduled": False, "reason": "fresh", "age_sec": age_sec}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()}
