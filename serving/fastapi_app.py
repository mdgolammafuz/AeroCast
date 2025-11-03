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

from model.gru_model import GRUWeatherForecaster
from utils.rca_logger import log_prediction

PREDICTIONS = Counter("predictions_total", "Total predictions", ["model_type"])
ERR_GAUGE = Gauge("forecast_error", "Absolute error", ["model_type"])
LAT_SUMMARY = Summary(
    "aerocast_predict_latency_ms",
    "Prediction latency (ms) for AeroCast GRU endpoint",
)

gru = GRUWeatherForecaster(input_dim=3, hidden_dim=16, output_dim=1)
ARTIFACT_PATH = os.path.join("artifacts", "gru_weather_forecaster.pt")
if os.path.exists(ARTIFACT_PATH):
    gru.load_state_dict(torch.load(ARTIFACT_PATH, map_location="cpu"))
gru.eval()

app = FastAPI(title="AeroCast++ API", version="1.0")


class SequenceIn(BaseModel):
    sequence: list[list[float]] = Field(..., min_items=5)


def _predict(model, seq):
    start = time.perf_counter()
    arr = np.array(seq, dtype=np.float32)  # (T, 3)
    tensor = torch.tensor(arr).unsqueeze(0)  # (1, T, 3)
    with torch.no_grad():
        out = model(tensor).item()
    dur_ms = (time.perf_counter() - start) * 1000.0
    LAT_SUMMARY.observe(dur_ms)
    return out


@app.post("/predict")
def predict_gru(data: SequenceIn):
    seq = data.sequence
    for step in seq:
        if len(step) != 3:
            raise HTTPException(
                status_code=400,
                detail="each timestep must have exactly 3 features: [temperature, humidity, rainfall]",
            )
    try:
        fc = round(_predict(gru, seq), 2)
        ERR_GAUGE.labels("GRU").set(0)
        PREDICTIONS.labels("GRU").inc()
        return {"forecast": fc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
def healthz():
    return {"ok": True}
