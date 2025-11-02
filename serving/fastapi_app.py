from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Gauge,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import torch, numpy as np
import time

from model.gru_model import GRUWeatherForecaster
from utils.rca_logger import log_prediction

# ---- Prometheus metrics ----
PREDICTIONS = Counter("predictions_total", "Total predictions", ["model_type"])
ERR_GAUGE = Gauge("forecast_error", "Absolute error", ["model_type"])
LAT_SUMMARY = Summary(
    "aerocast_predict_latency_ms",
    "Prediction latency (ms) for AeroCast GRU endpoint",
)

# ---- Load GRU model ----
# NOTE: input_dim=1 right now; multivariate will change later
gru = GRUWeatherForecaster(1, 16, 1)
gru.load_state_dict(torch.load("artifacts/gru_weather_forecaster.pt"))
gru.eval()

app = FastAPI(title="AeroCast++ API", version="1.0")


# ---- Input Validation ----
class SequenceIn(BaseModel):
    sequence: list[list[float]] = Field(..., min_items=5)


def _predict(model, seq):
    start = time.perf_counter()
    tensor = torch.tensor(np.array(seq, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor).item()
    dur_ms = (time.perf_counter() - start) * 1000.0
    LAT_SUMMARY.observe(dur_ms)
    return out


# ---------- GRU Endpoint ----------
@app.post("/predict")
def predict_gru(data: SequenceIn):
    try:
        fc = round(_predict(gru, data.sequence), 2)
        # we can set forecast_error later when live feeder sends actuals
        ERR_GAUGE.labels("GRU").set(0)
        PREDICTIONS.labels("GRU").inc()
        return {"forecast": fc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ---------- Prometheus Metrics ----------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------- Health ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}
