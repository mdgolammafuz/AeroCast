from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
import torch, numpy as np

from model.gru_model   import GRUWeatherForecaster
from utils.rca_logger  import log_prediction

# ---- Prometheus metrics ----
PREDICTIONS = Counter("predictions_total", "Total predictions", ["model_type"])
ERR_GAUGE   = Gauge("forecast_error", "Absolute error", ["model_type"])

# ---- Load GRU model ----
gru = GRUWeatherForecaster(1,16,1)
gru.load_state_dict(torch.load("artifacts/gru_weather_forecaster.pt"))
gru.eval()

app = FastAPI(title="AeroCast++ API", version="1.0")

# ---- Input Validation ----
class SequenceIn(BaseModel):
    sequence: list[list[float]] = Field(..., min_items=5)

def _predict(model, seq):
    tensor = torch.tensor(np.array(seq, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        return model(tensor).item()

# ---------- GRU Endpoint ----------
@app.post("/predict")
def predict_gru(data: SequenceIn):
    try:
        fc = round(_predict(gru, data.sequence), 2)
        ERR_GAUGE.labels("GRU").set(0)  # will be updated later
        PREDICTIONS.labels("GRU").inc()
        return {"forecast": fc}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# ---------- Prometheus Metrics ----------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
