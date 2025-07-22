from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import torch
import torch.nn as nn
import numpy as np

# -----------------------------
# Load GRU Model Definition
# -----------------------------
class GRUWeatherForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUWeatherForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n.squeeze(0))
        return out

# -----------------------------
# Load Model & Set to Eval
# -----------------------------
input_dim = 1
hidden_dim = 16
output_dim = 1

model = GRUWeatherForecaster(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("artifacts/gru_weather_forecaster.pt"))
model.eval()

# -----------------------------
# Initialize FastAPI App
# -----------------------------
app = FastAPI(
    title="AeroCast++ GRU Forecasting API",
    description="Serve real-time weather predictions using a trained GRU model.",
    version="1.0.0"
)

# -----------------------------
# Pydantic Input Schema with Validation
# -----------------------------
class WeatherInput(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        title="Sensor Input Sequence",
        description="2D list representing weather sensor time series: shape (seq_len, 1)",
        example=[[22.3], [22.1], [21.9], [22.0], [22.4]],
        min_items=5
    )

# -----------------------------
# Root Health Check
# -----------------------------
@app.get("/", summary="Health Check")
def read_root():
    return {"message": "✅ AeroCast++ GRU Predictor is Live"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict", summary="Generate Weather Forecast")
def predict(data: WeatherInput):
    try:
        sequence = np.array(data.sequence, dtype=np.float32)  # (seq_len, 1)
        sequence_tensor = torch.tensor(sequence).unsqueeze(0)  # (1, seq_len, 1)

        with torch.no_grad():
            forecast = model(sequence_tensor).item()

        return {
            "input_length": len(sequence),
            "forecast": round(forecast, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
