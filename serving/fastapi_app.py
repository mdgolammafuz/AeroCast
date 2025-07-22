from fastapi import FastAPI
from pydantic import BaseModel
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
# Define FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# Define Pydantic Input Schema
# -----------------------------
class WeatherInput(BaseModel):
    sequence: List[List[float]]

# -----------------------------
# Root Endpoint for Health
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "AeroCast++ GRU Predictor is running!"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: WeatherInput):
    sequence = np.array(data.sequence, dtype=np.float32)
    sequence_tensor = torch.tensor(sequence).unsqueeze(0)  # shape: (1, seq_len, input_dim)
    
    with torch.no_grad():
        forecast = model(sequence_tensor).item()

    return {"forecast": forecast}
