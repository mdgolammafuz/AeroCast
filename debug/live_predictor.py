import sys
import os

# Extend sys path to import from model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
from model.train_gru import GRURegressor

SEQ_LEN = 10
features = ["temperature", "humidity", "rainfall"]

def load_recent_sequence(file_path="data/processed/"):
    df = pd.read_parquet(file_path, engine="pyarrow")
    df = df.sort_values("timestamp")
    
    if len(df) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} rows for prediction. Found {len(df)}")
    
    recent_df = df[features].values[-SEQ_LEN:]
    recent_seq = torch.tensor(recent_df, dtype=torch.float32).unsqueeze(0)  # [1, 10, 3]
    return recent_seq

# Load trained model
model = GRURegressor(input_dim=3, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("model/gru_weather_forecaster.pt"))
model.eval()

print("🔁 Starting real-time prediction loop...\n")

while True:
    try:
        X_live = load_recent_sequence()
        with torch.no_grad():
            prediction = model(X_live).item()
        print(f"🌡️ Predicted next temperature → {prediction:.2f}°C")
    except Exception as e:
        print(f"⚠️ Prediction skipped due to error: {e}")

    time.sleep(5)
