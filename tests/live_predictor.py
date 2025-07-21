import sys
import os
# Extend sys path to import from model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import torch
from model.train_gru import GRURegressor
from monitoring.anomaly_detector import is_anomalous  # ✅ NEW

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

previous_pred = None  # ✅ NEW

while True:
    try:
        X_live = load_recent_sequence()
        with torch.no_grad():
            prediction = model(X_live).item()

        # ✅ Anomaly check
        if is_anomalous(prediction, previous_pred):
            print(f"🚨 Anomaly Detected → Prediction jumped to {prediction:.2f}°C")
        else:
            print(f"🌡️ Predicted next temperature → {prediction:.2f}°C")

        previous_pred = prediction

    except Exception as e:
        print(f"⚠️ Prediction skipped due to error: {e}")

    time.sleep(5)
