import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.train_gru import GRURegressor
import pandas as pd
import torch

print("🔍 Loading Parquet data...")
df = pd.read_parquet("data/processed/")
df = df.sort_values("timestamp")

SEQ_LEN = 10
features = ["temperature", "humidity", "rainfall"]
X = df[features].values

# Create sequences
X_seq = []
for i in range(len(X) - SEQ_LEN):
    X_seq.append(X[i:i+SEQ_LEN])
X_tensor = torch.tensor(X_seq, dtype=torch.float32)

# Load model
model = GRURegressor(input_dim=3, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("model/gru_weather_forecaster.pt"))
model.eval()

# Predict
with torch.no_grad():
    prediction = model(X_tensor[-1:])  # ✅ Already in [1, 10, 3] shape
    print("🌡️  Next predicted temperature:", prediction.item())
