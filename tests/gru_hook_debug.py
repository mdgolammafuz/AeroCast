import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from model.train_gru import GRURegressor

# Load full processed dataset (not just one file)
print("Loading full dataset from 'data/processed/' ...")
df = pd.read_parquet("data/processed/")  # ← reads entire folder if saved via Spark
df = df.sort_values("timestamp")

# Just grab the latest 10 rows
SEQ_LEN = 10
features = ["temperature", "humidity", "rainfall"]
latest_sequence = df[features].tail(SEQ_LEN).values  # shape (10, 3)

# Convert to tensor shape [1, 10, 3] for GRU
X_tensor = torch.tensor(latest_sequence, dtype=torch.float32).unsqueeze(0)

# Load trained model
model = GRURegressor(input_dim=3, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("model/gru_weather_forecaster.pt"))
model.eval()

# Make prediction
with torch.no_grad():
    prediction = model(X_tensor)
    print("🌡️ Next predicted temperature:", prediction.item())
