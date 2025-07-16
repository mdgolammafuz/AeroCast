import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# Step 1: Load preprocessed data
print("🔍 Loading data from 'data/processed/' ...")
df = pd.read_parquet("data/processed/")
print("✅ Data loaded:", df.shape)

# Step 2: Sort by timestamp
df = df.sort_values("timestamp")

# Step 3: Normalize features (optional for now, simple scaling)
features = ["temperature", "humidity", "rainfall"]
X = df[features].values

# Step 4: Create sequences
SEQ_LEN = 10

def create_sequences(data, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i+seq_len])
        y_seq.append(data[i+seq_len][0])  # Predict next temperature
    return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

X_tensor, y_tensor = create_sequences(X, SEQ_LEN)

# Step 5: Define Dataset
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = WeatherDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 6: Define GRU Model
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

model = GRURegressor(input_dim=3, hidden_dim=64, output_dim=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train
EPOCHS = 5
print("🚀 Training GRU...")
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in dataloader:
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📉 Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Step 8: Save the model
model_path = "model/gru_weather_forecaster.pt"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")
