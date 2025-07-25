# training/mlflow_gru_train.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import os
from datetime import datetime

# ----- Drift Hook Config -----
FLAG_PATH = "retrain.flag"
LOG_PATH = "logs/drift.log"

def check_and_reset_flag():
    if os.path.exists(FLAG_PATH):
        with open(LOG_PATH, "a") as f:
            f.write(f"[{datetime.now()}] Retraining triggered by drift detection.\n")
        os.remove(FLAG_PATH)
        return True
    return False

# ----- Training Config -----
SEQ_LENGTH = 5
BATCH_SIZE = 2
EPOCHS = 5
HIDDEN_DIM = 16
LR = 0.01

# ----- Simulate Minimal Data -----
series = np.array([i * 0.1 for i in range(20)])  # 0.0 to 1.9
X = np.array([series[i:i+SEQ_LENGTH] for i in range(len(series)-SEQ_LENGTH)])
y = np.array([series[i+SEQ_LENGTH] for i in range(len(series)-SEQ_LENGTH)])

# ----- Dataset Wrapper -----
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----- Model Definition -----
class GRUForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out

# ----- Training Loop -----
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ----- MLflow Logging -----
def run_training():
    dataset = TimeSeriesDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = GRUForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    mlflow.set_experiment("AeroCast-GRU-Training")
    with mlflow.start_run():
        mlflow.log_param("hidden_dim", HIDDEN_DIM)
        mlflow.log_param("lr", LR)
        mlflow.log_param("seq_length", SEQ_LENGTH)

        for epoch in range(EPOCHS):
            epoch_loss = train_one_epoch(model, loader, loss_fn, optimizer)
            mlflow.log_metric("loss", epoch_loss, step=epoch)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        # Save model
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/gru_weather_forecaster.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

# ----- Hook Entry Point -----
if __name__ == "__main__":
    if not check_and_reset_flag():
        print("No drift flag found. Skipping retraining.")
        exit()
    run_training()
