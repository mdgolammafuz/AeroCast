import os, sys, time
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
from mlflow.models.signature import infer_signature
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway

# Import Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.gru_model import GRUWeatherForecaster

# Config
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "data", "training_data.csv")
QUEUE_URL = os.environ.get("QUEUE_URL", "http://queue:8081")
PUSHGATEWAY = os.environ.get("PUSHGATEWAY_HOST", "localhost:9091")

# Metrics
registry = CollectorRegistry()
RETRAIN_COUNT = Gauge("aerocast_retrain_count", "Total retrains", registry=registry)

MODEL_NAME = "AeroCast-GRU-Model"
WINDOW = 24
N_FEATS = 3

class TSDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path).astype("float32")
        X = df.iloc[:, :-1].values
        self.X = torch.tensor(X).reshape(X.shape[0], WINDOW, N_FEATS)
        self.y = torch.tensor(df["target"].values).reshape(X.shape[0], 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_model(reason: str):
    print(f"[trainer] Starting training... Reason: {reason}")
    mlflow.set_experiment("AeroCast-GRU")
    
    try:
        ds = TSDataset(CSV)
        loader = DataLoader(ds, batch_size=16, shuffle=True)
    except Exception as e:
        print(f"[trainer] Failed to load data: {e}")
        return

    model = GRUWeatherForecaster(input_dim=N_FEATS, hidden_dim=16, output_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    with mlflow.start_run() as run:
        mlflow.set_tag("reason", reason)
        
        # Training Loop
        for epoch in range(15): # Fast training for demo
            model.train()
            for X, y in loader:
                opt.zero_grad()
                loss_fn(model(X), y).backward()
                opt.step()
        
        # Save Artifacts
        model.eval()
        os.makedirs(f"{ROOT}/artifacts", exist_ok=True)
        torch.save(model.state_dict(), f"{ROOT}/artifacts/gru_weather_forecaster.pt")
        
        # Log to MLflow
        try:
            sample = next(iter(loader))[0]
            sig = infer_signature(sample.numpy(), model(sample).detach().numpy())
            mlflow.pytorch.log_model(model, "model", signature=sig)
            mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
        except Exception as e:
            print(f"[trainer] MLflow logging warning: {e}")

    # Metrics
    try:
        RETRAIN_COUNT.inc()
        push_to_gateway(PUSHGATEWAY, job="aerocast_trainer", registry=registry)
    except: pass
    
    print(f"[trainer] Training complete. Artifact updated.")

def run_daemon():
    # 1. Initial Training (Ensure model exists on startup)
    print("[trainer] performing initial cold-start training...")
    train_model("initial_startup")

    # 2. Event Loop
    print(f"[trainer] entering event loop, listening to {QUEUE_URL}...")
    while True:
        try:
            resp = requests.get(f"{QUEUE_URL}/subscribe", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                event = data.get("event", "unknown")
                print(f"[trainer] Received event: {event}")
                train_model(event)
            elif resp.status_code == 204:
                time.sleep(1) # Queue empty, brief pause
            else:
                print(f"[trainer] queue error: {resp.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"[trainer] connection error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_daemon()