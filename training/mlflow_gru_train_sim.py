import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests  # NEW: For Queue polling

import mlflow
from mlflow.models.signature import infer_signature

from model.gru_model import GRUWeatherForecaster

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "data", "training_data_sim.csv")
FLAG = os.path.join(ROOT, "retrain.flag")
LAST_RETRAIN_FILE = os.path.join(ROOT, "last_retrain.txt")

# --- NEW: Queue Configuration ---
QUEUE_ENABLED = os.environ.get("QUEUE_ENABLED", "0") == "1"
QUEUE_URL = os.environ.get("QUEUE_URL", "http://queue:8081")

WINDOW = 24
N_FEATS = 3              # match API + fake 3D
MODEL_NAME = "AeroCast-GRU-SIM"


class SimTSDataset(Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path).astype("float32")
        X_raw = df.iloc[:, :-1].values
        N = X_raw.shape[0]
        self.X = torch.tensor(X_raw).reshape(N, WINDOW, N_FEATS)
        self.y = torch.tensor(df["target"].values).reshape(N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_reason_and_clear_flag() -> str:
    """
    Checks both the local file and the Go Event Queue for a retrain signal.
    """
    reason = "unknown"
    found = False

    # 1. Legacy: Check File
    if os.path.exists(FLAG):
        txt = open(FLAG, "r").read().strip()
        os.remove(FLAG)
        # Simple parsing of "drift <timestamp>" or "schedule <timestamp>"
        if txt.startswith("drift"):
            reason = "drift"
        elif txt.startswith("schedule"):
            reason = "schedule"
        found = True
        print(f"[train-sim] found flag file: {reason}")

    # 2. Modern: Check Queue (if enabled)
    if QUEUE_ENABLED and not found:
        try:
            # Poll the Go service
            resp = requests.get(f"{QUEUE_URL}/subscribe", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                event_msg = data.get("event", "")
                if "drift" in event_msg:
                    reason = "drift"
                elif "schedule" in event_msg:
                    reason = "schedule"
                found = True
                print(f"[train-sim] received event from queue: {event_msg}")
        except Exception as e:
            print(f"[train-sim] queue check failed: {e}")

    if not found:
        return "none"  # Signal caller to skip

    return reason


def train():
    # Check for trigger
    reason = get_reason_and_clear_flag()
    if reason == "none":
        # No work to do
        return

    mlflow.set_experiment("AeroCast-GRU-SIM")
    mlflow.pytorch.autolog(log_models=False)

    ds = SimTSDataset(CSV)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    model = GRUWeatherForecaster(input_dim=N_FEATS, hidden_dim=16, output_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.MSELoss()

    print(f"[train-sim] starting run. reason={reason}")
    
    with mlflow.start_run() as run:
        mlflow.set_tag("run_reason", reason)
        mlflow.log_param("window", WINDOW)
        mlflow.log_param("input_dim", N_FEATS)

        for epoch in range(30):
            tot = 0.0
            for X, y in loader:
                opt.zero_grad()
                out = model(X)
                loss = lossf(out, y)
                loss.backward()
                opt.step()
                tot += loss.item()
            mlflow.log_metric("loss", tot / len(loader), step=epoch)

        os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
        torch.save(
            model.state_dict(),
            os.path.join(ROOT, "artifacts", "gru_weather_forecaster.pt"),
        )

        X_sample, _ = next(iter(loader))
        sig = infer_signature(X_sample.numpy(), model(X_sample).detach().numpy())
        mlflow.pytorch.log_model(model, "model", input_example=X_sample.numpy(), signature=sig)
        mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)

    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(datetime.datetime.utcnow().isoformat())

    print(f"[train-sim] done. reason={reason}")


if __name__ == "__main__":
    train()