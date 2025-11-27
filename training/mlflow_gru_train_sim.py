import os
import sys
import time
import datetime
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
from mlflow.models.signature import infer_signature
import importlib.util

# Add root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.gru_model import GRUWeatherForecaster

# --- Configuration ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "data", "training_data_sim.csv")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "gru_weather_forecaster.pt")
LAST_RETRAIN_FILE = os.path.join(ROOT, "last_retrain.txt")

# Queue Config
QUEUE_ENABLED = os.environ.get("QUEUE_ENABLED", "0") == "1"
QUEUE_URL = os.environ.get("QUEUE_URL", "http://queue:8081")

# Model Hyperparameters
WINDOW = 24
N_FEATS = 3
MODEL_NAME = "AeroCast-GRU-SIM"


# Helper to run the data generation script programmatically
def regenerate_training_data():
    print("[train-sim] Regenerating CSV from latest Parquet data...")
    try:
        script_path = os.path.join(ROOT, "data", "generate_training_data_sim.py")
        if os.path.exists(script_path):
            spec = importlib.util.spec_from_file_location("gen_sim", script_path)
            gen_sim = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gen_sim)
            if hasattr(gen_sim, "main"):
                gen_sim.main()
            print("[train-sim] CSV regeneration complete.")
        else:
            print("[train-sim] Data generation script not found, skipping.")
    except Exception as e:
        print(f"[train-sim] Failed to regenerate data: {e}")


class SimTSDataset(Dataset):
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data not found at {path}")
            
        df = pd.read_csv(path).astype("float32")
        X_raw = df.iloc[:, :-1].values
        N = X_raw.shape[0]
        
        # Auto-pad features if we only have 1D data but model expects 3D
        if X_raw.shape[1] == WINDOW:
            X_expanded = torch.zeros((N, WINDOW, N_FEATS))
            X_tensor = torch.tensor(X_raw)
            X_expanded[:, :, 0] = X_tensor
            self.X = X_expanded
        else:
            self.X = torch.tensor(X_raw).reshape(N, WINDOW, N_FEATS)
            
        self.y = torch.tensor(df["target"].values).reshape(N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_reason_from_queue() -> str:
    if not QUEUE_ENABLED:
        return "none"

    try:
        resp = requests.get(f"{QUEUE_URL}/subscribe", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            event_msg = data.get("event", "")
            if "drift" in event_msg: return "drift"
            if "schedule" in event_msg: return "schedule"
            return "queue_event"
        elif resp.status_code == 204:
            return "none"
    except Exception:
        pass
    return "none"


def train_model(reason: str):
    print(f"[train-sim] Starting training run... (Reason: {reason})")
    
    # For setup/drift, we ensure data is fresh
    if reason in ["drift", "manual_setup"]:
        regenerate_training_data()

    mlflow.set_experiment("AeroCast-GRU-SIM")
    mlflow.pytorch.autolog(log_models=False)

    try:
        ds = SimTSDataset(CSV)
        if len(ds) == 0:
            print("[train-sim] Dataset empty. Skipping.")
            return
        batch_size = min(16, len(ds))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"[train-sim] Failed to load dataset: {e}")
        return

    model = GRUWeatherForecaster(input_dim=N_FEATS, hidden_dim=16, output_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    t0 = time.perf_counter()

    with mlflow.start_run() as run:
        mlflow.set_tag("run_reason", reason)
        
        model.train()
        for epoch in range(30):
            tot_loss = 0.0
            batch_count = 0
            for X, y in loader:
                opt.zero_grad()
                out = model(X)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                tot_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = tot_loss / batch_count
                mlflow.log_metric("loss", avg_loss, step=epoch)

        model.eval()
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        
        try:
            X_sample, _ = next(iter(loader))
            sig = infer_signature(X_sample.numpy(), model(X_sample).detach().numpy())
            mlflow.pytorch.log_model(model, "model", input_example=X_sample.numpy(), signature=sig)
            mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
        except Exception as e:
            print(f"[train-sim] MLflow model logging warning: {e}")

    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(datetime.datetime.utcnow().isoformat())

    dur = time.perf_counter() - t0
    print(f"[train-sim] Training complete in {dur:.2f}s.")


def run_daemon():
    print("[train-sim] Daemon started. Waiting for queue events...")
    
    # Cold start check (auto-heal on startup)
    if not os.path.exists(MODEL_PATH):
        print("[train-sim] Cold start: Training initial model.")
        train_model("initial_cold_start")

    while True:
        reason = get_reason_from_queue()
        if reason != "none":
            print(f"[train-sim] Received event from queue: {reason}")
            train_model(reason)
        time.sleep(5)

if __name__ == "__main__":
    # CLI ARGUMENT HANDLING
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        print("[train-sim] Running in One-Shot Setup Mode.")
        train_model("manual_setup")
    else:
        # Default to Daemon (Docker usage)
        run_daemon()