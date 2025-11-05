import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow
from mlflow.models.signature import infer_signature

from model.gru_model import GRUWeatherForecaster
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    push_to_gateway,
    Counter,
    Summary,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "data", "training_data.csv")
FLAG = os.path.join(ROOT, "retrain.flag")
LAST_RETRAIN_FILE = os.path.join(ROOT, "last_retrain.txt")
LOG_DIR = os.path.join(ROOT, "logs")
RETRAIN_COUNT_FILE = os.path.join(LOG_DIR, "retrain_count.txt")

os.makedirs(LOG_DIR, exist_ok=True)

registry = CollectorRegistry()

LOSS_GAUGE = Gauge("training_loss", "GRU loss", ["epoch", "reason"], registry=registry)
RETRAIN_TOTAL = Counter(
    "aerocast_retrain_total", "Total AeroCast GRU retrains (raw counter)", registry=registry
)
RETRAIN_DUR_S = Summary(
    "aerocast_retrain_duration_seconds",
    "Duration of GRU retrain runs (s)",
    registry=registry,
)
RETRAIN_COUNT_G = Gauge(
    "aerocast_retrain_count",
    "File-backed total retrains (authoritative for dashboard)",
    registry=registry,
)
LAST_RETRAIN_TS = Gauge(
    "aerocast_last_retrain_ts",
    "Unix timestamp (UTC) of last GRU retrain run",
    registry=registry,
)

WINDOW = 5
N_FEATS = 3
MODEL_NAME = "AeroCast-GRU-Model"


class TSDataset(Dataset):
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


def _load_retrain_count() -> int:
    if not os.path.exists(RETRAIN_COUNT_FILE):
        return 0
    try:
        return int(open(RETRAIN_COUNT_FILE, "r").read().strip() or "0")
    except Exception:
        return 0


def _save_retrain_count(n: int) -> None:
    with open(RETRAIN_COUNT_FILE, "w") as f:
        f.write(str(n))


def get_reason_and_clear_flag() -> str:
    if not os.path.exists(FLAG):
        return "initial"

    txt = open(FLAG, "r").read().strip()
    os.remove(FLAG)

    if txt.startswith("drift"):
        return "drift"
    if txt.startswith("schedule"):
        return "schedule"
    return "unknown"


def train():
    rsn = get_reason_and_clear_flag()
    mlflow.set_experiment("AeroCast-GRU")
    mlflow.pytorch.autolog(log_models=False)

    ds = TSDataset(CSV)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    model = GRUWeatherForecaster(input_dim=N_FEATS, hidden_dim=16, output_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.MSELoss()

    t0 = time.perf_counter()
    with mlflow.start_run() as run:
        mlflow.set_tag("run_reason", rsn)
        mlflow.set_tag("source_layer", "silver")
        mlflow.log_param("input_dim", N_FEATS)
        mlflow.log_param("window", WINDOW)

        for epoch in range(5):
            model.train()
            tot = 0.0
            for X, y in loader:
                opt.zero_grad()
                out = model(X)
                loss = lossf(out, y)
                loss.backward()
                opt.step()
                tot += loss.item()
            epoch_loss = tot / len(loader)
            mlflow.log_metric("loss", epoch_loss, step=epoch)
            LOSS_GAUGE.labels(str(epoch), rsn).set(epoch_loss)
            try:
                push_to_gateway("localhost:9091", job="aerocast_training", registry=registry)
            except Exception:
                pass

        model.eval()
        with torch.no_grad():
            preds = model(ds.X).numpy()
            targets = ds.y.numpy()
        rmse = float(((preds - targets) ** 2).mean() ** 0.5)
        mlflow.log_metric("rmse", rmse)

        os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ROOT, "artifacts", "gru_weather_forecaster.pt"))

        X_sample, _ = next(iter(loader))
        signature = infer_signature(
            X_sample.numpy(), model(X_sample).detach().numpy()
        )
        mlflow.pytorch.log_model(
            model, "model", input_example=X_sample.numpy(), signature=signature
        )
        mlflow.register_model(
            f"runs:/{run.info.run_id}/model",
            MODEL_NAME,
        )

    dur = time.perf_counter() - t0

    now = datetime.datetime.utcnow()
    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(now.replace(tzinfo=datetime.timezone.utc).isoformat())

    retrain_count = _load_retrain_count()
    if rsn in ("drift", "schedule"):
        retrain_count += 1
        _save_retrain_count(retrain_count)
        RETRAIN_TOTAL.inc()

    RETRAIN_COUNT_G.set(retrain_count)
    LAST_RETRAIN_TS.set(now.timestamp())
    try:
        push_to_gateway("localhost:9091", job="aerocast_training", registry=registry)
    except Exception:
        pass

    print(f"[train] done. reason={rsn} duration={dur:.2f}s retrain_count={retrain_count}")


if __name__ == "__main__":
    train()
