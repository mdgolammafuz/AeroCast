import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd, torch, mlflow, datetime, time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.gru_model import GRUWeatherForecaster
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    push_to_gateway,
    Counter,
    Summary,
)
from mlflow.models.signature import infer_signature

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "data", "training_data.csv")
FLAG = os.path.join(ROOT, "retrain.flag")
LAST_RETRAIN_FILE = os.path.join(ROOT, "last_retrain.txt")
LOG_DIR = os.path.join(ROOT, "logs")
RETRAIN_COUNT_FILE = os.path.join(LOG_DIR, "retrain_count.txt")  # NEW

os.makedirs(LOG_DIR, exist_ok=True)

registry = CollectorRegistry()

LOSS_GAUGE = Gauge("training_loss", "GRU loss", ["epoch", "reason"], registry=registry)
RETRAIN_TOTAL = Counter(
    "aerocast_retrain_total", "Total AeroCast GRU retrains (pushgateway raw)", registry=registry
)
RETRAIN_DUR_S = Summary(
    "aerocast_retrain_duration_seconds",
    "Duration of GRU retrain runs (s)",
    registry=registry,
)
# NEW: file-backed, authoritative for Grafana
RETRAIN_COUNT_G = Gauge(
    "aerocast_retrain_count",
    "File-backed total retrains (authoritative)",
    registry=registry,
)
# NEW: already there in your runs, keep it
LAST_RETRAIN_TS = Gauge(
    "aerocast_last_retrain_ts",
    "Unix timestamp (UTC) of last GRU retrain run",
    registry=registry,
)


class TS(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path).astype("float32")
        self.X = torch.tensor(df.values[:, :-1]).unsqueeze(-1)
        self.y = torch.tensor(df.values[:, -1]).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


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


def reason():
    if os.path.exists(FLAG):
        print(f"[train] {FLAG} FOUND at {FLAG}, marking run as drift and deleting it.")
        os.remove(FLAG)
        return "drift"
    return "initial"


def train():
    rsn = reason()
    mlflow.set_experiment("AeroCast-GRU")
    mlflow.pytorch.autolog(log_models=False)

    ds = TS(CSV)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    model = GRUWeatherForecaster(1, 16, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.MSELoss()

    t0 = time.perf_counter()
    with mlflow.start_run():
        mlflow.set_tag("run_reason", rsn)

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
            # we keep pushing to same job
            push_to_gateway("localhost:9091", job="aerocast_training", registry=registry)

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
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "AeroCast-GRU-Model",
        )

    dur = time.perf_counter() - t0
    RETRAIN_DUR_S.observe(dur)

    # --- NEW PART: file-backed counter + ts ---
    now = datetime.datetime.utcnow()
    now_ts = now.timestamp()
    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(now.isoformat())

    # load → maybe bump → save
    retrain_count = _load_retrain_count()
    if rsn == "drift":
        retrain_count += 1
        _save_retrain_count(retrain_count)
        # keep old counter too (even if pushgateway overwrites to 1)
        RETRAIN_TOTAL.inc()
    # always set gauges
    RETRAIN_COUNT_G.set(retrain_count)
    LAST_RETRAIN_TS.set(now_ts)
    # final push (overwrites is fine, it's gauges)
    push_to_gateway("localhost:9091", job="aerocast_training", registry=registry)

    print(f"[train] done. reason={rsn} duration={dur:.2f}s retrain_count={retrain_count}")


if __name__ == "__main__":
    train()
