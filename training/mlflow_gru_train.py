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

CSV = "data/training_data.csv"
FLAG = "retrain.flag"
LAST_RETRAIN_FILE = "last_retrain.txt"

registry = CollectorRegistry()

LOSS_GAUGE = Gauge("training_loss", "GRU loss", ["epoch", "reason"], registry=registry)
RETRAIN_TOTAL = Counter(
    "aerocast_retrain_total", "Total AeroCast GRU retrains", registry=registry
)
RETRAIN_DUR_S = Summary(
    "aerocast_retrain_duration_seconds",
    "Duration of GRU retrain runs (s)",
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


def reason():
    if os.path.exists(FLAG):
        print(f"[train] {FLAG} FOUND at {os.path.abspath(FLAG)}, marking run as drift and deleting it.")
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
            push_to_gateway("localhost:9091", job="aerocast_training", registry=registry)

        os.makedirs("artifacts", exist_ok=True)
        torch.save(model.state_dict(), "artifacts/gru_weather_forecaster.pt")

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
    if rsn == "drift":
        RETRAIN_TOTAL.inc()

    push_to_gateway("localhost:9091", job="aerocast_training", registry=registry)

    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(datetime.datetime.utcnow().isoformat())

    print(f"[train] done. reason={rsn} duration={dur:.2f}s")


if __name__ == "__main__":
    train()
