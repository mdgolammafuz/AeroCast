import os
import sys
import glob
import time
from datetime import datetime

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from utils.prophet_helper import load_prophet_model, forecast_next

API_URL = "http://localhost:8000/predict"
LOG_DIR = os.path.join(ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "prediction_history.csv")
PARQUET_DIR = os.path.join(ROOT, "data", "processed")

WINDOW = 5
os.makedirs(LOG_DIR, exist_ok=True)

prophet_model = load_prophet_model()


def ensure_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,sequence,forecast,actual,error,model\n")


def load_last_n_parquets(n=5) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "part-*.parquet")))
    if not files:
        raise FileNotFoundError("no parquet files yet in data/processed")
    chunk = files[-n:]
    dfs = [pd.read_parquet(f) for f in chunk]
    df = pd.concat(dfs, ignore_index=True)
    return df[["ts", "temperature"]]


def append_row(ts, seq, forecast, actual, error, model):
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},\"{seq}\",{forecast},{actual},{error},{model}\n")


def main():
    ensure_header()
    print("[Feeder] running… CTRL-C to stop")

    while True:
        try:
            df_latest = load_last_n_parquets(5)
        except FileNotFoundError:
            print("[Feeder] waiting for parquet…")
            time.sleep(5)
            continue

        temps = df_latest["temperature"].tolist()
        if len(temps) < WINDOW:
            print(f"[Feeder] only {len(temps)} readings, need {WINDOW}, retrying…")
            time.sleep(5)
            continue

        seq = temps[-WINDOW:]
        actual = float(temps[-1])

        resp = requests.post(
            API_URL,
            json={"sequence": [[x] for x in seq]},
            timeout=5,
        )
        resp.raise_for_status()
        gru_fc = float(resp.json()["forecast"])

        prophet_fc = forecast_next(df_latest, prophet_model, freq="5s")

        ts = datetime.utcnow().isoformat()

        err_gru = abs(gru_fc - actual)
        append_row(ts, seq, gru_fc, actual, err_gru, "GRU")

        err_prophet = abs(prophet_fc - actual)
        append_row(ts, seq, prophet_fc, actual, err_prophet, "Prophet")

        print(f"[Feeder] {ts}  GRU={gru_fc:.2f}  Prophet={prophet_fc:.2f}  actual={actual:.2f}")
        time.sleep(5)


if __name__ == "__main__":
    main()
