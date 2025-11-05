# streaming/client/live_predictor_feeder.py
import os
import sys
import glob
import time
from datetime import datetime

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

API_URL = "http://localhost:8000/predict"
EVAL_URL = "http://localhost:8000/eval"
LOG_DIR = os.path.join(ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "prediction_history.csv")

PARQUET_DIR = os.path.join(ROOT, "data", "processed", "noaa")

WINDOW = 5
os.makedirs(LOG_DIR, exist_ok=True)


def ensure_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,sequence,forecast,actual,error,model\n")


def _latest_parquet_files(n=5):
    files = glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))
    # keep only real files with nonzero size
    files = [f for f in files if os.path.getsize(f) > 0]
    if not files:
        return []
    files.sort(key=os.path.getmtime)
    return files[-n:]


def load_last_n_rows(n=5) -> pd.DataFrame:
    files = _latest_parquet_files(n)
    if not files:
        raise FileNotFoundError("no parquet files yet in data/processed/noaa/")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            # skip any half-written file
            continue
    if not dfs:
        raise FileNotFoundError("no readable parquet files in data/processed/noaa/")
    df = pd.concat(dfs, ignore_index=True)

    df = df[["ts", "temperature", "windspeed", "pressure"]]
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def append_row(ts, seq, forecast, actual, error, model):
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},\"{seq}\",{forecast},{actual},{error},{model}\n")


def main():
    ensure_header()
    print("[Feeder] running on NOAA bronze… CTRL-C to stop")

    while True:
        try:
            df_latest = load_last_n_rows(5)
        except FileNotFoundError:
            print("[Feeder] waiting for NOAA parquet…")
            time.sleep(5)
            continue

        if len(df_latest) < WINDOW:
            print(f"[Feeder] only {len(df_latest)} readings, need {WINDOW}, retrying…")
            time.sleep(5)
            continue

        window_df = df_latest.iloc[-WINDOW:]

        seq = []
        for _, r in window_df.iterrows():
            seq.append([
                float(r["temperature"]),
                float(r["windspeed"]),
                float(r["pressure"]),
            ])

        actual = float(window_df.iloc[-1]["temperature"])

        resp = requests.post(
            API_URL,
            json={"sequence": seq},
            timeout=5,
        )
        resp.raise_for_status()
        forecast = float(resp.json()["forecast"])

        try:
            requests.post(
                EVAL_URL,
                json={"predicted": forecast, "actual": actual},
                timeout=3,
            )
        except Exception:
            pass

        ts = datetime.utcnow().isoformat()
        err = abs(forecast - actual)
        append_row(ts, seq, forecast, actual, err, "GRU")

        print(f"[Feeder] {ts} GRU={forecast:.2f} actual={actual:.2f} err={err:.2f}")
        time.sleep(5)


if __name__ == "__main__":
    main()
