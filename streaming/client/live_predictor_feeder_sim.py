# streaming/client/live_predictor_feeder_sim.py
import os
import sys
import glob
import time
from datetime import datetime

import pandas as pd
import requests

# project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

API_URL = "http://localhost:8000/predict"

PARQUET_DIR = os.path.join(ROOT, "data", "processed")  # simulator writes here
LOG_DIR = os.path.join(ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "prediction_history.csv")
os.makedirs(LOG_DIR, exist_ok=True)

WINDOW = 5  # same as model/API

# prophet helper from your repo
try:
    from utils.prophet_helper import load_prophet_model, forecast_next
    _PROPhet = load_prophet_model(os.path.join(ROOT, "artifacts", "prophet_model.json"))
    HAS_PROPHET = True
    print("[Feeder-SIM] Prophet loaded.")
except Exception as e:
    HAS_PROPHET = False
    print(f"[Feeder-SIM] Prophet NOT loaded: {e}")


def ensure_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,sequence,forecast,actual,error,model\n")


def _latest_parquet_files(n=10):
    files = glob.glob(os.path.join(PARQUET_DIR, "part-*.parquet"))
    files = [f for f in files if os.path.getsize(f) > 0]
    files.sort(key=os.path.getmtime)
    return files[-n:]


def load_last_n_rows(n=5) -> pd.DataFrame:
    files = _latest_parquet_files(n)
    if not files:
        raise FileNotFoundError("no simulator parquet files yet in data/processed/")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            continue
    if not dfs:
        raise FileNotFoundError("no readable simulator parquet files")
    df = pd.concat(dfs, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df[["ts", "temperature"]]


def append_row(ts, seq, forecast, actual, error, model):
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},\"{seq}\",{forecast},{actual},{error},{model}\n")


def main():
    ensure_header()
    print("[Feeder-SIM] running on simulated heatwave… CTRL-C to stop")

    while True:
        try:
            df_latest = load_last_n_rows(20)
        except FileNotFoundError:
            print("[Feeder-SIM] waiting for simulator parquet…")
            time.sleep(3)
            continue

        if len(df_latest) < WINDOW:
            print(f"[Feeder-SIM] only {len(df_latest)} rows, need {WINDOW}…")
            time.sleep(3)
            continue

        window_df = df_latest.iloc[-WINDOW:]
        actual = float(window_df.iloc[-1]["temperature"])

        # 3D fake for NOAA-style API
        seq = []
        for _, r in window_df.iterrows():
            temp = float(r["temperature"])
            seq.append([
                temp,      # temperature
                5.0,       # fake windspeed
                101000.0,  # fake pressure
            ])

        # 1) GRU via API
        try:
            resp = requests.post(API_URL, json={"sequence": seq}, timeout=5)
            resp.raise_for_status()
            gru_fc = float(resp.json()["forecast"])
        except Exception as e:
            print(f"[Feeder-SIM] predict failed: {e}")
            time.sleep(3)
            continue

        ts = datetime.utcnow().isoformat()
        gru_err = abs(gru_fc - actual)
        append_row(ts, seq, gru_fc, actual, gru_err, "GRU")
        print(f"[Feeder-SIM] {ts} GRU={gru_fc:.2f} actual={actual:.2f} err={gru_err:.2f}")

        # 2) Prophet locally (for Grafana baseline)
        if HAS_PROPHET:
            try:
                # prophet wants the full recent df, not the faked 3D
                # we already have df_latest above
                prophet_fc = forecast_next(df_latest, _PROPhet, freq="5s")
                prophet_err = abs(prophet_fc - actual)
                append_row(ts, "[prophet]", prophet_fc, actual, prophet_err, "Prophet")
            except Exception as e:
                print(f"[Feeder-SIM] prophet forecast failed: {e}")

        time.sleep(5)


if __name__ == "__main__":
    main()
