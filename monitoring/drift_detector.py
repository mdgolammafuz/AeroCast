import os
import datetime as dt
import pandas as pd
import requests
import time
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    push_to_gateway,
    pushadd_to_gateway,
)

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "logs")
CSV_FILE = os.path.join(LOG_DIR, "prediction_history.csv")
LOG_FILE = os.path.join(LOG_DIR, "drift.log")
STATE_FILE = os.path.join(LOG_DIR, "last_drift_state.txt")
LAST_DRIFT_FILE = os.path.join(LOG_DIR, "last_drift.txt")
DRIFT_COUNT_FILE = os.path.join(LOG_DIR, "drift_count.txt")

# Settings
WINDOW = 6
TEMP_THRESHOLD = 38.0
PUSHGATEWAY = os.environ.get("PUSHGATEWAY_HOST", "localhost:9091")
QUEUE_URL = os.environ.get("QUEUE_URL", "http://queue:8081")
QUEUE_ENABLED = os.environ.get("QUEUE_ENABLED", "0") == "1"

# Metrics
gauge_reg = CollectorRegistry()
DRIFT_FLAG = Gauge("aerocast_drift_flag", "Drift flag (0/1)", ["city"], registry=gauge_reg)
RMSE_GRU = Gauge("aerocast_rmse_gru", "Rolling RMSE of GRU", registry=gauge_reg)
RMSE_PROP = Gauge("aerocast_rmse_prophet", "Rolling RMSE of Prophet", registry=gauge_reg)
LAST_DRIFT_TS = Gauge("aerocast_last_drift_ts", "Unix timestamp of last drift", registry=gauge_reg)
DRIFT_COUNT_G = Gauge("aerocast_drift_count", "Total drift events", registry=gauge_reg)

counter_reg = CollectorRegistry()
DRIFT_TOTAL = Counter("aerocast_drift_total", "Total drift events", registry=counter_reg)

def _log(msg: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def _read_state() -> str:
    if not os.path.exists(STATE_FILE): return "cool"
    try: return open(STATE_FILE, "r").read().strip() or "cool"
    except: return "cool"

def _write_state(state: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f: f.write(state)

def _rmse(s: pd.Series) -> float:
    return float(((s ** 2).mean()) ** 0.5)

def _load_count(path) -> int:
    if not os.path.exists(path): return 0
    try: return int(open(path, "r").read().strip() or "0")
    except: return 0

def _save_count(path, n):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(path, "w") as f: f.write(str(n))

def _trigger_retrain_event(reason: str):
    """Post drift event to Go Queue."""
    if not QUEUE_ENABLED:
        print(f"[drift] Queue disabled, skipping event: {reason}")
        return

    try:
        resp = requests.post(
            f"{QUEUE_URL}/publish", 
            json={"event": reason}, 
            timeout=2
        )
        if resp.status_code == 202:
            print(f"[drift] Event posted to queue: {reason}")
        else:
            print(f"[drift] Queue error: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[drift] Failed to connect to queue: {e}")

def detect_and_flag():
    if not os.path.exists(CSV_FILE):
        print("[drift] no csv yet")
        return

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception:
        return

    if not {"model", "error", "actual"}.issubset(df.columns):
        return

    # Calc Metrics
    gru_err = df[df["model"] == "GRU"]["error"].tail(WINDOW)
    prop_err = df[df["model"] == "Prophet"]["error"].tail(WINDOW)
    actuals = df["actual"].tail(WINDOW)
    mean_actual = float(actuals.mean()) if len(actuals) else 0.0

    is_hot = int(mean_actual >= TEMP_THRESHOLD)
    last_state = _read_state()
    current_count = _load_count(DRIFT_COUNT_FILE)

    # Push Gauges
    try:
        DRIFT_FLAG.labels("saarbruecken").set(is_hot)
        if len(gru_err): RMSE_GRU.set(_rmse(gru_err))
        if len(prop_err): RMSE_PROP.set(_rmse(prop_err))
        DRIFT_COUNT_G.set(current_count)
        push_to_gateway(PUSHGATEWAY, job="aerocast_drift_gauges", registry=gauge_reg)
    except: pass

    # Logic
    if not is_hot:
        if last_state == "hot": _write_state("cool")
        print(f"[drift] Status: OK (mean={mean_actual:.2f})")
        return

    if last_state == "hot":
        print(f"[drift] Status: HOT (ongoing)")
        return

    # --- NEW DRIFT DETECTED ---
    now_iso = dt.datetime.utcnow().isoformat()
    now_ts = dt.datetime.utcnow().timestamp()
    
    # 1. Trigger Event (Pure Queue)
    _trigger_retrain_event(f"drift {now_iso}")
    
    # 2. Update State
    _write_state("hot")
    _log(f"{now_iso} DRIFT DETECTED mean={mean_actual:.2f}")
    
    # 3. Update Counters
    current_count += 1
    _save_count(DRIFT_COUNT_FILE, current_count)
    
    try:
        LAST_DRIFT_TS.set(now_ts)
        DRIFT_COUNT_G.set(current_count)
        push_to_gateway(PUSHGATEWAY, job="aerocast_drift_gauges", registry=gauge_reg)
    except Exception:
        pass

    try:
        DRIFT_TOTAL.inc()
        pushadd_to_gateway(PUSHGATEWAY, job="aerocast_drift_counter", registry=counter_reg)
    except Exception:
        pass

    print(f"[drift] DRIFT DETECTED! Event sent. Count={current_count}")

if __name__ == "__main__":
    detect_and_flag()