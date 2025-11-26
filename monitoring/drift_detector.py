# Stable version: count only when we go from COOL -> HOT.
# + file-backed drift_count gauge for Grafana (because pushadd isn't accumulating).

import os
import datetime as dt
import pandas as pd
import requests  # NEW: For Go Queue
from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Counter,
    push_to_gateway,
    pushadd_to_gateway,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "logs")
CSV_FILE = os.path.join(LOG_DIR, "prediction_history.csv")
FLAG_PATH = os.path.join(ROOT, "retrain.flag")
LOG_FILE = os.path.join(LOG_DIR, "drift.log")
STATE_FILE = os.path.join(LOG_DIR, "last_drift_state.txt")   # "hot" / "cool"
LAST_DRIFT_FILE = os.path.join(LOG_DIR, "last_drift.txt")    # ISO timestamp
DRIFT_COUNT_FILE = os.path.join(LOG_DIR, "drift_count.txt")  # NEW

WINDOW = 6
TEMP_THRESHOLD = 38.0
PUSHGATEWAY = os.environ.get("PUSHGATEWAY_HOST", "localhost:9091")

# --- NEW: Queue Configuration ---
QUEUE_ENABLED = os.environ.get("QUEUE_ENABLED", "0") == "1"
QUEUE_URL = os.environ.get("QUEUE_URL", "http://queue:8081")

# 1) gauges → overwrite OK
gauge_reg = CollectorRegistry()
DRIFT_FLAG = Gauge("aerocast_drift_flag", "Drift flag (0/1)", ["city"], registry=gauge_reg)
RMSE_GRU = Gauge("aerocast_rmse_gru", "Rolling RMSE of GRU", registry=gauge_reg)
RMSE_PROP = Gauge("aerocast_rmse_prophet", "Rolling RMSE of Prophet", registry=gauge_reg)
LAST_DRIFT_TS = Gauge(
    "aerocast_last_drift_ts",
    "Unix timestamp (UTC) of last COOL→HOT drift",
    registry=gauge_reg,
)
DRIFT_COUNT_G = Gauge(
    "aerocast_drift_count",
    "File-backed total drift events (authoritative)",
    registry=gauge_reg,
)

# 2) counter
counter_reg = CollectorRegistry()
DRIFT_TOTAL = Counter("aerocast_drift_total", "Total drift events", registry=counter_reg)


def _log(msg: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def _read_state() -> str:
    if not os.path.exists(STATE_FILE):
        return "cool"
    try:
        return open(STATE_FILE, "r").read().strip() or "cool"
    except Exception:
        return "cool"


def _write_state(state: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        f.write(state)


def _rmse(s: pd.Series) -> float:
    return float(((s ** 2).mean()) ** 0.5)


def _load_last_drift_ts() -> float:
    if not os.path.exists(LAST_DRIFT_FILE):
        return 0.0
    try:
        iso = open(LAST_DRIFT_FILE, "r").read().strip()
        if not iso:
            return 0.0
        return dt.datetime.fromisoformat(iso).timestamp()
    except Exception:
        return 0.0


def _load_drift_count() -> int:
    if not os.path.exists(DRIFT_COUNT_FILE):
        return 0
    try:
        return int(open(DRIFT_COUNT_FILE, "r").read().strip() or "0")
    except Exception:
        return 0


def _save_drift_count(n: int) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(DRIFT_COUNT_FILE, "w") as f:
        f.write(str(n))

def _trigger_retrain(reason: str):
    """
    Hybrid trigger: Writes file (legacy) AND posts to queue (cloud-native)
    """
    # 1. Legacy File Path (Fallback)
    with open(FLAG_PATH, "w") as f:
        f.write(reason)
    print(f"[drift] wrote flag file: {reason}")

    # 2. Go Queue Path (Primary)
    if QUEUE_ENABLED:
        try:
            resp = requests.post(
                f"{QUEUE_URL}/publish", 
                json={"event": reason}, 
                timeout=2
            )
            if resp.status_code == 202:
                print(f"[drift] posted event to queue: {reason}")
            else:
                print(f"[drift] queue error: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"[drift] failed to connect to queue: {e}")

def detect_and_flag() -> bool:
    if not os.path.exists(CSV_FILE):
        print("[drift] no csv yet")
        return False

    df = pd.read_csv(CSV_FILE)
    if not {"model", "error", "actual"}.issubset(df.columns):
        print("[drift] csv missing columns -> skip")
        return False

    gru_err = df[df["model"] == "GRU"]["error"].tail(WINDOW)
    prop_err = df[df["model"] == "Prophet"]["error"].tail(WINDOW)
    actuals = df["actual"].tail(WINDOW)
    mean_actual = float(actuals.mean()) if len(actuals) else 0.0

    is_hot = int(mean_actual >= TEMP_THRESHOLD)
    last_state = _read_state()
    current_count = _load_drift_count()

    try:
        DRIFT_FLAG.labels("saarbruecken").set(is_hot)
        if len(gru_err):
            RMSE_GRU.set(_rmse(gru_err))
        if len(prop_err):
            RMSE_PROP.set(_rmse(prop_err))
        LAST_DRIFT_TS.set(_load_last_drift_ts())
        DRIFT_COUNT_G.set(current_count)
        push_to_gateway(PUSHGATEWAY, job="aerocast_drift_gauges", registry=gauge_reg)
    except Exception:
        pass

    # 1) COOL
    if not is_hot:
        _write_state("cool")
        print(f"[drift] no drift (mean={mean_actual:.2f} < {TEMP_THRESHOLD}) → state=cool")
        return False

    # 2) Hot but flag exists
    if os.path.exists(FLAG_PATH):
        _write_state("hot")
        print("[drift] hot but retrain.flag already exists → waiting for trainer")
        return False

    # 3) Still same hot episode
    if last_state == "hot":
        _write_state("hot")
        print("[drift] still same hot episode → no new count")
        return False

    # 4) REAL NEW DRIFT
    now = dt.datetime.utcnow()
    now_iso = now.isoformat()
    now_ts = now.timestamp()

    # NEW: Use trigger helper
    reason_msg = f"drift {now_iso}"
    _trigger_retrain(reason_msg)

    _write_state("hot")

    _log(f"{now_iso}  DRIFT  mean_actual={mean_actual:.2f}°C  >= {TEMP_THRESHOLD}°C")
    with open(LAST_DRIFT_FILE, "w") as f:
        f.write(now_iso + "\n")

    current_count += 1
    _save_drift_count(current_count)

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

    print(f"[drift] NEW drift → flag created | count={current_count} | mean={mean_actual:.2f}")
    return True


if __name__ == "__main__":
    detect_and_flag()