# monitoring/drift_detector.py
# Stable version: count only when we go from COOL -> HOT.

import os
import datetime as dt
import pandas as pd
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
STATE_FILE = os.path.join(LOG_DIR, "last_drift_state.txt")  # stores "hot" or "cool"

WINDOW = 6
TEMP_THRESHOLD = 38.0
PUSHGATEWAY = "localhost:9091"

# 1) gauges → safe to overwrite
gauge_reg = CollectorRegistry()
DRIFT_FLAG = Gauge("aerocast_drift_flag", "Drift flag (0/1)", ["city"], registry=gauge_reg)
RMSE_GRU = Gauge("aerocast_rmse_gru", "Rolling RMSE of GRU", registry=gauge_reg)
RMSE_PROP = Gauge("aerocast_rmse_prophet", "Rolling RMSE of Prophet", registry=gauge_reg)

# 2) counter → must be added, never overwritten
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


def detect_and_flag() -> bool:
    if not os.path.exists(CSV_FILE):
        print("[drift] no csv yet")
        return False

    df = pd.read_csv(CSV_FILE)
    if not {"model", "error"}.issubset(df.columns):
        print("[drift] csv missing columns -> skip")
        return False

    gru_err = df[df["model"] == "GRU"]["error"].tail(WINDOW)
    prop_err = df[df["model"] == "Prophet"]["error"].tail(WINDOW)
    actuals = df["actual"].tail(WINDOW)
    mean_actual = float(actuals.mean()) if len(actuals) else 0.0

    is_hot = int(mean_actual >= TEMP_THRESHOLD)
    last_state = _read_state()  # "hot" or "cool"

    # always push gauges so Grafana stays nonzero
    try:
        DRIFT_FLAG.labels("saarbruecken").set(is_hot)
        if len(gru_err):
            RMSE_GRU.set(_rmse(gru_err))
        if len(prop_err):
            RMSE_PROP.set(_rmse(prop_err))
        # use a separate job for gauges
        push_to_gateway(PUSHGATEWAY, job="aerocast_drift_gauges", registry=gauge_reg)
    except Exception:
        pass

    # case 1: cooled down → reset episode
    if not is_hot:
        _write_state("cool")
        print(f"[drift] no drift (mean={mean_actual:.2f} < {TEMP_THRESHOLD}) → state=cool")
        return False

    # now: is_hot == 1
    # case 2: trainer still needs to run
    if os.path.exists(FLAG_PATH):
        _write_state("hot")
        print("[drift] hot but retrain.flag already exists → waiting for trainer")
        return False

    # case 3: already hot in this episode → do NOT count again
    if last_state == "hot":
        _write_state("hot")
        print("[drift] still same hot episode → no new count")
        return False

    # case 4: transition COOL -> HOT → this is the only time we count
    # create flag
    open(FLAG_PATH, "w").close()
    _write_state("hot")

    _log(
        f"{dt.datetime.utcnow().isoformat()}  DRIFT  "
        f"mean_actual={mean_actual:.2f}°C  >= {TEMP_THRESHOLD}°C"
    )

    # increment counter and pushadd
    try:
        DRIFT_TOTAL.inc()
        # separate job for counter to avoid overwrite
        pushadd_to_gateway(PUSHGATEWAY, job="aerocast_drift_counter", registry=counter_reg)
    except Exception:
        pass

    print(
        f"[drift] NEW drift → flag created | last {WINDOW} = {list(actuals)} | mean={mean_actual:.2f} ≥ {TEMP_THRESHOLD}"
    )
    return True


if __name__ == "__main__":
    detect_and_flag()
