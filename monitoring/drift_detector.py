"""
Flag drift when GRU MAE exceeds Prophet MAE by >25%.
Creates 'retrain.flag' (at repo root) and logs to logs/drift.log.
"""

import os
import pandas as pd
import datetime as dt

# --- Make paths absolute to the repo root ---
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR  = os.path.join(ROOT, "logs")
CSV_FILE = os.path.join(LOG_DIR, "prediction_history.csv")
FLAG_PATH = os.path.join(ROOT, "retrain.flag")
LOG_FILE  = os.path.join(LOG_DIR, "drift.log")

WINDOW = 10    # last N predictions
RATIO  = 1.25  # 25% worse than Prophet baseline

def _log(msg: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def detect_and_flag() -> bool:
    """Return True if drift detected."""
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        return False

    # focus on most recent WINDOW rows for each model
    gru_err  = df[df.model == "GRU"     ].error.tail(WINDOW)
    prop_err = df[df.model == "Prophet"].error.tail(WINDOW)

    # if not enough data yet
    if len(gru_err) < WINDOW or len(prop_err) < WINDOW:
        return False

    mae_gru  = gru_err.mean()
    mae_prop = prop_err.mean()

    if mae_gru > RATIO * mae_prop:
        # create flag file only if not present
        if not os.path.exists(FLAG_PATH):
            open(FLAG_PATH, "w").close()

        _log(f"{dt.datetime.utcnow().isoformat()}  DRIFT  "
             f"GRU MAE={mae_gru:.3f}  > {RATIO}× Prophet MAE={mae_prop:.3f}")
        return True
    return False
