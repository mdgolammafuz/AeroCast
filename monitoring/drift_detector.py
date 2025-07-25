# monitoring/drift_detector.py

import os
import numpy as np
from datetime import datetime

# --- Configurable Threshold ---
DRIFT_THRESHOLD = 0.8  # MAE threshold
CHECK_EVERY_N = 10     # Check drift every N records

# --- File paths ---
FLAG_PATH = "retrain.flag"
LOG_PATH = "logs/drift.log"

# --- Ensure log directory exists ---
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# --- Drift Check Function ---
def check_drift(y_true_list, y_pred_list):
    """
    Compares true vs predicted values, and logs drift if MAE exceeds threshold.
    Creates a retrain.flag file if drift is detected.
    """
    if len(y_true_list) < CHECK_EVERY_N:
        return False  # Not enough data yet

    y_true = np.array(y_true_list[-CHECK_EVERY_N:])
    y_pred = np.array(y_pred_list[-CHECK_EVERY_N:])

    mae = np.mean(np.abs(y_true - y_pred))

    if mae > DRIFT_THRESHOLD:
        # --- Log drift ---
        with open(LOG_PATH, "a") as f:
            f.write(f"[{datetime.now()}] Drift detected! MAE = {mae:.4f}\n")

        # --- Create retrain flag ---
        with open(FLAG_PATH, "w") as f:
            f.write(f"Drift detected at {datetime.now()}\n")

        return True

    return False
