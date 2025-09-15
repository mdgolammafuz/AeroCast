"""
Root-Cause log. One CSV row per prediction.
Columns: ts,input_seq,forecast,actual,error,model
"""

import os, csv, json
from datetime import datetime

LOG_DIR  = "logs"
LOG_FILE = f"{LOG_DIR}/prediction_history.csv"
os.makedirs(LOG_DIR, exist_ok=True)

# create header if file missing
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["ts","sequence","forecast","actual","error","model"]).writeheader()

def log_prediction(seq, forecast, actual, model):
    error = abs(forecast - actual)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ts","sequence","forecast","actual","error","model"])
        w.writerow({
            "ts": datetime.utcnow().isoformat(),
            "sequence": json.dumps(seq),
            "forecast": forecast,
            "actual": actual,
            "error": error,
            "model": model,
        })
