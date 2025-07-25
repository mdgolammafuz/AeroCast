import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monitoring.drift_detector import check_drift

y_true_log = [22.1, 22.3, 22.5, 22.0, 22.2, 23.0, 24.5, 24.8, 25.1, 26.0]
y_pred_log = [22.0, 22.1, 22.2, 22.1, 22.0, 22.3, 22.4, 22.5, 22.6, 22.7]

if check_drift(y_true_log, y_pred_log):
    print("Drift detected. Retrain flag created.")
