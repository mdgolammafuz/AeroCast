
# Drift Detection and Auto-Retraining in AeroCast

This module adds a simple, production-aware mechanism for detecting model drift and automatically triggering retraining.

## Overview

- **Drift Detection**: A custom logic monitors rolling Mean Absolute Error (MAE) or other metrics and creates a `retrain.flag` file when drift exceeds a threshold.
- **Retrain Trigger**: The GRU training script checks for the `retrain.flag` file, retrains the model, and deletes the flag.
- **Log Tracking**: A log file (`logs/drift.log`) captures when drift was detected and when retraining occurred.

## File Summary

| File                                 | Description                                      |
|--------------------------------------|--------------------------------------------------|
| `monitoring/drift_detector.py`       | Checks MAE vs threshold, writes `retrain.flag`   |
| `tests/drift_detector_check.py`      | Simple test to simulate drift and trigger flag   |
| `training/mlflow_gru_train.py`       | GRU model training; now includes retrain hook    |
| `logs/drift.log`                     | Drift detection log (not committed to GitHub)    |
| `retrain.flag`                       | Auto-generated flag file for retraining trigger  |

## How It Works

1. The `drift_detector.py` script compares predicted vs. actual values.
2. If the drift metric exceeds a configured threshold, it creates a file called `retrain.flag`.
3. The training script checks for this flag on startup. If found:
    - Trains the GRU model on the existing (simulated or real) dataset.
    - Logs the retraining event.
    - Deletes the flag.

## Usage Example

### Detect Drift

```bash
python tests/drift_detector_check.py
```

Output:

```
Drift detected. Retrain flag created.
```

### Trigger Auto Retraining

```bash
python training/mlflow_gru_train.py
```

This will:
- Check for `retrain.flag`
- Retrain the model if drift was detected
- Log to `logs/drift.log`

## Notes

- `retrain.flag` and `drift.log` are runtime files. These are excluded from Git using `.gitignore`.
- The data used for training is still the minimal simulated sequence. For real data, adapt the training data loader accordingly.
