# AeroCast++: GRU vs Prophet Experimental Design

> This file documents the experimental design comparing **GRU** and **Prophet** models on both calm and heatwave datasets. It includes setup, criteria for drift, logging, and evaluation.

---

## Objective

Establish a baseline and demonstrate model performance degradation under drift using:

- GRU (deep learning sequence model)
- Prophet (classical time series model)

This helps justify **drift detection** and **auto-retraining** components of AeroCast++.

---

## Folder Structure

```
AeroCast/
├── training/
│   ├── mlflow_gru_train.py
│   └── train_prophet.py
├── utils/
│   └── prophet_helper.py
├── monitoring/
│   └── drift_detector.py
├── data/
│   ├── simulator_calm.py
│   └── simulator_heatwave.py
├── logs/
│   ├── drift.log
│   └── prediction_history.csv
```

---

## Experiment Steps

### 1. Calm Simulation
```bash
python simulator_calm.py
```
- Generates stable synthetic time series
- GRU expected to perform well
- **No drift triggered**

### 2. Heatwave Simulation
```bash
python simulator_heatwave.py
```
- Injects shock or seasonality into the signal
- GRU expected to underperform Prophet
- Drift detected → `retrain.flag` triggered

---

## Drift Detection Logic

```python
if MAE_GRU > 1.25 × MAE_Prophet:
    → Write to drift.log
    → Touch retrain.flag
```

- Rolling MAE calculated over last 10 predictions
- Uses FastAPI `/predict` + `prophet_helper.py`

---

## Visual Analysis

### MLflow UI (http://localhost:5000)

- Track loss over epochs
- Compare run tags: `run_reason = drift` vs `initial`
- Download trained model from registered model section

### Grafana Panels (http://localhost:3000)

- `training_loss`: from PushGateway (per epoch)
- `gru_prediction_error`: live MAE
- `drift_detected_flag`: toggled on detection

---

## Logs & Files

- `logs/drift.log`: One line per trigger with timestamp
- `logs/prediction_history.csv`: Stores actuals, GRU, Prophet for RCA
- `retrain.flag`: Temporary flag file to indicate drift

---

