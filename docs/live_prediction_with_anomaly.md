
# Real-Time Live Prediction with Anomaly Detection (Refactor Summary)

This document explains the updated `live_predictor.py` file that adds **anomaly detection** logic while preserving a **clean, production-ready design**.

---

## High-Level Symbolic Story

```
Let:
  - t = current time
  - Δt = prediction offset
  - SEQ_LEN = 10 (past timesteps used)
  - Xₜ = sequence at time t: [xₜ₋₁₀, ..., xₜ]
  - f(Xₜ) = GRU model's predicted temperature for (t + Δt)

Each loop:
  → Reads latest 10-step sequence: Xₜ
  → Predicts: f(Xₜ) → ŷₜ₊Δt
  → Compares ŷₜ₊Δt with ŷₜ₋₁₊Δt (previous prediction)
  → If sudden jump ⇒ raise anomaly
```

---

## Code Design Highlights

### Modular Imports
```python
from model.train_gru import GRURegressor
from monitoring.anomaly_detector import is_point_anomaly
```

### Sequence Loader
```python
recent_df = df[features].values[-SEQ_LEN:]
recent_seq = torch.tensor(recent_df, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 10, 3]
```

### Model Forward Pass
```python
prediction = model(X_live).item()  # → Float scalar prediction
```

### Anomaly Check Logic
```python
if is_anomalous(prediction, previous_pred):
    print(f"Anomaly Detected")
else:
    print(f"Temperature: {prediction:.2f}°C")
previous_pred = prediction
```

---

## Testing Strategy

To test before commit:

1. Ensure `data/processed/` has at least **10 rows** with valid timestamps.
2. Run the simulator:  
   ```bash
   python simulator/sensor_simulator.py
   ```
3. In separate terminal:  
   ```bash
   python debug/live_predictor.py
   ```
---

## Future Enhancements (Hooks)

| Feature       | To Be Added         |
|---------------|---------------------|
| Logging       | `logging.info()`    |
| MLflow        | `mlflow.log_metric` |
| Grafana       | Export logs via agent or REST |


---

> Status: `live_predictor.py` is now real-time + anomaly-aware + production-oriented.
