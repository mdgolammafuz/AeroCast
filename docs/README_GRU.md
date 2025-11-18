
# GRU Weather Forecaster – Internal Module of AeroCast

> Predicting tomorrow’s temperature using 10 days of weather sequences  
> Powered by PyTorch GRU | Deployed as part of AeroCast’s real-time forecasting pipeline

---

## Mental Model: How GRU Thinks

Imagine GRU as a **meteorologist with short-term memory**.

- It looks at **last 10 days** of weather:
  - Temperature
  - Humidity
  - Rainfall
- It learns **patterns** like:
  - "If it rained and humidity spiked, tomorrow is cooler."
- Then it **predicts** the next day’s temperature 📈

---

## Input Format: Shape Breakdown

Let’s decode how the GRU reads the input:

| Concept | Description | Shape |
|--------|-------------|-------|
| `seq_len` | Days we look back | 10 |
| `features` | Weather variables | 3 (`temperature`, `humidity`, `rainfall`) |
| `X_tensor` | Training input | `[samples, 10, 3]` |
| `y_tensor` | Ground truth temperature | `[samples]` |

Example:
```python
X_tensor.shape  # [974, 10, 3]
y_tensor.shape  # [974]
```

---

## GRU Internal Flow (Forward Pass)

### Symbolic Diagram

```
Input: [10 days × 3 features]
    ↓
GRU Layer (64 hidden units)
    ↓
Takes only the **last hidden state**
    ↓
Fully Connected Layer (fc → output_dim=1)
    ↓
Output: Temperature for Day 11
```

### Code Behind the Flow:

```python
out, _ = self.gru(x)          # out shape: [batch, seq_len, hidden_dim]
temp_pred = self.fc(out[:, -1, :])  # Pick last day’s hidden state
```

Why `out[:, -1, :]`?
> Because only the **final hidden state** contains the "summary" of all past 10 days.

---

## Why `.squeeze()` in Training?

In training loop:
```python
pred = model(xb).squeeze()
```

This removes extra dimension `[batch_size, 1] → [batch_size]` so that it matches `yb.shape`.

---

## Output Interpretation

Final output:
```python
   Next predicted temperature: 12.79 (in °C)
```

Used for:
- Real-time prediction (via FastAPI)
- Monitoring drift
- Triggering auto-retraining

---

## Integration in AeroCast++

This GRU engine is used in:
- Live forecast prediction (via `gru_hook_debug.py`)
- `model/train_gru.py` handles training + saving `.pt`
- Unit testing enabled via debug hooks
- Future: Used inside FastAPI (`api/main.py`) for serving prediction endpoint

---
