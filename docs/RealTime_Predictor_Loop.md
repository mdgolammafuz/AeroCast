# 🔁 Real-Time GRU Prediction Loop: Sliding Window Simulation

This document explains how our `live_predictor.py` and `sensor_simulator.py` synchronize in a **live forecasting scenario** using a **sliding window over time-series data**.

---

## 🎯 Goal

Predict the **next temperature (t+1)** based on the **last 10 time steps** of simulated data — and do it in real-time, as new sensor data arrives every 5 seconds.

---

## 🧱 Architecture

```
+-------------------------+      +-----------------------------+
| sensor_simulator.py     |      |  live_predictor.py          |
|-------------------------|      |-----------------------------|
| Writes 1 row every 5s   | ---> | Reads last 10 rows          |
| Parquet files in        |      | Predicts temperature at t+1 |
| data/processed/         |      | every 5s                    |
+-------------------------+      +-----------------------------+
```

---

## ⏱️ Time Sequence: Sliding Forecast

If the last data rows are:

| t | Temp |
|---|------|
| 1 | 25.1 |
| 2 | 26.3 |
| ... | ... |
| 10| 28.2 |

Then model predicts:  
→ **Temp at t = 11**

At next 5s tick:
- New row at t = 11
- Sliding window shifts to rows 2 to 11
- Predicts t = 12

---

## 🔣 Symbolic Formulation

Let `X = [x_{t-9}, ..., x_t] ∈ ℝ^{10×3}` where 3 = temperature, humidity, rainfall.  
Let `ŷ_{t+1} = GRU(X)` be the prediction.

This sliding window moves forward every 5 seconds, synced with sensor input.

---

## 🔄 Live Loop Implementation (Pseudo)

```python
while True:
    df = load_latest_data()
    X = last_10_rows(df)
    ŷ = model(X)
    print("🌡️ Predicted Temp:", ŷ)
    time.sleep(5)
```

---

## 🧪 Outcome

This creates a **real-time test environment**:
- Simulates a streaming weather station
- Forecasts before actual data arrives
- Mimics real-world time-series systems (IoT, Azure Event Hub, etc.)

