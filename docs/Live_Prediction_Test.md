# 🔁 Real-Time Prediction Test – GRU Forecasting (AeroCast++)

This test connects the GRU model with a live sensor simulator to mimic real-time weather forecasting based on streaming sensor data.

---

## 🧠 Purpose

To simulate a **real-time forecast system**, where:
- Synthetic weather data (temperature, humidity, rainfall) is generated every 5 seconds
- The GRU model predicts the next temperature based on the latest 10 records
- This prediction loop mimics real-world use cases like **smart farming**, **disaster forecasting**, etc.

---

## 🔄 Symbolic Flow

We use a **sliding window** of the latest 10 observations:

```
t₀  t₁  t₂  ... t₉  → GRU → predict t₁₀
```

With each new record at time `t_{now}`, the model uses:
- Last 10 time steps: `[t_{now-9}, ..., t_{now}]`
- To predict: `temperature_{t_{now+1}}`

---

## 🧪 Setup

### 🔧 Sensor Simulator (`simulator/sensor_simulator.py`)
- Appends a new `.parquet` file every 5 seconds to `data/processed/`
- Each file contains a row: timestamp + synthetic weather

```python
{
    "timestamp": "2025-07-18T12:34:56",
    "temperature": 28.5,
    "humidity": 45.3,
    "rainfall": 12.7
}
```

---

### 🤖 GRU Predictor (`debug/live_predictor.py`)
- Loads full `data/processed/` on each loop
- Sorts and picks latest 10 rows
- Predicts temperature for next time step using trained GRU
- Waits 5 seconds (`time.sleep(5)`) and repeats

---

## 📉 Example Prediction Log

```
🌡️ Predicted next temperature → 26.52°C
🌡️ Predicted next temperature → 28.10°C
🌡️ Predicted next temperature → 25.89°C
```

---

## 🧪 Sync Design Notes

- No explicit signal from sensor to predictor — just `sleep(5)`
- Both scripts **share the `data/processed/` folder**
- Guarantees freshness via sort → select last 10 rows
- Can be **replaced with Kafka or Azure Event Hub** for real-world production use

---

## 🚀 Next Steps

- Add monitoring + alerting (Grafana, anomaly detection)
- Replace local folder with Kafka/Azure streaming
- Optimize model reloading using `watchdog` or `mlflow` triggers