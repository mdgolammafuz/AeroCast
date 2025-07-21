# 🌾 AeroCast

🚀 **Real-time Sensor Ingestion and Forecasting Pipeline for Agro-Climate Monitoring**

AeroCast is a production-grade, modular data pipeline designed to simulate and process real-time IoT weather data for forecasting agro-climatic conditions. It integrates **Kafka**, **PySpark**, **GRU-based models**, **MLflow**, and **FastAPI** — making it suitable for real-world deployment in agriculture, climate risk, or smart irrigation systems.

---

## ✅ Key Features

- 🌦️ **Simulated Real-Time Sensor Data** (Temperature, Humidity, Rainfall)
- 📡 **Kafka-based Ingestion Pipeline** (`producer.py`)
- 🔥 **GRU-based Deep Learning Model** for Forecasting
- ⚙️ **PySpark Consumer Pipeline** for Stream Processing
- 📊 **MLflow Tracking** for Experiment Management
- ⚡ **FastAPI Interface** to Serve Forecasts
- 📈 **Grafana Monitoring** (Optional)
- 🧪 **Modular Folder Structure** with Notebooks, Logs, Docker, and APIs

---

## 📁 Current Project Structure

```bash
.
├── api/
├── artifacts/
├── cloud/
├── data/
├── datasets/
├── docker/
├── docs/
├── ingestion/
├── logs/
├── model/
├── monitoring/
├── notebooks/
├── processing/
├── requirements.txt
├── scripts/
├── serving/
├── streaming/
├── tests/
├── training/
├── README.md
```
    

🛠️ Tools & Technologies
| Component     | Technology                           |
| ------------- | ------------------------------------ |
| Ingestion     | `Kafka` + `kafka-python`             |
| Processing    | `PySpark`                            |
| Modeling      | `TensorFlow/Keras (GRU)`             |
| Serving       | `FastAPI`                            |
| Tracking      | `MLflow`                             |
| Monitoring    | `Grafana`                            |
| Notebook/Dev  | `JupyterLab`, `VSCode`, `virtualenv` |
| Infra (Local) | `Docker` + `Homebrew` (Mac)          |

🚧 Current Progress
✅ Kafka Installed and Running Locally

✅ Zookeeper Running

✅ Kafka Topic weather-data Created

✅ Sensor Stream Simulation Implemented in producer.py

✅ Folder Structure Modularized

✅ GitHub Repo Initialized

✅ Grafana, FastAPI, and MLflow Installed

🔜 Spark Consumer & GRU Model Training Pipeline Next

📌 Demo Snapshot
📡 Real-time sensor data being published to Kafka:
Sending: {
  "timestamp": "2025-07-14T11:07:31.419154",
  "temperature": 33.76,
  "humidity": 50.37,
  "rainfall": 9.53
}

🔄 How to Run the Kafka Producer
cd ingestion
python producer.py

🌐 Planned Features

    🔄 Stream Processing with PySpark

    📈 GRU Model Training on Real/Synthetic Sequences

    📤 FastAPI Interface with Forecast Route

    📉 MLflow Logging of Model Metrics

    📺 Grafana Dashboards for Sensor & Forecast Monitoring
---

## 📊 MLflow Integration

We have cleanly integrated MLflow for experiment tracking and visualization.

🔍 See: [docs/mlflow_integration_explained.md](docs/mlflow_integration_explained.md)

Includes screenshots of:
- Run Summary Page (`mlflow_run_summary.png`)
- Loss Curve View (`mlflow_loss_curve.png`)

📣 Author
MD Golam Mafuz
Aspiring Data Engineer & AI/ML Engineer
🔗 LinkedIn | GitHub

📌 License
MIT License. This is a learning + deployment showcase project.