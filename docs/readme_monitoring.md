# AeroCast++: Monitoring & Metrics with Prometheus + Grafana

> This standalone file explains everything related to **Prometheus**, **PushGateway**, **Grafana**, and how metrics are collected from **FastAPI** and **GRU training scripts**.

---

## Components Involved

```
FastAPI Endpoint (real-time)
↳ Exposes metrics like gru_prediction_error

mlflow_gru_train.py (batch)
↳ Pushes training_loss to Prometheus via PushGateway

prometheus.yml
↳ Configuration to scrape FastAPI and PushGateway

grafana
↳ Visual dashboard pulling from Prometheus
```

---

## Prometheus Setup

### File: `prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'aerocast-fastapi'
    static_configs:
      - targets: ['localhost:8000']

  - job_name: 'pushgateway'
    static_configs:
      - targets: ['localhost:9091']
```

### Start Prometheus

```bash
prometheus --config.file=prometheus/prometheus.yml
```

- UI: [http://localhost:9090](http://localhost:9090)
- Test queries:
  - `gru_prediction_error`
  - `drift_detected_flag`
  - `training_loss`

---

## PushGateway: Bridge for mlflow training

### Location: `training/mlflow_gru_train.py`

```python
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

LOSS_GAUGE = Gauge("training_loss", "GRU loss", ["epoch", "reason"])

...

LOSS_GAUGE.labels(str(epoch), reason).set(epoch_loss)
push_to_gateway("localhost:9091", job="aerocast_training")
```

**Why needed?** Training scripts are not long-lived services. PushGateway allows one-off pushes to be scraped later by Prometheus.

---

## FastAPI Metrics (Real-Time)

### Location: `serving/fastapi_app.py`

```python
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

PREDICT_ERROR = Gauge("gru_prediction_error", "MAE of GRU predictions")
LATENCY = Histogram("gru_prediction_latency_ms", "Prediction latency in ms")
```

Metrics are exposed via:

```python
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

→ Prometheus scrapes `localhost:8000/metrics` every 5s.

---

## Grafana Setup

### Start Grafana

```bash
brew services start grafana
```

- UI: [http://localhost:3000](http://localhost:3000)
- Default: `admin` / `admin`

### Add Prometheus as Data Source

- Go to: Gear icon → Data Sources → Add Prometheus
- URL: `http://localhost:9090`
- Save & Test

### Dashboards Panels

- `gru_prediction_error`
- `gru_prediction_latency_ms`
- `training_loss`
- `drift_detected_flag`

---

## Visual Flow Summary

```
Sensor Data / Simulated Stream
        ↓
   FastAPI → /predict  → exposes metrics
        ↓                          ↘
Prometheus ← PushGateway ← GRU Training (mlflow_gru_train.py)
        ↓
     Grafana Dashboards
```

---

## Notes

- **PushGateway** handles short-lived metrics like training
- **FastAPI** exposes live endpoint metrics like error/latency
- **Grafana** makes this visual and demo-ready

