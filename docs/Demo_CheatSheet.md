# AeroCast Demo Cheat Sheet

This is a concise guide to run and explain the full GRU + Prophet demo, with Prometheus + PushGateway + Grafana monitoring.

Paths assume we run commands from the AeroCast repo root.

---

## Architecture Overview

```
Simulator (calm or heatwave)  ->  Parquet files (data/processed/part-*.parquet)

generate_training_data.py     ->  Sliding-window CSV (data/training_data.csv)

mlflow_gru_train.py           ->  Train GRU, log to MLflow, push training_loss to PushGateway,
                                  save weights to artifacts/gru_weather_forecaster.pt

serving/fastapi_app.py        ->  /predict (GRU inference), /metrics (Prometheus exposition)

live_predictor_feeder.py      ->  Read last N parquet files, call GRU, compute Prophet,
                                  RCA log (logs/prediction_history.csv), drift detection

drift_detector.py             ->  If GRU MAE > 1.25x Prophet MAE over last WINDOW: create retrain.flag,
                                  log to logs/drift.log

Prometheus (prometheus.yml)   ->  Scrapes FastAPI /metrics and PushGateway /metrics

Grafana                       ->  Panels built on Prometheus queries
```

---

## Key Files (points to show in the demo)

- `data/simulator_calm.py` / `data/heatwave_simulator.py`  
  Produces one record every ~5s into `data/processed/` with `ts` and `temperature`.

- `data/generate_training_data.py`  
  Builds `data/training_data.csv` with sliding windows `[t0..t4,target]` from parquet stream.

- `model/gru_model.py`  
  Defines `GRUWeatherForecaster(1, 16, 1)` used for serving and training.

- `training/mlflow_gru_train.py`  
  Trains GRU on the CSV, logs to MLflow, pushes `training_loss` to PushGateway, saves weights to `artifacts/gru_weather_forecaster.pt`, registers a model version.

- `serving/fastapi_app.py`  
  Loads GRU weights, exposes `/predict` and `/metrics`. Prometheus scrapes `/metrics`.

- `utils/rca_logger.py`  
  Writes `logs/prediction_history.csv` with one row per prediction: `ts, sequence, forecast, actual, error, model`.

- `utils/prophet_helper.py`  
  Loads Prophet model from JSON and emits a one-step forecast for comparison.

- `streaming/client/live_predictor_feeder.py`  
  Reads last N parquet files, calls `/predict` (GRU), computes Prophet forecast, logs both to RCA, then calls the drift detector.

- `monitoring/drift_detector.py`  
  Computes last WINDOW MAE for GRU vs Prophet; if `GRU > 1.25 × Prophet`, writes `retrain.flag` (repo root) and appends to `logs/drift.log`. (Your updated version uses absolute paths based on repo root.)

- `prometheus/prometheus.yml`  
  Scrapes `localhost:8000` (FastAPI), `localhost:9091` (PushGateway), and `localhost:9090` (Prometheus self).

---

## Minimal Run Order

### 0) From repo root
```bash
cd /path/to/AeroCast
```

### 1) Start simulator (calm first)
```bash
python data/simulator_calm.py
```
Leave this running in its own terminal. It will print a line every ~5s.

### 2) Build training data CSV
```bash
python data/generate_training_data.py
```

### 3) Train GRU (logs to MLflow, pushes to PushGateway, saves weights)
```bash
python training/mlflow_gru_train.py
```

### 4) Start FastAPI (serving GRU + Prometheus /metrics)
```bash
pkill -f uvicorn || true
uvicorn serving.fastapi_app:app --host 0.0.0.0 --port 8000
```

### 5) Start feeder (RCA + drift detection)
```bash
python -u -m streaming.client.live_predictor_feeder
```
You should see a heartbeat line each loop.

### 6) Start Prometheus
```bash
prometheus --config.file="$(pwd)/prometheus/prometheus.yml"
# Open http://localhost:9090/targets and ensure the three targets are UP
```

### 7) Start Grafana (Docker-based is fine)
```bash
# If 3000 is busy, map to 3001
docker run -d -p 3001:3000 --name grafana grafana/grafana-oss
# Open http://localhost:3001, add Prometheus data source pointing to http://localhost:9090
```

### 8) Trigger drift
```bash
# Stop calm; optionally clear recent parquet for a clean regime change:
rm -f data/processed/part-*.parquet

# Start heatwave
python data/heatwave_simulator.py
```
While feeder runs, watch:
```bash
tail -f logs/drift.log
ls -l retrain.flag
```

### 9) Retrain on new regime (when flag is present)
```bash
python data/generate_training_data.py
python training/mlflow_gru_train.py
pkill -f uvicorn || true
uvicorn serving.fastapi_app:app --host 0.0.0.0 --port 8000
```

---

## Health Checks

FastAPI metrics:
```bash
curl -s http://127.0.0.1:8000/metrics | head
```

PushGateway metrics:
```bash
curl -s http://127.0.0.1:9091/metrics | grep '^training_loss' | tail
```

Recent MAE ratio (drift if > 1.25x):
```bash
python - <<'PY'
import pandas as pd; W=10
df=pd.read_csv('logs/prediction_history.csv')
g=df[df.model=='GRU'].error.tail(W).mean()
p=df[df.model=='Prophet'].error.tail(W).mean()
print('counts: ', len(df[df.model=='GRU'].tail(W)), len(df[df.model=='Prophet'].tail(W)))
print('MAE -> GRU=', round(g,3), ' Prophet=', round(p,3), ' ratio=', 'inf' if p==0 else round(g/p,2))
PY
```

RCA tail:
```bash
tail -n 5 logs/prediction_history.csv
```

Prometheus targets:
- http://localhost:9090/targets

---

## Grafana panel quick setup

1) Open Grafana (e.g., http://localhost:3001).
2) Add data source: Prometheus → URL `http://localhost:9090` → Save & test.
3) Dashboard → Add new panel → Visualization: Time series → Query examples:
   - `training_loss`
   - `rate(predictions_total[1m])`
   - Optional counters: `increase(predictions_total[5m])`

Save the dashboard. The two panels for screenshots: training loss over time and prediction throughput rate.

---

## What to screenshot for README

- Prometheus Targets page with all targets UP.
- Prometheus graph for `training_loss`.
- Grafana dashboard with:
  - `training_loss` time series
  - `rate(predictions_total[1m])` time series
- Optional CLI proof:
  - `tail -n 5 logs/prediction_history.csv` (alternating GRU/Prophet rows)
  - `tail -n 5 logs/drift.log` (DRIFT lines)

---

## Troubleshooting in one-liners

FastAPI not scraped:
```bash
curl -s http://127.0.0.1:8000/metrics | head
# If this works but Prometheus target is DOWN, check prometheus.yml and port
```

No training_loss in PushGateway:
```bash
curl -s http://127.0.0.1:9091/metrics | grep '^training_loss' || echo "no training_loss yet"
# Run: python training/mlflow_gru_train.py
```

No RCA rows:
```bash
python -u -m streaming.client.live_predictor_feeder  # watch heartbeat/errors
ls -ltr data/processed | tail -n 3                    # parquet freshness
```

Flag not appearing:
```bash
tail -f logs/drift.log
ls -l retrain.flag || echo "flag not present"
# Remember: training clears the flag at start by design
```

---

## Appendix: Prometheus config snippet

`prometheus/prometheus.yml`

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

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

---

## Metrics quick reference

- `predictions_total{model_type="GRU"}`
  Counter, incremented on each `/predict` call in FastAPI.

- `training_loss{epoch,reason}`
  Gauge, pushed to PushGateway each epoch during GRU training.

- FastAPI default Python client metrics are also exposed on `/metrics`
  (gc, process, etc.).

---

## Notes for the demo

- We compare GRU vs Prophet live on the latest window and record both to RCA for transparency.
- Drift is defined as GRU MAE > 1.25× Prophet MAE over the last WINDOW points; we write a `retrain.flag` to decouple detection from retraining.
- Training logs to MLflow, pushes a live `training_loss` to Prometheus via PushGateway, and saves the serving weights; serving is restarted to pick up new weights.
- Grafana pulls from Prometheus and gives us an at-a-glance view of training health and serving throughput.
