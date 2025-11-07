# AeroCast
AeroCast is a small-but-serious real-time forecasting pipeline: streaming in, landing to storage, training a model, watching it drift, and kicking off retrains — with Prometheus/Grafana watching. It includes the NOAA path (realistic, stable), the simulated path (drift-on-purpose), monitoring, and the self-heal loop.

--
- streaming-style ingestion (realistic and simulated),
- FastAPI serving surface,
- drift detection that raises a `retrain.flag`,
- trainers that log to MLflow and refresh artifacts,
- Prometheus + Pushgateway + Grafana,
- Helm chart for Kubernetes,
- Terraform wrapper to apply the chart,
- CI that tests and publishes GHCR images.

---

## 1. Repository layout (curated)

```text
.
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── serving/                 # FastAPI app
├── streaming/               # ingestion + client feeders
├── data/                    # parquet landing + CSV builders
├── training/                # GRU + Prophet trainers (MLflow)
├── monitoring/              # drift detector → retrain.flag
├── artifacts/               # model files already built
├── prometheus/              # prometheus.yml
├── grafana/                 # provisioning
├── helm/aerocast            # Helm chart
├── infra/main.tf            # Terraform → install chart
├── tests/                   # API + state file tests
└── docs/                    # deeper design notes
```

This is the surface that matters. Local state (`logs/`, `mlruns/`, `.terraform/`, `venv/`) is intentionally not front-and-center.

---

## 2. Verification flow for reviewers

This is the fastest path to check out that AeroCast is alive.

1. **Docker local run**
   ```bash
   docker compose up
   curl http://localhost:8000/healthz
   ```
   Expected: JSON like `{ "ok": true, ... }`.

2. **Metrics alive**
   ```bash
   curl http://localhost:8000/metrics
   ```
   Expected: `aerocast_*` Prometheus metrics.

3. **Trainer activity**
   ```bash
   docker logs -f aerocast-trainer
   ```
   Expected: a training run finishing with a reason (drift / schedule).

4. **Kubernetes install**
   ```bash
   helm upgrade --install aerocast ./helm/aerocast -n aerocast --create-namespace
   kubectl get pods -n aerocast
   ```
   Expected: API, Prometheus, Grafana pods in `Running` state.

5. **Port-forward checks**
   ```bash
   kubectl port-forward -n aerocast svc/aerocast-api 8000:8000
   kubectl port-forward -n aerocast svc/aerocast-grafana 3000:3000
   kubectl port-forward -n aerocast svc/aerocast-prometheus 9090:9090
   ```
   Then hit `http://localhost:8000/healthz`, open Grafana, open Prometheus.

6. **Prometheus sees Pushgateway**
   - Open Prometheus → Targets → `pushgateway` should be UP.

This is the minimal, observable checklist.

---

## 3. High-level architecture

```mermaid
flowchart TD
    subgraph Sources
        A[NOAA-like stream<br/>(ts, temperature, windspeed, pressure)]
        B[Simulated drift stream<br/>(ts, temperature)]
    end

    subgraph Storage
        C[data/processed/*.parquet]
    end

    subgraph Processing
        P1[Spark / PySpark style processing<br/>bronze → silver]
    end

    subgraph Training
        T1[data/generate_training_data.py]
        T2[data/generate_training_data_sim.py]
        T3[training/mlflow_gru_train.py]
        T4[training/train_prophet.py]
    end

    subgraph Serving
        S[FastAPI: /predict /eval /routine-retrain-check /metrics]
    end

    subgraph Monitoring
        M1[monitoring/drift_detector.py]
        M2[Pushgateway]
        M3[Prometheus]
        M4[Grafana]
    end

    A --> C
    B --> C
    C --> P1 --> T1
    C --> T2
    T1 --> T3 --> S
    T2 --> T3
    T4 --> S
    S --> M1
    M1 -->|"write retrain.flag"| T3
    S -->|"export metrics"| M3
    M1 --> M2 --> M3 --> M4
```

Two data shapes, one serving plane, with monitoring and retrain in the loop.

---

## 4. Two pipelines

### 4.1 NOAA-like (stable)
- Schema:
  - `ts: timestamp`
  - `temperature: float`
  - `windspeed: float`
  - `pressure: float`
- Stored under: `data/processed/noaa/*.parquet`
- CSV builder: `data/generate_training_data.py`
- Trainer: `training/mlflow_gru_train.py`
- Even though the data is stable, the API still exposes `/routine-retrain-check` so the loop can stay active.

### 4.2 Simulator (drift-on-purpose)
- Schema:
  - `ts: timestamp`
  - `temperature: float`
- Stored under: `data/processed/part-*.parquet`
- CSV builder: `data/generate_training_data_sim.py`
- Trainer: either overwrite the main CSV and run the normal trainer:
  ```bash
  cp data/training_data_sim.csv data/training_data.csv
  python training/mlflow_gru_train.py
  ```
  or use the dedicated sim trainer:
  ```bash
  python training/mlflow_gru_train_sim.py
  ```
- Feeder: `streaming/client/live_predictor_feeder_sim.py`  
  calls `/predict` and `/eval`, and writes `logs/prediction_history.csv`, which gives the drift detector something to analyze.

---

## 5. Runbook (demo)

This is the end-to-end demo path.

```bash
# 0) clean
rm -f retrain.flag last_retrain.txt
rm -f data/processed/part-*.parquet
rm -f logs/prediction_history.csv logs/last_drift_state.txt logs/last_drift.txt logs/drift_count.txt

# 1) collect calm data for both Prophet & GRU
CALM_ONLY=1 python streaming/ingestion/heatwave_simulator.py
# ... let it run 60–90s, then Ctrl+C

# 2) build CSV from calm
python data/generate_training_data_sim.py

# 3) train Prophet baseline
python training/train_prophet.py

# 4) train GRU on same calm window
cp data/training_data_sim.csv data/training_data.csv
python training/mlflow_gru_train.py

# 5) start simulated feeder (this hits /predict and /eval and writes logs/prediction_history.csv)
python streaming/client/live_predictor_feeder_sim.py

# 6) now start HEATWAVE version (no CALM_ONLY)
python streaming/ingestion/heatwave_simulator.py

# 7) run drift detector to turn heat into retrain.flag
python monitoring/drift_detector.py

# 8) consume the flag → retrain
python training/mlflow_gru_train.py
```

This path shows the “state-aware” nature: ingestion → model use → error observed → drift detected → flag created → trainer consumes flag → artifacts updated.

---

## 6. Self-healing loop

```mermaid
sequenceDiagram
    participant Feeder
    participant API as FastAPI
    participant Drift as drift_detector.py
    participant Trainer as mlflow_gru_train.py
    participant Store as artifacts/

    Feeder->>API: POST /predict {sequence}
    API-->>Feeder: forecast
    Feeder->>API: POST /eval {predicted, actual}
    API-->>Feeder: {rmse, drift_flagged}
    Drift->>logs/prediction_history.csv: scan recent errors
    alt error pattern changed
        Drift->>Trainer: create retrain.flag
        Trainer->>Trainer: read flag + reason
        Trainer->>Store: save .pt / model files
        Trainer->>MLflow: log run
        Trainer->>API: next reload picks up new artifact
    end
```

- the detector increments only on change to keep noise low,
- the trainer clears the flag to keep the channel single-shot,
- metrics are pushed so Prometheus/Grafana can show when retrain happened.

---

## 7. Serving (FastAPI)

Entrypoint:

```bash
uvicorn serving.fastapi_app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /healthz`
- `POST /predict`
- `POST /eval`
- `GET /metrics`
- `POST /routine-retrain-check`

Tested in CI with `pytest` and `httpx` (Starlette TestClient requires `httpx`).

---

## 8. Streaming / Kafka / Spark

- `streaming/ingestion/` and `streaming/processing/` mimic a Kafka/Spark ingestion path landing parquet under `data/processed/`.
- Folder structure under `data/` shows bronze/silver style layers, which is common in Spark or Delta pipelines.
- The docs folder contains extra notes:
  - `docs/kafka_startup.md`
  - `docs/RealTime_Predictor_Loop.md`
  - `docs/readme_monitoring.md`

---

## 9. Data prep (schemas)

### 9.1 NOAA → CSV
Source: `data/processed/noaa/*.parquet`  
Expected columns: `ts, temperature, windspeed, pressure`  
Windowed (e.g. 5 steps):

```text
t0_temp, t0_wind, t0_pressure,
t1_temp, t1_wind, t1_pressure,
t2_temp, t2_wind, t2_pressure,
t3_temp, t3_wind, t3_pressure,
t4_temp, t4_wind, t4_pressure,
target
```

This is exactly what `training/mlflow_gru_train.py` consumes.

### 9.2 Simulator → CSV
Source: `data/processed/part-*.parquet`  
Expected columns: `ts, temperature`  
Windowed (5 steps):

```text
t0_temp, t1_temp, t2_temp, t3_temp, t4_temp, target
```

This is what `data/generate_training_data_sim.py` writes.  
After that, either copy to the main CSV or call the sim trainer.

---

## 10. Deployment options

### 10.1 Local (compose)
- start everything:
  ```bash
  docker compose up
  ```
- verify:
  ```bash
  curl http://localhost:8000/healthz
  curl http://localhost:8000/metrics
  ```

### 10.2 Kubernetes (Helm)
- chart is in `helm/aerocast`
- install / upgrade:
  ```bash
  helm upgrade --install aerocast ./helm/aerocast     --namespace aerocast --create-namespace
  kubectl get pods -n aerocast
  ```
- port-forward to test API, Grafana, Prometheus.

### 10.3 Terraform (IaC wrapper)
- file: `infra/main.tf`
- providers: `kubernetes` and `helm`
- applies the local chart from `../helm/aerocast` into the `aerocast` namespace
- in practice:
  ```bash
  cd infra
  terraform init
  terraform apply
  ```
- if the release already exists, Terraform will recognize it and update in place; timeouts were increased, and `wait = false` was used to avoid long blocking applies on local clusters.

---

## 11. CI

Single workflow to keep things predictable: `.github/workflows/ci.yml`

- triggers on `main` and `helm`,
- installs Python 3.10,
- installs runtime + dev requirements,
- creates `serving/__init__.py` on CI to make imports resolvable,
- runs `pytest -q` (three tests in `tests/`),
- logs into GHCR with `${{ secrets.GITHUB_TOKEN }}`,
- builds:
  - `ghcr.io/mdgolammafuz/aerocast-api:latest`
  - `ghcr.io/mdgolammafuz/aerocast-worker:latest`

Extra docker-only workflows were removed to avoid parallel duplicate runs.

---

## 12. “Flat” GRU output note

A flat forecast band was observed during development. Probable causes:

1. training performed on a very short calm window → model learned a constant;
2. 1D simulated CSV was passed to the 3D trainer without aligning feature count;
3. old artifact files were still present and kept being loaded.

Current layout makes the flow explicit:

- simulator builds `data/training_data_sim.csv`,
- that file can be copied over `data/training_data.csv`,
- `training/mlflow_gru_train.py` can then be run,
- FastAPI will serve the refreshed artifact.

---

## 13. Security / production notes

- no secrets committed,
- `retrain.flag` is a local-file contract; a production system would move this to a message bus or DB,
- Pushgateway and Prometheus are cluster-internal in the Helm chart; ingress/auth is left out for brevity,
- MLflow in `mlruns/` is local; a remote tracking server and artifact store would be used in production,
- GHCR is used for images; in a company, an internal registry and OIDC login would be preferred.

---

## 14. License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
