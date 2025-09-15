# MLflow Integration in AeroCast++

This document explains how **MLflow** was integrated into the GRU-based time series forecasting pipeline in **AeroCast++**, and captures the execution path, UI outputs, key learnings, and debugging tips.

---

## Why We Use MLflow

MLflow was integrated to:

- Track model training experiments
- Log parameters (e.g., hidden dimension, learning rate)
- Track metrics (loss per epoch)
- Save and manage model artifacts
- View training summaries through a clean UI
- Compare multiple experiments visually

> "We don’t just run models — we track them professionally."

---

## Key Concepts

### Experiment
- A named container for related training runs
- Example: `AeroCast-GRU-Training`

### Run
- One training session = One run
- Each run has a **unique `run_id`**
- Run stores: params, metrics, loss plot, model artifact

---

## MLflow Directory Structure

Once a run completes:
```
mlruns/
└── 0/                       <- Experiment ID (0 = default or first created)
    └── [run_id]/
         ├── params/        <- All training parameters
         ├── metrics/       <- Epoch-wise loss
         └── artifacts/     <- Model + loss plot
```

You can have multiple run_ids under the same experiment.

---

## Logging Code Snippet

```python
with mlflow.start_run():
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_metric("loss", epoch_loss)
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact("loss_plot.png")
```

---

## How to Run

```bash
python training/mlflow_gru_train.py
```
Auto-creates experiment, logs everything.

Launch UI:
```bash
mlflow ui
```
Visit: `http://127.0.0.1:5000`

---

## UI Screenshot Examples

### Run Summary Page
Shows parameters, metrics, run metadata:

![Run Summary](docs/mlflow_run_summary.png)

### Loss Curve Visualization
Loss per epoch (logged manually or as artifact):

![Loss Curve](docs/mlflow_loss_curve.png)

---

## Key Lessons Learned

| Insight | Explanation |
|--------|-------------|
| `mlruns/0/` | Means experiment 0 — default unless specified |
| One run per script call | We start one run explicitly using `mlflow.start_run()` |
| UI Pages | Summary page ≠ Artifacts page (loss curve is nested) |
| Custom experiments | Use `mlflow.set_experiment("AeroCast-GRU-Training")` |
| Run multiple runs | Loop training over different seeds or hyperparams |

---

## Cleanup Tips

- Don't commit `mlruns/` unless needed for sharing results
- Add to `.gitignore`
- Use screenshots for README if showcasing

---

## Summary

MLflow transformed our local GRU training into a **tracked, reproducible, and professional pipeline**. This integration is now part of the AeroCast++ backbone — enabling monitoring, comparison, and scaling to more experiments as needed.

---

**Next Step → DVC Integration + Model Versioning + FastAPI Serving**


