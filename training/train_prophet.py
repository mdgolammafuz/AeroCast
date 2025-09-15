# training/train_prophet.py
import os, glob, json
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json  # <-- correct way to serialize
import mlflow

# 1. Load and concatenate all parquet files
parquet_files = sorted(glob.glob("data/processed/part-*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

# 2. Rename and format for Prophet
df["ts"] = pd.to_datetime(df["ts"])
df = df.rename(columns={"ts": "ds", "temperature": "y"})

# 3. Optional: Remove duplicates or sort by timestamp
df = df.sort_values("ds").drop_duplicates("ds")

# 4. Fit Prophet model
m = Prophet()
m.fit(df)

# 5. Forecast 1 step ahead (e.g., 5 seconds)
future = m.make_future_dataframe(periods=1, freq="5s")
forecast = m.predict(future)

# 6. Save artifacts + MLflow log
os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/prophet_model.json"

# >>> FIX: serialize to JSON instead of m.save()
with open(model_path, "w") as f:
    json.dump(model_to_json(m), f)

mlflow.set_experiment("AeroCast-Prophet")
with mlflow.start_run():
    mlflow.log_param("model_type", "Prophet")
    mlflow.log_artifact(model_path)
    print("Prophet model trained and logged.")
    print(f"Last forecast: {forecast[['ds','yhat']].tail(1).to_dict(orient='records')[0]}")
