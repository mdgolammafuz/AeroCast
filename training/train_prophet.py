import os, glob, json
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import mlflow

files = sorted(glob.glob("data/processed/part-*.parquet"))
if not files:
    raise SystemExit("no parquet files in data/processed")

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# keep only calm rows
df = df[df["temperature"] <= 35.0].copy()

df["ts"] = pd.to_datetime(df["ts"])
df = df.sort_values("ts").drop_duplicates("ts")
df = df.rename(columns={"ts": "ds", "temperature": "y"})

m = Prophet(
    growth="flat",
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=False,
)
m.fit(df)

# forecast +5s
last_ts = df["ds"].max()
future = pd.date_range(start=last_ts, periods=2, freq="5s")[1:]
future_df = pd.DataFrame({"ds": future})
forecast = m.predict(future_df)

os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/prophet_model.json"
with open(model_path, "w") as f:
    json.dump(model_to_json(m), f)

mlflow.set_experiment("AeroCast-Prophet")
with mlflow.start_run():
    mlflow.log_param("model_type", "Prophet-calm-only")
    mlflow.log_artifact(model_path)
    print("Prophet model (calm-only) trained and logged.")
    print(f"Last forecast: {forecast[['ds','yhat']].tail(1).to_dict(orient='records')[0]}")
