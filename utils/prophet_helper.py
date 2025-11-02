import json
import pandas as pd
from prophet.serialize import model_from_json  # <-- official loader

def load_prophet_model(path: str = "artifacts/prophet_model.json"):
    """Load a Prophet model saved via prophet.serialize.model_to_json()."""
    with open(path, "r") as f:
        return model_from_json(json.load(f))

def forecast_next(df: pd.DataFrame, model, freq: str = "5s") -> float:
    """
    Predict the next temperature using a loaded Prophet model.
    Expects df with columns: ['ts', 'temperature'].
    """
    if df.empty:
        raise ValueError("forecast_next() received empty dataframe")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates("ts")
    df_p = df.rename(columns={"ts": "ds", "temperature": "y"})

    last_ts = df_p["ds"].max()
    future = pd.date_range(start=last_ts, periods=2, freq=freq)[1:]  # next step
    future_df = pd.DataFrame({"ds": future})

    fc = model.predict(future_df)
    return round(float(fc["yhat"].iloc[0]), 2)
