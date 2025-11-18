# bench/bench_noaa_sample.py
#
# Sample benchmark on real weather for one station.
#
# CHANGES:
# - Uses full year of data.
# - Train = first ~9 months, Test = last 3 months.
# - WINDOW = 24 (daily cycle).
# - GRU: 30 epochs, hidden_dim=64, time features (hour/day-of-week).
# - No early stopping; just a straightforward 30-epoch fit.

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import json

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

from model.gru_model import GRUWeatherForecaster

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOAA_DIR = os.path.join(ROOT, "data", "processed", "noaa")
os.makedirs(NOAA_DIR, exist_ok=True)
RESULT_PATH = os.path.join(ROOT, "bench", "accuracy_noaa_sample.json")

STATION_ID = os.environ.get("AEROCAST_NOAA_STATION", "USW00094846")
YEAR = int(os.environ.get("AEROCAST_NOAA_YEAR", "2024"))

LCD_FILENAME = f"LCD_{STATION_ID}_{YEAR}.csv"
LCD_PATH = os.path.join(NOAA_DIR, LCD_FILENAME)
LCD_URL = (
    f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/"
    f"{YEAR}/{LCD_FILENAME}"
)

WINDOW = 24          # 24 hours of context
GRU_EPOCHS = 30      # let it actually train
GRU_BATCH_SIZE = 32


def download_lcd_if_needed() -> None:
    if os.path.exists(LCD_PATH) and os.path.getsize(LCD_PATH) > 0:
        print(f"[bench-noaa] using existing {LCD_PATH}")
        return

    print(f"[bench-noaa] downloading LCD data:\n  {LCD_URL}")
    resp = requests.get(LCD_URL, timeout=60)
    if resp.status_code != 200:
        raise SystemExit(
            f"Failed to download LCD data: HTTP {resp.status_code}\n"
            f"{resp.text[:300]}"
        )

    with open(LCD_PATH, "wb") as f:
        f.write(resp.content)
    print(f"[bench-noaa] saved to {LCD_PATH}")


def _pick_temp_column(df: pd.DataFrame) -> str:
    preferred = ["HourlyDryBulbTemperature", "DryBulbTemperature"]
    for col in preferred:
        if col in df.columns:
            return col

    candidates = [c for c in df.columns
                  if "temp" in c.lower() or "temperature" in c.lower()]
    if not candidates:
        raise RuntimeError(
            "Could not find a temperature column in LCD CSV. "
            f"Columns: {list(df.columns)[:20]}..."
        )
    return candidates[0]


def load_timeseries() -> pd.DataFrame:
    df = pd.read_csv(LCD_PATH)

    if "DATE" not in df.columns:
        raise RuntimeError(
            "Expected 'DATE' column in LCD CSV; got "
            f"{list(df.columns)[:20]}..."
        )

    temp_col = _pick_temp_column(df)
    print(f"[bench-noaa] using temperature column: {temp_col}")

    df["ts"] = pd.to_datetime(df["DATE"])
    df["temperature"] = pd.to_numeric(df[temp_col], errors="coerce")

    # Time features
    df["hour"] = df["ts"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df["ts"].dt.dayofweek  # 0=Monday
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df = df[["ts", "temperature", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]].dropna()
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid ts/temperature rows after cleaning.")

    print(
        f"[bench-noaa] full range: {df['ts'].min()} → {df['ts'].max()} "
        f"({len(df)} rows)"
    )
    return df


def split_train_test_full_year(df: pd.DataFrame):
    """
    Use full year:
    - Train: everything before last 3 months.
    - Test: last 3 months.
    """
    max_ts = df["ts"].max()
    test_start = max_ts - pd.DateOffset(months=3)

    train_df = df[df["ts"] < test_start].copy().reset_index(drop=True)
    test_df = df[df["ts"] >= test_start].copy().reset_index(drop=True)

    print(
        f"[bench-noaa] train: {train_df['ts'].min()} → {train_df['ts'].max()} "
        f"({len(train_df)} rows)"
    )
    print(
        f"[bench-noaa] test:  {test_df['ts'].min()} → {test_df['ts'].max()} "
        f"({len(test_df)} rows)"
    )

    if len(train_df) <= WINDOW + 10 or len(test_df) <= WINDOW + 10:
        raise RuntimeError(
            f"Not enough rows for split with WINDOW={WINDOW}."
        )

    return train_df, test_df


class GRUDataset(Dataset):
    """Multi-feature: X[i] = WINDOW × 5 features, y[i] = next temp."""

    def __init__(self, df: pd.DataFrame, window: int):
        df = df.reset_index(drop=True)

        feature_cols = ["temperature", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        features = df[feature_cols].astype("float32").values  # (N, 5)
        temps = df["temperature"].astype("float32").values    # (N,)

        if len(temps) <= window:
            raise ValueError("Not enough data for GRU dataset.")

        X_list, y_list = [], []
        for i in range(window, len(temps)):
            seq = features[i - window : i]  # (W, 5)
            target = temps[i]
            X_list.append(seq)
            y_list.append(target)

        X_arr = np.array(X_list, dtype=np.float32)  # (N, W, 5)
        y_arr = np.array(y_list, dtype=np.float32)  # (N,)

        self.X = torch.tensor(X_arr)                 # (N, W, 5)
        self.y = torch.tensor(y_arr).unsqueeze(-1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_gru(train_df: pd.DataFrame) -> GRUWeatherForecaster:
    train_ds = GRUDataset(train_df, WINDOW)
    train_loader = DataLoader(train_ds, batch_size=GRU_BATCH_SIZE, shuffle=True)

    model = GRUWeatherForecaster(input_dim=5, hidden_dim=64, output_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lossf = nn.MSELoss()

    model.train()
    for epoch in range(GRU_EPOCHS):
        train_loss = 0.0
        for X, y in train_loader:
            opt.zero_grad()
            out = model(X)
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f"[bench-noaa][GRU] epoch={epoch} train_loss={avg_train_loss:.4f}")

    model.eval()
    return model


def eval_gru(model: GRUWeatherForecaster, full_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    full_df = full_df.reset_index(drop=True)
    test_start_ts = test_df["ts"].min()

    feature_cols = ["temperature", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    features = full_df[feature_cols].astype("float32").values  # (N, 5)
    temps = full_df["temperature"].astype("float32").values
    ts = full_df["ts"].values

    errors_sq = []
    count = 0

    for i in range(WINDOW, len(temps)):
        if ts[i] < test_start_ts:
            continue

        seq = features[i - WINDOW : i]  # (W, 5)
        arr = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, W, 5)

        with torch.no_grad():
            pred = float(model(arr).item())

        actual = float(temps[i])
        err = pred - actual
        errors_sq.append(err * err)
        count += 1

    if count == 0:
        raise RuntimeError("No GRU test samples produced.")

    rmse = math.sqrt(sum(errors_sq) / count)
    print(f"[bench-noaa][GRU] test_samples={count} rmse={rmse:.4f}")
    return rmse


def eval_naive(full_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    full_df = full_df.reset_index(drop=True)
    test_start_ts = test_df["ts"].min()

    temps = full_df["temperature"].astype("float32").values
    ts = full_df["ts"].values

    errors_sq = []
    count = 0

    for i in range(1, len(temps)):
        if ts[i] < test_start_ts:
            continue
        pred = temps[i - 1]
        actual = temps[i]
        err = pred - actual
        errors_sq.append(err * err)
        count += 1

    if count == 0:
        raise RuntimeError("No Naive test samples produced.")

    rmse = math.sqrt(sum(errors_sq) / count)
    print(f"[bench-noaa][Naive] test_samples={count} rmse={rmse:.4f}")
    return rmse


def eval_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float | None:
    if not HAS_PROPHET:
        print("[bench-noaa][Prophet] prophet not installed, skipping.")
        return None

    df_train = train_df[["ts", "temperature"]].rename(columns={"ts": "ds", "temperature": "y"})
    df_test = test_df[["ts", "temperature"]].rename(columns={"ts": "ds", "temperature": "y"})

    m = Prophet(
        growth="linear",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    m.fit(df_train)

    future = df_test[["ds"]].copy()
    forecast = m.predict(future)

    merged = df_test.merge(
        forecast[["ds", "yhat"]],
        on="ds",
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("Prophet: no overlapping timestamps for test merge.")

    y_true = merged["y"].astype(float).values
    y_pred = merged["yhat"].astype(float).values

    errors_sq = ((y_pred - y_true) ** 2).tolist()
    rmse = math.sqrt(sum(errors_sq) / len(errors_sq))
    print(f"[bench-noaa][Prophet] test_samples={len(errors_sq)} rmse={rmse:.4f}")
    return rmse


def run_benchmark() -> dict:
    download_lcd_if_needed()
    df = load_timeseries()
    train_df, test_df = split_train_test_full_year(df)

    gru_model = train_gru(train_df)
    rmse_gru = eval_gru(gru_model, df, test_df)
    rmse_naive = eval_naive(df, test_df)
    rmse_prophet = eval_prophet(train_df, test_df)

    result = {
        "station": STATION_ID,
        "year": YEAR,
        "window_months_total": 12,
        "train_months": 9,
        "test_months": 3,
        "window": WINDOW,
        "models": {
            "GRU": {"rmse": rmse_gru},
            "Naive": {"rmse": rmse_naive},
        },
    }
    if rmse_prophet is not None:
        result["models"]["Prophet"] = {"rmse": rmse_prophet}

    return result


def main():
    print(
        f"[bench-noaa] station={STATION_ID} year={YEAR} "
        "(9+3 month benchmark: full year, last 3 months as test)"
    )
    result = run_benchmark()

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[bench-noaa] wrote result to {RESULT_PATH}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
