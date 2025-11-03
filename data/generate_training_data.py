import os
import glob
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_CSV = os.path.join(ROOT, "data", "training_data.csv")

# where Spark is writing NOAA parquet
PARQUET_DIRS = [
    os.path.join(ROOT, "data", "processed"),
    os.path.join(ROOT, "data", "processed", "noaa"),
    os.path.join(ROOT, "data", "processed", "weather"),
]

WINDOW = 5  # must match feeder + model
TARGET_COL = "temperature"  # we forecast temperature


def _collect_parquets() -> pd.DataFrame:
    files = []
    for d in PARQUET_DIRS:
        if os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, "*.parquet")))
    if not files:
        raise FileNotFoundError("no parquet files found in data/processed*/")

    dfs = [pd.read_parquet(f) for f in sorted(files)]
    df = pd.concat(dfs, ignore_index=True)

    # drop producer-only stuff
    for col in ["v", "station"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ts must exist
    if "ts" not in df.columns:
        raise ValueError("expected 'ts' column in parquet data")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    # temperature must exist
    if "temperature" not in df.columns:
        raise ValueError("expected 'temperature' column in parquet data")

    # make synthetic cols so CSV shape is stable
    if "humidity" not in df.columns:
        base = df["temperature"].rolling(5, min_periods=1).mean().fillna(df["temperature"])
        df["humidity"] = (base + 20).clip(lower=30.0, upper=90.0)

    if "rainfall" not in df.columns:
        df["rainfall"] = 0.0

    return df[["ts", "temperature", "humidity", "rainfall"]]


def build_training_csv():
    df = _collect_parquets()

    rows = []
    for i in range(len(df) - WINDOW):
        window_df = df.iloc[i : i + WINDOW]
        target_row = df.iloc[i + WINDOW]

        feats = []
        for _, r in window_df.iterrows():
            feats.extend([
                float(r["temperature"]),
                float(r["humidity"]),
                float(r["rainfall"]),
            ])

        target = float(target_row[TARGET_COL])
        rows.append(feats + [target])

    # column names
    cols = []
    for k in range(WINDOW):
        cols.extend([f"t{k}_temp", f"t{k}_hum", f"t{k}_rain"])
    cols.append("target")

    out_df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[generate_training_data] wrote {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    build_training_csv()
