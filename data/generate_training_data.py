import os
import glob
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_CSV = os.path.join(ROOT, "data", "training_data.csv")

PARQUET_DIRS = [
    os.path.join(ROOT, "data", "processed", "noaa"),
]

WINDOW = 24  # was 5; must match feeder + model
FEATURES = ["temperature", "windspeed", "pressure"]
TARGET_COL = "temperature"


def _collect_parquets() -> pd.DataFrame:
    files = []
    for d in PARQUET_DIRS:
        if os.path.isdir(d):
            # skip 0-byte files
            for f in glob.glob(os.path.join(d, "*.parquet")):
                if os.path.getsize(f) > 0:
                    files.append(f)

    if not files:
        raise FileNotFoundError("no parquet files found in data/processed/noaa/")

    dfs = [pd.read_parquet(f) for f in sorted(files)]
    df = pd.concat(dfs, ignore_index=True)

    for col in ["v", "station"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "ts" not in df.columns:
        raise ValueError("expected 'ts' column in parquet data")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)

    for f in FEATURES:
        if f not in df.columns:
            raise ValueError(f"expected '{f}' column in parquet data")

    return df[["ts"] + FEATURES]


def build_training_csv():
    df = _collect_parquets()

    rows = []
    for i in range(len(df) - WINDOW):
        window_df = df.iloc[i : i + WINDOW]
        target_row = df.iloc[i + WINDOW]

        feats = []
        for _, r in window_df.iterrows():
            for f in FEATURES:
                feats.append(float(r[f]))

        target = float(target_row[TARGET_COL])
        rows.append(feats + [target])

    cols = []
    for k in range(WINDOW):
        for f in FEATURES:
            if f == "temperature":
                cols.append(f"t{k}_temp")
            elif f == "windspeed":
                cols.append(f"t{k}_wind")
            elif f == "pressure":
                cols.append(f"t{k}_pressure")
            else:
                cols.append(f"t{k}_{f}")
    cols.append("target")

    out_df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[generate_training_data] station rows={len(out_df)} -> {OUT_CSV}")


if __name__ == "__main__":
    build_training_csv()
