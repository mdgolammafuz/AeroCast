import os
import glob
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_CSV = os.path.join(ROOT, "data", "training_data_sim.csv")

PARQUET_DIR = os.path.join(ROOT, "data", "processed")  # simulator output
WINDOW = 5
TARGET_COL = "temperature"


def _collect_parquets() -> pd.DataFrame:
    files = []
    if os.path.isdir(PARQUET_DIR):
        for f in glob.glob(os.path.join(PARQUET_DIR, "*.parquet")):
            if os.path.getsize(f) > 0:
                files.append(f)

    if not files:
        raise FileNotFoundError("no parquet files found in data/processed/")

    dfs = [pd.read_parquet(f) for f in sorted(files)]
    df = pd.concat(dfs, ignore_index=True)

    if "ts" not in df.columns or "temperature" not in df.columns:
        raise ValueError("expected ts and temperature in simulated parquet")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df[["ts", "temperature"]]


def build_training_csv():
    df = _collect_parquets()

    rows = []
    for i in range(len(df) - WINDOW):
        window_df = df.iloc[i : i + WINDOW]
        target_row = df.iloc[i + WINDOW]

        feats = []
        for _, r in window_df.iterrows():
            temp = float(r["temperature"])
            feats.append(temp)        # real temp
            feats.append(5.0)         # fake wind
            feats.append(101000.0)    # fake pressure

        target = float(target_row[TARGET_COL])
        rows.append(feats + [target])

    cols = []
    for k in range(WINDOW):
        cols.append(f"t{k}_temp")
        cols.append(f"t{k}_wind")
        cols.append(f"t{k}_pressure")
    cols.append("target")

    out_df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[generate_training_data_sim] wrote {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    build_training_csv()
