import os
import glob
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_CSV = os.path.join(ROOT, "data", "training_data.csv")

PARQUET_DIRS = [
    os.path.join(ROOT, "data", "processed", "noaa"),
]

WINDOW = 5  # must match feeder + model
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

    # ---- station-aware filtering (NEW) ----
    station_id = os.environ.get("AEROCAST_NOAA_STATION")
    if "station" in df.columns:
        if not station_id:
            raise RuntimeError(
                "AEROCAST_NOAA_STATION is not set but 'station' column exists in NOAA data. "
                "Set AEROCAST_NOAA_STATION to the station id you want to train on."
            )
        df = df[df["station"] == station_id].copy()
        if df.empty:
            raise RuntimeError(
                f"No rows found for station '{station_id}' in NOAA parquet files."
            )
        # drop station column after filtering
        df = df.drop(columns=["station"])
    else:
        # if there's no station column, we silently train on whatever is there
        # (e.g. per-station-directory layout). That's fine.
        pass

    # drop any extra "v" column if present
    if "v" in df.columns:
        df = df.drop(columns=["v"])

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
    print(
        f"[generate_training_data] station={os.environ.get('AEROCAST_NOAA_STATION', 'UNSPECIFIED')} "
        f"rows={len(out_df)} -> {OUT_CSV}"
    )


if __name__ == "__main__":
    build_training_csv()
