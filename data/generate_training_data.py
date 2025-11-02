import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import glob

def build_training_csv(
    output_path="data/training_data.csv",
    window=5,
    max_temp=35.0,          # <-- keep only calm data
):
    # Read all parquet files in chronological order
    files = sorted(glob.glob("data/processed/part-*.parquet"))
    if not files:
        print("No parquet files found in data/processed/")
        return

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # ---- NEW: keep only calm rows ----
    df = df[df["temperature"] <= max_temp].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    if len(df) <= window:
        print("Not enough CALM data to build windows.")
        return

    # Build windows
    rows = []
    for i in range(len(df) - window):
        seq = df["temperature"].iloc[i:i+window].tolist()
        target = df["temperature"].iloc[i+window]
        rows.append(seq + [target])

    cols = [f"t{i}" for i in range(window)] + ["target"]
    pd.DataFrame(rows, columns=cols).to_csv(output_path, index=False)
    print(f"Wrote CALM training data ({len(rows)} rows, temp ≤ {max_temp}) to {output_path}")

if __name__ == "__main__":
    build_training_csv()
