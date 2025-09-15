import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import glob

def build_training_csv(output_path="data/training_data.csv", window=5):
    # Read all parquet files in chronological order
    files = sorted(glob.glob("data/processed/part-*.parquet"))
    # Concatenate into one DataFrame
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # Build windows
    rows = []
    for i in range(len(df) - window):
        seq = df["temperature"].iloc[i:i+window].tolist()
        target = df["temperature"].iloc[i+window]
        rows.append(seq + [target])
    # Write CSV
    cols = [f"t{i}" for i in range(window)] + ["target"]
    pd.DataFrame(rows, columns=cols).to_csv(output_path, index=False)
    print(f"Wrote training data ({len(rows)} rows) to {output_path}")

if __name__ == "__main__":
    build_training_csv()
