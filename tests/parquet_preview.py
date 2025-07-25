import pandas as pd
import os

parquet_folder = "data/processed/"

# Find latest parquet file
files = sorted([
    f for f in os.listdir(parquet_folder)
    if f.endswith(".parquet")
], reverse=True)

if not files:
    print("No parquet files found.")
else:
    latest_file = os.path.join(parquet_folder, files[0])
    print(f"\nPreviewing file: {latest_file}\n")
    df = pd.read_parquet(latest_file)
    print(df.head())
