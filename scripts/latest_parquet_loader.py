import os
import glob
import pandas as pd

def load_latest_parquet(parquet_dir="data/processed/"):
    """
    Load the most recent Parquet file from a given directory.

    Parameters:
        parquet_dir (str): Directory containing parquet files.

    Returns:
        pd.DataFrame: Loaded DataFrame from the latest parquet file.
    """
    print("🔍 Loading latest Parquet data...")

    # Find all .parquet files
    parquet_files = glob.glob(os.path.join(parquet_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}")

    # Get the most recently modified parquet file
    latest_file = max(parquet_files, key=os.path.getmtime)
    print(f"📁 Latest Parquet file: {latest_file}")

    # Load and return as DataFrame
    df = pd.read_parquet(latest_file)
    print(f"✅ Loaded data: {df.shape}")
    return df
