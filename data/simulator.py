import os
import time
import random
import pandas as pd
from datetime import datetime

DATA_DIR = "data/processed/"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_and_save():
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "temperature": round(random.uniform(20.0, 40.0), 2),
        "humidity": round(random.uniform(30.0, 90.0), 2),
        "rainfall": round(random.uniform(0.0, 20.0), 2)
    }
    filename = f"{DATA_DIR}/part-{time.time()}.parquet"
    df = pd.DataFrame([record])
    df.to_parquet(filename, index=False)
    print(f"Wrote: {filename} → {record}")

if __name__ == "__main__":
    print("Starting sensor data simulator...")

    # Step 1: Initial backfill — generate 50 records fast (fake historical context)
    for _ in range(50):
        generate_and_save()
        time.sleep(0.1)

    # Step 2: Live simulation — 1 record every 5 seconds
    while True:
        generate_and_save()
        time.sleep(5)
