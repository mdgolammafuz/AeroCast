"""
Calm version: Uniform(20,30) ONLY — no spikes.
Used for Loop-A baseline.
"""
import os, time, random, pandas as pd
from datetime import datetime

DIR = "data/processed"
os.makedirs(DIR, exist_ok=True)

print("[Simulator] Started generating calm data to:", DIR)

while True:
    ts = datetime.utcnow().isoformat()
    temp = round(random.uniform(20, 30), 2)
    rec = {"ts": ts, "temperature": temp}
    
    # Save to Parquet
    filepath = f"{DIR}/part-{time.time()}.parquet"
    pd.DataFrame([rec]).to_parquet(filepath, index=False)
    
    # Print the log
    print(f"[Simulator] → {ts} | temperature = {temp}°C")

    time.sleep(5)
