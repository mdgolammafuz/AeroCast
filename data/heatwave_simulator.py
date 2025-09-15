"""
Simulated weather sensor stream with occasional heat‑wave spikes.
- Normal temp: Uniform(20, 30)
- Spike: 45–50°C every 60 records  ➜ causes drift
"""
import os, time, random, pandas as pd
from datetime import datetime

DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

COUNT = 0
def generate_and_save():
    global COUNT
    COUNT += 1
    # 1. create normal reading
    temp = round(random.uniform(20, 30), 2)
    # 2. Inject heat‑wave spike every 60th reading
    if COUNT % 60 == 0:
        temp = round(random.uniform(45, 50), 2)
    rec = {
        "ts": datetime.utcnow().isoformat(),
        "temperature": temp,
    }
    fname = f"{DATA_DIR}/part-{time.time()}.parquet"
    pd.DataFrame([rec]).to_parquet(fname, index=False)
    print(f"Saved {fname} — {rec}")

if __name__ == "__main__":
    print("Streaming sensor … CTRL‑C to stop")
    while True:
        generate_and_save()
        time.sleep(5)              # 5‑sec cadence
