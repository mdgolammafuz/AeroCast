"""
Simulated weather sensor stream designed to trigger drift robustly, without globals.

Phases:
- Warmup (calm): Uniform(20, 30) for RAMP_AFTER records
- Heatwave: immediate upward BUMP, then linear RAMP_STEP per tick
- Periodic spikes to accentuate GRU error vs Prophet

Result: GRU MAE > 1.25× Prophet MAE over the recent window -> retrain.flag
"""

import os
import time
import random
import pandas as pd
from datetime import datetime

# ----- Config -----
DATA_DIR     = "data/processed"
CADENCE_SEC  = 5
RAMP_AFTER   = 20         # calm points before regime shift
BUMP         = 12.0       # immediate jump at shift
RAMP_STEP    = 0.6        # degrees per reading after shift
NOISE_STD    = 0.6        # small noise around trend
SPIKE_EVERY  = 7          # every Nth heatwave reading becomes a spike
SPIKE_TEMP   = (46.0, 52.0)
SEED         = 42         # reproducible demo

random.seed(SEED)
os.makedirs(DATA_DIR, exist_ok=True)

def _calm_temp() -> float:
    return round(random.uniform(20.0, 30.0), 2)

def _heatwave_temp(k_since_shift: int) -> float:
    # Linear trend + small Gaussian noise
    base  = 25.0 + BUMP + k_since_shift * RAMP_STEP
    noise = random.gauss(0.0, NOISE_STD)
    val   = base + noise
    if k_since_shift > 0 and (k_since_shift % SPIKE_EVERY == 0):
        val = random.uniform(*SPIKE_TEMP)
    return round(val, 2)

def main():
    print("Streaming sensor … CTRL-C to stop")
    print(f"Warmup (calm) for {RAMP_AFTER} points, then heatwave ramp + spikes.")

    count = 0  # local state
    while True:
        count += 1

        if count <= RAMP_AFTER:
            phase = "CALM"
            temp = _calm_temp()
        else:
            phase = "HEATWAVE"
            temp = _heatwave_temp(count - RAMP_AFTER)

        rec = {"ts": datetime.utcnow().isoformat(), "temperature": temp}
        fname = f"{DATA_DIR}/part-{time.time()}.parquet"
        pd.DataFrame([rec]).to_parquet(fname, index=False)
        print(f"[{phase}] {rec['ts']}  temp={temp}°C  -> {fname}")

        time.sleep(CADENCE_SEC)

if __name__ == "__main__":
    main()
