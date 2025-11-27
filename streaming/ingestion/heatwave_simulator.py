"""
Simulated weather stream with Atomic Writes to prevent read errors.

Normal mode (default):
- first RAMP_AFTER rows = calm (20–30 °C)
- then heatwave: jump + slow ramp + occasional spikes
- BUT capped so it never goes absurd

CALM_ONLY=1 mode:
- stays in calm forever so you can collect baseline for Prophet/GRU
  example:
    CALM_ONLY=1 python streaming/ingestion/heatwave_simulator.py
"""

import os
import time
import random
import pandas as pd
from datetime import datetime

DATA_DIR = "data/processed"
CADENCE_SEC = 5

# Calm phase length (36 steps ensures Feeder buffer of 24 + 12 visible steps)
RAMP_AFTER = 36

# Heatwave shape
BUMP = 12.0
RAMP_STEP = 0.6
NOISE_STD = 0.6
SPIKE_EVERY = 7
SPIKE_TEMP = (46.0, 52.0)
MAX_TEMP = 55.0

SEED = 42
random.seed(SEED)
os.makedirs(DATA_DIR, exist_ok=True)

CALM_ONLY = os.environ.get("CALM_ONLY", "0") == "1"


def _calm_temp() -> float:
    return round(random.uniform(20.0, 30.0), 2)


def _heatwave_temp(k_since_shift: int) -> float:
    base = 25.0 + BUMP + k_since_shift * RAMP_STEP
    noise = random.gauss(0.0, NOISE_STD)
    val = base + noise
    if k_since_shift > 0 and (k_since_shift % SPIKE_EVERY == 0):
        val = random.uniform(*SPIKE_TEMP)
    if val > MAX_TEMP:
        val = MAX_TEMP
    return round(val, 2)


def main():
    mode = "CALM ONLY" if CALM_ONLY else "36 Calm -> Heatwave"
    print(f"Streaming: {mode}. CTRL-C to stop.")

    count = 0
    while True:
        count += 1

        if CALM_ONLY:
            phase = "CALM"
            temp = _calm_temp()
        else:
            if count <= RAMP_AFTER:
                phase = "CALM"
                temp = _calm_temp()
            else:
                phase = "HEATWAVE"
                temp = _heatwave_temp(count - RAMP_AFTER)

        rec = {"ts": datetime.utcnow().isoformat(), "temperature": temp}
        
        # --- ATOMIC WRITE PATTERN ---
        # 1. Define filename
        fname = f"{DATA_DIR}/part-{time.time()}.parquet"
        # 2. Write to a temporary hidden file first
        temp_fname = f"{fname}.tmp"
        
        try:
            pd.DataFrame([rec]).to_parquet(temp_fname, index=False)
            # 3. Rename is atomic: The Feeder will never see a partial file
            os.rename(temp_fname, fname)
            print(f"[{phase}] {rec['ts']}  temp={temp}°C  -> {fname}")
        except Exception as e:
            print(f"Error writing parquet: {e}")
            # Cleanup temp file on failure
            if os.path.exists(temp_fname):
                os.remove(temp_fname)

        time.sleep(CADENCE_SEC)


if __name__ == "__main__":
    main()