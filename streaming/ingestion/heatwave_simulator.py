# streaming/ingestion/heatwave_simulator.py
"""
Simulated weather stream.

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

# calm phase
RAMP_AFTER = 20  # ~100 seconds @ 5s cadence

# heatwave shape (but we will cap)
BUMP = 12.0
RAMP_STEP = 0.6
NOISE_STD = 0.6
SPIKE_EVERY = 7
SPIKE_TEMP = (46.0, 52.0)

# hard safety cap so we don't see 200°C
MAX_TEMP = 55.0

SEED = 42
random.seed(SEED)
os.makedirs(DATA_DIR, exist_ok=True)

CALM_ONLY = os.environ.get("CALM_ONLY", "0") == "1"


def _calm_temp() -> float:
    return round(random.uniform(20.0, 30.0), 2)


def _heatwave_temp(k_since_shift: int) -> float:
    # baseline ramp
    base = 25.0 + BUMP + k_since_shift * RAMP_STEP
    noise = random.gauss(0.0, NOISE_STD)
    val = base + noise

    # every Nth reading, make it look dramatic
    if k_since_shift > 0 and (k_since_shift % SPIKE_EVERY == 0):
        val = random.uniform(*SPIKE_TEMP)

    # hard cap
    if val > MAX_TEMP:
        val = MAX_TEMP

    return round(val, 2)


def main():
    if CALM_ONLY:
        print("Streaming CALM ONLY … use this to pretrain Prophet/GRU. CTRL-C to stop.")
    else:
        print("Streaming calm → capped heatwave … CTRL-C to stop.")

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
        fname = f"{DATA_DIR}/part-{time.time()}.parquet"
        pd.DataFrame([rec]).to_parquet(fname, index=False)
        print(f"[{phase}] {rec['ts']}  temp={temp}°C  -> {fname}")

        time.sleep(CADENCE_SEC)


if __name__ == "__main__":
    main()
