import os
import time
import requests
import pandas as pd
from collections import deque

# --- CONFIG ---
API_URL = os.environ.get("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"

# Paths adjusted relative to this script
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
HISTORY_FILE = os.path.join(LOG_DIR, "prediction_history.csv")

os.makedirs(LOG_DIR, exist_ok=True)
WINDOW_SIZE = 24

def get_latest_parquet():
    try:
        if not os.path.exists(DATA_DIR): return None
        files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and f.startswith("part-")]
        if not files: return None
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)), reverse=True)
        return os.path.join(DATA_DIR, files[0])
    except Exception:
        return None

def main():
    print(f"[Feeder-SIM] Targeting API at: {API_URL}")
    print(f"[Feeder-SIM] Buffering {WINDOW_SIZE} steps before first prediction...")
    
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            f.write("ts,model,prediction,actual,error\n")

    seen_ts = set()
    buffer = deque(maxlen=WINDOW_SIZE)

    while True:
        fpath = get_latest_parquet()
        if not fpath:
            time.sleep(1)
            continue

        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"[Feeder-SIM] Read error: {e}")
            time.sleep(1)
            continue
        
        df = df.sort_values("ts")
        
        for _, row in df.iterrows():
            ts = str(row["ts"])
            if ts in seen_ts:
                continue
            
            seen_ts.add(ts)
            
            # 1. Update Buffer (Fake 3D feature: [temp, 0, 0])
            temp = float(row["temperature"])
            buffer.append([temp, 0.0, 0.0])

            # 2. Wait for buffer to fill
            if len(buffer) < WINDOW_SIZE:
                if len(buffer) % 5 == 0:
                    print(f"[Feeder-SIM] Buffering... ({len(buffer)}/{WINDOW_SIZE})")
                continue

            # 3. Predict
            try:
                # Send list of lists
                payload = {"sequence": list(buffer)}
                resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=2)
                
                if resp.status_code != 200:
                    print(f"[Feeder-SIM] API Error {resp.status_code}: {resp.text}")
                    continue
                
                data = resp.json()
                
                # Robust parsing: Handle list vs float response
                forecast = data.get("forecast")
                if isinstance(forecast, list):
                    pred_val = float(forecast[0])
                else:
                    pred_val = float(forecast)
                    
                model_name = data.get("model_name", "unknown")
                
            except Exception as e:
                print(f"[Feeder-SIM] Request failed: {e}")
                continue

            # 4. Log Result
            error = abs(pred_val - temp)
            
            with open(HISTORY_FILE, "a") as f:
                f.write(f"{ts},{model_name},{pred_val:.2f},{temp:.2f},{error:.4f}\n")
            
            print(f"[Feeder] Actual={temp:.1f} | Pred={pred_val:.1f} | Err={error:.2f}")
            
            # Throttle to match simulator cadence
            time.sleep(0.5)

if __name__ == "__main__":
    main()