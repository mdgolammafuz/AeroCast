import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import pandas as pd

from monitoring.drift_detector import TEMP_THRESHOLD, WINDOW as DRIFT_WINDOW

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "logs")

PRED_CSV = os.path.join(LOG_DIR, "prediction_history.csv")
DRIFT_LOG = os.path.join(LOG_DIR, "drift.log")

REPORT_DIR = os.path.join(ROOT, "bench")
REPORT_PATH = os.path.join(REPORT_DIR, "drift_eval.json")


def _load_predictions() -> pd.DataFrame:
    if not os.path.exists(PRED_CSV):
        raise FileNotFoundError(
            f"prediction history not found: {PRED_CSV}. "
            "Run the simulator + live_predictor_feeder_sim first."
        )
    df = pd.read_csv(PRED_CSV)
    if not {"timestamp", "actual", "model"}.issubset(df.columns):
        raise ValueError("prediction_history.csv missing required columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # focus on GRU rows for ground truth actuals
    df = df[df["model"] == "GRU"].copy()
    return df


def _load_drift_events() -> list:
    """Parse drift.log lines like:
    2025-... DRIFT  mean_actual=... >= 38.0°C
    """
    if not os.path.exists(DRIFT_LOG):
        return []
    events = []
    with open(DRIFT_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "DRIFT" not in line:
                continue
            # timestamp is first token
            ts_str = line.split()[0]
            try:
                ts = pd.to_datetime(ts_str)
                events.append(ts)
            except Exception:
                continue
    events.sort()
    return events


def _ground_truth_drifts(df: pd.DataFrame) -> list:
    """
    Ground truth: COOL->HOT based on actual >= TEMP_THRESHOLD.

    We approximate: at any row where actual >= threshold and previous row < threshold,
    a new drift episode begins.
    """
    actual = df["actual"].astype(float).values
    ts = df["timestamp"].values

    gt_starts = []
    was_hot = False
    for i in range(len(actual)):
        is_hot = actual[i] >= TEMP_THRESHOLD
        if is_hot and not was_hot:
            # COOL -> HOT
            gt_starts.append(ts[i])
        was_hot = is_hot
    return gt_starts


def _match_events(gt_starts, detected):
    """
    Greedy matching:
    - For each ground truth start, find the first detection >= that time.
    - That pair is a TP, remove detection from pool.
    - Remaining detections are FP.
    - Ground truth without a match are FN.
    """
    detected_sorted = list(sorted(detected))
    tp_pairs = []
    fp = 0
    fn = 0

    used_idx = set()

    for gt in gt_starts:
        match_idx = None
        for i, det in enumerate(detected_sorted):
            if i in used_idx:
                continue
            if det >= gt:
                match_idx = i
                break
        if match_idx is None:
            fn += 1
        else:
            used_idx.add(match_idx)
            tp_pairs.append((gt, detected_sorted[match_idx]))

    fp = len(detected_sorted) - len(used_idx)

    return tp_pairs, fp, fn


def eval_drift() -> dict:
    df = _load_predictions()
    gt_starts = _ground_truth_drifts(df)
    detected = _load_drift_events()

    tp_pairs, fp, fn = _match_events(gt_starts, detected)
    tp = len(tp_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    latencies_min = []
    for gt, det in tp_pairs:
        delta = (det - gt).total_seconds() / 60.0
        if delta >= 0:
            latencies_min.append(delta)

    avg_latency = float(sum(latencies_min) / len(latencies_min)) if latencies_min else None

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "avg_latency_minutes": avg_latency,
        "threshold": TEMP_THRESHOLD,
        "window_used_by_detector": DRIFT_WINDOW,
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    result = eval_drift()
    with open(REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[eval_drift_sim] wrote drift metrics to {REPORT_PATH}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
