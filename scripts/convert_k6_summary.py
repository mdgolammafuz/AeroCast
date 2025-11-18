import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(ROOT, "bench", "k6_summary_raw.json")
OUT_PATH = os.path.join(ROOT, "bench", "k6_summary.json")


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run k6 with --summary-export first."
        )

    with open(RAW_PATH, "r") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})

    # Inspect once for sanity
    print("[convert_k6_summary] available metric keys:")
    for k in sorted(metrics.keys()):
        print(f"  - {k}")

    http_req = metrics.get("http_req_duration", {})
    http_reqs = metrics.get("http_reqs", {})

    # k6 usually stores percentiles directly on the metric
    # e.g. "p(95)" or "p(95.0)"
    p95 = (
        http_req.get("p(95)")
        or http_req.get("p(95.0)")
        or http_req.get("p95")
    )

    # RPS is usually under http_reqs["rate"]
    rps = http_reqs.get("rate")

    print(f"[convert_k6_summary] http_req_duration = {http_req}")
    print(f"[convert_k6_summary] http_reqs        = {http_reqs}")

    summary = {
        "rps": rps,
        "p95_latency_ms": p95,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[convert_k6_summary] wrote {OUT_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
