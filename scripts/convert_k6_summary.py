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

    data = json.load(open(RAW_PATH, "r"))

    metrics = data.get("metrics", {})
    http_req = metrics.get("http_req_duration", {}).get("values", {})
    http_reqs = metrics.get("http_reqs", {}).get("values", {})

    p95 = http_req.get("p(95)") or http_req.get("p(95.0)") or http_req.get("p95")
    rps = http_reqs.get("rate")

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
