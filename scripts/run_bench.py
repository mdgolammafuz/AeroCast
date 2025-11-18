import argparse
import subprocess
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd, cwd=None):
    print(f"[run_bench] → {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        print(f"[run_bench] command failed with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noaa",
        action="store_true",
        help="run NOAA accuracy benchmark",
    )
    parser.add_argument(
        "--drift-sim",
        action="store_true",
        help="run simulator drift benchmark (uses existing logs)",
    )
    parser.add_argument(
        "--k6-summary",
        action="store_true",
        help="convert k6 summary_export JSON to k6_summary.json",
    )
    args = parser.parse_args()

    # default: run all
    if not (args.noaa or args.drift_sim or args.k6_summary):
        args.noaa = args.drift_sim = args.k6_summary = True

    if args.noaa:
        _run([sys.executable, "scripts/bench_noaa_sample.py"])

    if args.drift_sim:
        _run([sys.executable, "scripts/eval_drift_sim.py"])

    if args.k6_summary:
        _run([sys.executable, "scripts/convert_k6_summary.py"])

    print("[run_bench] done.")


if __name__ == "__main__":
    main()
