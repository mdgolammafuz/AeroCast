#!/usr/bin/env bash
set -e
rm -f logs/last_drift_state.txt \
      logs/last_drift.txt \
      logs/retrain_count.txt \
      logs/drift.log \
      retrain.flag
echo "[reset-demo] file state cleared."
