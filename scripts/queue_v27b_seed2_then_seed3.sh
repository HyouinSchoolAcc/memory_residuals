#!/usr/bin/env bash
# Launch v27b-seed2 NOW on GPU 1, wait for it, then launch v27b-seed3.
# Independent of the GPU-0 queue (which is on v27c right now).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/queue_v27b_seeds.log
> "$LOG"

echo "[q27b] $(date +%H:%M:%S) launching v27b-seed2" | tee -a "$LOG"
bash Scripts/train_v27b_v24a_no_probe_seed2_0p6b_frozen_local.sh 2>&1 | tee -a "$LOG"
PID=$(pgrep -af "chain_v27b_v24a_no_probe_seed2_0p6b_frozen_local" | awk '{print $1}' | head -1)
if [ -z "${PID:-}" ]; then
    echo "[q27b] ERROR: pid for seed2 not found" | tee -a "$LOG"
    exit 1
fi
echo "[q27b] $(date +%H:%M:%S) v27b-seed2 pid=$PID" | tee -a "$LOG"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "[q27b] $(date +%H:%M:%S) v27b-seed2 finished" | tee -a "$LOG"
sleep 15

echo "[q27b] $(date +%H:%M:%S) launching v27b-seed3" | tee -a "$LOG"
bash Scripts/train_v27b_v24a_no_probe_seed3_0p6b_frozen_local.sh 2>&1 | tee -a "$LOG"
PID=$(pgrep -af "chain_v27b_v24a_no_probe_seed3_0p6b_frozen_local" | awk '{print $1}' | head -1)
echo "[q27b] $(date +%H:%M:%S) v27b-seed3 pid=$PID" | tee -a "$LOG"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "[q27b] $(date +%H:%M:%S) v27b-seed3 finished" | tee -a "$LOG"
echo "[q27b] $(date +%H:%M:%S) DONE" | tee -a "$LOG"
