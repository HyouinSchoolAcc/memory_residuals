#!/usr/bin/env bash
# Polls for v23a/v23b completion (free GPU 0/1) and launches v24a/v24c
# as soon as each GPU frees up. Independent per GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/queue_v24.log
mkdir -p logs
> "$LOG"

free_gpu_when_ready() {
    local gpu=$1
    local launcher=$2
    local marker_pid=$3
    echo "[queue_v24] gpu $gpu: waiting for pid=$marker_pid (v23) to finish, then launching $launcher" | tee -a "$LOG"
    while kill -0 "$marker_pid" 2>/dev/null; do sleep 30; done
    echo "[queue_v24] gpu $gpu: pid=$marker_pid finished" | tee -a "$LOG"
    # Wait a few seconds for GPU memory to clear
    sleep 15
    echo "[queue_v24] gpu $gpu: launching $launcher" | tee -a "$LOG"
    bash "$launcher" 2>&1 | tee -a "$LOG"
}

V23A_PID=525697
V23B_PID=525701

free_gpu_when_ready 0 Scripts/train_v24a_v21c_lme_seed1_0p6b_frozen_local.sh "$V23A_PID" &
free_gpu_when_ready 1 Scripts/train_v24c_v21c_lmemsc_seed1_0p6b_frozen_local.sh "$V23B_PID" &
wait
echo "[queue_v24] both v24 launches dispatched" | tee -a "$LOG"
