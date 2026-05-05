#!/usr/bin/env bash
# Sequential queue runner for Paper C v27a ablation seeds 2,3,4 on local GPU 0.
# Each cell takes ~1.5h on H100; total ~4.5h.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue_paper_c_v27a_gpu0.log"

CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES

{
  echo "[$(date -Iseconds)] queue start, GPU=$CUDA_VISIBLE_DEVICES, host=$(hostname)"
  for SEED in 2 3 4; do
    SCRIPT="Scripts/train_v27a_v24a_no_depth_seed${SEED}_0p6b_frozen_local.sh"
    echo "[$(date -Iseconds)] launching v27a-seed${SEED}: $SCRIPT"
    bash "$SCRIPT"
    rc=$?
    echo "[$(date -Iseconds)] v27a-seed${SEED} exit=$rc"
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 42 ]; then
      echo "[$(date -Iseconds)] non-collapse failure (rc=$rc); aborting queue"
      exit "$rc"
    fi
  done
  echo "[$(date -Iseconds)] queue ALL DONE"
} >> "$QLOG" 2>&1
