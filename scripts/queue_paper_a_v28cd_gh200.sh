#!/usr/bin/env bash
# Sequential queue runner for Paper A v28c, v28d (1.7B no-F3) seeds 3,4 on GH200.
# Each cell takes ~3-6h; total ~12h.
# Run inside `tmux` per OVERNIGHT_STATUS.md guidance — survives SSH drops.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
QLOG="$LOG_DIR/queue_paper_a_v28cd_gh200.log"

CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES

# GH200 has its torch/transformers stack inside ~/venv (system python lacks both).
if [ -f "$HOME/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/venv/bin/activate"
fi

{
  echo "[$(date -Iseconds)] queue start, GPU=$CUDA_VISIBLE_DEVICES, host=$(hostname), python=$(which python)"
  for CELL in "v28c:3" "v28d:4"; do
    LETTER=${CELL%:*}
    SEED=${CELL#*:}
    SCRIPT="Scripts/train_${LETTER}_v25a_no_probe_seed${SEED}_1p7b_frozen_gh200.sh"
    echo "[$(date -Iseconds)] launching ${LETTER}-seed${SEED}: $SCRIPT"
    bash "$SCRIPT"
    rc=$?
    echo "[$(date -Iseconds)] ${LETTER}-seed${SEED} exit=$rc"
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 42 ]; then
      echo "[$(date -Iseconds)] non-collapse failure (rc=$rc); aborting queue"
      exit "$rc"
    fi
  done
  echo "[$(date -Iseconds)] queue ALL DONE"
} >> "$QLOG" 2>&1
