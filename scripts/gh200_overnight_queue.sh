#!/usr/bin/env bash
# GH200 overnight queue: ~15h sequential ablation cells, all on the
# single GH200 GPU. Runs synchronously, one cell at a time. Logs to
# logs/gh200_overnight.log; per-cell logs in logs/<run>.log.
#
# Sequence:
#   1. v27b-seed3   0.6B F3-off LME    ~1.5h   verify v27b-seed1 finding
#   2. v27b-seed4   0.6B F3-off LME    ~1.5h   third verification seed
#   3. v28a         1.7B F3-off LME    ~6h     does F3-off scale?
#   4. v28b         1.7B F3-off LME    ~6h     second 1.7B seed
#
# Run on GH200 inside ~/memory_residuals after `source ~/venv/bin/activate`:
#     nohup bash scripts/gh200_overnight_queue.sh > logs/gh200_overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=logs/gh200_overnight.log
mkdir -p logs
echo "[gh200_queue] $(date -Iseconds) starting" | tee -a "$LOG"

run_cell() {
    local label=$1; shift
    local script=$1; shift
    echo "[gh200_queue] $(date -Iseconds)  begin  $label" | tee -a "$LOG"
    bash "$script" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    echo "[gh200_queue] $(date -Iseconds)  end    $label rc=$rc" | tee -a "$LOG"
    sleep 30
}

run_cell "v27b_seed3" "Scripts/train_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200.sh"
run_cell "v27b_seed4" "Scripts/train_v27b_v24a_no_probe_seed4_0p6b_frozen_gh200.sh"
run_cell "v28a"       "Scripts/train_v28a_v25a_no_probe_seed1_1p7b_frozen_gh200.sh"
run_cell "v28b"       "Scripts/train_v28b_v25a_no_probe_seed2_1p7b_frozen_gh200.sh"

echo "[gh200_queue] $(date -Iseconds) ALL DONE" | tee -a "$LOG"
