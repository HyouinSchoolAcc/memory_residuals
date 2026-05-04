#!/usr/bin/env bash
# Local overnight: adds two more v24a seeds (seed4 on GPU1, seed5 on GPU0)
# as soon as the corresponding GPUs free up. Tightens the 0.6B headline
# variance estimate to n=5 seeds.
#
# GPU 1 frees first (v27b-seed2 finishes ~04:00 EDT).
# GPU 0 frees later (v27c-seed1 finishes ~05:30 EDT).
#
# Usage:
#   nohup bash scripts/local_overnight_extra_seeds.sh > logs/local_overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/local_overnight_extra_seeds.log
mkdir -p logs
echo "[local] $(date -Iseconds) start" | tee -a "$LOG"

wait_for_gpu_free() {
    local gpu=$1 label=$2
    while true; do
        # Check whether any python process is using GPU `gpu`.
        local n
        n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
            | wc -l)
        # Simpler: query free memory threshold
        local free_mb
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
        if [ -n "$free_mb" ] && [ "$free_mb" -gt 60000 ]; then
            echo "[local] $(date -Iseconds) GPU $gpu free (${free_mb} MB free) for $label" | tee -a "$LOG"
            return 0
        fi
        echo "[local] $(date -Iseconds) GPU $gpu busy (${free_mb:-?} MB free); sleep 60 [$label]" >> "$LOG"
        sleep 60
    done
}

launch_seed4() {
    wait_for_gpu_free 1 "v24a_seed4"
    echo "[local] $(date -Iseconds) launching v24a_seed4 on GPU 1" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 bash Scripts/train_v24a_v21c_lme_seed4_0p6b_frozen_local.sh 2>&1 | tee -a "$LOG"
}

launch_seed5() {
    wait_for_gpu_free 0 "v24a_seed5"
    echo "[local] $(date -Iseconds) launching v24a_seed5 on GPU 0" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 bash Scripts/train_v24a_v21c_lme_seed5_0p6b_frozen_local.sh 2>&1 | tee -a "$LOG"
}

# Run launchers in parallel (each blocks on its own GPU).
launch_seed4 &
P4=$!
launch_seed5 &
P5=$!
wait "$P4" "$P5"
echo "[local] $(date -Iseconds) both v24a seeds 4+5 trainers handed off; daemon exits" | tee -a "$LOG"
