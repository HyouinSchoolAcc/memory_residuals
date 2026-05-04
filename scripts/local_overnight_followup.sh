#!/usr/bin/env bash
# Local overnight follow-up: AFTER v24a-seed4 (GPU 1) and v24a-seed5 (GPU 0)
# finish, launches v25a-seed3 (1.7B) on GPU 0 and v25a-seed4 (1.7B) on GPU 1
# in parallel for two more 1.7B headline seeds.
#
# Polls every 60s for free GPU memory (>40GB).
#
# Usage:
#   nohup bash scripts/local_overnight_followup.sh > logs/local_followup.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/local_overnight_followup.log
mkdir -p logs
echo "[followup] $(date -Iseconds) start" | tee -a "$LOG"

wait_for_gpu_and_done() {
    # Wait for: (a) the previous v24a seed on this GPU to have a final/ ckpt, and
    # (b) the GPU to be free (>40GB free).
    local gpu=$1 prev_outdir=$2 label=$3
    while true; do
        local free_mb
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
        if [ -n "$free_mb" ] && [ "$free_mb" -gt 40000 ] && [ -d "$prev_outdir/final" ]; then
            echo "[followup] $(date -Iseconds) GPU $gpu free + $prev_outdir/final exists -> launching $label" | tee -a "$LOG"
            return 0
        fi
        echo "[followup] $(date -Iseconds) GPU $gpu free=${free_mb:-?}MB | prev_final=$([ -d "$prev_outdir/final" ] && echo yes || echo no) | sleep 60 [$label]" >> "$LOG"
        sleep 60
    done
}

launch_v25a_seed3() {
    wait_for_gpu_and_done 0 "output/chain_v24a_v21c_lme_seed5_0p6b_frozen_local" "v25a_seed3"
    CUDA_VISIBLE_DEVICES=0 bash Scripts/train_v25a_v21c_lme_seed3_1p7b_frozen_local.sh 2>&1 | tee -a "$LOG"
    sleep 5
    # Wait for it to actually finish.
    while [ ! -d "output/chain_v25a_v21c_lme_seed3_1p7b_frozen_local/final" ]; do sleep 120; done
    echo "[followup] $(date -Iseconds) v25a_seed3 done" | tee -a "$LOG"
}

launch_v25a_seed4() {
    wait_for_gpu_and_done 1 "output/chain_v24a_v21c_lme_seed4_0p6b_frozen_local" "v25a_seed4"
    CUDA_VISIBLE_DEVICES=1 bash Scripts/train_v25a_v21c_lme_seed4_1p7b_frozen_local.sh 2>&1 | tee -a "$LOG"
    sleep 5
    while [ ! -d "output/chain_v25a_v21c_lme_seed4_1p7b_frozen_local/final" ]; do sleep 120; done
    echo "[followup] $(date -Iseconds) v25a_seed4 done" | tee -a "$LOG"
}

launch_v25a_seed3 &
P3=$!
launch_v25a_seed4 &
P4=$!
wait "$P3" "$P4"
echo "[followup] $(date -Iseconds) both followup 1.7B seeds done" | tee -a "$LOG"
