#!/usr/bin/env bash
# Polls every 2 min for newly-finished training cells (local + GH200-pulled)
# and runs eval_callback.py against the patched lme_val_s512_evpos.pt.
# Emits results JSON; idempotent and resumable.
#
# Watches:
#   chain_v27c_v24a_no_floor_seed1_0p6b_frozen_local      (local; tail-end of orig queue)
#   chain_v27b_v24a_no_probe_seed2_0p6b_frozen_local      (local; in-flight confirmation)
#   chain_v24a_v21c_lme_seed4_0p6b_frozen_local           (local; overnight extra)
#   chain_v24a_v21c_lme_seed5_0p6b_frozen_local           (local; overnight extra)
#   chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200           (GH200, rsync-pulled)
#   chain_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200      (GH200, rsync-pulled)
#   chain_v27b_v24a_no_probe_seed4_0p6b_frozen_gh200      (GH200, rsync-pulled)
#   chain_v28a_no_probe_seed1_1p7b_frozen_gh200           (GH200, rsync-pulled)
#   chain_v28b_no_probe_seed2_1p7b_frozen_gh200           (GH200, rsync-pulled)
#
# Outputs: results/eval_v25_seed_pack_evpos/<tag>_lme_val_evpos.json
#
# Usage:
#   nohup bash scripts/watcher_eval_overnight.sh > logs/watcher_overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR=results/eval_v25_seed_pack_evpos
mkdir -p "$OUT_DIR" logs

GPU="${WATCHER_GPU:-1}"

run_eval() {
    local ckpt=$1 tag=$2
    local out="$OUT_DIR/${tag}_lme_val_evpos.json"
    if [ -f "$out" ]; then return; fi
    if [ ! -d "$ckpt" ]; then return; fi
    echo "[watcher] $(date +%H:%M:%S)  evaluating $tag ($ckpt)"
    CUDA_VISIBLE_DEVICES="$GPU" python tools/eval_callback.py \
        --model_path "$ckpt" \
        --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \
        --names lme_val \
        --output "$out" 2>&1 | tail -3
}

declare -a RUNS=(
    "output/chain_v27c_v24a_no_floor_seed1_0p6b_frozen_local v27c_no_floor_best best"
    "output/chain_v27c_v24a_no_floor_seed1_0p6b_frozen_local v27c_no_floor_final final"
    "output/chain_v27b_v24a_no_probe_seed2_0p6b_frozen_local v27b_no_probe_seed2_best best"
    "output/chain_v27b_v24a_no_probe_seed2_0p6b_frozen_local v27b_no_probe_seed2_final final"
    "output/chain_v24a_v21c_lme_seed4_0p6b_frozen_local      v24a_seed4_best best"
    "output/chain_v24a_v21c_lme_seed4_0p6b_frozen_local      v24a_seed4_final final"
    "output/chain_v24a_v21c_lme_seed5_0p6b_frozen_local      v24a_seed5_best best"
    "output/chain_v24a_v21c_lme_seed5_0p6b_frozen_local      v24a_seed5_final final"
    "output/chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200      v25a_seed7_best best"
    "output/chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200      v25a_seed7_final final"
    "output/chain_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200 v27b_no_probe_seed3_best best"
    "output/chain_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200 v27b_no_probe_seed3_final final"
    "output/chain_v27b_v24a_no_probe_seed4_0p6b_frozen_gh200 v27b_no_probe_seed4_best best"
    "output/chain_v27b_v24a_no_probe_seed4_0p6b_frozen_gh200 v27b_no_probe_seed4_final final"
    "output/chain_v28a_no_probe_seed1_1p7b_frozen_gh200      v28a_no_probe_seed1_best best"
    "output/chain_v28a_no_probe_seed1_1p7b_frozen_gh200      v28a_no_probe_seed1_final final"
    "output/chain_v28b_no_probe_seed2_1p7b_frozen_gh200      v28b_no_probe_seed2_best best"
    "output/chain_v28b_no_probe_seed2_1p7b_frozen_gh200      v28b_no_probe_seed2_final final"
)

while true; do
    for entry in "${RUNS[@]}"; do
        set -- $entry
        outdir="$1"; tag="$2"; sub="$3"
        ckpt="$outdir/$sub"
        out_json="$OUT_DIR/${tag}_lme_val_evpos.json"
        if [ -d "$ckpt" ] && [ ! -f "$out_json" ]; then
            run_eval "$ckpt" "$tag"
        fi
    done

    all_done=true
    for entry in "${RUNS[@]}"; do
        set -- $entry
        tag="$2"
        out_json="$OUT_DIR/${tag}_lme_val_evpos.json"
        if [ ! -f "$out_json" ]; then
            all_done=false
        fi
    done
    if $all_done; then
        echo "[watcher] $(date +%H:%M:%S) all evals complete; exiting."
        break
    fi
    sleep 120
done
