#!/usr/bin/env bash
# Polls every 60s for ablation training runs to finish (i.e. emit a
# `final/` directory), then runs eval_callback.py against the patched
# lme_val_s512_evpos.pt and emits results JSON. Sleeps when nothing new.
#
# Watches:
#   chain_v27a_v24a_no_depth_seed1_0p6b_frozen_local
#   chain_v27b_v24a_no_probe_seed1_0p6b_frozen_local
#   chain_v27c_v24a_no_floor_seed1_0p6b_frozen_local
#   chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200          (rsync-pulled from GH200)
#
# Outputs: results/eval_v25_seed_pack_evpos/<tag>_lme_val_evpos.json
#
# Usage:
#   nohup bash scripts/watcher_eval_ablations.sh > logs/watcher_eval.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR=results/eval_v25_seed_pack_evpos
mkdir -p "$OUT_DIR" logs

GPU="${WATCHER_GPU:-1}"

run_eval() {
    local ckpt=$1 tag=$2
    local out="$OUT_DIR/${tag}_lme_val_evpos.json"
    if [ -f "$out" ]; then
        echo "[watcher] $tag already evaluated -> skip"
        return
    fi
    if [ ! -d "$ckpt" ]; then
        echo "[watcher] $tag ckpt $ckpt missing -> skip"
        return
    fi
    echo "[watcher] $(date +%H:%M:%S)  evaluating $tag ($ckpt)"
    CUDA_VISIBLE_DEVICES="$GPU" python tools/eval_callback.py \
        --model_path "$ckpt" \
        --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \
        --names lme_val \
        --output "$out" 2>&1 | tail -3
}

# Poll loop.
declare -a RUNS=(
    "output/chain_v27a_v24a_no_depth_seed1_0p6b_frozen_local v27a_no_depth_best best"
    "output/chain_v27a_v24a_no_depth_seed1_0p6b_frozen_local v27a_no_depth_final final"
    "output/chain_v27b_v24a_no_probe_seed1_0p6b_frozen_local v27b_no_probe_best best"
    "output/chain_v27b_v24a_no_probe_seed1_0p6b_frozen_local v27b_no_probe_final final"
    "output/chain_v27c_v24a_no_floor_seed1_0p6b_frozen_local v27c_no_floor_best best"
    "output/chain_v27c_v24a_no_floor_seed1_0p6b_frozen_local v27c_no_floor_final final"
    "output/chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200      v25a_seed7_best     best"
    "output/chain_v25a_v21c_lme_seed7_1p7b_frozen_gh200      v25a_seed7_final    final"
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

    # If everything is evaluated, exit.
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
        echo "[watcher] all evals complete; exiting."
        break
    fi
    sleep 60
done
