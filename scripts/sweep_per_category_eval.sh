#!/usr/bin/env bash
# Tier-1 (3) within-category Δ_sh + Tier-4 (10) per-category Δ_sh
# breakdown sweep over the existing v24a (with F3) and v27b/v28 (no F3)
# checkpoints. Outputs per-checkpoint JSONs to results/eval_per_category/
# and a summary aggregate at the end.
#
# Wallclock: ~75 s per 0.6B ckpt + ~120 s per 1.7B ckpt × ~11 ckpts = ~20 min.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
OUT_DIR="results/eval_per_category"
mkdir -p "$OUT_DIR" logs

CORPUS="paper_artifacts/chains/lme_val_s512_evpos.pt"

run_eval() {
    local label="$1"
    local ckpt_dir="$2"
    local out_json="${OUT_DIR}/${label}.json"
    if [ ! -d "$ckpt_dir" ]; then
        echo "[skip] $label : $ckpt_dir does not exist"
        return 0
    fi
    if [ -f "$out_json" ]; then
        echo "[skip] $label : $out_json already exists"
        return 0
    fi
    echo "[run]  $label : $ckpt_dir -> $out_json"
    python tools/eval_callback_categories.py \
        --model_path "$ckpt_dir" \
        --corpora "$CORPUS" \
        --names lme_val \
        --output "$out_json" 2>&1 | tail -12
}

# v27b (no F3 — the canonical recipe): all 4 seeds.
run_eval "v27b_seed1_0p6b_lme_val" "runs/chain_v27b_v24a_no_probe_seed1_0p6b_frozen_local/final"
run_eval "v27b_seed2_0p6b_lme_val" "runs/chain_v27b_v24a_no_probe_seed2_0p6b_frozen_local/final"
run_eval "v27b_seed3_0p6b_lme_val" "runs/chain_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200/final"
run_eval "v27b_seed4_0p6b_lme_val" "runs/chain_v27b_v24a_no_probe_seed4_0p6b_frozen_gh200/final"

# v28 (1.7B, no F3): all available seeds.
run_eval "v28a_seed1_1p7b_lme_val" "runs/chain_v28a_no_probe_seed1_1p7b_frozen_gh200/final"
run_eval "v28b_seed2_1p7b_lme_val" "runs/chain_v28b_no_probe_seed2_1p7b_frozen_gh200/final"
run_eval "v28c_seed3_1p7b_lme_val" "runs/chain_v28c_no_probe_seed3_1p7b_frozen_gh200/final"
run_eval "v28d_seed4_1p7b_lme_val" "runs/chain_v28d_v25a_no_probe_seed4_1p7b_frozen_gh200/final"

# v24a (with F3 — reference for ablation): all available seeds.
run_eval "v24a_seed1_0p6b_lme_val" "runs/chain_v24a_v21c_lme_seed1_0p6b_frozen_local/final"
run_eval "v24a_seed2_0p6b_lme_val" "runs/chain_v24a_v21c_lme_seed2_0p6b_frozen_local/final"
run_eval "v24a_seed3_0p6b_lme_val" "runs/chain_v24a_v21c_lme_seed3_0p6b_frozen_local/final"
run_eval "v24a_seed5_0p6b_lme_val" "runs/chain_v24a_v21c_lme_seed5_0p6b_frozen_local/final"

echo
echo "=== sweep done ==="
ls -1 "$OUT_DIR"
