#!/usr/bin/env bash
# Cross-corpus eval for v27b/v28 headline checkpoints.
# Runs callback-aware eval on the three callback-annotated OOD validation
# corpora (synthd4, synthd4v2, synthd5) and full-CE eval on MSC + LoCoMo.
#
# Usage:
#   bash scripts/eval_cross_corpus.sh <ckpt_dir> <tag> [<gpu_id>]
#
# Outputs to results/eval_v27_v28_cross_corpus/<tag>_{cb,fullce}.json
set -euo pipefail
ckpt="${1:?need ckpt}"
tag="${2:?need tag}"
gpu="${3:-0}"
cd "$(dirname "$0")/.."

if [ ! -d "$ckpt" ]; then
    echo "ckpt $ckpt does not exist" >&2; exit 2
fi

OUT_DIR="results/eval_v27_v28_cross_corpus"
mkdir -p "$OUT_DIR"
out_cb="$OUT_DIR/${tag}_cb.json"
out_fc="$OUT_DIR/${tag}_fullce.json"

export CUDA_VISIBLE_DEVICES="$gpu"

echo "[eval $tag GPU $gpu] callback-aware on synthd4, synthd4v2, synthd5..."
python tools/eval_callback.py \
    --model_path "$ckpt" \
    --corpora \
        paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
        paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt \
        paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --names synthd4_val synthd4v2_val synthd5_val \
    --output "$out_cb"

echo "[eval $tag GPU $gpu] full-CE on locomo, msc_test..."
python tools/eval_chain.py \
    --model_path "$ckpt" \
    --corpora \
        paper_artifacts/chains/locomo_s512.pt \
        paper_artifacts/chains/msc_test_s512.pt \
    --names locomo msc_test \
    --score_window 4 \
    --oracle_window 4 \
    --output "$out_fc"

echo "[eval $tag] done -> $out_cb $out_fc"
