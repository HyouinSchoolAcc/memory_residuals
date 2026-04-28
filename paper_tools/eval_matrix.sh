#!/usr/bin/env bash
# Standalone eval matrix runner for Memory Residuals.
#
# Runs eval_chain.py (with RAG) and callback_probe.py for one checkpoint
# across the standard eval suite (PG-19 validation, PG-19 test, LoCoMo),
# then collects horizon-bucketed analyses.
#
# Usage:
#   bash paper_tools/eval_matrix.sh <ckpt_dir> <tag>
# e.g.:
#   bash paper_tools/eval_matrix.sh output/chain_v2_phaseB_residual/best phaseB_residual
set -euo pipefail

CKPT="${1:-}"
TAG="${2:-}"
if [ -z "$CKPT" ] || [ -z "$TAG" ]; then
  echo "Usage: $0 <ckpt_dir> <tag>" >&2
  exit 2
fi

DEVICE="${DEVICE:-cuda:0}"
EVAL_DIR=paper_artifacts/eval

mkdir -p "$EVAL_DIR" logs

# Pair-format data for the callback probe: prepared once from the
# validation+test PG-19 chains via prepare_pairs.py. Cached at
# paper_artifacts/pairs/stage1_validation.jsonl.
PAIR_DATA="paper_artifacts/pairs/stage1_validation.jsonl"
if [ ! -f "$PAIR_DATA" ]; then
  echo "  preparing pair data for callback probe..."
  python paper_tools/prepare_pairs.py \
    --data-root ../memory_residuals_data \
    --out-dir paper_artifacts/pairs \
    --max-stage1-train 0 --max-stage1-eval 256 --max-stage2-train 0 --max-stage2-eval 0
fi

echo "==[ chain eval (PG-19 val + PG-19 test + LoCoMo) ]=="
python paper_tools/eval_chain.py \
  --model_path "$CKPT" \
  --corpora paper_artifacts/chains/stage1_validation_s512.pt \
            paper_artifacts/chains/stage1_test_s512.pt \
            paper_artifacts/chains/locomo_s512.pt \
  --names pg19_validation pg19_test locomo \
  --score_window 4 --oracle_window 4 \
  --do_rag --rag_top_k 3 --rag_prefix_len 1024 \
  --device "$DEVICE" \
  --output "$EVAL_DIR/${TAG}_chain_eval.json"

echo "==[ horizon-bucketed analysis ]=="
python paper_tools/horizon_analysis.py \
  --inputs "$EVAL_DIR/${TAG}_chain_eval.json"

echo "==[ callback probe ]=="
python paper_tools/callback_probe.py \
  --model_path "$CKPT" \
  --data_path "$PAIR_DATA" \
  --num_samples 256 --history_len 1024 --current_len 512 \
  --device "$DEVICE" \
  --output "$EVAL_DIR/${TAG}_callback_probe.json"

echo ""
echo "==[ DONE: $TAG ]=="
echo "  chain eval     : $EVAL_DIR/${TAG}_chain_eval.json"
echo "  horizon table  : $EVAL_DIR/${TAG}_chain_eval_horizon.md"
echo "  callback probe : $EVAL_DIR/${TAG}_callback_probe.json"
