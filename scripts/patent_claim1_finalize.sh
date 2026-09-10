#!/usr/bin/env bash
# Wait for the claim-1 evidence queue, add the accuracy re-eval of the
# headline checkpoint, then rebuild figures + the patent document.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[wait] claim1 queue"
while ! grep -q CLAIM1_QUEUE_DONE logs/patent_claim1_queue.log 2>/dev/null; do
  sleep 120
done
echo "[ready] claim1 queue"

echo "=== headline re-eval with accuracy metric ==="
CUDA_VISIBLE_DEVICES=0 python tools/eval_callback.py \
  --model_path runs/chain_v27b_v24a_no_probe_seed3_0p6b_frozen_gh200/final \
  --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \
  --names lme_val \
  --output results/patent_experiments/eval_trained_k128_seed3_acc.json

echo "=== rebuild figures + patent doc ==="
python tools/patent_make_figures.py
python tools/patent_write_docx.py \
  --src paper_artifacts/patent_draft_src.docx \
  --out "一种基于注意力读出与竞争式更新的语言模型外置记忆方法（发明，含实验验证）.docx"

echo CLAIM1_FINALIZE_DONE
