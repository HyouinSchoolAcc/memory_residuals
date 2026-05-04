#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/exx/Desktop/fine-tune/memory_residuals

CORPUS=paper_artifacts/chains/synthd4v2_persona_callback_val_s512.pt
N=32
STEPS=50
LR=1e-2
OUT=results/ttt_mc_v17pre

for ckpt_name in chain_v14k_d4v2_norm_no_warmup_local chain_v15a_d4v2_norm_replicate_local; do
  CKPT=output/${ckpt_name}/best
  for init in writer iid; do
    echo "=========================================="
    echo "RUN: $ckpt_name init=$init"
    echo "=========================================="
    python tools/eval_ttt_mc.py \
      --ckpt "$CKPT" \
      --eval_corpus "$CORPUS" \
      --n_chains "$N" \
      --ttt_steps "$STEPS" \
      --ttt_lr "$LR" \
      --init_mode "$init" \
      --seed 0 \
      --out "${OUT}/${ckpt_name}_${init}.json"
  done
done
echo "GPU0 sweep DONE"
