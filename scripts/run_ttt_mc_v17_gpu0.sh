#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /home/exx/Desktop/fine-tune/memory_residuals

N=32
STEPS=50
LR=1e-2
OUT=results/ttt_mc_v17pre

for ckpt_name in chain_v17a_f2_codes_0p6b_frozen_local chain_v17b_f2_codes_0p6b_joint_local; do
  CKPT=output/${ckpt_name}/best
  for corpus_tag in synthd5_random_codes_val_s512 synthd4v2_persona_callback_val_s512; do
    CORPUS=paper_artifacts/chains/${corpus_tag}.pt
    for init in writer iid; do
      tag=${ckpt_name}__${corpus_tag}__${init}
      echo "=========================================="
      echo "RUN: $tag"
      echo "=========================================="
      python tools/eval_ttt_mc.py \
        --ckpt "$CKPT" \
        --eval_corpus "$CORPUS" \
        --n_chains "$N" \
        --ttt_steps "$STEPS" \
        --ttt_lr "$LR" \
        --init_mode "$init" \
        --seed 0 \
        --out "${OUT}/${tag}.json"
    done
  done
done
echo "GPU0 v17 sweep DONE"
