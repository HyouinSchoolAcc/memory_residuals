#!/usr/bin/env bash
# v7 P0 V3BIAS — compression curriculum with v3's original router bias.
#
# A/B partner of train_v7_p0_softerbias.sh. Single-knob diff:
#
#   --router_mem_bias_init   -2 (softerbias)  -> -4 (this; v3 default)
#
# Everything else identical. Together the two cells answer:
#   Q1: Does the compression curriculum (P0) alone open the channel,
#       independent of bias relaxation?  -> if THIS cell shows
#       gate_max > 0 / Δ_sh-m > 0 within 1k steps with v3's -4 bias,
#       the bias was never the binding constraint and the bottleneck
#       was always credit assignment.
#   Q2: Does bias relaxation alone help?  -> if SOFTERBIAS cell strictly
#       beats this one on gate_max, the bias matters too and we should
#       test mem=0 in v7 cell 3.
#   Q3: If both cells fail to open the channel, the next move is NOT
#       another bias/curriculum tweak -- it's an architectural ablation
#       (simple_gate baseline on LME) because attention_parity routing
#       is the structural problem.
#
# See train_v7_p0_softerbias.sh header for the full failure analysis
# and the design of P0.
set -eu
cd "${REPO:-/home/exx/Desktop/fine-tune/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --memres_update_mode gated \
    --router_mem_bias_init -4 \
    --router_recent_bias_init 4 \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --window_k 2 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 100 \
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 10.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v7_p0_v3bias \
    --out_dir output/chain_v7_p0_v3bias
