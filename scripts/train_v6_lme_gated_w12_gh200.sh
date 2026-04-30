#!/usr/bin/env bash
# v6 GH200 variant: gated memory + LME callback supervision + DEEPER window.
#
# Replaces the LME+MSC mix variant which was killed because at window_k=12,
# MSC chains (only 5 sessions) are ineligible for sampling, so the mix
# silently degraded to LME + PG-19 (books, wrong domain) + TV. With LME-
# only and window_k=12 we get a clean architectural axis vs the local
# v6 cells:
#   local v6 GATED: LME, window_k=8, gated
#   local v6 COMPETITIVE: LME, window_k=8, competitive  (architecture A/B)
#   GH200 v6 GATED-DEEP: LME, window_k=12, gated  (window-depth ablation)
#
# At window_k=12, each TBPTT window covers 12 sessions (~6k tokens of
# dialogue), so the M_c being read by the callback session was compressed
# from 11 prior sessions of TBPTT-tracked context. With LME chains
# averaging 50 sessions, the callback bias still works (it pins window
# end to the callback session).
#
# Routing is attention_parity at -4/+4 (the only routing config we have
# empirical confidence in: ReZero variants are strictly worse, hard ±32
# saturates bf16).
set -eu
cd "${REPO:-/home/ubuntu/memory_residuals}"
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
    --window_k 12 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --steps 8000 \
    --warmup 100 \
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --carry_state \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 10.0 \
    --callback_window_bias 0.7 \
    --burn_in_max 32 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 12 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v6_lme_gated_callback_w12 \
    --out_dir output/chain_v6_lme_gated_callback_w12
