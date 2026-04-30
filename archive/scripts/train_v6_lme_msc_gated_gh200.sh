#!/usr/bin/env bash
# v6 GH200 variant: gated memory + LME callback supervision + MSC mix.
#
# Differs from local v6 GATED (LME-only) on two axes that GH200's larger
# memory budget makes affordable:
#   - --train_chains  v6_lme_msc_train_s512.pt  (450 LME + 5928 MSC = 6378 chains)
#                     Source weights upweight LME (4.0) over MSC (1.0) so the
#                     callback-supervised chains are visited proportionally
#                     (MSC contributes ~half the sampling but provides
#                     conversational distribution diversity, no callback loss).
#   - --window_k 12   (vs local 8) — exploits GH200 96GB to reach deeper
#                     into the callback session's prior context. A window
#                     of 12 covers the callback session + 11 prior sessions,
#                     so the M_c being read by the callback is a state
#                     compressed from ~6k tokens of dialogue.
# Everything else matches local v6 GATED so the cells are comparable.
#
# Why same architecture (attention_parity at -4/+4):
#   ReZero-style routing was empirically WORSE than attention_parity at
#   -4/+4 in our earlier ablations (v3 trajectory). Hard ±32 parity
#   saturated bf16 in v4 (gate_max stuck at 0). So -4/+4 with the parity
#   topology is the only routing config we have empirical confidence in.
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
    --train_chains paper_artifacts/chains/v6_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --source_weights '{"longmemeval":4.0,"msc":1.0}' \
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
    --eval_window 8 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v6_lme_msc_gated_callback \
    --out_dir output/chain_v6_lme_msc_gated_callback
