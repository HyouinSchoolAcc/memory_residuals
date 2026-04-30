#!/usr/bin/env bash
# Headline run v5: MemRes-Hidden + MSC, SOFT parity init.
#
# Replaces scripts/train_headline.sh after the v4 hard-init run
# (chain_v4_hidden14_msc, killed at ~step 5500/6000 on 2026-04-29
# 15:30 UTC) was diagnosed as bf16-saturating on the depth router.
#
# Why soft init: at hard mem_bias=-32, recent_bias=+32 the depth
# router softmax mass on memory is exp(-64) ~ 1.6e-28, ~20 orders of
# magnitude below bf16 representability (~6e-8). Backward gradient
# through that softmax onto mem_bias is bf16-zero -- the contrastive
# ramp cannot move it. EVAL across all 27 in-trainer evals in v4
# (step 200 -> 5400) showed mem == nomem == shuffle to 4 decimals,
# Delta_sh-m = +0.0000 throughout. See:
#   logs/agent_session_20260429_1011/writer/findings_alpha_mem.md
#   logs/agent_session_20260429_1011/monitor/concern_memory_channel_closed.md
#   logs/chain_v4_hidden14_msc_final.log
#
# At soft +-4 the init mass on memory is ~ exp(-8)/N ~ 6.2e-5 (or
# ~1.7e-4 with the recent_bias correctly in the denominator). Well
# above bf16 representability. v3 soft-parity reached alpha_mem
# ~ 4.7e-4 in 4400 steps, so the gradient is non-vanishing and
# memory routing is recruitable.
#
# This script is a clean A/B against the running v4 config: only the
# two router init flags differ. window_k=3 + dropouts + carry_state
# are kept as-is (they match the running headline). A separate run
# at window_k=8 + no dropouts (the README recipe) is a follow-on
# ablation.
set -eu
cd "${REPO:-/home/ubuntu/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_mem_bias_init -4 \
    --router_recent_bias_init 4 \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/stage1_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/stage1_msc_val_s512.pt \
    --source_weights '{"pg19":1.0,"tv":4.0,"msc":8.0}' \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --steps 6000 \
    --warmup 100 \
    --memory_dropout 0.10 \
    --context_dropout 0.30 \
    --carry_state \
    --neg_chain_weight 0.5 \
    --neg_chain_warmup_steps 1000 \
    --neg_chain_initial_weight 0.05 \
    --burn_in_max 8 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 4 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v5_softhidden14_msc \
    --out_dir output/chain_v5_softhidden14_msc
