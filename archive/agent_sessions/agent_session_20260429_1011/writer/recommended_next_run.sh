#!/usr/bin/env bash
# DO NOT RUN WITHOUT HUMAN APPROVAL.
# Candidate 1 — soft parity init recipe rerun (chain_v5_softhidden14_msc).
# See logs/agent_session_20260429_1011/writer/recommended_next_run.md
# for full justification, predictions, and decision triggers.
#
# Differences from scripts/train_headline.sh:
#   --router_mem_bias_init     -32 -> -4
#   --router_recent_bias_init  +32 -> +4
#   --run_name                 chain_v4_hidden14_msc -> chain_v5_softhidden14_msc
#   --out_dir                  output/chain_v4... -> output/chain_v5_softhidden14_msc
# Everything else is identical to the running headline.

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
