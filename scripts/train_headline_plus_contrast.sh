#!/usr/bin/env bash
# Headline run A+: same as A (chain_v4_hidden14_msc) plus the
# intra-chain perturbation contrastive loss.
#
# A vs A+ is a clean A/B test of "does the contrast loss further
# sharpen recall on top of the contextualised extract + MSC corpus?"
#
# Designed to run AFTER A on GH200 (single GPU; cannot share with A).
# Same compute footprint as A plus ~30%% per-step overhead for the
# extra TBPTT chain build with grad.
set -eu
cd "${REPO:-/home/ubuntu/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_mem_bias_init -32 \
    --router_recent_bias_init 32 \
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
    --in_chain_contrast_weight 0.5 \
    --in_chain_contrast_warmup_steps 1000 \
    --in_chain_contrast_initial_weight 0.05 \
    --in_chain_contrast_margin 0.05 \
    --in_chain_perturb_strategy random_earlier \
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
    --run_name chain_v4_hidden14_msc_contrast \
    --out_dir output/chain_v4_hidden14_msc_contrast
