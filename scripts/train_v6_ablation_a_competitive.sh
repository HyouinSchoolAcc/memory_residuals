#!/usr/bin/env bash
# v6 ablation A: same recipe as train_v6_lme_gated_callback.sh but with
# --memres_update_mode competitive instead of gated. This isolates
# whether the gated update mechanism is doing real work, or whether the
# corpus + callback loss alone is sufficient.
#
# Expected outcome at step 1000:
#   - If competitive matches gated on Δ_sh-m: gated is unnecessary
#     overhead and we drop it for v7.
#   - If competitive lags by > 1 stderr: gated is doing real work on
#     long horizons (50-session chains can clobber a session-1 fact
#     under competitive's zero-sum overwrite).
#   - If both fail: corpus / callback loss isn't enough and we need
#     more architectural surgery (e.g. an external persistent buffer).
set -eu
cd "${REPO:-/home/exx/Desktop/fine-tune/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --memres_update_mode competitive \
    --router_mem_bias_init -4 \
    --router_recent_bias_init 4 \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --window_k 8 \
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
    --burn_in_max 24 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v6_lme_competitive_callback \
    --out_dir output/chain_v6_lme_competitive_callback
