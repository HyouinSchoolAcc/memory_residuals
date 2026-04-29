#!/usr/bin/env bash
# Ablation C: MemRes-Hidden + PG-19/TV (no MSC).
# Same as headline A except --train_chains points at the legacy PG-19+TV
# corpus.  Holds extract source fixed at hidden_14; isolates the
# contribution of conversational training data.
#
# Designed for one H100 NVL.  Same hyper-params as A so the contrast is
# clean.  window_k=8 here because PG-19 chains are long and we want to
# train at the same TBPTT depth as the v3 baselines.
set -eu
cd "${REPO:-/home/exx/Desktop/fine-tune/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_mem_bias_init -32 \
    --router_recent_bias_init 32 \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/stage1_train_s512.pt \
    --eval_chains  paper_artifacts/chains/stage1_validation_s512.pt \
    --source_weights '{"pg19":1.0,"tv":4.0}' \
    --window_k 8 \
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
    --run_name chain_v4_hidden14_pgtv \
    --out_dir output/chain_v4_hidden14_pgtv
