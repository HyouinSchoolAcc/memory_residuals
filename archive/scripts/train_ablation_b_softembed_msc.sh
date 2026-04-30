#!/usr/bin/env bash
# Local ablation B (cell B of 2x2): MemRes-Embed + MSC, SOFT parity init.
#
# Purpose: companion to the GH200 v5 headline (cell A). Holds the corpus
# fixed at PG-19+TV+MSC and varies extract source: embed (this) vs
# hidden_14 (cell A). With both at soft +-4 init, the column contrast
# A vs B isolates the contribution of contextualised extraction.
#
# All knobs except --memres_extract_source match scripts/train_headline_softinit.sh
# so the contrast is clean.
#
# Replaces scripts/train_ablation_b_embed_msc.sh which used the broken
# hard +-32 init (see scripts/train_headline_softinit.sh header for the
# bf16-saturation diagnosis).
#
# Designed for one H100 NVL (GPU index passed via CUDA_VISIBLE_DEVICES).
set -eu
cd "${REPO:-/home/exx/Desktop/fine-tune/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_mem_bias_init -4 \
    --router_recent_bias_init 4 \
    --memres_extract_source embed \
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
    --run_name chain_v5_softembed_msc \
    --out_dir output/chain_v5_softembed_msc
