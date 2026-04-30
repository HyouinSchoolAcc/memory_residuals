#!/usr/bin/env bash
# Cell B prime: cell B with the two suspect regularisers OFF.
#
# Why this run exists:
#   The original cell B (chain_v5_softembed_msc) diverged at step ~1200
#   under the canonical recipe (mem_drop=0.10, ctx_drop=0.30, neg_chain
#   ramp 0.05->0.5 over 1000 steps). Trajectory:
#     step 200..600  Δ_sh-m near zero, ce_mem stable ~3.10
#     step 800..1000 Δ_sh-m goes negative (memory actively unhelpful)
#     step 1200..1400 ce_mem runs from 3.10 to 3.62 (catastrophic),
#                     ce_nomem stays ~3.14 (fine without memory).
#   So memory is poisoning the prediction; nomem-mode is healthy.
#   Killed at step 1460. Final state preserved at:
#     logs/chain_v5_softembed_msc_KILLED_step1460.log
#     output/chain_v5_softembed_msc/{best,step-500,step-1000}
#
# Hypothesis being tested:
#   The dropout stack (especially context_dropout=0.30) drives early-
#   stage posterior collapse on the memory channel. Once the model
#   learns to ignore memory, the contrastive ramp peaking at step 1000
#   pushes mem != shuffle in pathological directions, satisfying the
#   margin loss by making BOTH bad. carry_state then propagates the
#   broken M_c across minibatches.
#
#   v3 chain_v3_softparity_full reached Δ_sh-m=+0.0177 at step 1000
#   on the same architecture with mem_drop=0, ctx_drop=0, no
#   neg_chain. So removing those two regularisers should at minimum
#   restore the v3-soft trajectory shape (allowing for the harder MSC
#   eval distribution depressing the absolute number).
#
# Single-knob diff vs cell B:
#   --memory_dropout 0.10 -> 0.0
#   --context_dropout 0.30 -> 0.0
#   --neg_chain_weight 0.5 -> 0.0  (drops contrastive entirely)
#   --neg_chain_warmup_steps and --neg_chain_initial_weight removed
#                                 (no ramp without the loss)
#   --run_name + --out_dir         (new identity)
#
# Everything else (window_k=3, carry_state, lr=2e-4, lr_backbone=2e-5,
# burn_in_max=8, mask_padding_loss, save_best=composite, hidden args)
# is kept identical so the contrast is clean.
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
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --carry_state \
    --neg_chain_weight 0.0 \
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
    --run_name chain_v5_softembed_msc_noreg \
    --out_dir output/chain_v5_softembed_msc_noreg
