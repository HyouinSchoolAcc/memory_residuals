#!/usr/bin/env bash
# v6 PURIST — strips the last v5 recipe inheritances on top of v6 GATED.
#
# Function: clean attribution of which v5-era knobs were genuinely
# necessary vs which were dead weight. v3 (chain_v3_softparity_full)
# climbed Δ_sh-m monotonically to +0.0379 by step 4400 with a much
# leaner recipe than v5 cell C, on the SAME corpus. v6 already
# dropped three v5 pieces (memory_dropout, context_dropout,
# neg_chain_weight) and went back to v3's window_k=8. This run drops
# the remaining three:
#
#   knob              v3   v5 cell C   v6 GATED   v6 PURIST (this)
#   ---------------------------------------------------------------
#   carry_state       F     T          T          F   <-- new
#   burn_in_max       0     8          24         0   <-- new
#   burn_in_resample  F     T          T          F   <-- new
#   lr (memres)       3e-4  2e-4       2e-4       3e-4 <-- new
#   lr (backbone)     3e-5  2e-5       2e-5       3e-5 <-- new
#
# Everything else matches v6 GATED:
#   - LME corpus
#   - --memres_update_mode gated
#   - attention_parity at -4/+4
#   - --callback_loss_weight 10.0
#   - --callback_window_bias 0.7
#   - --window_k 8
#
# Outcome interpretation:
#   - PURIST > GATED on Δ_sh-m at step 1000 → carry_state/burn_in/lr
#     are dead weight; recipe paper's prescriptions simplify.
#   - PURIST ≈ GATED → those knobs are neutral; document but don't
#     emphasise.
#   - PURIST < GATED → at least one of the three was load-bearing;
#     run a finer ablation to find which.
#
# This is a fire-and-forget cell: queued on GH200 watchdog, runs
# automatically after chain_v6_lme_gated_callback_w12 (GATED-DEEP)
# completes.
set -eu
cd "${REPO:-/home/ubuntu/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --memres_update_mode gated \
    --router_mem_bias_init -4 \
    --router_recent_bias_init 4 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --window_k 8 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 3e-4 \
    --lr_backbone 3e-5 \
    --steps 8000 \
    --warmup 100 \
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 10.0 \
    --callback_window_bias 0.7 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v6_lme_gated_purist \
    --out_dir output/chain_v6_lme_gated_purist
