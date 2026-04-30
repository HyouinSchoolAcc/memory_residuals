#!/usr/bin/env bash
# v8b MIXED SIMPLE_GATE + readout RMSNorm
#
# The headline candidate cell. Combines:
#   - Architecture fix (RMSNorm on MemoryReadout output, see v8a header
#     for the diagnostic rationale).
#   - Routing mode that actually recruits memory (simple_gate; the only
#     mode in the v3-v7 lineage where gate / alpha_mem moved off zero
#     under the chain trainer).
#   - Mixed-bias curriculum (--curriculum_evidence_bias 0.5
#     --window_k 8): half the windows are P0 evidence+callback
#     (compressed credit-assignment path between fact and recall),
#     half are full contiguous window_k=8 chains. The mix is the v8
#     option (1) from
#     experiments/exp2_long_horizon_recipe/runs.md: train on both
#     regimes simultaneously so the readout learns to handle both
#     M_c distributions, instead of overfitting to one and collapsing
#     on the other (the v7 P0 train/eval mismatch trap).
#   - Phase-aligned save_best (the v8 default).
#   - Same gated writer / hidden_14 extract / no burn-in / no carry
#     state as v7 P0 cells, so the only knobs that differ from v7
#     SIMPLE_GATE are: RMSNorm, mixed_bias=0.5, window_k=8.
#
# Window structure when curriculum_evidence_bias=0.5 fires (50% of
# steps for chains with cb_pos >= 7):
#   [evidence_session, intermediate_1, ..., intermediate_6, callback]
# i.e. 1 fact + 6 distractors + 1 recall. The writer must defend the
# evidence-bearing M_c slot through 6 intervening judge updates
# before the callback supervision pulls on it. This is the
# architectural challenge the paper is ultimately trying to solve --
# v7 P0 (window_k=2) was a bootstrap; v8b mixed at window_k=8 is the
# real test.
#
# Other 50% (and all chains with cb_pos < 7): contiguous window_k=8
# uniform sampling. No curriculum, no callback alignment, just the
# raw long-context training distribution. The model sees the full
# eval-shaped M_c regime regularly and the readout learns to handle
# it.
#
# Decision triggers (sharp):
#   step 500: pa_cb_dsh > +0.005 (curriculum callback discrimination
#             survives the harder distractor depth)
#   step 1000: pa_cb_dsh > +0.020 AND legacy delta_shuffle_minus_mem
#              > 0 (i.e. the harder eval ALSO starts moving)
#   step 2000: legacy delta_shuffle_minus_mem > +0.010
#              -> ship as v9 baseline.
#
# Memory budget: window_k=8 with simple_gate (no depth router on the
# forward path) and bs=2 grad_accum=4 fits in ~50-55 GiB on H100 NVL
# with gradient checkpointing.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --window_k 8 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 100 \
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 10.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.5 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 48 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v8b_mixed_simplegate_rmsnorm \
    --out_dir output/chain_v8b_mixed_simplegate_rmsnorm \
    2>&1 | tee logs/chain_v8b_mixed_simplegate_rmsnorm.log
