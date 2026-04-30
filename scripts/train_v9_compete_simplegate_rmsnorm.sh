#!/usr/bin/env bash
# v9 PURE-COMPETITION SIMPLE_GATE + readout RMSNorm
#
# User directive 2026-04-29 ~23:00 UTC: "we separately learn the
# summarizer and the competition; the individual problems then can
# generalize." This cell implements the COMPETITION half of that
# decomposition.
#
# What's new vs v8b / v8c (which used --curriculum_evidence_bias):
#
# The v8 mixed-bias curriculum [evidence, ...intermediates, callback]
# trains the writer + readout but NEVER trains the judge layer's
# competitive function in isolation:
#   - In the P0 sub-window (window_k=2), the judge step has
#     M_c_prev = 0 (fresh), so judge degenerates to a no-competition
#     aggregation of C_t into K slots. The keep-vs-write decision
#     is never explicitly tested.
#   - In the contiguous sub-window, the judge fires multiple times
#     in sequence with diffuse credit assignment.
#
# v9 fixes this by sampling JUDGE-COMPETITION pairs:
#
#   Sample A (KEEP-PREV, ~50%): [evidence, distractor, callback]
#     -- judge step at distractor MUST keep evidence in M_c
#     -- gradient: callback CE rewards small write_gate at this step
#
#   Sample B (WRITE-NEW, ~50%): [noise, evidence, callback]
#     -- judge step at evidence MUST overwrite noise in M_c
#     -- gradient: callback CE rewards large write_gate at this step
#
# Both samples score the same callback session, so the gradient
# directly trains the write_gate / judge weights to be content-aware.
# The signal is the SAME callback CE used in v8a/b/c, but the
# pairing structure forces the judge to make a content-discriminative
# keep-or-write decision rather than learning a degenerate solution.
#
# Why pure (not mixed) competition: hypothesis that the judge subproblem
# can be trained cleanly in isolation; if so, the resulting checkpoint
# becomes a warm-start for v10 (mixed competition + compacting +
# contiguous). If the judge can NOT be trained cleanly even in
# isolation, that's a sharp diagnostic of architectural insufficiency.
#
# Knobs vs v8c:
#   - window_k          : 3 (was 8)         curriculum is a 3-session pair
#   - curriculum_competition_bias : 1.0 (NEW)  pure competition
#   - curriculum_evidence_bias    : 0.0       no overlap with v8 curriculum
#   - lr (memres)       : 5e-5 (was 1e-4)   even slower; the judge has 4 small parameters per layer (write_gate + judging W_Q/K/V/O), should be trained gently
#   - callback_loss_weight : 3.0 (same as v8c)  preserves selective gradient
#   - corpus            : v6 mixed (same as v8c)  data diversity helps
#   - source_weights    : same as v8c
#   - dropouts          : same as v8c (memory_dropout=0.10, context_dropout=0.05)
#
# Decision triggers (sharp):
#   step 200 : pa_cb_dsh > 0 AND ||m^t||/||embed|| stable (not exploding)
#   step 500 : pa_cb_dsh > +0.005
#   step 1000: pa_cb_dsh > +0.020 -- judge is content-discriminative
#   step 2000: standard delta_shuffle_minus_mem > +0.005 -- generalizes
#              -> ship checkpoint as warm-start for v10 (composed curriculum)
#   anywhere : train loss < 0.6 -> kill (overfitting detected)
#
# Memory budget: window_k=3, bs=4 grad_accum=2, fits in ~20 GiB on H100.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/v6_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 200 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.0 \
    --curriculum_competition_bias 1.0 \
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
    --run_name chain_v9_compete_simplegate_rmsnorm \
    --out_dir output/chain_v9_compete_simplegate_rmsnorm \
    2>&1 | tee logs/chain_v9_compete_simplegate_rmsnorm.log
