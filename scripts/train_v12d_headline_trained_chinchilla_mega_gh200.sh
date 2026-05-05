#!/usr/bin/env bash
# v12d-headline-TRAINED (GH200, 0.6B) -- slot_attention writer
#                                      * TRAINED BACKBONE
#                                      * CHINCHILLA-BUDGETED
#                                      * MEGA CORPUS
# Trained-backbone arm of the v12d frozen-vs-trained comparison study.
# Pairs with train_v12d_headline_frozen_chinchilla_mega_gh200.sh.
# Single-axis diff:
#   frozen arm:  --freeze_backbone     --lr_backbone 0
#   trained arm: (no freeze flag)      --lr_backbone 2e-5
# Everything else (memres lr 5e-5, 25k steps, mega corpus, k=4,
# carry_state, burn_in_max=12, burn_in_resample, attention_parity,
# +4/0 router bias, slot_attention writer with 3 iters) is identical.
#
# Why this is necessary even though v11l-fix already showed frozen
# breaks the original writer
# --------------------------------------------------------------------
# v11l-fix (still running) demonstrated that with the *original*
# writer, freezing the backbone collapses alpha_mem_max (5e-4 at step
# 600) -- backbone co-evolution was NOT what was crowding memory out
# of the depth softmax; the writer subsystem itself is the bottleneck.
# That answer was specific to the original writer architecture.
#
# v12a's D2 diagnostic (runs.md "v12 campaign" §) shows the slot-
# attention writer ALSO collapses to the symmetric uniform fixed point
# (row_entropy_norm 0.998 -> 0.999, eff_rank 1.02 -> 1.01, keep_mean
# pinned at 0.500) by step 800, even though it briefly produced
# content-aware memory at step 200 (PA CB Δsh-m = +0.0205 -- the only
# such observation in the campaign).  The slot writer's softmax-over-
# slots breaks symmetry at init but the per-slot GRUCell with shared
# weights lets slots drift back together as training progresses.
#
# Hypothesis this cell tests
# --------------------------
# Does backbone gradient flow help BREAK the slot-attention symmetry
# by pushing M_c toward content-relevant directions from the LM-loss
# side?
#
#   * If trained beats frozen on PA CB Δ_sh-m / α_mem_max stability:
#     LM gradient through W_V_read and W_V_judge provides the symmetry-
#     breaking pressure the writer subsystem cannot generate on its own.
#     Decision: v13 keeps backbone training and considers warm-up
#     sched. or auxiliary content-aware probing loss.
#
#   * If frozen beats trained:
#     backbone co-evolution still corrupts the channel even with the
#     new writer.  The architectural fix needs to be deeper -- either
#     untie GRU weights across slots, add slot-specific bias, or move
#     m^t out of the depth softmax (the route the user pre-emptively
#     ruled out as out-of-spec).
#
# Decision triggers (mirrors frozen arm exactly)
# ----------------------------------------------
#   step 1500  : alpha_mem_max > 0 AND ||m^t||/||embed|| in [0.3, 50]
#   step 5000  : alpha_mem_max > 1e-2 AND pa_cb_dnm > +0.05
#   step 12500 : pa_cb_dsh > +0.020 AND alpha_mem_max NOT decaying
#                AND D2 row_entropy_norm < 0.98
#   step 25000 : standard Delta_sh-m > +0.010
#   KILL: step 5000 with pa_cb_dsh < +0.005 AND row_entropy_norm > 0.98
#         -- both writer kinds and both backbone regimes hit the
#         symmetric uniform fixed point; the slot-attention idea
#         needs deeper modification (untied GRU, slot-bias, etc).
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
WRITER_KIND="${1:-slot_attention}"
case "$WRITER_KIND" in
    slot_attention|slot_attention_full) ;;
    *) echo "writer_kind must be slot_attention or slot_attention_full" >&2; exit 2 ;;
esac
# GH200 layout is flat (train_chain.py at project root; no src/ dir).
# Local layout has src/. Detect.
if [ -f train_chain.py ]; then
    TRAIN=train_chain.py
elif [ -f src/train_chain.py ]; then
    TRAIN=src/train_chain.py
else
    echo "train_chain.py not found in . or src/" >&2; exit 2
fi
exec python -u "$TRAIN" \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind "$WRITER_KIND" \
    --memres_slot_attention_iters 3 \
    --train_chains paper_artifacts/chains/v11_mega_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0, "ultrachat": 2.0, "pippa": 2.0, "soda": 1.5, "lmsys": 2.0, "synthdlg": 1.5}' \
    --window_k 4 \
    --carry_state \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 25000 \
    --warmup 1500 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 3.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 0.0 \
    --curriculum_competition_bias 1.0 \
    --burn_in_max 12 \
    --burn_in_resample \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 500 \
    --save_every 1500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name "chain_v12d_${WRITER_KIND}_trained_chinchilla_mega_gh200" \
    --out_dir "output/chain_v12d_${WRITER_KIND}_trained_chinchilla_mega_gh200"
