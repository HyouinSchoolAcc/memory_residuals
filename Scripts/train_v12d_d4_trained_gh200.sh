#!/usr/bin/env bash
# v12d-D4-TRAINED (GH200) -- slot_attention writer * TRAINED BACKBONE
#                                                  * D4 SYNTH CORPUS
#                                                  * MODEST BUDGET
#
# Trained-backbone arm of the v12d frozen-vs-trained comparison study.
# Pairs with train_v12d_d4_frozen_gh200.sh.  See that script for the
# full diagnosis.  Single-axis diff:
#   frozen:  --freeze_backbone     --lr_backbone 0
#   trained: (no freeze flag)      --lr_backbone 2e-5
#
# Effectively a re-run of v12a with the same corpus + recipe; the
# previous v12a was running locally and was killed (per "move
# everything from H100s to GH200" directive) so the comparison study
# is re-launched cleanly on GH200 alongside its frozen counterpart.
#
# Hypothesis this cell tests (vs frozen)
# --------------------------------------
# Does backbone gradient flow help BREAK the slot-attention symmetry
# by pushing M_c toward content-relevant directions from the LM-loss
# side?
#
#   * Trained beats frozen on D2 row_entropy_norm / pa_cb_dsh:
#     LM gradient through W_V_read and W_V_judge provides the
#     symmetry-breaking pressure the writer subsystem cannot
#     self-generate -- v13 keeps backbone training and adds
#     curriculum / auxiliary content-aware probing loss.
#   * Frozen beats trained: backbone co-evolution still corrupts the
#     channel even with the new writer; architectural fix needs to
#     be deeper (untied per-slot GRU, slot-specific bias) or the
#     spec-strict constraint must be relaxed (m^t out of depth softmax).
#   * Both collapse to row_entropy_norm > 0.98 by step 1000:
#     KILL.  Slot-attention as currently formulated does not break the
#     symmetric uniform fixed point.  v13 architectural pivot.
#
# Decision triggers identical to frozen arm (mirrors exactly).
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
WRITER_KIND="${1:-slot_attention}"
case "$WRITER_KIND" in
    slot_attention|slot_attention_full) ;;
    *) echo "writer_kind must be slot_attention or slot_attention_full" >&2; exit 2 ;;
esac
if   [ -f train_chain.py ];     then TRAIN=train_chain.py
elif [ -f src/train_chain.py ]; then TRAIN=src/train_chain.py
else echo "train_chain.py not found in . or src/" >&2; exit 2
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
    --train_chains paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
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
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name "chain_v12d_${WRITER_KIND}_d4_trained_gh200" \
    --out_dir "output/chain_v12d_${WRITER_KIND}_d4_trained_gh200"
