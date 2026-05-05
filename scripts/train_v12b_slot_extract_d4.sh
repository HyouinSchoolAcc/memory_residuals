#!/usr/bin/env bash
# v12b-slot-extract-D4 (LOCAL or GH200) -- Slot-Attention judge AND
# extract on the D4 synthetic persona-callback corpus.
#
# Conditional cell.  Launches only if v12a (slot judge alone) shows
# non-trivial signal at step 1000 but does not yet hit the step-2000
# success bar.  Tests whether also replacing Stage-1 extraction with
# the Slot Attention primitive closes the remaining gap.
#
# Hypothesis: v11g D2 found judge entropy at the symmetric uniform
# fixed point.  Stage-1 extraction has the same shape (K queries
# attending over N inputs with softmax over inputs), and so the same
# inductive-bias argument applies: at init every M_in slot becomes a
# uniform average over the session tokens, producing K nearly-identical
# extracted "candidates" with no architectural pressure to specialise.
# slot_attention_full pivots BOTH stages to softmax-over-slots, which
# means M_new is also forced to tile the input pool into K disjoint
# factors before the judge sees it.  If v12a is gradient-bottlenecked
# on the writer's *input* (homogeneous M_new), v12b lifts that ceiling.
#
# Single-knob diff vs v12a:
#   --memres_writer_kind slot_attention_full   (was slot_attention)
#
# Decision triggers (mirrors v12a, expects equal-or-better)
# ---------------------------------------------------------
#   step 1000 : pa_cb_dnm >= v12a@1000 + 1.0 nat
#                                 (extract specialisation lifts ceiling)
#   step 2000 : pa_cb_ce_mem < 1.5 nats  (vs v12a target of 2.0)
#   KILL @ step 1000 with pa_cb_dnm < v12a@1000:
#                                 the extract-side change is harmful or
#                                 a wash; fall back to slot_attention
#                                 for v12c / v12d.
#
# Param budget: SlotAttentionWriter is ~6 d^2 + 6 d (vs CrossAttention's
# 3 d^2).  With L_E + 1 = 5 cross-attention layers replaced by ONE
# SlotAttentionWriter doing 5 iterations, we save 9 d^2 of params and
# halve forward FLOPs in the extract stage.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention_full \
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
    --run_name chain_v12b_slot_extract_d4 \
    --out_dir output/chain_v12b_slot_extract_d4 \
    2>&1 | tee logs/chain_v12b_slot_extract_d4.log
