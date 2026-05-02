#!/usr/bin/env bash
# v13a-d4-trained-local (LOCAL H100 GPU 1) -- The v13 full stack on D4
# with trained backbone.  First direct test of the problems.md §5
# recommendations composed together.
#
# v13 thesis (see problems.md §2a-§2g and §5):
#
#   Every v1..v12 cell hits the same "symmetric uniform judge /
#   content-blind writer" attractor.  The root cause has three
#   mutually-reinforcing parts (problems.md §2a-§2c):
#
#     §2a  M_in and M_judge receive ~10^-8 of the backbone gradient
#          under LM-only joint training.  They never move from their
#          nn.init.normal_(std=d^-0.5) init, so the writer has no
#          semantic prior.
#     §2b  With random M_judge + softmax over inputs, the judge is
#          permutation-equivariant at every symmetric configuration.
#          Uniform 1/(2K) attention is a gradient fixed point that
#          10^-8 noise cannot escape.
#     §2f  The depth-softmax in attention_parity (Eq. 9) puts m^t in
#          zero-sum competition with the backbone's block summaries.
#          As the backbone fine-tunes, it wins the competition and
#          m^t is crowded out.
#
# v13 attacks all three simultaneously:
#
#     O (objective):  --writer_warmup_steps 500
#         Phase 1 freezes backbone + LM head + embed, forces
#         mem_bias=+4 (memory dominates depth softmax), and trains
#         ONLY the memres subsystem (writer + readout + router).
#         The writer receives LM gradient 5+ orders of magnitude
#         larger than in joint training.  After 500 steps, anneal
#         mem_bias back to init and (optionally) unfreeze backbone.
#
#     S (symmetry): --memres_queries_init orthogonal
#                   --memres_slot_positional
#         M_in and M_judge rows are orthogonal at init (qr-based);
#         the judge softmax is no longer permutation-equivariant
#         (rows produce distinct attention patterns from t=0).  A
#         learnable Fourier-pattern per-slot positional address is
#         added before Q/K/V projections, giving each slot a
#         permanent identity beyond its initial direction.
#
#     R (routing): --memres_mode simple_gate
#         m^t is pulled OUT of the Eq. 9 depth softmax and given a
#         dedicated per-sublayer scalar gate.  m^t no longer
#         competes with mature block summaries for route mass;
#         memory can be recruited without structurally disfavoring
#         it against pretraining-bearing signals.  v9c (the one
#         cell with monotonic Δ_nm-m growth to +0.18) used
#         simple_gate -- this is not an exotic path, it is the one
#         path with empirical priors.
#
# Writer kind stays at slot_attention (v12's contribution) and
# slot_positional + orthogonal addresses the "GRU weight-tie pulls
# slots back to uniform" weakness that killed v12a at step 800.
#
# Decision triggers (sharp; D4 is gold-standard ground truth, floor
# pa_cb_ce_mem = log(256) = 5.55 nats)
# --------------------------------------------------------------------
#   step  200 (mid-warmup):
#       |g_M_in| / |g_bb_in_phase_2| > 1e-4
#                      AND D2 row_entropy_norm < 0.98
#       -- writer is actually getting gradient; symmetry breaking
#   step  500 (end of warmup):
#       pa_cb_dnm > +2.0  AND  pa_cb_evidence_lift > +0.5
#                         AND  D2 row_entropy_norm < 0.95
#       -- memory is content-specific, not chain-identity hash
#   step 1000 (post-anneal):
#       pa_cb_ce_mem < 3.5 nats  AND  pa_cb_evidence_lift > +0.5
#       AND D2 eff_rank > 4  AND  alpha_mem channel open
#   step 2000:
#       pa_cb_ce_mem < 2.0 nats  (~35% of log(256))
#   step 4000:
#       pa_cb_ce_mem < 1.0 nats  (~18% of log(256))
#
# KILL @ step 1000 with D2 row_entropy_norm > 0.98 AND
#                        pa_cb_evidence_lift < 0.01:
#   the v13 3-lever stack does not escape the uniform attractor
#   on the cleanest possible corpus -> the architecture is the
#   wall and v14 must go off-spec (e.g. prefix-conditioning,
#   untied per-slot GRU, or abandoning learned compression for
#   frozen-KV memory).
#
# Hyperparameters copy v12d_d4_trained EXCEPT the v13 additions
# and simple_gate routing.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=1
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
    --router_recent_bias_init 0 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention \
    --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal \
    --memres_slot_positional \
    --writer_warmup_steps 500 \
    --writer_warmup_router_bias 4.0 \
    --writer_warmup_anneal_steps 200 \
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
    --run_name chain_v13a_d4_trained_local \
    --out_dir output/chain_v13a_d4_trained_local \
    2>&1 | tee logs/chain_v13a_d4_trained_local.log
