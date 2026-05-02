#!/usr/bin/env bash
# v13b-d4-frozen-local (LOCAL H100 GPU 0) -- v13 full stack + frozen
# backbone on D4.  Stacks on the v12d_d4_frozen result (first positive
# evidence_lift in the whole campaign, step 4000 = +0.0267) by adding
# the three problems.md §5 levers on top.
#
# Single-knob difference vs v13a_d4_trained_local:
#   v13a:  backbone trainable, writer_warmup unfreezes backbone @ step 500
#   v13b:  --freeze_backbone + --writer_warmup_keep_backbone_frozen
#          (backbone stays frozen throughout)
#
# Intent:
# - v12d_d4_frozen (slot_attention + frozen_backbone + AP routing) hit
#   evidence_lift = +0.0267 at step 4000 -- a small but real first
#   signal in the whole campaign.  The v11l-fix frozen-AP cell hit
#   zero.  So "frozen + slot" is the first architectural regime where
#   something isn't immediately destroyed by backbone co-evolution.
# - v13b adds the writer_warmup + orthogonal + slot_positional +
#   simple_gate stack ON TOP of that frozen regime.  If the frozen
#   regime is necessary but insufficient (as the evidence suggests),
#   the three v13 levers should push evidence_lift from +0.03 to
#   something much larger.
# - If v13b hits pa_cb_evidence_lift > +1.0 by step 1000 and
#   pa_cb_ce_mem < 3.5 nats, the frozen v13 regime is the
#   architectural recipe for the paper.
#
# Decision triggers (same as v13a; frozen backbone should make the
# step-500 evidence-lift gate EASIER to clear because no backbone
# co-evolution is crowding m^t out):
#   step  500:  pa_cb_dnm > +2.0  AND  pa_cb_evidence_lift > +0.5
#   step 1000:  pa_cb_ce_mem < 3.5 nats  AND  evidence_lift > +1.0
#   step 2000:  pa_cb_ce_mem < 2.0 nats
#   step 4000:  pa_cb_ce_mem < 1.0 nats
#
# KILL @ step 1000 if evidence_lift < 0.1 (frozen regime can't break
# the uniform attractor either -> architecture-level pivot required).
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=0
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
    --writer_warmup_keep_backbone_frozen \
    --freeze_backbone \
    --train_chains paper_artifacts/chains/synthd4_persona_callback_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd4_persona_callback_val_s512.pt \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 0 \
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
    --run_name chain_v13b_d4_frozen_local \
    --out_dir output/chain_v13b_d4_frozen_local \
    2>&1 | tee logs/chain_v13b_d4_frozen_local.log
