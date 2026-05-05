#!/usr/bin/env bash
# v12a-slot-judge-D4 (LOCAL H100 GPU 1) -- Slot-Attention judge on the
# D4 synthetic persona-callback corpus.
#
# v12 thesis (results/exp2_chain_recipe/runs.md "v11 campaign -- first
# 7 cells finished, post-mortem"):
#
#   The post-mortem found "the writer is content-blind".  D2 audit on
#   v11g/best showed the Judge layer is decision-less by *construction*:
#   row_entropy/log(2K) = 0.999, eff_rank 1.02, keep_mean = 0.500
#   exactly across all rows.  This is a structural fixed point of the
#   original judge (cross-attention with K queries over 2K keys, softmax
#   over the inputs axis = uniform average).
#
#   v12 replaces the Stage-2 judge with Slot Attention (Locatello et
#   al. 2020, NeurIPS): softmax over the *slots* axis, weighted-mean
#   renormalisation, GRU keep/write per slot.  This forces slots to
#   specialise on disjoint pieces of the input -- the PDF's "zero-sum
#   competition" claim (Section 2.1) realised at the architectural
#   level rather than by hopeful gradient.
#
# Why D4 first, not LME
# ---------------------
# The v11r/q post-mortem proved that LME's noise + sparse callback +
# chain-identity confounder makes diagnostic signals nearly useless.
# D4 is the synthetic gold-standard already shipped with the
# diagnostic stack:
#
#   tools/build_synthetic_persona_callback.py
#   paper_artifacts/chains/synthd4_persona_callback_train_s512.pt
#   paper_artifacts/chains/synthd4_persona_callback_val_s512.pt
#
# 5000 chains x 9 sessions, 256-item closed-set callback, deterministic
# ground truth.  Irreducible callback_ce floor = log(256) = 5.55 nats.
# If the slot-attention writer cannot move pa_cb_ce_mem significantly
# toward this floor in 2000 steps, the writer subsystem is not the
# bottleneck and we pivot to the v12-spec-flexible scope.
#
# Single-knob diff vs v11g_ap_baseline_gh200
# ------------------------------------------
# Only architectural change:
#   --memres_writer_kind slot_attention
# Stage-2 judge replaced; Stage-1 extraction (CrossAttention layers)
# unchanged.  v12b adds slot-attention to extract; v12a is the cleaner
# attribution test (does the judge fix alone close the gap?).
#
# Decision triggers (sharp; D4 has clean ground truth)
# ----------------------------------------------------
#   step  200 : pa_cb_dnm > +0.5  AND  D2 row_entropy_norm < 0.95
#                                 (slots are no longer uniform)
#   step 1000 : pa_cb_dnm > +2.0  AND  pa_cb_evidence_lift > +1.0
#                                 AND  judge_keep_mean != 0.500 +/- 0.05
#                                 (judge actually decides)
#   step 2000 : pa_cb_ce_mem < 2.0 nats  (35% of log(256))
#                                 -- the architecture works in principle
#   KILL @ step 1000 with D2 row_entropy_norm > 0.98:
#                                 slot attention also stays uniform;
#                                 the diagnosis was wrong; pivot to
#                                 ground-truth-first scope.
#
# Hyperparameter copies of v11g (P0+P2, AP +4/0, hidden_14, k=3)
# preserved exactly so the only causal axis is the writer kind.
# Source weights set to 1.0 on every dataset key D4 might emit
# (harmless on chains whose sources aren't recognised; chain_name
# in the synth corpus is "synth_persona_callback_<i>" so the sampler
# falls through to the default weight).
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=1
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention \
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
    --run_name chain_v12a_slot_judge_d4_local \
    --out_dir output/chain_v12a_slot_judge_d4_local \
    2>&1 | tee logs/chain_v12a_slot_judge_d4_local.log
