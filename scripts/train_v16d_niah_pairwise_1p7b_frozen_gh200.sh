#!/usr/bin/env bash
# v16d_niah_pairwise_1p7b_frozen_gh200 (GH200 GPU 0) -- pair-wise NIAH probe.
#
# Same recipe as v16b at 1.7B-large frozen on synthd5_random_codes,
# with ONE deliberate diff: --window_k 2 (the trainer-supported
# minimum) instead of v16{a,b}'s window_k 3.
#
# Why pair-wise (window_k=2):
#   The v16{a,b} cells use window_k=3, which means the LM attends to
#   sessions [cb_pos-2, cb_pos-1, cb_pos] at the callback step. With
#   the synthd5 builder placing 2 evidence sessions at random body
#   positions in [0, 8] (callback at 9), at window_k=3 ~38% of chains
#   have at least one needle session inside the LM-visible window
#   instead of in M_c -- the LM can answer those callbacks by direct
#   attention without ever exercising the memory pathway. At
#   window_k=2 the LM-visible prefix is just session 8, so for the
#   ~78% of chains where neither needle landed at body position 8,
#   M_c is the *only* path to the answer span. This is the cleanest
#   pair-wise NIAH formulation the chain trainer supports
#   (window_k=1 would force pure memory but is rejected by the
#   curriculum sampler, see src/train_chain.py line 1209).
#
# Step budget is a 2000-step probe (vs v16b's 3500). If the
# evidence_lift trajectory shows symmetry breaking + a positive
# evidence_lift gradient by ~step 1000, this checkpoint becomes the
# warmup init for a phase-2 chain run on D4v2/D5 with an unfrozen
# backbone. If by step 2000 evidence_lift is still ~0 (and v16b at
# window_k=3 is also ~0), the writer/readout pathway has a deeper
# degeneracy that pair-wise window_k alone doesn't fix and we
# triage from there.
#
# Decision rule (logged in results/exp2_chain_recipe/runs.md v16d):
#   * step 2000 evidence_lift > +0.05 (1.7x v14k's headline)
#       -> relaunch v16e: 6000-step continuation resuming from best/,
#          same flags, same scale.
#   * step 2000 evidence_lift in [0, +0.05]
#       -> warmup is too short; relaunch as v16e at 6000 steps from
#          scratch, same flags. The probe didn't win but didn't lose.
#   * step 2000 evidence_lift <= 0
#       -> rebuild synthd5 with body_positions := range(body_len-1)
#          so evidence is strictly < callback_pos-1; retry as v16f.
#       -> if v16f also <= 0, the writer/readout has a degeneracy
#          that the corpus + window_k can't cure; revisit
#          architecture (extract path, slot writer, readout
#          competition) before scaling further.
#
# Frozen backbone, slot_attention writer, extract_input_norm,
# alpha_mem_floor, InfoNCE, judge_qk_layernorm OFF default per
# README architectural priors bullet 8 -- all unchanged from v16b.
#
# GH200 repo layout is FLAT (train_chain.py at root, not src/).
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
[ -f "$HOME/venv/bin/activate" ] && . "$HOME/venv/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python -u train_chain.py \
    --preset qwen3-1.7b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_extract_input_norm \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention \
    --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal \
    --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --writer_warmup_steps 0 \
    --writer_warmup_router_bias 0.0 \
    --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --alpha_mem_floor_aux_weight 0.01 \
    --alpha_mem_floor_target 0.05 \
    --contrastive_infonce_weight 0.5 \
    --contrastive_infonce_temperature 1.0 \
    --contrastive_infonce_callback_only \
    --contrastive_infonce_initial_weight 0.5 \
    --contrastive_infonce_warmup_steps 0 \
    --train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 2 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 1e-4 \
    --lr_backbone 0 \
    --steps 2000 \
    --warmup 200 \
    --max_norm 1.0 \
    --memory_dropout 0.10 \
    --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 5.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 \
    --curriculum_competition_bias 0.0 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --mask_evidence_session_loss \
    --kill_on_memory_collapse \
    --kill_on_memory_collapse_min_step 200 \
    --eval_every 100 \
    --save_every 500 \
    --eval_n_chains 24 \
    --eval_window 8 \
    --phase_aligned_eval_n_chains 32 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric evidence_lift \
    --run_name chain_v16d_niah_pairwise_1p7b_frozen_gh200 \
    --out_dir output/chain_v16d_niah_pairwise_1p7b_frozen_gh200 \
    2>&1 | tee logs/chain_v16d_niah_pairwise_1p7b_frozen_gh200.log
