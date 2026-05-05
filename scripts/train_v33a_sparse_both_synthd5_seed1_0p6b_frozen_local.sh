#!/usr/bin/env bash
# v33a: v32a (sparse writer top-k=8) + sparse READOUT top-k=8 on
# synthd5. The friend's option 3: "Hard slot-routing in the readout —
# make the readout's cross-attention sparse (top-k or Gumbel) so the
# readout has to select slots rather than mix them. Chain-specific
# facts then live in identifiable slots."
#
# Diff from v32a/seed1: --memres_readout_topk_slot_softmax 8 (only).
#
# Motivation: v32a/final on synthd5 gave Δ_cb = +1.33 nats but
# evidence_lift = +0.0017 (writer puts something useful into M_c but
# it's 'this is an ID' priors, not the actual ID). TTT-on-Mc on the
# v32a/final returned NEGATIVE (-0.078): even with M_c TTT-optimised
# against the evidence session, the readout cannot extract chain-
# specific recall.  This says the readout pathway (dense soft-mixing
# over K=128 slots) lacks the *capacity* to localise a chain-specific
# fact in M_c.  Sparse readout gives the readout a SELECT-don't-mix
# inductive bias, the same inductive bias the writer now has.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v33a_sparse_both_synthd5_seed1_0p6b_frozen_local"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" nohup python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 --router_mem_bias_init 0 \
    --memres_update_mode gated --memres_extract_source hidden_14 \
    --memres_extract_input_norm \
    --memres_gate_init 0.0 --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --memres_judge_topk_slot_softmax 8 \
    --memres_readout_topk_slot_softmax 8 \
    --memres_readout_depth 4 \
    --writer_warmup_steps 0 --writer_warmup_router_bias 0.0 --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --readout_probe_enabled \
    --readout_probe_loss_weight 0.0 \
    --readout_probe_warmup_steps 200 \
    --alpha_mem_floor_aux_weight 0.5 \
    --alpha_mem_floor_target 0.10 \
    --train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 --batch_size 4 --grad_accum 2 \
    --lr 1e-4 --lr_backbone 0 --steps 1500 --warmup 200 --max_norm 1.0 \
    --memory_dropout 0.10 --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 5.0 --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 --curriculum_competition_bias 0.0 \
    --burn_in_max 0 --mask_padding_loss --score_tail_frac 1.0 \
    --mask_evidence_session_loss \
    --kill_on_memory_collapse --kill_on_memory_collapse_min_step 200 \
    --eval_every 100 --save_every 500 \
    --eval_n_chains 24 --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric evidence_lift \
    --seed 1 \
    --run_name chain_v33a_sparse_both_synthd5_seed1_0p6b_frozen_local \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v33a_sparse_both_synthd5_seed1_0p6b_frozen_local.log" 2>&1 &
echo "v33a sparse-both synthd5 seed=1 launched pid=$!"
