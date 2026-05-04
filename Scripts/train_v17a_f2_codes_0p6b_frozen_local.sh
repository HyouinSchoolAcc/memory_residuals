#!/usr/bin/env bash
# v17a_f2_codes_0p6b_frozen_local (LOCAL GPU 0)
# F2 / WriterProbeHead writer-side extractive supervision.
#
# Diagnosis recap:
#   * v15a (0.6B frozen) D5 readout-TTT: callback CE 4.77 -> 4.23,
#     -11.2% (MIXED: writer is partially content-blind).
#   * v15e (1.7B frozen) D5 readout-TTT: 4.36 -> 4.01, -8.1%
#     (LIKELY-W: writer is the bottleneck).
#   * v16a (0.6B frozen, leak-free random-codes corpus):
#     evidence_lift = +0.005 after 2500 steps (no improvement).
#   * v16b (1.7B frozen, leak-free random-codes corpus):
#     evidence_lift = +0.013 after 3500 steps (no improvement).
#
#   Both v16 cells removed the D4v2 category-cue leak but the writer
#   still failed to learn chain-specific content.  This confirms the
#   v17_wildcards diagnosis that the v11..v16 LM-NLL gradient is the
#   wrong signal (content-blind LM head; the writer never has a
#   reason to extract the chain's needle).
#
# F2 fix (this cell):
#   --writer_probe_enabled with --writer_probe_loss_weight 1.0 adds
#   a separate gradient channel that supervises the writer to make
#   M_c probe-decodable to the chain's first answer token id.  The
#   probe head is independent of the LM head, so a content-blind
#   M_c cannot fool it with a learned prior.  Eval is unchanged
#   (probe is unused at scoring time).
#
# Pier-agent prescription compliance:
#   * KILLED alpha_mem_floor_aux_weight (was knob the joint-train cell
#     gamed by spreading alpha thinly).
#   * KILLED contrastive_infonce_weight (NCE through content-blind LM
#     head was random by construction).
#   * RAISED phase_aligned_eval_n_chains to 64 (sign-off cell, not a
#     sweep cell).
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v17a_f2_codes_0p6b_frozen_local"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 nohup python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 --router_mem_bias_init 0 \
    --memres_update_mode gated --memres_extract_source hidden_14 \
    --memres_extract_input_norm \
    --memres_gate_init 0.0 --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --writer_warmup_steps 0 --writer_warmup_router_bias 0.0 --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --writer_probe_enabled \
    --writer_probe_loss_weight 1.0 \
    --writer_probe_warmup_steps 200 \
    --writer_probe_n_queries 1 \
    --train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 --batch_size 4 --grad_accum 2 \
    --lr 1e-4 --lr_backbone 0 --steps 2500 --warmup 200 --max_norm 1.0 \
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
    --run_name chain_v17a_f2_codes_0p6b_frozen_local \
    --out_dir "$OUT_DIR" \
    > "$LOG_DIR/chain_v17a_f2_codes_0p6b_frozen_local.log" 2>&1 &
echo "v17a launched pid=$!"
