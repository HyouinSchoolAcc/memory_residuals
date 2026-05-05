#!/usr/bin/env bash
# v17e_f2_codes_1p7b_frozen_gh200 (GH200 GPU 0)
# F2 / WriterProbeHead writer-side extractive supervision at 1.7B.
#
# Sister cell to v17a (0.6B local).  v15e (1.7B frozen) D5 readout
# audit returned LIKELY-W (-8.1% callback CE drop with TTT on
# readout): the 1.7B writer is the bottleneck.  v17e tests whether
# adding the F2 writer-side gradient channel (probe head supervision
# of M_c -> first answer-token id) cures the writer-content-blind
# pathology at the 1.7B scale.
#
# All other knobs match v16b (1.7B baseline on synthd5_random_codes)
# so the F2 head is the sole confound.
#
# Pier-agent prescription compliance: alpha_mem_floor_aux_weight and
# contrastive_infonce_weight both DROPPED.
#
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="output/chain_v17e_f2_codes_1p7b_frozen_gh200"

exec python -u src/train_chain.py \
    --preset qwen3-1.7b-large \
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
    --window_k 3 --batch_size 2 --grad_accum 4 \
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
    --run_name chain_v17e_f2_codes_1p7b_frozen_gh200 \
    --out_dir "$OUT_DIR"
