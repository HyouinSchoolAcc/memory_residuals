#!/usr/bin/env bash
# Patent-evidence training cell launcher (foreground).
#
# Usage: patent_train_cell.sh <run_name> <seed> [extra trainer flags...]
#
# Recipe = v27b headline (F3-off, frozen 0.6B backbone) minus the
# collapse kill-switch (ablation cells are EXPECTED to underperform and
# must still train the full 1000 steps). Pass either
#   --preset qwen3-0.6b-large
# or the manual size flags (--pretrained/--memres_num_vectors/...) plus
# any single-variable change (e.g. --memres_update_mode overwrite) in
# the extra args.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="$1"; SEED="$2"; shift 2

mkdir -p logs
python -u src/train_chain.py \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 --router_mem_bias_init 0 \
    --memres_extract_source hidden_14 \
    --memres_extract_input_norm \
    --memres_gate_init 0.0 --memres_readout_norm_init 0.05 \
    --memres_writer_kind slot_attention --memres_slot_attention_iters 3 \
    --memres_queries_init orthogonal --memres_slot_positional \
    --memres_judge_qk_layernorm \
    --memres_readout_depth 4 \
    --writer_warmup_steps 0 --writer_warmup_router_bias 0.0 --writer_warmup_anneal_steps 0 \
    --freeze_backbone \
    --readout_probe_enabled \
    --readout_probe_loss_weight 0.0 \
    --readout_probe_warmup_steps 200 \
    --alpha_mem_floor_aux_weight 0.5 \
    --alpha_mem_floor_target 0.10 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt \
    --window_k 3 --batch_size 4 --grad_accum 2 \
    --lr 1e-4 --lr_backbone 0 --steps 1000 --warmup 200 --max_norm 1.0 \
    --memory_dropout 0.10 --context_dropout 0.05 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 5.0 --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 --curriculum_competition_bias 0.0 \
    --burn_in_max 0 --mask_padding_loss --score_tail_frac 1.0 \
    --mask_evidence_session_loss \
    --eval_every 100 --save_every 500 \
    --eval_n_chains 24 --eval_window 8 \
    --phase_aligned_eval_n_chains 64 \
    --diag_routing_n_chains 8 \
    --diagnose_grad_groups --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric evidence_lift \
    --seed "$SEED" \
    --run_name "$RUN_NAME" \
    --out_dir "output/$RUN_NAME" \
    "$@" \
    > "logs/$RUN_NAME.log" 2>&1
