#!/usr/bin/env bash
# v11k (GH200, 0.6B) -- attention_parity +4/0 with NO evidence labels
#
# A/B against v11g_ap_baseline. Reverts to the legacy P0-broken corpus:
#   --train_chains v6_lme_msc_train_s512.pt   (no chain_evidence_positions)
#   --eval_chains  lme_val_s512.pt            (no chain_evidence_positions)
#
# When chain_evidence_positions is absent, the competition curriculum
# falls back to uniform sampling (this is exactly the v9/v10 behaviour
# pre-fix). The phase-aligned eval also falls back to uniform "evidence"
# session selection -- the same ambiguous metric the v9 ablations used.
#
# This is the cleanest possible test of the P0 hypothesis: same code
# path, same router, same initializers, same curriculum -- only the
# evidence labels differ. If v11g >> v11k on pa_cb_dnm and pa_cb_dsh,
# we have proof that data quality (P0) is load-bearing for memory
# learning.
#
# Decision triggers:
#   step 1000 : if alpha_mem_max < 1e-3 -> EXPECTED collapse, P0 is
#               confirmed as the load-bearing fix
#   step 2000 : if alpha_mem_max > 1e-2 AND pa_cb_dnm > +0.005, P0 is
#               NOT load-bearing (P2/bias did the work alone) -- update
#               theory
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --train_chains paper_artifacts/chains/v6_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
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
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v11k_ap_no_evidence_gh200 \
    --out_dir output/chain_v11k_ap_no_evidence_gh200
