#!/usr/bin/env bash
# v10b ATTENTION_PARITY +4/-4 ON DIVERSE CORPUS (LOCAL H100 GPU 1)
#
# Sister cell to v10a (composed simple_gate) and a direct 0.6B proxy
# for the GH200 8B cell's routing choice. Tests the central
# hypothesis of the v10 campaign: on a diverse, memory-requiring
# corpus (v6_lme_msc, the only v9 cohort member with good
# survivability), does the original paper-spec attention_parity
# routing with softer +4/-4 biases break the v3-v9 router-collapse
# pattern?
#
# Diff vs v9c:
#   - memres_mode simple_gate -> attention_parity   (load-bearing change)
#   - router_recent_bias_init  = +4                 (softer than default +32)
#   - router_mem_bias_init     = -4                 (v3 default, 700x softmax
#                                                     ratio favouring recent
#                                                     at init -- relaxed
#                                                     enough for gradient to
#                                                     find the memory branch)
#   - keeps diverse v6_lme_msc corpus and the pure-competition curriculum
#     (those are the two v9c axes that survived).
#
# If this cell produces non-zero α_mem and any positive PA CB Δ_sh-m by
# step 500, attention_parity is viable on diverse data and the 8B GH200
# cell's routing choice is validated. If it collapses again (α_mem=0,
# ||m^t||->0 drift), we fall back to simple_gate for the 8B and the
# paper ships "simple_gate + RMSNorm readout" as the canonical primitive.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=1
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init -4 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/v6_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --source_weights '{"longmemeval": 4.0, "msc": 3.0, "pg19": 1.0, "tv": 4.0, "realtalk": 1.0}' \
    --window_k 3 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 5e-5 \
    --lr_backbone 2e-5 \
    --steps 6000 \
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
    --phase_aligned_eval_n_chains 48 \
    --diag_routing_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v10b_attnparity_pm4_diverse_local \
    --out_dir output/chain_v10b_attnparity_pm4_diverse_local \
    2>&1 | tee logs/chain_v10b_attnparity_pm4_diverse_local.log
