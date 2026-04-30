#!/usr/bin/env bash
# v11g (GH200, 0.6B) -- attention_parity +4/0 BASELINE
#
# This is the v11 GH200 baseline cell. It composes the three v11 fixes:
#   P0  evidence-aware curriculum (uses chain_evidence_positions)
#   P2  readout magnitude rescale (out_norm_init=0.05, ||m^t|| ~ 4 not ~73)
#   v3  routing per user directive: attention_parity, recent=+4, mem=0
#
# Sister to local v11 (which uses simple_gate). The motivation for
# attention_parity over simple_gate: simple_gate puts a learnable scalar
# directly on the residual stream which scales linearly with whatever
# ||m^t|| the readout produces; attention_parity routes m^t through the
# block depth softmax so its mass is automatically clipped to [0, 1] per
# sublayer and competes against (b_0, b_1, ..., b_{n-1}). The user's v3
# routing-trace data showed alpha_mem ~ 4.7e-4 averaged across 55
# sublayers under +4/-4 -- with +4/0 the initial alpha_mem mass is
# ~4x larger (mem doesn't get the -4 penalty), giving the readout a
# bigger "voice" at step 0.
#
# Decision triggers (sharp; sequential):
#   step 200  : alpha_mem_max > 0     AND ||m^t||/||embed|| in [0.3, 50]
#   step 500  : alpha_mem_max > 5e-3  AND pa_cb_dnm > +0.005
#                                     AND pa_cb_evidence_lift > +0.005
#   step 1000 : alpha_mem_max > 1e-2  AND pa_cb_dnm > +0.020
#                                     AND pa_cb_dsh > +0.005
#                                     (memory is content-specific)
#   step 2000 : standard Δ_sh-m > +0.005 (deployment-distribution gap
#                                          starting to close)
#   KILL: step 1000 with alpha_mem_max < 1e-3   -> attention_parity
#         is collapsed even with P0/P2 -- relaunch as simple_gate.
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
    --train_chains paper_artifacts/chains/v11_lme_msc_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512_v11.pt \
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
    --run_name chain_v11g_ap_baseline_gh200 \
    --out_dir output/chain_v11g_ap_baseline_gh200
