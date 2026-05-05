#!/usr/bin/env bash
# v11r (GH200, 0.6B) -- attention_parity +4/0 WITH readout-warmup + InfoNCE
#
# Architectural fix A (revised), motivated by the 2026-05-01 audit on
# v11g/best (results/exp2_chain_recipe/v11g_diag_synth.json,
# v11g_d5_ttt.json):
#
#   D2: judge softmax row_entropy / log(2K) = 0.999 (uniform).
#   D3: M_c pair_dist / self_norm = 0.022 (chains look near-identical
#       structurally, but ...)
#   D5: 300 steps of TTT on the readout alone drops callback CE 48%
#       (8.20 -> 4.26 nats on synthd4) -- the WRITER does encode
#       chain-specific content; the READOUT is the bottleneck.
#
#   Chain of reasoning behind v11r:
#     - In a normal v11g-style joint training, the router closes the
#       memory pathway (alpha_mem_max ~ 0.0092 at step 4000) BEFORE
#       the readout has converged enough to pull useful signal from
#       M_c.  The system locks in a "memory disabled" solution
#       because every step the readout is wrong, it makes the LM
#       loss WORSE, so the router learns to route around it.
#     - Fix A breaks the lock-in by giving the readout a head start:
#       freeze writer + router + backbone + LM head for a 500-step
#       warmup, force-open the routing (mem_bias = +4), and only let
#       the readout update.  After 500 steps the readout has actual
#       structure and the router sees a useful m^t when we unfreeze.
#     - InfoNCE contrastive supervision (--contrastive_infonce_*) is
#       on from step 0, providing dense chain-discriminative
#       gradient that does NOT require the LM head at all -- it
#       acts as a watchdog on the writer/readout's content
#       specificity even during the warmup window.
#
# Decision triggers (sharp):
#   step  200 (mid-warmup): nce_gap > +0.5 (readout is learning to
#                                          discriminate chains).
#   step  500 (end-warmup): alpha_mem_max set by the schedule, NOT
#                            yet a learning signal.  Check
#                            ||m^t||/||embed|| in [0.3, 50] and
#                            judge_row_entropy / log(2K) starting
#                            to drop below 1.0 (D2 fix).
#   step 1000 (post-anneal): pa_cb_dsh > +0.020 AND
#                             pa_cb_evidence_lift > +0.020 AND
#                             alpha_mem_max stays > 0.05 (the
#                             router didn't immediately re-close).
#   step 2000              : pa_cb_dsh > +0.050 (path is real).
#   step 4000              : pa_cb_dsh > +0.100  (head-line goal).
#
#   KILL: step 1000 with alpha_mem_max < 0.01 means the warmup
#         didn't escape the lock-in -- relaunch with longer warmup
#         (1000 steps) or stronger force-open bias (+6).
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --router_recent_bias_init 4 \
    --router_mem_bias_init 0 \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.0 \
    --memres_readout_norm_init 0.05 \
    --readout_warmup_steps 500 \
    --readout_warmup_router_bias 4.0 \
    --readout_warmup_anneal_steps 200 \
    --contrastive_infonce_weight 0.5 \
    --contrastive_infonce_initial_weight 0.05 \
    --contrastive_infonce_warmup_steps 500 \
    --contrastive_infonce_temperature 1.0 \
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
    --diagnose_grad_groups \
    --diagnose_memory_dynamics \
    --diagnose_memory_dynamics_n_chains 8 \
    --gradient_checkpointing \
    --save_best_metric phase_aligned \
    --run_name chain_v11r_ap_readout_warmup_gh200 \
    --out_dir output/chain_v11r_ap_readout_warmup_gh200
