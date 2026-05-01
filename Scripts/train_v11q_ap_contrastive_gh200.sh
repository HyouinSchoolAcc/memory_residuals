#!/usr/bin/env bash
# v11q (GH200, 0.6B) -- attention_parity +4/0 BASELINE + multi-negative
# InfoNCE contrastive loss.
#
# Diff vs v11g (single-knob):  --contrastive_infonce_weight 0.5 with linear
# warmup 0.05 -> 0.5 over the first 500 steps.  Everything else identical
# to v11g (preset, routing, biases, P0+P2 fixes, corpus, schedule).
#
# Why this run.  v11g's grow-then-decay was diagnosed as a stacking of
# (a) backbone co-evolution crowding m^t out of the AP softmax and
# (b) sparse / indirect supervision on the memory subsystem (gradient
# only through callback-token CE, which is < 5% of training tokens, and
# routes through three softmaxes).  v11l/p attack (a) by freezing the
# backbone; this cell attacks (b) by injecting dense supervision
# directly onto Δ_sh-m.
#
# What InfoNCE does.  Within the batch of B=4 chains, after the
# standard TBPTT pass produces per-chain M_c[1..B] and L_match[i] =
# NLL(last_session_i | M_c[i]), we additionally score every (input_i,
# M_c[j]) pair (B*B forward on the last session, ~25% per-step
# overhead measured locally).  The B*B NLL matrix L gets cross-entropy
# loss with diagonal=positive: chain i's M_c should fit chain i's last
# session better than any other chain's M_c.  Three pressures via
# gradient: (1) L[i,i] -> down (use own M_c well), (2) L[i,j] -> up
# for j != i (other M_c is unhelpful), (3) M_c[j] for j != i is
# pushed AWAY from chain i's content (M_c becomes chain-specific).
# The third pressure is the one that should directly move pa_cb_dsh
# off zero -- it's a training-time analogue of the eval-time shuffle
# baseline, except every chain in the batch is everyone else's
# shuffle so we get B-1 negatives per positive instead of one.
#
# Why callback-only scoring.  We mean-NLL on callback-supervision
# tokens only (with fall-back to all-valid when no callback is
# present).  Concentrates the contrastive signal on the tokens that
# require memory; LM-distribution noise on filler tokens is invariant
# to which chain's M_c we use, so including those tokens dilutes the
# contrast.
#
# Why temperature 1.0.  NLL values live in [1, 4] nats so T=1.0 makes
# softmax dynamic range natural; lower T (e.g. 0.5) is the next
# escalation if v11q shows pa_cb_dsh gain but plateaus early.
#
# Why warmup 0.05 -> 0.5 over 500 steps.  Pure InfoNCE from step 0
# can produce degenerate solutions where the readout learns to be
# chain-discriminative on TRIVIAL features (like the chain's source
# token distribution) before the writer/judge are functional.  500
# steps lets the standard P0/P1/P2 stack open the channel first
# (matches v11g's step-200/step-500 trigger schedule); the
# contrastive objective then refines the readout's content-
# discrimination on top of an already-learned baseline.
#
# Decision triggers (sharper than v11g; contrastive directly attacks
# Δ_sh-m):
#   step 200  : pa_cb_dsh > 0 AND ||m^t||/||embed|| in [0.3, 50] AND
#               nce_gap (off - diag) > 0
#   step 500  : pa_cb_dsh > +0.005 AND nce_gap > +0.05
#   step 1000 : pa_cb_dsh > +0.020 (vs v11g's +0.005 -- the contrastive
#               objective should hit this faster because we're
#               OPTIMISING this directly)
#   step 2000 : pa_cb_dsh > +0.030 AND standard Δ_sh-m > +0.005
#   step 4000 : standard Δ_sh-m > +0.010 (final calibration test)
#   KILL: step 1000 with pa_cb_dsh < 0 OR nce_gap <= 0 -> contrastive
#         failed to make the readout content-specific even with dense
#         supervision -> the architecture is the wall, not pedagogy;
#         pivot to B1/B2/B5 (per-block memory refresh / move m^t out
#         of depth softmax / two-tier memory).
#
# This is the Phase 1 #1 calibration experiment from the post-v11i
# rundown: cleanly resolves whether v3-v11g's grow-then-decay pattern
# is an architectural ceiling or a supervision deficit.
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
    --contrastive_infonce_weight 0.5 \
    --contrastive_infonce_initial_weight 0.05 \
    --contrastive_infonce_warmup_steps 500 \
    --contrastive_infonce_temperature 1.0 \
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
    --run_name chain_v11q_ap_contrastive_gh200 \
    --out_dir output/chain_v11q_ap_contrastive_gh200
