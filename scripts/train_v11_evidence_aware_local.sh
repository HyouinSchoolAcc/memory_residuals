#!/usr/bin/env bash
# v11 EVIDENCE-AWARE COMPETITION CURRICULUM + BOOTSTRAP-FIX SCALES
# (LOCAL H100 GPU 0)
#
# This is the post-v10-audit recipe -- the v10 cohort (v10a / v10b /
# 4b_mega) was killed mid-training on 2026-04-30 ~20:50 UTC after a
# careful re-read of the trainer + corpus pipeline exposed three
# *causally* independent failures (README "Stop everything"):
#
#   P0 (data, ~100x leverage). The corpus builder threw away
#       ``answer_session_ids`` (LongMemEval-S ships per-chain
#       evidence labels). The competition-curriculum sampler then
#       picked "evidence" UNIFORMLY from [0, cb_pos) -- mean
#       haystack = 47.7 sessions, mean evidence = 1.9 ⇒ 96%+ of
#       training windows had M_c built from sessions that
#       demonstrably did not contain the answer. The LM-loss-
#       optimal policy on that distribution is "ignore memory",
#       and the v3-v10 routing collapses we attributed to
#       architecture were actually the writer/judge/readout
#       converging to that policy.
#
#   P1 (gate/readout/writer chicken-and-egg). All three modules
#       multiply each other in the forward path:  h += g * m^t,
#       so gradient on g is multiplied by m^t and gradient on
#       W_V^read is multiplied by g. With g=0 at init AND
#       W_V^read=normal(d^-0.5) at init, gate sees zero gradient
#       at step 0 and stays zero at step N because it has nothing
#       to push against. Pair training was supposed to bootstrap
#       the writer/readout but it doesn't transfer to chain.
#
#   P2 (readout magnitude). MemoryReadout.out_norm is RMSNorm
#       with weight init 1.0, so ||m^t||/||embed|| ~ 73 at d=1024.
#       Useful gate operating range becomes [0, ~0.014] -- too
#       narrow for AdamW's natural step size.
#
# v11 fixes ALL THREE on a 0.6B proxy. Single coherent recipe; no
# composing five recipes simultaneously.
#
# Knob diff vs v9c (the only "alive" v9 cell):
#
#   1. Corpus: v6_lme_msc_train -> v11_lme_msc_train (same chains;
#      LME 450 re-tokenised to preserve answer_session_ids and
#      mark per-turn has_answer tokens in the callback mask).
#   2. ChainSampler.sample_window competition branch: uses
#      chain_evidence_positions when available; falls back to
#      uniform on non-LME chains. Both Sample A (KEEP-PREV) and
#      Sample B (WRITE-NEW) now sample the "evidence" slot from
#      true evidence positions and the "distractor"/"noise" slot
#      from non-evidence positions. Verified on the smoke test:
#      187/187 keep-prev anchors are real evidence vs ~3.6% pre-v11.
#   3. MemoryGate init: 0.0 -> 0.005 (positive). Gives the readout
#      a small but real forward influence at step 0; LM gradient
#      can flow back to W_V^read / writer / judge from step 1.
#   4. MemoryReadout.out_norm.weight init: 1.0 -> 0.05.
#      Downscales ||m^t||/||embed|| from ~73 to ~3.6, putting the
#      gate in [0, ~0.3] -- well within AdamW step size.
#      Verified by smoke test (build model, measure ratio).
#   5. Eval: phase-aligned eval picks evidence from
#      chain_evidence_positions and reports an extra
#      "evidence-absent floor" (CB Δ_nm-m_floor) and the
#      "evidence_lift" diagnostic = (Δ_nm-m with evidence) -
#      (Δ_nm-m without evidence). Lift > 0 means the readout is
#      content-specific to evidence-bearing M_c.
#
# Routing mode: simple_gate. Only mode that has demonstrably
# opened in any v3-v10 cell. attention_parity is an axis to test
# *after* P0 / P1 / P2 are validated -- v9d showed it's
# architecturally equivalent to dead under v9 conditions, and we
# don't want to compose two unproven knob changes simultaneously.
#
# Decision triggers (sharp, falsifiable):
#   step  200: pa_cb_dnm > +0.005  AND  ||m^t||/||embed|| in [0.3, 50]
#                                  AND  gate_max > 1e-4 (P1 fix delivers)
#   step  500: pa_cb_dnm > +0.020  AND  pa_cb_evidence_lift > +0.005
#                                  AND  gate_max > 1e-3
#   step 1000: pa_cb_dsh > +0.010  AND  pa_cb_evidence_lift > +0.010
#                                       (memory is content-specific
#                                        AND chain-specific)
#   step 2000: pa_cb_dsh > +0.020  -> ship as v11 baseline; promote
#                                     to attention_parity ablation
#                                     (v11b) and the 4B GH200 cell.
#   KILL trigger: step 1000 with gate_max < 1e-4 (P1 fix didn't take).
#                 This would mean even a positive gate init couldn't
#                 break the chicken-and-egg; rethink readout init or
#                 add an explicit retrieval-objective warm-up phase.
set -eu
cd "$(dirname "$0")/.."
mkdir -p output logs
export CUDA_VISIBLE_DEVICES=0
exec python -u src/train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode simple_gate \
    --memres_update_mode gated \
    --memres_extract_source hidden_14 \
    --memres_gate_init 0.005 \
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
    --run_name chain_v11_evidence_aware_local \
    --out_dir output/chain_v11_evidence_aware_local \
    2>&1 | tee logs/chain_v11_evidence_aware_local.log
