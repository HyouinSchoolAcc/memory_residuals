#!/usr/bin/env bash
# v7 P0 SOFTERBIAS — compression curriculum + relaxed router bias
#
# Why: v3-v6 all collapsed at the depth router (gate_max=0.0000 across
# all v6 cells through step 1080+; v3 standalone trace had α_mem ≈ 4.7e-4
# average). Two diagnoses from the failure analysis:
#
#   FAILURE 1 (router): with router_recent=+4, router_mem=-4, the depth
#       softmax starts ~saturated against memory and never relaxes. The
#       per-sublayer pseudo-query gradient is small because the LM CE
#       reward for opening memory is delivered indirectly through tokens
#       whose causal predictor is mostly the recent-block source.
#   FAILURE 2 (writer): even when memory IS opened in v3, α_mem on
#       correct vs shuffled M_c is essentially identical — the readout
#       has not learned to discriminate "right chain" from "any chain".
#       Hypothesis: M_c is mostly noise because credit assignment for
#       "compress prior session into M_c so it pays off later" is too
#       long-range to learn from chain TBPTT alone.
#
# v7 P0 attacks BOTH at once:
#
#   (a) BIAS RELAXATION — router_mem -4 -> -2 (router_recent stays +4).
#       Memory mass per sublayer at init: e^{-2}/(e^4 + 8*e^0 + e^{-2}) ≈ 0.002,
#       up from v3's ~3e-5 — about 70x lift. Still a saturated softmax
#       against recent, but enough non-trivial mass that gradient on
#       the memory pseudo-query is meaningful from step 0. (recent=+4,
#       mem=0 was discussed but routes ~1.6% per sublayer through random
#       M_c at init, which floods the residual stream; -2 is a safer
#       middle ground for the first cell.)
#
#   (b) COMPRESSION CURRICULUM PHASE 0 — every training window is
#       [evidence_session, callback_session] (window_k=2). This is the
#       absolute shortest credit-assignment chain: M_c starts fresh,
#       compresses ONE evidence session, then the next session contains
#       the callback whose answer span is the only place callback_loss_weight
#       supervision lives. If P0 can produce gate_max > 0 and Δ_sh-m > 0
#       within 500 steps, the architecture is salvageable and we promote
#       to P1 (window_k=3, +1 distractor session between evidence and
#       callback) by re-launching with --window_k 3. If P0 fails on this
#       cell AND on the v3-bias cell, the architecture story has to be
#       rewritten (next move would be simple_gate baseline on LME).
#
# Decision triggers (vs v6 GATED at matched step counts):
#   step  200: gate_max > 1e-4 (any memory channel signal at all). v6 GATED
#              was at gate_max=0.0000 here; if v7 P0 still 0.0000, the
#              bias relaxation isn't enough and we lift more next cell.
#   step  500: Δ_sh-m on lme_val > +0.005. v6 GATED was at +0.0005 / step 1000.
#   step 1000: Δ_sh-m > +0.01 AND gate_max > 1e-3. If both met, promote
#              to P1 (--window_k 3) on next launch. If only Δ_sh-m met,
#              ablate gated vs competitive at P0 next.
#
# Knob diff vs scripts/train_v6_lme_gated_callback.sh (smallest set):
#   --router_mem_bias_init        -4    -> -2     [BIAS RELAXATION]
#   --window_k                     8    -> 2      [P0 curriculum]
#   --burn_in_max                 24    -> 0      [no burn-in, clean M_c]
#   --burn_in_resample            on    -> off    [N/A with burn=0]
#   (REMOVED) --carry_state                       [fresh M_c per window]
#   --callback_window_bias         0.7  -> 0.0    [superseded by curriculum]
#   --curriculum_evidence_bias     0    -> 1.0    [NEW; pure P0]
#   --steps                       8000  -> 4000   [P0 should converge fast]
#   --eval_window                  8    -> 8      [unchanged: eval on FULL]
set -eu
cd "${REPO:-/home/exx/Desktop/fine-tune/memory_residuals}"
mkdir -p output logs
exec python -u train_chain.py \
    --preset qwen3-0.6b-large \
    --memres_mode attention_parity \
    --memres_update_mode gated \
    --router_mem_bias_init -2 \
    --router_recent_bias_init 4 \
    --memres_extract_source hidden_14 \
    --train_chains paper_artifacts/chains/lme_train_s512.pt \
    --eval_chains  paper_artifacts/chains/lme_val_s512.pt \
    --window_k 2 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --steps 4000 \
    --warmup 100 \
    --memory_dropout 0.0 \
    --context_dropout 0.0 \
    --neg_chain_weight 0.0 \
    --callback_loss_weight 10.0 \
    --callback_window_bias 0.0 \
    --curriculum_evidence_bias 1.0 \
    --burn_in_max 0 \
    --mask_padding_loss \
    --score_tail_frac 1.0 \
    --eval_every 200 \
    --save_every 500 \
    --eval_n_chains 32 \
    --eval_window 8 \
    --gradient_checkpointing \
    --save_best_metric composite \
    --run_name chain_v7_p0_softerbias \
    --out_dir output/chain_v7_p0_softerbias
