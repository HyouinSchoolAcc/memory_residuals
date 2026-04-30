# Verification: writer's `next_iteration_proposal.md` and
# `recommended_next_run.md`

Read both files at 10:30 local. Below is per-claim cross-check.

## Verified ✅

1. **"`α_mem ≈ exp(-64) ≈ 1.6e-28` at strict ±32 init."**
   Math is correct. Re-derived from `modeling_memres.py` + the launch
   command: pseudo-query is zero-init (`BlockAttnResRouter` `w` set to
   zero in `_init_memres_params`); routing softmax over N+1 sources
   gets logits `[mem_bias=-32, 0, ..., 0, recent_bias=+32]`; max-logit
   pivot is +32; mass on mem source = `exp(-32-32) / Z` where Z is
   dominated by `exp(0)` ≈ 1; ≈ `exp(-64) ≈ 1.6e-28`.

2. **"v3 init mass ≈ `exp(-8)/N ≈ 6.2e-5`."**
   The overnight_traces_writeup formula is approximate; the more
   careful version is `exp(-8) / (1 + (N-1)*exp(-4))` ≈ `1.7e-4`.
   Either way, 8–9 orders of magnitude above bf16's denormal floor,
   well within the trainable regime. **Writer's value is propagated
   from the existing writeup; the *correct* value 1.7e-4 strengthens
   their argument** (the trainable initial mass is even larger than
   they cite). Manuscript should fix the formula.

3. **Recipe-drift table.** All 11 flag-deltas in the writer's table
   match the `cmdline` I pulled from `/proc/11737`. ✅

4. **v3 in-trainer Δ_sh-m at step 4400 = +0.0379.** Cross-checked
   against `paper_artifacts/eval/chain_v3_training_summary.md` row
   "**4400**" for `chain_v3_softparity_full`. ✅

5. **v3 routing-trace `α_mem = 4.7e-4` on PG-19 val.** Cross-checked
   against `paper_artifacts/eval/overnight_traces_writeup.md`
   "Headline 1" table (4.7e-4). ✅

6. **Δ_sh-m bit-zero at every EVAL.** Cross-checked end-to-end against
   the on-host log (steps 200, 2200, 5000, 5400). ✅

## Caveats / quibbles

7. **"`exp(-32-32)` for mem_bias gradient" -- minor.**
   The gradient onto `mem_bias` itself is `α_mem (1 − α_mem) * grad_softmax_mem`,
   ≈ `α_mem` ≈ `1.6e-28` (since `α_mem` ≈ 0). Writer wrote
   "`softmax(-32)` gradient is ~`1e-14`" in `STATUS.md`, which conflates
   single-bias `softmax(-32) ≈ 1.3e-14` with the actual joint-softmax
   value `≈ exp(-64) = 1.6e-28`. The conclusion (gradient is bf16 zero)
   is the same; the number reported in the manuscript should be the
   joint value, not the single-bias one.

8. **Cost estimate for Candidate 3 (24 h).**
   At window_k=3 the running run does ~6 s/step on a GH200 (24 576 toks/
   step at 2.8 k toks/s with the eval-period dilation). At
   window_k=8, the per-step recurrent unroll is 8/3 ≈ 2.7× longer; it
   isn't a strict 2.7× wallclock multiplier because the burn-in
   amortises and bf16 attention scales sub-linearly, but expect ≥ 2×.
   So Candidate 3 is closer to **28–35 h**, ≈ $70–87 cloud cost, not
   24 h / $60. Doesn't change the recommendation order, but the
   manuscript's compute-budget table should be honest.

9. **Candidate 2 implementation cost ("30–50 LoC, 1 h to write/test").**
   Plumbing `α_mem` out of `BlockAttnResRouter.route()` with grad
   requires careful handling: the router's `route()` already returns
   `(routed, alpha_mem)` (line 1180) and the model collects an
   `alpha_trace` list (line 1184), but only when
   `collect_alpha_trace=True`. To add an aux loss on top, we need
   that path active during training, plus an aggregation reduction
   (mean over (B, S, sublayer)) that supports backward. This is
   feasible but I'd budget **2–4 h** including a sanity check that
   adding the aux term does not break init-parity at step 0.

## Disagreements (substantive)

10. **"adopt `scripts/train_headline.sh` as the canonical recipe and
   update the README to match."**
   This is the writer's pragmatic fix for the recipe-drift
   concern. I would push back: the running recipe has
   `--window_k 3`, but the paper's *title* is "long-horizon
   dialogue recall" and the paper itself declares "honest scope
   is k=8 sessions of 512 tokens each" (exp2 README,
   §"What's explicitly *not* in this paper"). Promoting `window_k 3`
   to canonical retrofits the recipe to the implementation, **not the
   other way around.** A more honest path:
   - Keep README's `window_k=8` as the canonical recipe.
   - Re-run with `window_k=8` + soft init (this is Candidate 3, not
     Candidate 1) as the *headline* attempt for the next iteration.
   - Run Candidate 1 as a *cheap diagnostic* to check whether init
     is the only issue, but do NOT promote it to the headline.
   See `concern_writer_pinning_to_short_horizon.md` for the longer
   argument.

11. **"Δ_sh-m > +0.005 by step 1500" trigger for Candidate 1.**
   This is reasonable on PG-19 val but may be too aggressive on the
   PG-19+TV+MSC val mix. v3 reached +0.0177 by step 1000 on
   PG-19+TV val. The new mix has 4000 MSC chains (much harder
   signal). I'd prefer the trigger to be:
   - Δ_sh-m > +0.002 by step 1500 (relaxed pass), OR
   - Δ_sh-m > +0.005 with α_mem > 1e-4 by step 2000 (corroborated
     pass).
   Otherwise we may kill a run that is actually moving but slower
   than v3 due to the harder corpus.

12. **No prediction for LoCoMo Δ_sh-m.** The whole reason exp2 exists
   is that v2sp's LoCoMo Δ_sh-m was within bootstrap CI of zero.
   Candidate 1's prediction list (recommended_next_run.md) covers
   PG-19 val and MSC val but **does not commit to a falsifiable
   LoCoMo result.** The paper's main external eval is LoCoMo;
   without a LoCoMo prediction, "Candidate 1 succeeded" can be
   declared without addressing the original failure mode that
   motivated the paper. **Add: post-train standalone eval on LoCoMo
   should produce Δ_sh-m bootstrap 95% CI excluding zero.**

## Net call

The writer's analysis of the failure mode (bf16 saturation at strict
init) is correct, well-argued, and supported by the v3 baseline. The
recommendation to flip to soft init is sound.

The recommendation to *also* keep window_k=3 + dropouts as part of the
canonical recipe is where I disagree most strongly. That is a separate
choice that should be made on its own merits, not folded into the
"soft init unblocks training" claim.

I will not block Candidate 1 if it gets approved -- it is a cheap
diagnostic and the highest-priority next probe -- but the FINAL_REPORT
will recommend that the paper's headline claim be defended by a run
that matches the README recipe (Candidate 3-like), not by Candidate 1.
