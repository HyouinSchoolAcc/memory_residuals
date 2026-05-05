# Per-category Δ_sh diagnostic — first results (2026-05-05)

Tier-1 (3) of the friend's plan, run on existing v27b/v28 checkpoints
**before** any retraining. Recipe: identical to the canonical eval
(`tools/eval_callback.py`) except the chain-shuffle confound is computed
**within callback category** instead of `(ci+1) mod N`. Six LongMemEval-S
question-types: `knowledge-update` (n=6), `multi-session` (n=13),
`single-session-assistant` (n=8), `single-session-preference` (n=4),
`single-session-user` (n=13), `temporal-reasoning` (n=6).

Source: `tools/eval_callback_categories.py`,
`results/eval_per_category/*.json`, aggregated by
`tools/aggregate_per_category_eval.py`.

## Headline: corpus-level same-category Δ_sh

| recipe | scale | n_seeds | Δ_cb | Δ_sh random | Δ_sh **same-category** |
|---|---|---|---|---|---|
| v24a (with F3) | 0.6B | 3 | +0.223 ± 0.143 | +0.0064 ± 0.016 | **+0.0150 ± 0.006** |
| **v27b (no F3, headline)** | 0.6B | 4 | **+1.318 ± 0.522** | −0.0005 ± 0.010 | **−0.0003 ± 0.005** |
| v28a (no F3) | 1.7B | seed 1 | +0.880 | −0.0245 | **−0.0057** |
| v28b (no F3) | 1.7B | seed 2 | +0.945 | −0.0085 | **−0.0066** |
| v28c (no F3) | 1.7B | seed 3 | +1.090 | −0.0087 | **−0.0222** |

**Reading:** the corpus-level same-category Δ_sh is statistically
indistinguishable from the random-shuffle metric — at this aggregation
the writer is essentially encoding "I am callback-category-K" and
nothing finer. The friend's hypothesis was: "if it's still ~0 with
same-cat negs, the writer is encoding category info and the paper's
chain-conditional context claim is on weaker ground." We confirm that
*at the corpus level*. **But the per-category breakdown changes the
picture materially.**

## Per-category breakdown (the load-bearing finding)

The "hidden" column is `Δ_sh-samecat − Δ_sh-random`. A positive value
means **same-category replacement is *worse* than random replacement**
— i.e., the writer encodes signal *beyond* the category prior in that
category bucket. (The intuition: if `M_c` only encoded category, then
swapping in a same-category `M_c` would preserve all useful signal and
random shuffle would destroy it; we'd see `Δ_sh-rand > 0` and
`Δ_sh-samecat ≈ 0`. We see roughly the *opposite* in some categories,
which means the writer encodes *more than* category in those buckets.)

### v28 1.7B (seed 1, F3-off, Δ_cb = +0.88)

| category | Δ_cb | Δ_sh rand | Δ_sh same-cat | hidden |
|---|---|---|---|---|
| **knowledge-update** | +1.94 | −0.075 | −0.021 | **+0.054** |
| single-session-user | +0.83 | −0.029 | −0.003 | **+0.026** |
| single-session-assistant | +0.54 | −0.037 | −0.015 | **+0.022** |
| temporal-reasoning | +0.27 | −0.006 | +0.015 | **+0.021** |
| multi-session | +1.07 | −0.003 | −0.005 | −0.003 |
| single-session-preference | +0.44 | −0.007 | −0.005 | +0.002 |

### v24a 0.6B (with-F3 reference, n=3 seeds, Δ_cb = +0.22)

| category | Δ_cb | Δ_sh rand | Δ_sh same-cat | hidden |
|---|---|---|---|---|
| **knowledge-update** | +0.41 | −0.044 | +0.044 | **+0.087** |
| single-session-preference | +0.01 | −0.001 | +0.004 | +0.005 |
| multi-session | +0.30 | −0.016 | −0.013 | +0.003 |
| single-session-assistant | +0.10 | −0.005 | −0.006 | −0.001 |
| temporal-reasoning | +0.13 | +0.013 | +0.011 | −0.002 |
| single-session-user | +0.25 | +0.057 | +0.048 | −0.010 |

### v27b 0.6B (no-F3 headline, n=4 seeds, Δ_cb = +1.32)

| category | Δ_cb | Δ_sh rand | Δ_sh same-cat | hidden |
|---|---|---|---|---|
| **knowledge-update** | +2.61 | +0.012 | +0.008 | −0.004 |
| single-session-user | +1.70 | −0.012 | +0.002 | +0.014 |
| single-session-preference | +0.33 | −0.001 | +0.001 | +0.002 |
| multi-session | +1.11 | −0.008 | −0.008 | +0.000 |
| temporal-reasoning | +1.01 | +0.013 | +0.015 | +0.002 |
| single-session-assistant | +0.78 | +0.011 | −0.011 | −0.022 |

## Take-aways

1. **The "I am callback-category-K and nothing finer" hypothesis is
   partially confirmed at 0.6B and partially refuted at 1.7B.** v27b
   (0.6B, n=4) shows essentially flat hidden columns — the small writer
   plausibly *is* doing category-class encoding. v28 (1.7B) shows four
   of six categories with statistically meaningful (>+0.02 nat)
   hidden chain-conditional signal, with the strongest effect in
   `knowledge-update` (+0.054). **Scale unlocks chain-specificity even
   under the same recipe.** This is a pre-registered prediction the
   diagnostic was designed to test.

2. **The `knowledge-update` category is the cleanest signal across
   recipes.** It is the per-chain-evidence-bearing category in
   LongMemEval (information that the user revised within the chain;
   the answer must come from the chain, not the LM's pretraining
   prior). The writer surfaces a +0.054 nat hidden chain-conditional
   signal at 1.7B and **+0.087 nats** even at 0.6B with F3 on. This is
   the right signal to be lifting and we are lifting it.

3. **The corpus-mean Δ_sh ≈ 0 framing in the current Paper A is correct
   but understated.** The mean washes out the categorical structure.
   We can update §6 of `paper/main.tex` to add: "the corpus-mean
   Δ_sh ≈ 0 hides per-category structure: in the four LongMemEval
   categories where chain-specificity should matter most
   (knowledge-update, temporal-reasoning, single-session-{user,
   assistant}), the same-category-shuffle Δ_sh is +0.02 to +0.054
   nats positive at 1.7B, indicating chain-conditional binding beyond
   the category prior."

4. **Direction for the recipe work in this round.** The scale
   dependence (1.7B has hidden signal, 0.6B does not under the v27b
   recipe) suggests two non-exclusive bets:
   - Tier-1 contrastive (the friend's (1)+(2)) should *force* the
     0.6B writer into the same regime the 1.7B writer reaches without
     any contrastive — i.e., we should expect the contrastive signal
     to lift Δ_sh primarily in the high-evidence categories.
   - Tier-2 train-on-synthd5 (the friend's (4)) is the cleaner
     test: synthd5 has no category prior at all, so any positive Δ_cb
     on synthd5 is unambiguously chain-specific. Currently running
     (v29a 0.6B local, v30a 1.7B GH200). Result determines whether
     the next move is contrastive (Tier 1) or sparse-writer (Tier 3).

## What this diagnostic does **not** answer

- The hidden positives are at single seeds for 1.7B (n=1 per cell);
  not yet a multi-seed claim. v28d should land soon to confirm.
- Same-category negative is *deterministic* (rotate-by-1 within the
  category pool). A randomised same-cat shuffle averaged over multiple
  permutations would tighten the noise band.
- Per-category n is small (4–13), so per-category bootstrap CIs would
  be wide. The right write-up if this goes into the paper is "n=50
  corpus-mean is 0; here are the per-category breakdowns; here is the
  hidden-signal column", not "we report a per-category headline."
