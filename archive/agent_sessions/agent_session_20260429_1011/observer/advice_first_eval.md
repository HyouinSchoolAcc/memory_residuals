# Advice — first-EVAL signal at step 200 (all three cells)

Time: 2026-04-29 10:55 (UTC-5).

## 1. Observation

The first in-trainer EVAL line (n=124 score positions, eval_window=4)
landed in all three v5 soft-init cells.

| cell | extract | corpus | mem CE | nomem CE | shuffle CE | oracle CE | Δ_nm-m | Δ_sh-m | Δ_or-m |
|---|---|---|------:|------:|------:|------:|------:|------:|------:|
| **v4 ref (hard ±32)** | hidden_14 | +MSC | 3.1231 | 3.1231 | 3.1231 | 2.9174 | +0.0000 | +0.0000 | -0.2057 |
| **v3 ref (soft ±4)** | embed | no MSC | 3.0859 | 3.0878 | 3.0858 | 2.9243 | +0.0019 | -0.0001 | -0.1616 |
| A (HEADLINE) | hidden_14 | +MSC | 3.1147 | 3.1142 | 3.1137 | 2.9282 | -0.0005 | **-0.0010** | -0.1865 |
| B | embed | +MSC | 3.1183 | 3.1159 | 3.1188 | 2.9294 | -0.0024 | **+0.0005** | -0.1888 |
| C | hidden_14 | no MSC | 3.1408 | 3.1397 | 3.1417 | 2.9396 | -0.0011 | **+0.0008** | -0.2012 |

The CE absolute level differs (different eval sets — A/B use the
MSC-mixed val, C uses PG-19 val only); the Δ's are intra-cell so they
*are* comparable across cells.

## 2. Interpretation

**The most important fact: in all three cells, mem != nomem !=
shuffle to 4 decimals.** This is the qualitative "the gate is no
longer bit-saturated" signal. v4 hard ±32 sat on `mem == nomem ==
shuffle` for 5400 steps. v5 soft ±4 broke that on the *first* EVAL.

That alone vindicates the writer's `findings_alpha_mem.md` thesis:
the bf16 saturation of the depth-router softmax at `±32` was the
proximate blocker. Switching from `±32` to `±4` makes the gradient
through the routing softmax representable, and even at step 200 — well
inside the contrastive-ramp warmup (cur_neg_weight ≈ 0.05 + 0.45 ×
200/1000 = **0.14**) — the routing decision distinguishes
mem / nomem / shuffle.

**Less important but worth flagging: cell A is currently the WORST
of the three on Δ_sh-m at step 200**:

- A (HEADLINE: hidden_14 + MSC) = -0.0010
- B (embed + MSC)               = +0.0005
- C (hidden_14, no MSC)         = +0.0008

This is order-of-magnitude noise (n=124, σ on a 4-th decimal CE diff
is in the high 1e-3's), so it would be wrong to read this as "cell A
is failing." But it is also not the prior the recipe paper assumes.
The HEADLINE expectation is that cell A — full recipe with both
hidden extract and MSC — should be best. At step 200, cell A is
**mildly worse than both A→B (drop hidden_14 → embed) and A→C
(drop MSC)**. Three reasons this is OK at step 200 but should be
re-checked at step 1000:

1. v3 soft-init at step 200 was also at Δ_sh-m=-0.0001 (also
   "negative-ish noise"), and went on to +0.0177 at step 1000.
   Cell A's -0.0010 is in the same envelope at this point in
   training.
2. Cell A's mem CE (3.1147) is slightly *better* than B's (3.1183)
   and C's (3.1408), so the model is fitting fine; the routing
   has just not picked up a positive signal yet.
3. Δ_or-m on A is -0.1865, on B is -0.1888 — A is slightly closer
   to oracle than B even at step 200, consistent with hidden_14
   being a richer representation.

## 3. What it means for the recipe paper

- The "init parity preservation requires hard ±32" clause of the
  v4-era recipe is effectively retracted. The headline next-iteration
  story is: *parity is preserved softly enough at ±4, and that is the
  only setting where the contrastive ramp can do anything.*
  The first EVAL is consistent with that.
- **Cell A's slight underperformance at step 200 is not yet
  interpretable.** If at step 1000 cell A is still negative or below
  cells B / C, the writer's "hidden_14 + MSC is the headline" story
  has to be re-defended — possibly with an explicit "the column
  contrast (B → A) costs nothing on books and helps on dialogue at
  long horizons, but doesn't show up at step 200" caveat.
- Cell B's Δ_sh-m=+0.0005 already exceeds v3 soft-init's step-200
  reading of -0.0001 (within noise), with the *added* contrastive
  ramp + carry_state + dropouts. So none of the v4/v5 knob additions
  appear to have *regressed* below the v3 soft-init baseline at
  matched step 200 — which is good.

## 4. Recommended action

**No action; continue.**

Specifically:

- Do **not** re-prioritise the headline based on step-200 numbers.
  This is too early.
- Step 400 is the next checkpoint. By step 400, v3 soft-init reached
  Δ_sh-m=+0.0042; if any of A/B/C is below +0.001 at step 400, that's
  the first early-but-actionable signal of trouble in that cell.
- The **decision trigger 1 (step 1500, Δ_sh-m > +0.005)** remains the
  binding test. We are far below that step right now.
- Continue passive observation cadence (every 10–15 min). No probes,
  no checkpoint loads. Cell A's GH200 has zero spare GPU; the local
  H100s are saturated by B / C.

## 5. Side note: the order *is* interesting

If by step 1000 the order is C > B > A on Δ_sh-m (as it currently is
at step 200), the recipe paper has an unexpected story: hidden
extraction matters most when there is *less* dialogue noise in the
training distribution, and adding MSC to the corpus *hurts*
in-trainer Δ_sh-m at fixed step despite being closer to the eval
target. That would be a paper-rewriting result. **Wait for step
1000 before saying anything.**
