# Concern: cell B's Δ_sh-m trajectory is materially below v3 at every comparable step, and going negative

**Severity: high.** Cell B (`chain_v5_softembed_msc`) is the closest
v5-knob match to the v3 soft-init reference. If v5 cell B cannot
match v3, the v4-→-v5 transition has not just failed to *improve* the
recipe — it has actively regressed it. That makes the v3 numbers
(soft init + plain corpus + no contrastive ramp + no dropouts +
window_k=8) the more honest paper-2 baseline, and the v5 knob set
becomes *part* of the negative result rather than the path forward.

## Evidence

In-trainer EVAL (n=124, eval_window=4, score_tail_frac=1.0) on the
same shared codepath, both runs at soft `±4` init:

| step | v3 soft (`embed`, no MSC, no dropouts, no ctve, window_k=8, lr_bb=3e-6 eff.) | v5 cell B (`embed`, +MSC, mem_drop=0.1, ctx_drop=0.3, ctve 0.05→0.5, window_k=3, carry_state, lr_bb=2e-5) | gap (v3 − v5) |
|-----:|--:|--:|--:|
| 200 | -0.0001 | +0.0005 | -0.0006 |
| 400 | +0.0042 | +0.0006 | **+0.0036** |
| 600 | +0.0089 | +0.0011 | **+0.0078** |
| 800 | +0.0107 | -0.0019 | **+0.0126** |
| 1000 | +0.0177 | -0.0039 | **+0.0216** |

The gap is monotonically widening. Cell B's curve **peaks at step 600
(+0.0011), then falls and goes negative**, while v3's curve climbs
monotonically through step 4400 to its final +0.0379.

A second corroborating signal in the same direction: cell B's
Δ_nm-m (mem helps vs no mem) flipped from positive (+0.0027 at
step 400) to negative (-0.0021 at step 1000). At step 1000 in cell
B, training on a true memory chain produces a *worse* held-out
mem-conditional CE than the same model with memory ablated. That is
the "memory is harmful" failure mode, not the "memory is unused"
failure mode of v4 hard ±32. They are different failures.

**Cell C corroborates the same pattern.** C's Δ_nm-m is sharply
negative at step 400 (-0.0105) and deepens at step 600 (-0.0129),
i.e. memory becomes harmful even faster on the books-only eval set.
C and B share every dropout/contrastive/window_k/carry_state/lr knob
— they only differ in extract source and corpus. So the failure is
plausibly in the *shared* knob set, not in extract or corpus.

## Why this is a concern (not just an early-step blip)

1. The trajectory is not noisy fluctuation — it is monotonically
   declining post-step-600 with growing |Δ| and a sign flip.
2. v3 at the same step on the same codepath had no such reversal;
   it climbed monotonically.
3. Both Δ_sh-m and Δ_nm-m have flipped — independent corroboration
   from the same EVAL.
4. mem CE in cell B is fitting normally (3.12 → 3.09) while *only*
   the memory channel is degrading. This is a localised failure of
   the routing / extraction pieces, not a global training failure.
5. Cell C reproduces the Δ_nm-m sign flip on a different eval set
   (PG-19 only, no MSC), removing "MSC corpus is the cause" as a
   single explanation.

## Decision-trigger 1 will likely fail in cell B

Cell B is at step ~1040. Decision trigger 1 reads "step ~1500 with
Δ_sh-m > +0.005." Linear extrapolation from -0.0039 at step 1000 with
a *negative* slope crosses +0.005 only if the slope reverses
sharply within 460 steps — possible but on current evidence not
expected. Eval cadence is every 200 steps, so the binding readings
are step 1200 (~6 min from now), step 1400 (~12 min), step 1600
(~18 min from now). One of step 1400 or 1600 will determine the
trigger 1 outcome.

**Cell B is escalation-worthy at the next EVAL.** The kill decision
remains with the human. I am not recommending a kill in this
document; I am flagging that cell B is the strongest candidate to be
re-run with one fewer knob if the human chooses.

## Hypothesis ranking (per writer's brief)

| # | hypothesis | evidence for | evidence against |
|---|-----------|--------------|------------------|
| **H1** | `--context_dropout 0.30` is starving signal | (a) ctx_drop=0.30 means 30% of training the model can't predict from in-context tokens, so it has to use M_c. But (b) M_c at this step is essentially noise. The model learns to *ignore* M_c because trusting it on dropout-corrupted prefixes hurts more than it helps. (c) v3 had no ctx_drop and climbed monotonically. (d) Cell C also has ctx_drop=0.30 and shows the same Δ_nm-m flip. | If ctx_drop alone were the cause we'd expect a strong negative impact on mem CE itself, but mem CE is descending normally. So ctx_drop is interacting with something else, not single-handedly destroying. |
| **H2** | Contrastive ramp (peaking 0.5 at step 1000) is destabilising | The Δ_sh-m sign flip in cell B happened *exactly* at the step (~600-800) where cur_neg_weight was climbing past ~0.30, and reached its worst at step 1000 when cur_neg_weight hit its peak 0.5. Mechanistically: the margin loss compares mem and shuffle CEs through the readout; if those are nearly identical (early training), the margin-loss gradient is high-noise and pushes routing in random directions. | Cell C's Δ_sh-m has not flipped sign; C also has the same ramp. Different timing? Different magnitude? Possibly because C's eval set has higher CE absolute values (3.16 vs 3.10) and the in-trainer contrastive penalty is on training chains not held-out, so eval-set CE level isn't the cause. |
| **H3** | `lr_backbone=2e-5` co-adapts the backbone away from memory utility | (a) v3 effectively had 3e-6, we are at 7×. The LM head can drift more between memory-on vs memory-off forward passes if the backbone is moving fast. (b) But the *first* EVAL at step 200 already broke v4's bit-zero — so backbone drift from soft init wasn't necessary to open the gate. The mid-run regression is consistent with backbone drift continuing to "fix" predictions that don't need memory, hollowing out the memory niche. | If H3 alone were the cause, mem CE in cell B would be higher than nomem CE *at every step* in equal proportion to backbone movement. But mem CE drops from 3.12 to 3.09 in lockstep with nomem CE. The sign flip is in the *gap*, not the absolute. So H3 is contributory at most. |
| **H4** | `--window_k 3` is too shallow | window_k=3 means the trainer sees chains of length 3 sessions while the eval scores 4-session windows. Generalisation gap. v3 had window_k=8. | This would predict a constant sub-v3 gap, not a sign flip with growing magnitude. So H4 is probably not the *primary* cause but could explain why even soft init can't get above +0.001 in cell B; it would not by itself flip the sign negative. |
| **H5** | mem_dropout 0.10 + carry_state interact badly | When carry_state is on, M_c persists across minibatches; if 10% of the time M_c is zeroed, the next minibatch starts with a "post-zero" M_c that's a different distribution from a normally-carried M_c. The trainer doesn't separate these cases in loss accounting. | Cell C has the same dropout/carry_state and has not yet sign-flipped on Δ_sh-m, so if H5 were dominant we'd expect both cells to flip together. So H5 is plausible but not the leader. |

## Most likely diagnosis

Combining the above: **H2 (contrastive ramp) is the most likely
primary driver** because (a) the timing of cell B's sign flip
matches the ramp's mid-warmup, peaking *exactly* as cur_neg_weight
peaks; (b) the mechanism — pushing routing randomly in a regime where
mem and shuffle are nearly identical — naturally produces a *negative*
Δ_sh-m, not just a stuck-at-zero one; (c) cell C's eval set has
higher CE so the model's mem-vs-shuffle distinction is even noisier,
which would explain why C is *also* damaged (Δ_nm-m flip) just on a
slightly different schedule.

H1 (context dropout) is the most likely *secondary* driver — if
ctx_drop=0 the contrastive ramp would still inject noise, but the
model would have a healthier in-context fitting baseline to fall
back on, and the sign flip would be smaller / later.

H3 (lr_backbone) is a slow accumulator and would not explain the
*sign-flip* timing.

## Falsifiable next experiments — single-knob ablations

Each is described as a one-line diff against
`scripts/train_headline_softinit.sh`. Ranked by **cost on a freed
local H100 NVL**, cheapest first. Wallclock estimates assume
cell B's current 7.9k tok/s.

### F1 (cheapest, highest prior) — drop the contrastive ramp

```diff
- --neg_chain_weight 0.5 --neg_chain_initial_weight 0.05 --neg_chain_warmup_steps 1000 --neg_chain_margin 0.05
+ --neg_chain_weight 0.0
```

Falsifies H2. Cost: ~5 h on H100 NVL (same as cell B). If at step
1000 Δ_sh-m > +0.005 with this change alone, H2 was the dominant
cause and we keep the rest of the v5 knob set (still drop the
contrastive piece from the recipe). If still negative or stuck at
zero, contrastive isn't doing the damage — pivot to F2.

### F2 — drop context dropout

```diff
- --context_dropout 0.30
+ --context_dropout 0.0
```

Falsifies H1. Cost: ~5 h. If F1 has been run and pinned H2, run F2
*alongside* F1 (drop both) only if F1 alone wasn't sufficient.
Otherwise F2 in isolation isolates ctx_drop.

### F3 — restore v3-equivalent backbone learning rate

```diff
- --lr_backbone 2e-5
+ --lr_backbone 3e-6
```

Falsifies H3. Cost: ~5 h. Single-knob, mechanistically clean.

### F4 — drop memory dropout

```diff
- --memory_dropout 0.10
+ --memory_dropout 0.0
```

Falsifies H5 (the half about mem_drop). Cost: ~5 h. Lowest prior.

### F5 (most expensive, lowest prior) — restore window_k=8

```diff
- --window_k 3 --batch_size 4 --grad_accum 4
+ --window_k 8 --batch_size 4 --grad_accum 4
```

Falsifies H4. Cost: ~12-15 h on H100 NVL (per-step compute scales
with window_k; 8/3 ≈ 2.7× per step). Tokens-per-step also rises 8/3×,
so steps-to-target scales differently — fairer comparison is at
matched tokens, not matched steps. This is the v3 recipe exactly
(modulo MSC corpus and contrastive ramp).

## Recommended order (if a local GPU frees up)

If I were the human deciding, I would queue F1 first (drops the
contrastive ramp — single largest mechanistic suspect, cheapest, and
the contrastive piece is one of the four advertised recipe pieces, so
if it's the destabiliser the paper has to address it). If F1 doesn't
recover Δ_sh-m by step 1000, queue F2 next.

I am not running anything. The above is documentation.

## What this means for the recipe paper if cell B confirms negative

The recipe paper as currently written says the four pieces are:
1. hidden extract source
2. mixed conversational corpus
3. negative-chain contrastive ramp
4. parity-preserving init

The v4 hard-init result (alongside writer's bf16-saturation analysis)
already retracted (4) in favour of soft `±4`. If cell B confirms a
sustained negative trajectory, then **(3) the contrastive ramp may
also need to be retracted, or scoped tightly**: it is not "always
useful," it is "useful at v3-style soft init *if and only if* the
optimisation budget is large enough to overcome the early-warmup
noise it injects."

That makes the publishable recipe smaller — possibly just (1) hidden
extract + (2) MSC corpus + soft init, with the contrastive ramp
demoted to "we tried it; not in the recipe." The headline gain over
v3 would have to come from extract + corpus alone.

The cleanest possible paper-2 outcome at this point:

- v3 baseline (embed, no MSC, no contrastive, soft init): Δ_sh-m =
  +0.0379 at step 4400 on PG-19+TV val.
- v5 cell A converted-recipe (hidden_14, +MSC, no contrastive, soft
  init): if F1 lands and we re-run cell A without the contrastive
  ramp, the headline number would be a clean A-vs-v3 contrast at
  fixed init, varying only extract source and corpus.

This is a smaller, cleaner paper. I am not recommending the human
make that call yet — the current cells need to play out — but the
escalation path is being laid down here.

## What I'm tracking next

- Cell B step 1200, 1400, 1600 EVALs (decision trigger 1 evaluation
  point).
- Whether cell A (HEADLINE, hidden_14 + MSC) follows cell B into
  negative Δ_sh-m at steps ~600, ~800. Hidden_14 may or may not
  matter if the destabilising knob is shared (H2 / H1).
- Whether cell C's Δ_sh-m (currently still positive at +0.0005, but
  with an actively negative Δ_nm-m) follows cell B into Δ_sh-m
  negative territory in another 200-400 steps.
