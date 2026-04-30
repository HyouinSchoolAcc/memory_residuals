# Next-iteration training proposal

**Author:** writer agent, 2026-04-29 10:35 CDT
**Headline run referenced:** `chain_v4_hidden14_msc` (GH200, step ~5500/6000 as of writing).
**Headline finding:** `Δ_nm-m = Δ_sh-m = +0.0000` to 4 decimals at every
EVAL from step 200 → 5400. The depth router is saturated by the strict
`mem_bias=-32 / recent_bias=+32` init: the bf16 gradient through the
softmax onto `mem_bias` is `~exp(-64) ≈ 1.6e-28`, well below bf16's
`~6e-8` representable floor. The contrastive ramp 0.05 → 0.5 over
1000 steps cannot escape this saturation.

---

## Confounds in the current run (must be addressed by the next run)

The actual `scripts/train_headline.sh` differs from the
`experiments/exp2_long_horizon_recipe/README.md` "Train the headline run (A)"
block in several knobs (also raised by the monitor in
`monitor/concern_recipe_drift.md`):

| flag | README says | scripts/train_headline.sh runs |
|---|---|---|
| `--window_k` | 8 | **3** |
| `--lr` | (default 3e-4) | 2e-4 |
| `--lr_backbone` | (default 3e-5) | 2e-5 |
| `--warmup` | 200 | 100 |
| `--memory_dropout` | unset (=0) | **0.10** |
| `--context_dropout` | unset (=0) | **0.30** |
| `--carry_state` | unset | set |
| `--save_best_metric` | unset (`ce_mem`) | `composite` |
| `--burn_in_max` | unset | 8 |
| `--eval_n_chains` | (default 24) | 32 |
| `--eval_every` | 250 | 200 |

The next iteration should pin one recipe and stick to it. My
recommendation is to **adopt `scripts/train_headline.sh` as the
canonical recipe** (it incorporates real-world experimentation since
the README was written) and update the README to match, but in this
proposal each candidate is described as "a one-flag delta from the
current strict-init run" so the next iteration cleanly attributes the
effect of the change.

---

## Three candidate next-iteration runs

For each candidate I give the prior, the cost on GH200, the
falsifiable prediction, and what evidence in the current logs supports
the prior.

### Candidate 1 (TOP RECOMMENDATION) — soft parity init, otherwise identical

**Delta from current run.** Only two flags change:
`--router_mem_bias_init -4` and `--router_recent_bias_init 4`
(was `-32` / `+32`). Everything else (`hidden_14`, MSC mix,
contrastive ramp, dropouts, `window_k=3`, `carry_state`, `lr=2e-4`)
is unchanged from the running headline.

**Prior.** The v3 `chain_v3_softparity_full` run with `mem_bias=-4`
moved `α_mem` from the analytic init floor `exp(-8)/N ≈ 6.2e-5` to a
measured `4.7e-4` at step 4400 on PG-19 val (overnight_traces_writeup.md),
and produced in-trainer `Δ_sh-m = +0.0379` at step 4400
(chain_v3_training_summary.md). The bf16 gradient at `mem_bias=-4` is
`~6e-5` — comfortably representable. Soft init has been *empirically
demonstrated to be trainable on this codepath* on the legacy v3
configuration; the question is whether the v4 recipe additions
(hidden_14 + MSC + contrastive) produce a *better* result on top of
soft init, or whether they are inert.

**GPU-h cost.** ~14 h on the GH200 (single seed, 6000 steps,
~$35 cloud cost). Same as the current headline.

**Falsifiable prediction.**
- In-trainer `Δ_sh-m > +0.005` (positive, not noise) by step 1500.
- In-trainer `Δ_sh-m > +0.020` by step 4000.
- Held-out routing-trace `α_mem > 1e-3` on PG-19 val at the best
  checkpoint (vs 1.6e-28 nominal under strict init).
- Held-out routing-trace `α_mem > 1e-3` on **MSC val** — this is the
  novel claim of the v4 recipe vs v3. v3 on PG-19+TV showed alpha_mem
  open on books only; if hidden_14 + MSC + contrastive helps, alpha_mem
  on MSC val should be in the same order of magnitude as on PG-19 val.

**Failure modes.**
- If `Δ_sh-m` is positive on PG-19 val but still zero on MSC val, then
  hidden_14 + contrastive is necessary but not sufficient for
  dialogue, and we need a stronger contrastive (Candidate 2) or to
  switch the loss form entirely.
- If `Δ_sh-m` is positive on both PG-19 val and MSC val, the recipe
  works and Paper 2 has its headline.

**Evidence in current logs supporting this candidate.**
- `chain_v4_hidden14_msc` shows `Δ_sh-m = +0.0000` at every step,
  consistent with bf16 saturation at `mem_bias=-32`.
- `chain_v3_softparity_full` (same architecture, soft init, no
  hidden_14, no MSC, no contrastive) reached `Δ_sh-m = +0.0379` on
  PG-19 in 4400 steps. Soft init was demonstrably enough to open the
  channel on books.
- `α_mem` measured directly at step 4400 of v3 was `4.7e-4` (PG-19
  val); positive mem-vs-shuffle gap of +4.79%.

**Risk.** Low. This is the highest-prior candidate; the only risk is
that the contrastive ramp + hidden_14 combine in a way we did not
anticipate. The v3 baseline gives strong empirical support that soft
init is trainable.

### Candidate 2 — soft parity init + auxiliary memory-utilisation loss

**Delta from Candidate 1.** Add a new auxiliary term to the loss that
directly penalises α_mem-near-zero. Concrete proposal:

```
L_util = - log(mean_pos_sublayer(α_mem))    # collected from depth_router.route()
total_loss += w_util * L_util               # w_util ramped 0 → 0.1 over 2000 steps
```

This requires a small code change in `train_chain.py` to expose
`alpha_mem` from the model forward and add it as an auxiliary
gradient. About 30–50 lines of code; the routing trace already
reports α_mem so the plumbing exists.

**Prior.** Even with soft init, the contrastive loss only generates
gradient signal when shuffle-NLL > match-NLL by less than the margin.
If the model has not yet recruited memory at all, `loss_match ≈
loss_shuffle ≈ baseline` and the contrastive penalty is `m_neg = 0.05`
constant (clamped at the margin), giving zero useful gradient.
Adding an explicit `-log(α_mem)` term creates an
*always-on* signal that the router is being graded on memory mass,
independent of whether the contrastive comparison is informative
yet. This is the textbook "ignore-memory equilibrium escape" trick.

**GPU-h cost.** ~14 h on the GH200 + ~1 h to write/test the loss term.

**Falsifiable prediction.**
- α_mem reaches `> 5e-3` by step 2000 (5x faster than Candidate 1).
- `Δ_sh-m` matches Candidate 1's by step 4000.
- BUT: if α_mem opens up before the contrastive signal becomes
  meaningful, the model may end up routing to memory content that is
  not chain-specific — i.e. Δ_sh-m may stay flat while α_mem opens.
  This is the worst-of-both-worlds outcome.

**Failure modes.**
- The aux loss may pull α_mem open without inducing chain-specific
  representations. The cleanest sanity check: monitor (Δ_sh-m on
  MSC val) per eval; if α_mem opens and Δ_sh-m stays flat, kill the
  run and try Candidate 3.
- Implementation bug: the routing softmax is computed inside the
  Block AttnRes router; piping `alpha_mem` out as a tensor with grad
  requires reading `modeling_memres.py:1136` ff. carefully.

**Evidence in current logs supporting this candidate.**
- The current strict-init run shows the contrastive loss alone is
  insufficient to escape saturation.
- v3 soft-init reached `α_mem = 4.7e-4` at step 4400; an aux loss
  could plausibly push this 10x further.

**Risk.** Medium — code change introduces a new bug surface, and the
"open α_mem without inducing specificity" failure is a real worry.

### Candidate 3 — soft parity init + window_k=8 + drop heavy dropouts

**Delta from Candidate 1.** Three flag changes:
- `--window_k 8` (was 3 in script; matches README recipe).
- Drop `--memory_dropout 0.10` and `--context_dropout 0.30` (set to 0).
- Keep `--carry_state` (it's a separate concern; carry-state propagates
  detached state across minibatches, is reasonable to keep).

**Prior.** The recipe paper is *about long-horizon dialogue recall*,
yet the run trains on chains of only 3 sessions. Doubling the
training horizon to 8 sessions matches the scoring window and
gives the recurrent state more chances to be exercised. The
30% context dropout on the extract input is aggressive — it zero-
masks 30% of the prefix tokens that feed the extract source, which
may starve the extractor of the contextual signal that hidden_14 was
introduced to provide.

**GPU-h cost.** ~24 h on the GH200 (window_k=8 is ~2× slower than
window_k=3 because the per-step recurrent unroll dominates wall-clock;
3 sessions × 4 batch × 4 grad-accum = 48 sessions/step at window=3 vs
128/step at window=8). About $60 cloud cost.

**Falsifiable prediction.**
- α_mem opens at the same rate as Candidate 1 (init effect dominates).
- BUT in-trainer `Δ_sh-m` (which scores at eval window 4) should be
  *smaller* on PG-19 val and *larger* on MSC val than Candidate 1,
  because the longer training window forces the model to use memory
  for genuinely cross-session content rather than the very-recent
  prior session that window_k=3 effectively encodes.
- Held-out counterfactual `Δ(d=8) > 0` (this is in the noise for v3;
  Candidate 1 alone is unlikely to fix it; longer training horizon is).

**Failure modes.**
- 24 h is more than the current cloud budget allows for a single
  ablation.
- If the dropouts were actually load-bearing for some other reason
  (e.g. preventing the extractor from collapsing onto the most-recent
  token), removing them might destabilise extract.

**Evidence in current logs supporting this candidate.**
- The launch script header says `window_k4` while the actual flag is
  `window_k 3` — there's already an inconsistency in the codebase
  about what the right window is.
- The recipe paper's own scope claims "k=8 sessions of 512 tokens
  each" (README §"What's explicitly *not* in this paper").

**Risk.** Medium — runs longer and changes more variables at once.
Cleanest as a *third* iteration after Candidates 1/2 give a baseline.

---

## Candidates considered and rejected

- **Larger K, deeper L_E, more N blocks.** None of the in-trainer
  evidence points at a memory-capacity bottleneck; the channel is
  closed at the input, not at the bottleneck. Postpone.
- **Different extract layer (hidden_8 vs hidden_14 vs hidden_20).**
  Worth ablating eventually, but not until α_mem is known to be > 0.
  Postpone to post-Candidate-1.
- **Curriculum on chain depth / window_k.** Mechanically sound, but
  the simpler fix (Candidate 1) has stronger prior support. Postpone.
- **Higher contrastive ramp (target weight > 0.5, longer warmup).**
  Doesn't help under strict init (the gradient through the softmax is
  bf16 zero regardless of margin weight). Could marginally help under
  soft init but the v3 evidence shows soft init alone is trainable
  without an extreme contrastive weight. Skip.

## Comparison summary

| Candidate | Δ from current | GPU-h | Prior strength | Falsifiable signal | Risk |
|---|---|---|---|---|---|
| 1 (TOP) — soft init | flip 2 flags | ~14 h | strong (v3 ran successfully) | α_mem > 1e-3 + Δ_sh-m > 0.02 by step 4000 | low |
| 2 — soft init + α_mem aux | flip 2 flags + ~50 LoC | ~15 h | medium (no prior on this codepath) | α_mem > 5e-3 by step 2000 | medium |
| 3 — soft init + window_k=8 + drop dropouts | flip 5 flags | ~24 h | medium (changes 3 vars at once) | longer-horizon `Δ(d=8)` > 0 | medium |

**Recommendation:** run Candidate 1 first; if α_mem opens but Δ_sh-m
on MSC val stays flat, run Candidate 2; if Candidate 1 succeeds on
PG-19 but is mediocre on dialogue, run Candidate 3 to clarify the
horizon dependence. Each iteration's outcome dictates the next.

See `recommended_next_run.md` for the full launch command for Candidate 1.
