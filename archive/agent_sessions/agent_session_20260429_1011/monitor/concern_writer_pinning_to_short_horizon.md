# Concern: writer's recommendation pins paper headline to a short-horizon recipe

**Severity:** medium-high. Independent of the soft-vs-strict init
debate, the writer's `recommended_next_run.md` (Candidate 1) keeps
`--window_k 3` and proposes adopting `scripts/train_headline.sh` as
the canonical recipe. Both choices weaken the paper's headline
claim.

## What the paper claims (per docs)

- Title (exp2 README §"This paper claims"): *"a training recipe for
  long-horizon dialogue recall."*
- Honest-scope statement (exp2 README §"What's explicitly *not* in
  this paper"): *"k=8 sessions of 512 tokens each (4096 tokens of
  effective in-stream context, augmented by M_c)."*
- README launch block (exp2 README §"Train the headline run (A)"):
  `--window_k 8`.

## What is actually running

`/proc/11737/cmdline`: `--window_k 3`. `--eval_window 4`.

So the model is being trained on chains of 3 sessions and evaluated on
chains of 4 sessions. The recipe's claimed scope is 8 sessions.

## Why this matters

1. **Mechanistic.** A recurrent state's job is to *survive interference*
   from intervening sessions. With window_k=3, the model only sees
   chains where one session's content needs to survive at most 1
   interfering session. Generalisation to 8-session chains is not
   trained for. The cleanest analogy: TBPTT on RNNs at chunk length
   3 is known to under-train long-horizon dependence.

2. **Rhetorical.** "Trained on 3-session chains, evaluated on 4-
   session chains" is awkward to defend in front of a reviewer
   asking *"why call this long-horizon?"*. The honest framing of the
   paper at that point is closer to *"a training recipe for 3-session
   dialogue recall"*, which is much less compelling.

3. **Reproducibility.** The README and the script disagree. The
   script's own header comment says `window_k4` while the body has
   `--window_k 3`. Anyone reading the README and trying to reproduce
   will hit the wrong recipe.

## Writer's stated rationale for Candidate 1's window_k=3

> "the cleanest scientific comparison against the strict-init data we
> already have is to vary one knob (the init)."
> — `recommended_next_run.md` §"What this run does NOT change"

This is sound *as a debugging step*: yes, varying one knob isolates
the cause of the strict-init failure. But the resulting checkpoint is
a debugging artefact, **not the paper's headline.** Treating it as
the headline ("a separate ablation against the README recipe is
Candidate 3, after Candidate 1's outcome is known") inverts the
priority order: the headline claim is being deferred behind a
debugging probe.

## What I would recommend instead

- **Run Candidate 1 *as a diagnostic*** (cheap, one-flag delta) to
  confirm that soft init unblocks training in the running config.
  Do not write its results into the headline of the manuscript.
- **Run Candidate 3 (window_k=8 + soft init + drop dropouts) as the
  paper's actual headline,** even at ~$75 cost. This is the recipe
  the manuscript describes.
- **If Candidate 1 succeeds (α_mem > 1e-3, Δ_sh-m > 0.02 on PG-19
  val) but Candidate 3 fails (e.g. window_k=8 + no dropouts is
  actually unstable),** that's a valuable result that should be
  reported as-is — but the paper's claim must then narrow to match
  what was actually demonstrated.
- **Update the README and `scripts/train_headline.sh` together** so
  they describe the same recipe. If `window_k=3 + dropouts` is the
  empirically best we can do, the README's scope statement
  ("k=8 sessions of 512 tokens each") must be revised down.

## What I will not claim

- I do not have evidence that window_k=8 will succeed; only that the
  paper's claim is about long-horizon and the chosen recipe is short-
  horizon. The next iteration's design should make a deliberate
  choice, not implicitly accept the drift.
- The writer is acting reasonably under "isolate one knob" doctrine.
  My disagreement is about which knob is the headline, not about the
  scientific care of the recommendation.
