# Concern: the headline run is NOT running the README recipe

**Severity:** high. The paper's claimed experimental conditions for cell A
of the 2 × 2 factorial differ materially from the actual command on the
GH200. Any conclusion drawn from this run -- including the writer's
"hard-±32 saturates the router" conclusion -- needs to be qualified
because the run also varies four other knobs vs the documented recipe.

## Evidence

Independent verification by reading `/proc/11737/cmdline` of the
running training process and `~/memory_residuals/scripts/train_headline.sh`.
Captured to `monitor/training_cmdline.log`.

| flag | README §"Train the headline run (A)" | actual `scripts/train_headline.sh` (running) | impact |
|---|---|---|---|
| `--window_k` | **8** | **3** | **biggest issue.** The recipe paper is *about long-horizon dialogue recall*. window_k=3 means the routed memory only ever sees a chain of 3 sessions during training. The flagship eval window is 4–8. Training-eval window mismatch + much shorter training horizon. |
| `--lr` | (default; effectively 3e-4) | 2e-4 | minor |
| `--lr_backbone` | (default 3e-5) | 2e-5 | minor |
| `--warmup` | 200 | 100 | minor |
| `--memory_dropout` | not set (=0) | **0.10** | undocumented in recipe; could mask alpha_mem signal during training |
| `--context_dropout` | not set (=0) | **0.30** | undocumented; aggressive noise injected into the extract input |
| `--carry_state` | not set | **set** | TBPTT-style state carrying between minibatches; not in recipe |
| `--save_best_metric` | (default `ce_mem`) | **composite** | reasonable, but undocumented; the v3 baseline used a different metric |
| `--burn_in_max` | not set | **8** | not in recipe |
| `--eval_n_chains` | (default 24) | **32** | minor |
| `--eval_every` | 250 | 200 | minor |
| `--neg_chain_margin` | 0.05 | (default 0.05) | OK, no drift |

Even the launch script's *own header* documents `window_k4`, while its
flag is `--window_k 3`. The script body has drifted from its comment.

## Why this matters

The writer's `findings_alpha_mem.md` correctly identifies the hard ±32
init as the proximate cause of `Δ_sh-m = 0` and recommends switching to
soft init. That recommendation is sound *for the failure mode it
diagnoses*, but the diagnosis is being made against a run that also
- has 60% shorter chain unroll than the recipe (window_k=3 vs 8)
- adds 30% context dropout that the recipe does not call for
- adds 10% memory dropout that the recipe does not call for

If a soft-init rerun is launched with `--window_k 8` and dropouts removed
(the actual recipe), it may behave differently than under window_k=3 +
heavy dropouts. Or, conversely, the dropouts may be the dominant cause
of memory-channel collapse and not the hard init.

The clean question to answer next is: *which of these knobs is doing
the damage?* That can only be answered with a controlled rerun.

## What the writer says about this

The writer flagged the discrepancy in
`writer/commands_run.log` (line 6, parenthetical) and noted
"differs from README" but did not surface it as a primary finding.
Their `next_iteration_proposal.md` (not yet on disk as of this
concern) and `recommended_next_run.md` should explicitly say *which
recipe* the next run will follow: the README recipe, or the
`scripts/train_headline.sh` recipe + soft init. Without that, the
"v4 hard init falsifies the recipe" conclusion is a confound.

## Recommended next-run guardrails (independent of init choice)

1. **Pin a recipe.** Either update the README to match
   `scripts/train_headline.sh`, or update the script to match the
   README. Running both as if they are the same recipe is a paper-
   readability bug that will confuse readers.
2. **Ablate window_k separately.** A run at window_k=8 + soft ±4 init
   isolates the init effect. A run at window_k=3 + soft ±4 init mirrors
   the current run's other knobs.
3. **Drop or motivate the dropouts.** If `memory_dropout=0.10` and
   `context_dropout=0.30` are stage-2 additions, document the rationale.
   If they are leftover from an unrelated debugging session, remove them.
4. **`--save_best_metric composite` is a good default; promote it to
   the README recipe.**

## What I will not claim

I do not have evidence that the dropouts cause the routing to collapse;
the writer's softmax-saturation argument is correct on its own terms
(`exp(-64)` is below bf16 representability). The recipe drift is a
confound that does not falsify the writer's primary diagnosis -- it
just means the next run's design has to remove the confound to be
sure.
