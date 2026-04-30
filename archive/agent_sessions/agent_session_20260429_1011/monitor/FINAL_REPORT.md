# Monitor — FINAL REPORT

Session: 2026-04-29 10:11 → 10:35 CDT (15:11 → 15:35 UTC).
Author: monitor agent, working independently of the writer agent.

---

## 1. Headline GH200 run state (independently verified)

**Run:** `chain_v4_hidden14_msc` (cell A of the exp2 2 × 2 factorial),
tmux `cwd-chain_v4_hidden14_msc`, PID 11737, ELAPSED 14:04:45 (alive).
GH200, 1 process, GPU util 24–34%, 44–47 GB / 98 GB GPU memory used,
~190–280 W. No OOM, no crashes since the failed 01:11 UTC start
(the surviving run started 01:14 UTC).

**Step:** **5560 / 6000** as of 15:35 UTC. ~440 steps remain.
**ETA to step 6000:** between 11:10 and 11:40 local, depending on
eval-period dilation. Likely **after my 1 h monitor budget closes**;
this report does not include step-6000 numbers.

**Latest EVAL (step 5400):**

| metric | value |
|---|---|
| ce_mem | 3.0894 |
| ce_nomem | 3.0894 |
| ce_shuffle | 3.0894 |
| ce_oracle_concat | 2.9103 |
| Δ_nm-m (no-mem − mem) | **+0.0000** |
| Δ_sh-m (shuffle − mem) | **+0.0000** |
| Δ_or-m (oracle − mem) | −0.1790 |

Independently verified against
`~/memory_residuals/paper_tools/cloud_watchdog/logs/chain_v4_hidden14_msc.log`.
Spot-checks of step 200, 2200, 5000, 5400 match the writer's
`findings_loss_curve.md` table.

**Checkpoints:** step-500 through step-5500 saved on `--save_every 500`
cadence; step-5500 saved at 15:17 UTC. Watchdog queue empty; no
follow-up jobs queued.

## 2. Did α_mem move meaningfully vs the v3 baseline?

**No — the in-trainer signal is *more* collapsed than v3.**

| metric | v3sp (step 4400, soft ±4) | v4 (step 5400, strict ±32) |
|---|---|---|
| Routing-trace `α_mem` (PG-19 val, mem) | **4.7e-4** [measured] | **not yet measured**, predicted ≤ 1e-15 |
| Routing-trace mem-vs-shuffle gap (PG-19) | +5% | not yet measured |
| In-trainer Δ_sh-m (val) | **+0.0379** | **+0.0000** |
| In-trainer Δ_nm-m (val) | +0.0131 | +0.0000 |
| In-trainer ce_mem | 2.9342 | 3.0894 |

The strict `mem_bias=-32, recent_bias=+32` init places the routing
softmax mass on memory at `≈ exp(-64) ≈ 1.6e-28` at step 0, ~20
orders of magnitude below bf16's representable floor. The gradient
through the softmax onto `mem_bias` is thus numerically zero in
bf16, so neither the contrastive ramp (0.05 → 0.5 over 1000 steps)
nor 4400 subsequent full-strength steps could move the bias off
init. The eval `Δ_sh-m = +0.0000` is the operational confirmation:
swapping the chain's M_c with another chain's has zero measurable
effect on next-token NLL.

I have NOT run `paper_tools/routing_trace.py` against a v4 ckpt
(read-only on remote; the GH200 is busy with the live run, and 1.2 GB
checkpoint download + local re-run was outside budget). The v4 α_mem
will be measured by the post-train pipeline when the run finishes.

## 3. Where I agree with the writer

1. **Δ_sh-m = +0.0000 across all 27 in-trainer EVAL lines.** The writer's
   `findings_loss_curve.md` table is faithful to the source.
2. **Failure mechanism: bf16 saturation of the depth-router softmax under
   strict ±32 init.** Math is correct (`α_mem(0) ≈ exp(-64) ≈ 1.6e-28`).
3. **Soft init (~`±4`) is the right primary intervention.** v3sp at ±4
   reached α_mem = 4.7e-4, in-trainer Δ_sh-m = +0.0379 — the same
   codepath, demonstrably trainable.
4. **The mem CE decline 3.12 → 3.09 is backbone adapting to the corpus
   mix, not memory recruitment.** Writer correctly does not cite this
   as evidence of memory utilisation.
5. **Recipe drift exists.** The writer flagged the README-vs-script
   delta in their proposal once I raised it; their proposal docs the
   table.
6. **Candidate 1 (soft init only) is the right *first* probe.** Cheap,
   low risk, isolates one variable, has strong v3 prior.

## 4. Where I disagree with the writer

### 4.1 Promoting `scripts/train_headline.sh` to canonical recipe
(`concern_writer_pinning_to_short_horizon.md`)

The writer recommends adopting the running script as canonical and
updating the README to match. The script runs `--window_k 3`. The
paper title is *"long-horizon dialogue recall"* and the README's
honest-scope statement says "k=8 sessions". Demoting the README to
match a 3-session script narrows the paper's claim implicitly without
acknowledging the change. **I would instead make the README the source
of truth and re-run with `--window_k 8` for the actual headline.**

### 4.2 Treating Candidate 1 as the next-iteration headline

Candidate 1 (soft init, otherwise identical to running script) is a
good *diagnostic* to confirm the soft-init hypothesis. It is **not**
the right *paper-headline* run because it inherits the recipe drift
(window_k=3, 0.10 mem dropout, 0.30 ctx dropout, carry_state).
The headline run for the manuscript should be the README recipe with
soft init: window_k=8, no dropouts, soft ±4 — i.e. a Candidate 3-shaped
run (with one knob from the Candidate 3 list dropped: keep the
recipe, change init.). Run Candidate 1 first to debug, then run the
README+soft-init run for the headline.

### 4.3 Manuscript pivot to "negative result"

Writer's `findings_manuscript.md` states they wrote a 600-line draft
including a §6 "Negative result: strict ±32 init is operationally
untrainable" with the bf16 saturation analysis. I have not read the
draft (it is outside my read-only `monitor/` scope by user
constraint), but the writer's own description suggests the negative
result is being elevated to "the most concrete contribution."

This is **fine as a section** but should not become the paper's
headline. A pivot from "training recipe for long-horizon dialogue
recall" to "init-choice gotcha for parity-preserving routers" is a
substantive scope reduction. Reviewers will ask "why is this paper
about init calibration?" If the soft-init rerun (Candidate 1 or 3)
gives a positive Δ_sh-m on MSC val + LoCoMo, the paper's headline
should remain the recipe; the strict-init negative is a §6 not a §1.

### 4.4 Numerical / formula nits

- v3 init mass `exp(-8)/56 ≈ 6.2e-5` (writer + overnight writeup) is
  the rough-bound version; the more careful softmax with both biases
  in the denominator gives ≈ 1.7e-4. Doesn't change the conclusion;
  manuscript should fix the formula.
- Writer's `STATUS.md` line "softmax(-32) gradient is ~1e-14"
  conflates single-bias `softmax(-32) ≈ 1.3e-14` with the joint-
  softmax `exp(-64) ≈ 1.6e-28`. The findings file uses the joint
  form correctly; STATUS.md should be reconciled.
- Candidate 3's "~24 h on GH200" is optimistic. window_k=8 is
  ~2.5–3× wallclock at fixed batch/grad-accum due to the linear
  scaling of the recurrent unroll; expect 28–35 h.
- Candidate 2's "30–50 LoC, 1 h" implementation cost is also
  optimistic; pumping `α_mem` out of `BlockAttnResRouter.route()`
  with grad and adding an aux term (ramped) is closer to **2–4 h**
  including init-parity sanity check.

## 5. Risks the writer did NOT flag

1. **LoCoMo dialogue generalisation.** v2sp (the only ckpt with paper-
   grade bootstrap CIs so far) showed Δ_sh-m positive on PG-19 books
   but `+0.0025 [-0.0015, +0.0087]` on LoCoMo — within bootstrap CI
   of zero. The whole reason exp2 exists is that books-trained
   memory does not transfer to dialogue. **Soft init alone may
   reproduce this failure**: open α_mem on books, still flat on
   LoCoMo. Writer's prediction list (Candidate 1) covers PG-19 val
   and MSC val but commits to **no falsifiable LoCoMo prediction**.
   That's a paper-headline gap.

2. **MSC val noise.** MSC has many short sessions with EOS padding
   (which is why `--mask_padding_loss` is essential); standalone
   eval on MSC val may have larger CIs than on PG-19. The
   "Δ_sh-m > +0.005 by step 1500" trigger may be too aggressive.
   Suggest relaxing to `+0.002 by 1500` *or* requiring routing-trace
   `α_mem > 1e-4` corroboration by step 2000.

3. **`--save_best_metric=composite` interaction with α_mem=0.**
   With Δ_nm-m and Δ_sh-m both bit-zero, the composite score
   `-(Δ_nm-m + 2·Δ_sh-m) = 0.0` is a tie at every eval. Whichever
   first eval landed wins `best/`. Looking at remote `ls -la`,
   `best/` was last touched at 01:44 UTC (step 200's eval). The
   `best/` checkpoint for v4 is therefore effectively a step-200
   snapshot, which has no meaningful "best" interpretation. The
   post-train pipeline will fall back to `final/` step-6000 — fine
   in this case but worth noting.

4. **Sequential cost path: $35 → $35 → $75 = $145 minimum.**
   Candidate 1 alone is $35, Candidate 2 if needed is +$35-40,
   Candidate 3 if needed is +$70-85. The remaining cloud budget is
   ~$260 ($500 - $240 already spent). Tight but feasible. If the
   writer's Candidate 1 is approved and fails the step-1500 trigger,
   the kill-and-pivot decision needs human approval within ~2.5 h
   of launch — and there is no document in writer/ that names who
   owns that decision.

5. **Reproducibility.** README and script disagree. Anyone trying to
   reproduce will hit the wrong recipe. Whichever recipe is chosen
   as canonical, both files must be updated together.

6. **`gate_mean +0.0000` log column is the unused `MemoryGate`
   parameter under `attention_parity` mode.** Writer caught this in
   their reply (good). The training log itself is misleading — the
   logged metric is irrelevant for the running architecture. Worth a
   doc comment in `train_chain.py` so future readers don't mis-read
   the column.

## 6. My independent recommendation for next iteration

I would launch a **two-step plan**, not a single Candidate 1:

### Step A (diagnostic, ~14 h, $35) — confirm soft init unblocks
Same as writer's Candidate 1: flip ONLY `--router_mem_bias_init` to
`-4` and `--router_recent_bias_init` to `4`; otherwise identical to
the running script. **Decision triggers:**
- At step 1500: in-trainer Δ_sh-m **≥ +0.002** on the MSC-mixed val
  AND `gate_mean` (still uninformative — but) `α_mem` from a
  routing_trace dump on step-1500 ckpt **≥ 1e-4**. *Either* triggers
  pass; if *both* fail, kill and rethink.
- At step 4000: Δ_sh-m ≥ +0.015. (v3 reached +0.0365 at step 4000;
  the new mix is harder, so 0.015 is the floor.)
- At step 6000: `α_mem` from routing_trace **≥ 1e-3** on PG-19 val
  AND **≥ 1e-3** on MSC val (the novel claim).

### Step B (headline, ~30–35 h, $75–87) — if Step A passes
Re-run with the README recipe + soft init: `window_k=8`,
`memory_dropout=0`, `context_dropout=0`, `mem_bias=-4`,
`recent_bias=+4`, otherwise as the README. Carry over `--carry_state`
and `--save_best_metric composite` and `--mask_padding_loss` from
the script (these are sensible additions). This is the *manuscript's*
headline; Step A is just the proof that soft init does its job.

### Falsifiable predictions for Step B's success
- Δ_sh-m on PG-19 val standalone: **bootstrap 95% CI excludes zero.**
- Δ_sh-m on MSC val standalone: **bootstrap 95% CI excludes zero.**
- Δ_sh-m on LoCoMo standalone: **bootstrap 95% CI excludes zero.**
  (this is the originally-failing eval the recipe is meant to fix.)
- Routing-trace α_mem on LoCoMo at the best ckpt: **mean ≥ 1e-3**;
  mem-vs-shuffle gap **≥ 5%**.
- Counterfactual Δ(d=ALL) on LoCoMo: **bootstrap 95% CI excludes zero.**
  (the v3sp value was -0.0069 ± 0.0103, indistinguishable from 0.)

If any of these LoCoMo predictions fail, the recipe paper's headline
claim is not defensible and the paper should narrow scope to PG-19/
MSC val and not promise LoCoMo transfer.

### What I would NOT do as the next iteration
- Promote Candidate 2 (aux α_mem loss) above Step A. Rationale: code
  change introduces bug surface; v3 evidence shows soft init alone
  is trainable on the same codepath. Aux loss is a backup if Step A
  shows α_mem opening but Δ_sh-m staying flat (the
  "open-but-not-specific" failure), not a primary path.
- Promote `--window_k 3` to canonical. This implicitly weakens the
  paper's "long-horizon" claim.
- Treat Step A's pass as a paper-ready result. Step A is debugging.

## 7. One-paragraph summary for caller

The headline GH200 run `chain_v4_hidden14_msc` is healthy at step
~5560/6000 and will likely finish ~30 min after my budget ends, but
its in-trainer Δ_sh-m has been bit-zero for all 5400 evaluated
steps — memory is doing nothing. The proximate cause is correctly
identified by the writer as bf16 saturation of the depth-router
softmax under the strict `±32` init (`α_mem(0) ≈ 1.6e-28`,
gradient < bf16 floor). I agree with the writer's primary
recommendation to switch to soft `±4` init. I disagree with two
choices: (a) running the next iteration at `--window_k 3` rather
than the README's `8`, which weakens the paper's "long-horizon"
claim; (b) framing the next iteration as a one-knob delta from the
running script rather than as a return to the README recipe + soft
init. I would launch a cheap soft-init diagnostic at the running
script's hyperparameters to confirm the init theory, then a
README-recipe + soft-init headline run as the actual paper-shipping
attempt. The biggest risk that nobody is flagging is that soft init
alone might still leave LoCoMo Δ_sh-m within bootstrap CI of zero
(as v2sp did), which is the original failure mode that motivated
exp2 in the first place; the next iteration's success criteria
should commit to a falsifiable LoCoMo prediction.

## 8. Files written by monitor

- `STATUS.md` — running observation log (10:18, 10:23 entries).
- `concern_recipe_drift.md` — README-vs-script discrepancy.
- `concern_memory_channel_closed.md` — the bit-zero Δ_sh-m story
  with mechanism, plus the `gate_mean` log-column caveat.
- `concern_writer_pinning_to_short_horizon.md` — disagreement with
  writer's choice to keep window_k=3 in Candidate 1.
- `verification_writer_proposal.md` — per-claim cross-check of
  writer's `next_iteration_proposal.md` and `recommended_next_run.md`.
- `independent_investigations.md` — six independent probes (cmdline
  verification, EVAL-table cross-check, v3 baseline lookup, ETA,
  watchdog history, post-train pipeline plan).
- `training_log_1015.log` — verbatim tail of the on-host watchdog
  log (3000 lines).
- `training_cmdline.log` — verbatim `/proc/<pid>/cmdline` of the
  training process.
- `FINAL_REPORT.md` — this file.
