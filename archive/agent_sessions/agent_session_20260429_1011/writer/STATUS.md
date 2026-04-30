# Writer/Trainer agent — STATUS

## 2026-04-29 10:13 CDT (15:13 UTC) — initial bring-up

- Read `README.md`, `experiments/exp2_long_horizon_recipe/README.md`,
  `experiments/exp2_long_horizon_recipe/draft.tex`,
  `paper_artifacts/eval/overnight_traces_writeup.md`,
  `paper_artifacts/eval/chain_v3_training_summary.md`,
  `experiments/exp1_drop_in_primitive/manuscript.tex` (to fold in).
- SSH'd to GH200; tmux `cwd-chain_v4_hidden14_msc` is alive (pid 11737),
  but `tmux capture-pane` of pane 0 returns empty (the job uses
  `python -u` with stdout redirected to the cloud watchdog log file,
  so the pane is blank). Real log is
  `~/memory_residuals/paper_tools/cloud_watchdog/logs/chain_v4_hidden14_msc.log`.
- Pulled the log: 367 lines, last EVAL @ step 5400, last train line at
  step 5460. Throughput ~2.8 k tok/s. Latest checkpoint
  `output/chain_v4_hidden14_msc/step-5000` (14:01 UTC); step-5500
  expected within ~10 min.

## Headline observation (do not bury)

**Δ_nm-m and Δ_sh-m are exactly +0.0000 at EVERY EVAL from step 200
through step 5400.** Mem == nomem == shuffle to 4 decimals. With
`--router_mem_bias_init -32 --router_recent_bias_init 32` the depth
router is fully saturated and the contrastive ramp 0.05 → 0.5 over
1000 steps never moved it — `softmax(-32)` gradient is ~`1e-14`,
indistinguishable from numerical zero in bf16.

So v4_hidden14_msc, run as specified, is a **stronger** version of
the v3 alpha_mem≈0 failure: the hard-init router cannot be unstuck
by any of the recipe's training-time interventions. The MSC corpus
mix and contextual hidden_14 extract source are the right fixes
*if* the router were learning anything; they are doing zero work
under hard init.

This contradicts the v4 README claim that "bit-exact at step 0 and
lets the contrastive gradient drive memory recruitment from a clean
baseline." The contrastive gradient cannot escape the exp(-32)
saturation. The v3 soft-init (-4/+4) reached α_mem ~ 5e-4; the v4
hard-init shows zero in-trainer mem-vs-nomem to 4 decimals.

## Job A status: in progress

- Pulled training log and EVAL trajectory. (DONE)
- Need to dump α_mem from the live step-5000 ckpt to confirm the
  saturation reading. (NEXT — but tier-2 priority since the +0.0000
  in-trainer numbers already prove the channel is closed.)
- Drafting `findings_alpha_mem.md` and `findings_loss_curve.md` next.
- `next_iteration_proposal.md` and `recommended_next_run.md` after
  manuscript work.

## Job B status: in progress

- Read exp1 manuscript; will adapt §2 (related work) and §3 (method)
  into a "background / architectural primitive" section in exp2.
- exp2 `draft.tex` is currently 22 lines. Plan:
  intro + folded-in primitive + four-piece training recipe with
  numerical anchors from chain_v3_training_summary.md and
  overnight_traces_writeup.md.

## What's next (in order)

1. Write `findings_loss_curve.md` and `findings_alpha_mem.md`.
2. Push `draft.tex` to a real outline + 3 sections.
3. Write `next_iteration_proposal.md` (≥3 candidates with cost,
   prior, falsifiable prediction).
4. Pick best, document in `recommended_next_run.md`, prepare (do
   not run) `recommended_next_run.sh`.

## Blockers

None yet. SSH key works, log is readable, no monitor concerns yet.

---

## 2026-04-29 10:26 CDT — mid-session checkpoint

GH200 is now at step ~5540 / 6000. step-5500 ckpt landed at 15:17 UTC.
Same `Δ_sh-m = +0.0000` story; nothing has moved.

### Reply to monitor concerns

- **`concern_memory_channel_closed.md`** — monitor's gate_mean caveat
  is right and important. I have updated `findings_loss_curve.md`
  (item 6) to flag explicitly that the `gate_mean` log column is the
  unused `simple_gate` ReZero parameter and is *not* evidence on its
  own; the honest in-trainer signal is `mem == nomem == shuffle to 4
  decimals`. My manuscript draft (§"Negative result") and my
  `findings_alpha_mem.md` already only cite the EVAL `Δ` values, not
  the gate column, so this is internally consistent. No deliverable
  walkback needed.
- **`concern_recipe_drift.md`** — monitor is correct that
  `scripts/train_headline.sh` differs materially from the README
  recipe (window_k=3 vs 8; +0.10 mem dropout; +0.30 ctx dropout;
  carry_state; lr; warmup; save_best_metric; etc.). I have addressed
  this directly in `next_iteration_proposal.md` ("Confounds in the
  current run") and in `recommended_next_run.md` ("What this run
  does NOT change"). My recommended Candidate 1 deliberately keeps
  ALL of those auxiliary knobs identical to the running headline,
  flipping only `mem_bias` / `recent_bias` from ±32 to ±4. That is
  the cleanest scientific comparison. A separate ablation against
  the README recipe (window_k=8, no dropouts) is documented as
  Candidate 3 in the proposal and should be run after Candidate 1.

### Deliverables now on disk

- `STATUS.md` (this file)
- `commands_run.log`
- `findings_loss_curve.md`  (with monitor caveat folded in)
- `findings_alpha_mem.md`
- `findings_manuscript.md`
- `next_iteration_proposal.md`  (3 candidates, table of deltas)
- `recommended_next_run.md`     (Candidate 1, with predictions)
- `recommended_next_run.sh`     (NOT chmod'd, NOT executed)

### Manuscript progress

- `experiments/exp2_long_horizon_recipe/draft.tex` rewritten from
  22 lines → ~600 lines, ~3700 words, 9 sections + 2 appendices.
- Folded in §2 "Background: the architectural primitive" using the
  exp1 manuscript's content (own derivation, not imported). Sections
  include Intro, Background/Primitive, four-piece Recipe, Setup,
  Experiments (placeholder for soft-init numbers), Negative result
  (the strict-init bf16-saturation story, fully written and
  numerical), Discussion.
- Compiled cleanly with `pdflatex` (TinyTeX 2026, neurips_2024.sty
  copied from exp1). Output at
  `experiments/exp2_long_horizon_recipe/draft.pdf` (8 pages, ~300 KB).
- Citation references (`liu2025drop`, `liu2025memres`,
  `duan2025memres`) are placeholders not yet in references.bib;
  produces `Citation undefined` warnings but compiles. \todo items
  are marked in red throughout the PDF for follow-up.

### Job A complete (modulo: not running the run).
### Job B at 80%: structure + 3 sections written, headline numbers
   pending the soft-init rerun.

Returning final summary to caller next.
