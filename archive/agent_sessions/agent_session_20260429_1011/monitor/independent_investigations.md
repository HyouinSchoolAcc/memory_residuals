# Independent investigations log

Each entry: question, method, command(s), result, takeaway.

---

## I1. Verify the GH200 process is the documented training and is alive

**Question:** Is `cwd-chain_v4_hidden14_msc` actually running the
headline recipe described in `experiments/exp2_long_horizon_recipe/README.md`?

**Method:** Pull `/proc/<pid>/cmdline` and `ps -p ... -o cmd` for the
training PID. Compare against the README launch block. Also pull the
script the watchdog actually launched.

**Commands:**
```
ssh ubuntu@192.222.50.225 'ps -p 11737 -o pid,etime,cmd | tail -5'
ssh ubuntu@192.222.50.225 'cat /proc/11737/cmdline | tr "\000" "\n"'
ssh ubuntu@192.222.50.225 'cat ~/memory_residuals/scripts/train_headline.sh'
ssh ubuntu@192.222.50.225 'cat ~/memory_residuals/paper_tools/cloud_watchdog/running/1777425221_chain_v4_hidden14_msc.json'
```

**Result:** PID 11737, ELAPSED 14:04:45. Process running. The launched
command differs from the README in eight non-trivial ways
(`window_k=3` vs 8, dropouts present, `--carry_state`,
`--save_best_metric=composite`, ...).
See `concern_recipe_drift.md` for the full table. The watchdog job
spec calls `bash scripts/train_headline.sh`; the script's *own header*
documents `window_k4` while its body has `--window_k 3` (drift even
inside the script itself).

**Takeaway:** Any next-iteration recommendation needs to specify which
recipe variant it ablates against. The writer's "soft-±4 init"
recommendation is sound but does not specify which other knobs it
inherits from `train_headline.sh`.

---

## I2. Independently parse the EVAL trajectory from step 200 → 5400

**Question:** Does the writer's EVAL table (in
`writer/findings_loss_curve.md`) match the on-host log?

**Method:** Pull the watchdog log directly, grep `EVAL @ step`, count
lines and inspect endpoints.

**Commands:**
```
ssh ubuntu@192.222.50.225 'tail -3000 ~/memory_residuals/paper_tools/cloud_watchdog/logs/chain_v4_hidden14_msc.log' \
    > monitor/training_log_1015.log
grep -c 'EVAL @ step' monitor/training_log_1015.log
```

**Result:** 16 EVAL lines visible in the last 3000 lines (full log
~9000+ lines but the older steps have been overwritten in my window;
writer reports 27 EVAL lines starting from step 200, which is
plausible at eval_every=200 over 5400 steps). Spot-checks at step 200,
2200, 5000, 5400:

| step | mem CE | nomem CE | shuffle CE | oracle CE | Δ_sh-m | Δ_or-m |
|---:|---:|---:|---:|---:|---:|---:|
| 200  (writer's table) | 3.1231 | 3.1231 | 3.1231 | 2.9174 | +0.0000 | -0.2057 |
| 2200 (my pull)        | 3.1037 | 3.1037 | 3.1037 | 2.9206 | +0.0000 | -0.1831 |
| 5000 (my pull)        | 3.0902 | 3.0902 | 3.0902 | 2.9108 | +0.0000 | -0.1793 |
| 5400 (my pull)        | 3.0894 | 3.0894 | 3.0894 | 2.9103 | +0.0000 | -0.1790 |

Endpoints match the writer's findings_loss_curve.md exactly. **Writer's
EVAL table is faithful to the source.** ✅

**Takeaway:** I trust the writer's headline numerical claim:
Δ_sh-m and Δ_nm-m are bit-zero throughout. ce_mem decays from 3.12 to
3.09 (the backbone adapting to the corpus mix); Δ_or-m sits around
-0.18 throughout (the gap memory was meant to close).

---

## I3. v3 baseline α_mem and Δ_sh-m for honest comparison

**Question:** What were the v3sp `chain_v3_softparity_full` numbers at
step 4400 (the comparable late-training checkpoint)? The v4 paper's
"after" needs an honest "before".

**Method:** Read `paper_artifacts/eval/routing_v3sp_val.json` (alpha_mem)
and `paper_artifacts/eval/chain_v3_training_summary.md` (in-trainer
Δ_sh-m).

**Result:**

| metric | v3sp @ step 4400 | v4 @ step 5400 |
|---|---|---|
| α_mem (mem) on PG-19 val   | **4.7e-4** (routing_trace) | **not yet measured**; predicted ≤ 1e-15 |
| α_mem (shuffle) on PG-19 val | 4.5e-4 | not yet measured |
| α_mem mem-vs-shuffle gap   | **5%** of α_mem (essentially zero) | not yet measured |
| in-trainer Δ_sh-m on PG-19 val | **+0.0379** | **+0.0000** |
| in-trainer Δ_nm-m on PG-19 val | +0.0131 | +0.0000 |
| in-trainer Δ_or-m | -0.1197 | -0.1790 |
| in-trainer ce_mem | 2.9342 | 3.0894 |

**Takeaway:** v4 is *worse* than v3 on the in-trainer headline metric.
v3sp had a real (small) Δ_sh-m of +0.0379 on PG-19 val at step 4400 --
not enough to defend the paper's dialogue claim, but a measurable
non-zero. v4 has Δ_sh-m of exactly zero. The paper's "v3 → v4"
upgrade as currently configured is a regression, not a progression.

The writer's framing in `findings_alpha_mem.md` --
"**stronger** statement than the v3 routing-trace finding" --
is technically correct (v4's "0" is more emphatic than v3's "5e-4")
but should not be confused with a research win. v3's 5e-4 is at least
above the bf16 floor; v4's exp(-64) is below it.

⚠ Caveat: v3 was trained on PG-19+TV (the books/TV mix), not on
PG-19+TV+MSC. So strictly v3 vs v4 is also confounded by corpus.
The cleanest before-vs-after would require a v4_embed_msc run (cell
B in the 2 × 2) on the same MSC mix. That run is queued for local
GPU per the README but not yet reported on.

---

## I4. Will the run finish on time?

**Question:** At ~step 5460 with ~2.8 k tok/s, is the run on track to
hit step 6000 within the 1-hour monitor budget?

**Method:** From the watchdog log: `step 5460 | ... | 2.8k tok/s`.
With effective batch = 4 batch * 4 grad_accum * window_k=3 *
session_len=512 = 24 576 tokens / step. So ~ 24576 / 2800 = 8.8 s /
optimizer step. Remaining 540 steps × ~8.8 s ≈ 4750 s ≈ **79 min**.

Hmm. Let me re-check: the on-line printed values say "2.8k tok/s" and
`step 6 sec/step` worth of throughput. The log also shows "1.7k tok/s"
right after each EVAL block (eval cost amortised). Also, `--carry_state`
plus 8-step burn-in increases tokens/step; effective tokens/step
include burn-in beyond the 24 576 window.

**Result:** Best-case ~50 min remaining; worst-case ~80 min. The
1-hour monitor budget might end before the run finishes. Step-5500
checkpoint should appear within ~5–10 min from now (10:21 → ~10:30).

**Takeaway:** Don't promise FINAL_REPORT.md will include step-6000
numbers. It will at most include step-5500 or step-6000 from the
log. The post-train pipeline (routing_trace etc.) will not have run
within my budget; I should not gate my final synthesis on it.

---

## I5. Watchdog history sanity check

**Question:** Has the run been restarted / OOM'd / rolled back during
the night?

**Method:** Read watchdog log, queue/running/done/failed JSONs.

**Commands:**
```
ssh ubuntu@192.222.50.225 'tail -50 ~/memory_residuals/paper_tools/cloud_watchdog/logs/watchdog.log'
ssh ubuntu@192.222.50.225 'ls ~/memory_residuals/paper_tools/cloud_watchdog/{queue,running,done,failed}'
ssh ubuntu@192.222.50.225 'cat ~/memory_residuals/paper_tools/cloud_watchdog/failed/1777425061_chain_v4_hidden14_msc.json'
```

**Result:**
- One `failed/` job: `1777425061_chain_v4_hidden14_msc.json`,
  enqueued and "started" at 01:11:01 UTC. The same job re-enqueued at
  01:13:41 and started at 01:14:02; that's the running one.
- No restarts after 01:14. So we have one ~14-hour continuous training
  episode -- no checkpoint replay, no carry-state mismatch.
- Watchdog has been quiet since `overnight_traces` finished 03:12 UTC.

**Takeaway:** The "first attempt failed at 01:11" might have been a
rapid crash (~2 min to the restart). Worth checking why if I have
time -- could be a cmdline-typo issue that caused argparse to bomb,
or an OOM at startup. But it's not affecting the current run.

---

## I6. Next checkpoint and post-train pipeline

**Question:** What runs automatically after the training process exits?

**Method:** Read `paper_tools/post_train_pipeline.sh` and the
trainer's final-step behaviour.

**Result (from earlier read of train_chain.py and README):**
- On step 6000 the trainer writes `final/` and runs a final eval.
- `paper_tools/post_train_pipeline.sh chain_v4_hidden14_msc` is the
  documented post-train pipeline; it runs `eval_chain.py` on PG-19 +
  MSC + LoCoMo, callback probe, horizon analysis, figures.
- The watchdog daemon is the same one that ran overnight_traces.sh; if
  it is configured to chain post_train onto headline completion that
  would happen automatically. The watchdog log does not show a queued
  follow-up job, so it likely needs manual enqueue.

**Takeaway:** The writer should explicitly include the post-train
pipeline as the last step in `recommended_next_run.md` -- and do so
with a note that the pipeline must include `routing_trace.py` against
the new ckpt (otherwise we won't have a v4 α_mem number to corroborate
or refute the recipe diagnosis).
