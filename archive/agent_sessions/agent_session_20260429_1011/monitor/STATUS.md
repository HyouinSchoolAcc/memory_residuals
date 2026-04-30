# Monitor STATUS

Session start: 2026-04-29 10:11 local (15:11 UTC).
Monitor agent operating independently of writer.

---

## 10:18 local — first independent observation

### What I observed (sources: `tail -3000` of remote watchdog log
`~/memory_residuals/paper_tools/cloud_watchdog/logs/chain_v4_hidden14_msc.log`)

- Headline run `chain_v4_hidden14_msc` (tmux `cwd-chain_v4_hidden14_msc`)
  is alive, GPU busy.
  - `nvidia-smi`: GH200, 47306 MiB / 97871 MiB, 24% util, 191 W,
    one python PID 11737. Healthy, no OOM signature.
  - Latest step in log: **5460 / 6000** (~91% through).
  - Most recent eval visible: **step 5400**, `n=124  mem=3.0894
    nomem=3.0894  shuffle=3.0894  oracle=2.9103  Δnm-m=+0.0000
    Δsh-m=+0.0000  Δor-m=-0.1790`.
  - Throughput: ~2.8–2.9 k tok/s, dropping to ~1.7–1.8 k tok/s on the
    log line right after eval (eval cost). step time ~6 s, so ~540
    steps remaining → **~54 min wallclock to step 6000**, finishing
    ~11:10 local. Comfortable inside the 1 h monitor budget.
  - Checkpoints landed on cadence `--save_every 500`: step-500 …
    step-5000 plus `best/`. Next checkpoint expected at step-5500.
- Watchdog: queue empty; `running/1777425221_chain_v4_hidden14_msc.json`
  is the only active job; no failures since `1777425061` (an earlier
  failed attempt before the current 01:14 UTC restart). Watchdog
  daemon last logged 03:12:08 UTC (overnight_traces completed). No
  intermediate restarts since 01:14, so the current run has been
  uninterrupted ~14 h and is on its first life.

### Numerical state (v4 headline at step 5400 on PG-19+TV+MSC val)

| metric | value |
|---|---|
| ce_mem | 3.0894 |
| ce_nomem | 3.0894 |
| ce_shuffle | 3.0894 |
| ce_oracle_concat | 2.9103 |
| Δ_nm-m (no-mem − mem) | +0.0000 |
| Δ_sh-m (shuffle − mem) | +0.0000 |
| Δ_or-m (oracle − mem) | −0.1790 |

Across **every** logged eval from step 2200 → step 5400 (16 evals),
both Δ_nm-m and Δ_sh-m round to exactly +0.0000 at 4 decimal places.
ce_mem drifted 3.1037 → 3.0894 (~0.014 nat) which is the LM head
adapting to the conversational mix. Memory contributes nothing.

### Alpha_mem / gate signal

**Trap I would flag immediately if the writer reports `gate_mean=0` as
evidence.** `gate_mean` and `gate_max` in the training log are reading
`model.memory_gate.gate` (`MemoryGate`, `modeling_memres.py:262`).
That parameter is only consumed by the **`simple_gate` memres mode**
(`modeling_memres.py:1077`). In `attention_parity` mode (which the
headline is using, per `experiments/exp2_long_horizon_recipe/README.md`
and the v4 launch command) the forward pass goes through
`route_if_needed()` / `depth_router.route(...)` (lines 1176–1185,
1200+) and **never touches `memory_gate.gate`**. So a flat zero in
the train log is *uninformative* for v4 — it's just the unused
ReZero parameter sitting at its zero init.

The actual signal we care about (`alpha_mem` from
`BlockAttnResRouter`) is **not** in the training stdout stream. It
only shows up in the per-eval `BaseModelOutput.alpha_trace` collected
by `paper_tools/routing_trace.py`, which writes
`paper_artifacts/eval/routing_*.json`. We have v3 baseline numbers
there but no v4 numbers yet.

### Why the eval Δ's are bit-zero

`mem_bias_init=-32` in the v4 recipe (vs `-8` in v3). At init,
alpha_mem ~ exp(−32)/N ≈ 4e-15, so memory enters the routed value
pool with effectively zero weight. Even a 4-order-of-magnitude move
in alpha_mem during 5000 steps would still leave it < 1e-10 — far
below where the routed memory term has any measurable effect on
NLL. This is consistent with `Δ_sh-m = +0.0000` being printed at
4-decimal precision: the routing softmax is so far in the saturated
tail that the contrastive gradient through `alpha_mem` is vanishing.

This is **the** central concern, see
`concern_memory_channel_closed.md`.

### Where I expect the writer might mislead

1. Reporting in-trainer `gate_mean ≈ 0` as evidence of memory
   collapse on a v4 run when in fact the gate is unused.
2. Treating eval `ce_mem` decline (3.10 → 3.09) as "the model is
   learning" without separating LM-head adaptation to MSC token
   distribution from any memory-routed gain. Δ_sh-m is the only
   honest signal and it's zero.
3. Recommending a higher `--neg_chain_weight` as the fix, when the
   underlying problem is more likely the `mem_bias_init=-32`
   saturation. A larger margin loss does not help if the
   alpha_mem gradient is vanishing through the softmax.

### Next monitor actions

- Pull `paper_tools/routing_trace.py` and read its CLI; see if I can
  point it at the latest checkpoint (step-5000) and obtain a v4
  alpha_mem number to compare against the v3 0.00047 baseline. This
  is the cleanest independent measurement.
- Re-pull the watchdog log every 10–15 min and parse the latest
  step / loss / Δ_sh-m.
- Read writer's `STATUS.md` once it appears and corroborate / push
  back per their numerical claims.

---

## 10:23 local — second observation + cross-check

### What's new on the GH200

- step-5500 checkpoint was saved at 15:17 UTC (just now). Latest
  step in log: **5520 / 6000**. About 480 steps remain. At ~6–9 s
  per optimiser step (with eval cost amortised), the run finishes
  in **50–70 min wallclock**, i.e. somewhere between 11:13 and
  11:33 local. **Run is likely to finish AFTER my 1 h monitor
  budget ends at ~11:11.**
- Most recent EVAL @ step 5400 (no new EVAL since step 5400 yet;
  next EVAL at step 5600 expected in ~15 min):
  `mem=3.0894 nomem=3.0894 shuffle=3.0894 oracle=2.9103
  Δnm-m=+0.0000 Δsh-m=+0.0000 Δor-m=-0.1790`. Still flat zero on
  the headline contrasts.

### Independent verification: actual training command vs README

I pulled `/proc/11737/cmdline` and `scripts/train_headline.sh`. The
running command **deviates from the README documented launch in 8
ways**: `--window_k 3` (not 8), `--memory_dropout 0.10`,
`--context_dropout 0.30`, `--carry_state`, `--save_best_metric
composite`, `--lr 2e-4` / `--lr_backbone 2e-5`, `--warmup 100`,
`--burn_in_max 8`. The script's own header even says `window_k4`
while its body has `--window_k 3`. See
`monitor/concern_recipe_drift.md`.

The writer noticed this in their `commands_run.log` (line 6) but did
not surface it as a finding; their `findings_*.md` files implicitly
treat the run as the README recipe.

### Where I agree with the writer

1. **Δ_sh-m and Δ_nm-m are bit-zero across all 27 EVALs.** Verified
   independently against the same log. Writer's
   `findings_loss_curve.md` is a faithful render of the source.
2. **Hard `±32` init saturates the depth-router softmax.** Writer's
   math (mass on memory ≈ exp(-64) ≈ 1.6e-28 at init, well below
   bf16 representability) is correct; the contrastive ramp can't
   move `mem_bias` because `softmax(-32)` ≈ 0 to float precision.
3. **Switching to soft init (~`±4`) is the right primary
   intervention.** v3sp at `±4` reached α_mem ~ 4.7e-4 on PG-19 val
   per `routing_v3sp_val.json`; that's small but not zero. Soft init
   gives a non-vanishing gradient.
4. **The `mem CE` decline 3.12 → 3.09 is backbone-on-corpus, not
   memory recruitment.** Writer correctly separates these.

### Where I disagree / push back

1. **Recipe drift is a confound the writer hasn't surfaced.** The
   diagnosis "hard ±32 saturates the router" is true on its own
   terms, but is being read off a run that also varies window_k,
   carry_state, and dropout. Soft-init alone may not be sufficient
   if the dropouts also choke memory. (`concern_recipe_drift.md`).
2. **`gate_mean` log column is the unused `MemoryGate.gate`
   parameter for `attention_parity` mode.** Writer's findings don't
   make this mistake explicitly, but a paper reader looking at the
   log would. The honest in-trainer signal is the EVAL `Δ` columns,
   not the per-step `gate_mean`. (`concern_memory_channel_closed.md`,
   "Caveat the writer should note".)
3. **v3 init mass calc.** The overnight writeup says
   `exp(-8)/56 ≈ 6.2e-5` at v3 init. Writer's `findings_alpha_mem.md`
   reproduces this. The actually-correct calc with both `±4` biases
   in the softmax denominator is closer to 1.7e-4. Doesn't change
   the conclusion (3 orders of magnitude > v4's 1.6e-28) but the
   manuscript should fix the overnight-writeup formula.
4. **"v4 hard-init result is not a wasted run, it's the empirical
   case for soft init" framing.** This is too soft. The v4 run was
   the *headline* of the paper. Calling its failure a successful
   ablation post-hoc is OK if and only if the next iteration ships
   with a passing α_mem. It is not OK as the manuscript's headline.

### What I still need to do

- Read writer's `next_iteration_proposal.md` and
  `recommended_next_run.md` once they exist and cross-check whether
  the proposal addresses recipe drift, not just init.
- Pull EVAL for step 5600 when it lands.
- Write FINAL_REPORT.md by ~11:00 local.

---

## 10:35 local — verification of writer's proposal + final synthesis

### Writer's Candidate 1 (TOP) summary

Two-flag delta from the running script: `--router_mem_bias_init -4
--router_recent_bias_init 4`, otherwise identical (window_k=3,
dropouts kept, carry_state kept). Predictions: in-trainer Δ_sh-m
> +0.005 by step 1500, > +0.020 by step 4000; routing-trace α_mem
> 1e-3 on PG-19 val and MSC val.

### Verified ✅
- bf16 saturation math (`α_mem(0) ≈ exp(-64)`).
- Recipe-drift table.
- v3 baseline numbers (4.7e-4, +0.0379).
- EVAL-table fidelity.

### My disagreements (filed as concerns)
1. Promoting `scripts/train_headline.sh` to canonical (with window_k=3)
   silently demotes the paper's "long-horizon" claim. See
   `concern_writer_pinning_to_short_horizon.md`.
2. Candidate 1 inherits all 8 recipe-drift knobs. It is a *diagnostic*,
   not a paper-headline run. Headline should match README + soft init.
3. The writer's prediction list does not commit to a LoCoMo result.
   LoCoMo is the eval that motivated this paper's existence; without
   a falsifiable LoCoMo prediction, "Candidate 1 succeeded" can be
   declared without addressing the original failure mode.
4. Numerical nits in writer's docs (joint vs single-bias softmax,
   Candidate 3 cost estimate, Candidate 2 LoC estimate). See
   `verification_writer_proposal.md`.

### Final report

`monitor/FINAL_REPORT.md` written at 10:35 local. Contains:
- Independently verified state of GH200 run.
- α_mem v3-vs-v4 comparison (with the v4 number "not yet measured"
  caveated).
- Per-section agreement / disagreement with writer.
- Risks the writer did not flag (LoCoMo, MSC noise,
  `save_best_metric=composite` ties at zero, sequential cost,
  reproducibility, `gate_mean` log-column trap).
- A two-step recommendation (cheap soft-init diagnostic, then
  README-recipe + soft-init headline run) with falsifiable LoCoMo
  predictions.

Final summary delivered to caller.
