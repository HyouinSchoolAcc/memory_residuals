# Recommended next training run

## Decision

**Run Candidate 1: soft parity init, otherwise identical to the current
strict-init headline.** Run name: `chain_v5_softhidden14_msc`.

## One-line justification

Strict init `±32` is provably bf16-saturating on the depth-router
softmax (gradient `~exp(-64)` on `mem_bias` → numerical zero); v3's
soft init `±4` is empirically trainable on the same codepath
(`α_mem` moved 7.6× off init in 4400 steps, in-trainer `Δ_sh-m =
+0.0379`). Flipping just two flags isolates the init effect from
every other recipe knob and is the cheapest, highest-prior next step.

## What this run does NOT change

I am intentionally **keeping** the current strict-init run's:
- `--window_k 3` (not 8 from the README, but matching the running config so the comparison is clean)
- `--memory_dropout 0.10`, `--context_dropout 0.30`, `--carry_state`
- `--lr 2e-4`, `--lr_backbone 2e-5`, `--warmup 100`
- `--save_best_metric composite`, `--burn_in_max 8`,
  `--burn_in_resample`, `--mask_padding_loss`, `--score_tail_frac 1.0`
- `--neg_chain_weight 0.5 --neg_chain_warmup_steps 1000
   --neg_chain_initial_weight 0.05 --neg_chain_margin 0.05`

This is **not the README recipe** (which has `--window_k 8` and no
dropouts). It is the *running headline's recipe minus strict init*.
Rationale: the cleanest scientific comparison against the strict-init
data we already have is to vary one knob (the init).

A separate ablation against the README recipe (`window_k=8`, no
dropouts) is Candidate 3 in `next_iteration_proposal.md`; it should
be run only after Candidate 1's outcome is known.

## When to launch

The current strict-init run is at step ~5500 / 6000 as of writing
and should finish within ~30 min. Do **not** launch the soft-init run
until the strict-init run has:

1. Produced its final `step-6000` checkpoint.
2. Triggered the post-train pipeline (`paper_tools/post_train_pipeline.sh
   chain_v4_hidden14_msc`) which runs the routing-trace and
   counterfactual probes — confirming `α_mem` is indeed at the
   saturation floor (~`1e-15` to `1e-26`, not just `+0.0000` in the
   in-trainer eval).
3. Been reviewed by a human; this proposal is not a blanket approval
   to launch.

The strict-init run's best ckpt + post-train pipeline output is the
"before" data for the recipe paper's strict-vs-soft init negative
result section (manuscript §"Negative result").

## Predictions (recorded so we can score them)

If Candidate 1 succeeds:

- in-trainer `Δ_sh-m > +0.005` by step 1500
- in-trainer `Δ_sh-m > +0.020` by step 4000
- post-train `α_mem^true > 1e-3` on PG-19 val (vs 1.6e-28 strict)
- post-train mem-vs-shuffle gap > +5% on PG-19 val
- post-train `α_mem^true > 1e-3` on MSC val (the v3 unknown)
- post-train counterfactual `Δ(d=ALL) > 0.010` on MSC val

If those land, the recipe paper has its headline. If `α_mem` opens
but `Δ_sh-m` on MSC val stays flat, escalate to Candidate 2 (auxiliary
α_mem loss). If `α_mem` opens *and* MSC `Δ_sh-m > 0` *but* held-out
counterfactual at depth ≥ 2 stays flat, escalate to Candidate 3
(longer training horizon).

## Run this command (DO NOT RUN WITHOUT HUMAN APPROVAL)

See `recommended_next_run.sh` for the exact bash. The change from
`scripts/train_headline.sh` is two lines:

```diff
-    --router_mem_bias_init -32 \
-    --router_recent_bias_init 32 \
+    --router_mem_bias_init -4 \
+    --router_recent_bias_init 4 \
```

and the `--run_name` / `--out_dir` change from `chain_v4_hidden14_msc`
→ `chain_v5_softhidden14_msc`.

## Safety / guardrails

- The current GH200 watchdog will pick up the new run from its queue;
  human should `paper_tools/cloud_watchdog/enqueue_job.sh
  chain_v5_softhidden14_msc bash scripts/train_headline_softinit.sh`
  *or* push `recommended_next_run.sh` and run it via the existing
  watchdog flow — exact mechanism is the operator's call.
- Estimated cost: ~14 h × $2.49/h ≈ **$35**.
- If after step 1500 the in-trainer `Δ_sh-m` is still `+0.0000` to 4
  decimals, kill the run; the soft-init theory is wrong and we need
  to rethink rather than burn the rest of the budget. (Decision
  trigger; the v3 baseline reached `Δ_sh-m = +0.0177` by step 1000,
  so anything materially below that at step 1500 is a clear failure
  signal even before factoring in the new contrastive ramp.)
