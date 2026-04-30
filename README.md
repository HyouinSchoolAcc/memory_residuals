# Memory Residuals

A fixed-size recurrent memory matrix $M_c \in \mathbb{R}^{K \times d}$
that gets updated end-to-end through the language-modelling loss — no
retrieval index, no separate memory controller, no hand-engineered
gating heuristic. A learned compression of past sessions that reads
into the depth-wise residual stream of a pretrained transformer.

## What's active right now

**Experiment 2 — *A training recipe for long-horizon dialogue
recall*.** Four v6 cells running across local 2× H100 + GH200,
testing the four-axis pivot (LongMemEval-S corpus + gated memory
update + callback-supervised loss + callback-aligned window
sampling). Live state in
[`experiments/exp2_long_horizon_recipe/runs.md`](experiments/exp2_long_horizon_recipe/runs.md).
The recipe and bar are documented in
[`experiments/exp2_long_horizon_recipe/README.md`](experiments/exp2_long_horizon_recipe/README.md).

Stage-1 (the architectural primitive paper,
[`experiments/exp1_drop_in_primitive/`](experiments/exp1_drop_in_primitive/))
is dormant — folded in as background / baseline for stage 2.

## File map

```
memory_residuals.pdf / .txt  position paper (the architectural spec)
atn_residuals.pdf            Block Attention Residuals reference
COMPREHENSIVE.md             deep technical context, prior runs, every
                             architectural decision and failure mode

modeling_memres.py           architecture (config, model, init)
presets.py                   named (backbone, K, L_E, N) tuples
train_chain.py               recurrent chain TBPTT trainer (the active one)
train_phase1.py              pair-based warm-up trainer (stage-1 baseline)

scripts/                     active v6 launchers (4 cells)
paper_tools/                 active eval / probes / corpus builders
paper_tools/cloud_watchdog/  remote-survivable job queue + ntfy daemon

paper_artifacts/chains/      pretokenised v6 corpora (lme_*, realtalk_*, v6_lme_msc_*)
paper_artifacts/eval/        eval JSONs + bootstrap CIs that paper text cites

experiments/exp1_drop_in_primitive/    paper-1 manuscript + figures (dormant baseline)
experiments/exp2_long_horizon_recipe/  paper-2 draft, README, runs.md (active)

output/                      training checkpoints (gitignored)
logs/                        training & sync logs (gitignored)

archive/                     everything not immediately needed (KILLED v3-v5
                             scripts, old datasets, old eval JSONs, stale
                             agent notes)
```

## How "is the memory working?" gets demonstrated

Three independent lines of evidence per checkpoint, not one:

1. **Aggregate.** Δ_sh-m bootstrap 95% CI excludes zero on at least
   one held-out corpus. Done for v2 phaseA softparity_b4 on PG-19
   val (+0.0529 [+0.0246, +0.0915]) and PG-19 test (+0.0279 [+0.0221,
   +0.0338]).
2. **Mechanistic.** α_mem (memory-source weight in the depth-wise
   routing softmax) is non-trivially > 0 at *some* sublayer × position
   on held-out data. Tool: `paper_tools/routing_trace.py`.
3. **Causal.** Perturbing session $t-k$ (replace with another
   chain's session) measurably moves NLL on session $t$, with the
   effect curve giving an honest *horizon* in sessions. Tool:
   `paper_tools/counterfactual_eval.py`.

Plus per-token zoom: callback NLL on named entities that re-appear
in the scoring session (`paper_tools/callback_probe.py`).

## Compute

- **Local.** 2 × H100 NVL (94 GB) at the lab box. ~16 h/day usable
  (residential power-down ~8 h overnight).
- **Cloud.** 1 × NVIDIA GH200 480 GB at `192.222.50.225`. Hosts the
  cloud watchdog and one v6 cell; survives SSH drops and lab-box
  power-offs (jobs run in detached `tmux`).

## Stop everything

```bash
pkill -f train_chain.py
ssh ubuntu@192.222.50.225 'tmux kill-session -t cwd-daemon; pkill -f heartbeat.sh'
```

## Reference papers (top-level, on purpose)

When you find yourself nose-down in a particular run or knob and want
to step back, read these:

- [`memory_residuals.pdf`](memory_residuals.pdf) /
  [`memory_residuals.txt`](memory_residuals.txt) — the original
  position paper. The formal spec for the architecture (Eqs. 1-10,
  the two-stage QKV competition, the off-sequence depth-wise
  injection). When in doubt about *what the architecture is supposed
  to do*, this is the source.
- [`atn_residuals.pdf`](atn_residuals.pdf) — the Block Attention
  Residuals paper. The depth-wise routing pool that paper 1's
  `attention_parity` mode generalises. Read this when the routing
  trace looks weird or you're tuning router init biases.

## Reading order if you're new

1. This README — what's active and where things live.
2. [`memory_residuals.pdf`](memory_residuals.pdf) — the position
   paper. The architectural spec everything else implements.
3. [`experiments/exp2_long_horizon_recipe/README.md`](experiments/exp2_long_horizon_recipe/README.md)
   — the active paper's claim, recipe, and decision triggers.
4. [`experiments/exp2_long_horizon_recipe/runs.md`](experiments/exp2_long_horizon_recipe/runs.md)
   — which cell is doing what right now.
5. [`COMPREHENSIVE.md`](COMPREHENSIVE.md) — every prior run, every
   architectural decision, every failure mode (long; reference, not
   reading order).
