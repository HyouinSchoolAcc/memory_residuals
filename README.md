# Memory Residuals

A fixed-size recurrent memory matrix $M_c \in \mathbb{R}^{K \times d}$
that gets updated end-to-end through the language-modelling loss — no
retrieval index, no separate memory controller, no hand-engineered
gating heuristic. Just a learned compression of past sessions that
reads into the depth-wise residual stream of a pretrained transformer.

This repository covers a multi-stage research program around that
idea. Each stage is a separate paper.

## Stages

### Stage 1 — *Memory Residuals: a drop-in recurrent memory primitive*

[`papers/paper1_drop_in_primitive/`](papers/paper1_drop_in_primitive/)
&nbsp;·&nbsp; **status: NeurIPS-bound, ~6 days to submission**

The architectural primitive: parity-preserving Block AttnRes routing
pool, two-stage QKV competition for memory updates, init-equivalent to
the bare backbone (max\|Δ_logit\| = 0.000 at step 0). The empirical
claim is *sample-efficiency at matched compute on PG-19* against three
comparable architectural memory mechanisms. **The paper does not
require dialogue results to ship.**

Headline number in hand: soft-init `attention_parity` reaches the
terminal plateau of a ReZero scalar gate in ~40% of the steps and
exceeds it from there (Δ_sh-m = +0.0379 vs +0.0249 at matched
compute on PG-19 in-trainer eval; bootstrap CIs on standalone eval
in [`paper_artifacts/eval/`](paper_artifacts/eval/)).

### Stage 2 — *A training recipe for long-horizon dialogue recall*

[`papers/paper2_long_horizon_recipe/`](papers/paper2_long_horizon_recipe/)
&nbsp;·&nbsp; **status: active research; headline run training on cloud GH200**

The architecture from stage 1 is *necessary but not sufficient* for
long-horizon dialogue recall. Stage 2 defends a four-piece training
recipe: contextual extract source (`hidden_<L>` instead of token
embeddings), mixed conversational corpus (PG-19 + TV + MSC) with
explicit per-source weighting, negative-chain contrastive loss ramped
over warmup, and init-parity-preserving router init. Headline ablation
is a 2 × 2 factorial over `{embed, hidden_14}` × `{PG-19+TV,
PG-19+TV+MSC}`; cell A (`hidden_14` × `PG-19+TV+MSC`) is currently
training on the GH200 under the cloud watchdog.

### Stage 3 — *Humanoid recall: rationalities, emotional shifts, intimacy deltas*

Speculative future work. Probes whether the learned $M_c$ tracks
character-level state (preferences, beliefs, emotional valence,
relationship distance) across sessions, not just topical or factual
recall. No paper draft.

## How "is the memory working?" gets demonstrated

A persistent reviewer concern is that $\Delta_{sh-m}$ effect sizes
are small (a few hundredths of a nat). The defense, common to both
stage 1 and stage 2, is **three independent lines of evidence per
checkpoint**, not one:

1. **Aggregate** — Δ_sh-m bootstrap 95% CI excludes zero on at least
   one held-out corpus. Already done for v2 phaseA softparity_b4 on
   PG-19 val (+0.0529 [+0.0246, +0.0915]) and PG-19 test (+0.0279
   [+0.0221, +0.0338]).
2. **Mechanistic** — α_mem (memory-source weight in the depth-wise
   routing softmax) is non-trivially > 0 at *some* sublayer × position
   on the held-out corpus. Routing-trace tool: `paper_tools/routing_trace.py`
   (TODO).
3. **Causal** — perturbing session $t-k$ (replace with another chain's
   session) measurably moves NLL on session $t$, with the effect
   curve giving an honest *horizon* in sessions. Counterfactual tool:
   `paper_tools/counterfactual_eval.py` (TODO).

Plus, when aggregate effect sizes are small, zoom into per-token
metrics: callback NLL on named entities that re-appear in the scoring
session (`paper_tools/callback_probe.py`, already implemented).

The two tier-2 tools (routing trace + counterfactual eval) are the
biggest open items for stage 1's NeurIPS readiness; both are < 100 LOC
each.

## Shared infrastructure

Code, training data, and eval pipeline are shared between stages:

```
modeling_memres.py                  architecture (config, model, init)
presets.py                          named (backbone, K, L_E, N) tuples
train_chain.py                      recurrent chain TBPTT trainer
train_phase1.py                     pair-based warm-up trainer

paper_tools/                        eval, probes, RAG baselines, parity test
paper_tools/cloud_watchdog/         remote-survivable job queue + ntfy daemon
paper_tools/cloud_handoff.sh        rsync best/ ckpts to cloud + queue eval
paper_tools/build_msc_chains.py     MSC parquet -> chain JSONLs (stage 2)
paper_tools/merge_chain_corpora.py  concat multiple .pt corpora (stage 2)
paper_tools/build_figures.py        per-paper figure generation
paper_tools/cost_analysis.py        params, FLOPs, wall-clock (stage 1, TODO)
paper_tools/routing_trace.py        alpha_mem heatmap (stage 1 + 2, TODO)
paper_tools/counterfactual_eval.py  causal sensitivity probe (stage 1 + 2, TODO)

paper_artifacts/eval/               eval JSONs, bootstrap CIs, NIAH grids
paper_artifacts/chains/             pretokenised chain corpora

scripts/                            sentinel + train_*.sh launchers (stage 2)
output/                             training checkpoints (gitignored)
logs/                               training & sync logs (gitignored)

papers/paper1_drop_in_primitive/    stage 1 manuscript + figures + tables
papers/paper2_long_horizon_recipe/  stage 2 manuscript draft + recipe doc

docs/COMPREHENSIVE.md               deep technical context, prior runs
docs/paper1_calendar.md             stage-1 6-day calendar with decision triggers
COMPREHENSIVE.md                    legacy location (kept as symlink target)
memory_residuals.{pdf,txt}          position paper
atn_residuals.pdf                   Block Attention Residuals reference paper
```

## Compute resources

- **Local.** 2 × H100 NVL (94 GB) at the lab box. ~16 h/day usable
  (residential power-down ~8 h overnight).
- **Cloud.** 1 × NVIDIA GH200 480 GB at `192.222.50.225`, rented
  ~$2.49/h × ~96 h = $240 of a $500 budget. Hosts the cloud watchdog
  and the stage-2 headline training run; survives SSH drops and lab-
  box power-offs (jobs run in detached `tmux` sessions).

## Stop everything

```bash
# local
pkill -f train_chain.py
pkill -f paper_tools/watchdog.sh

# cloud
ssh ubuntu@192.222.50.225 'tmux kill-session -t cwd-daemon; pkill -f heartbeat.sh'
```

## Reading order if you're new

1. [`memory_residuals.pdf`](memory_residuals.pdf) — the position paper.
2. [`papers/paper1_drop_in_primitive/README.md`](papers/paper1_drop_in_primitive/README.md) — what stage 1 is and isn't.
3. [`papers/paper2_long_horizon_recipe/README.md`](papers/paper2_long_horizon_recipe/README.md) — what stage 2 is going after.
4. [`COMPREHENSIVE.md`](COMPREHENSIVE.md) — every prior run, every architectural decision, every failure mode (long).
