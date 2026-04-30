# Paper 1 — Memory Residuals: a drop-in recurrent memory primitive

## What this paper claims

A fixed-size recurrent memory matrix
$M_c \in \mathbb{R}^{K \times d}$ can be strapped onto a pretrained
transformer backbone such that:

1. **Init-equivalent.** At step 0 the augmented model is *bit-exactly*
   identical to the bare backbone (max\|Δ_logit\| = 0.000 on
   `attention_parity` and `simple_gate`-with-gate-zero modes). No
   destructive change to the pretrained weights.
2. **Cheap.** O(K·d) extra parameters; one cross-attn per session
   boundary at write time; one K-row cross-attn per token at read
   time. Wall-clock throughput (10.8 k tok/s on H100 at 0.6B) is
   indistinguishable from the bare backbone.
3. **Trainable end-to-end with vanilla LM loss.** No retrieval head,
   no auxiliary objectives, no curriculum gymnastics. Plain next-token
   NLL across a chain of $k$ sessions, with $M_c$ updated recurrently
   between sessions.
4. **Sample-efficient vs comparable architectural memory.** The
   soft-init `attention_parity` routing pool reaches the terminal
   plateau of a ReZero-style scalar gate (the closest comparable
   parametric memory mechanism) in roughly **40% of the training
   steps**, and exceeds it from there.

The paper is **not** about long-horizon dialogue recall. The training
recipe needed to land long-horizon recall on conversational benchmarks
is the subject of [experiment 2](../exp2_long_horizon_recipe/README.md).

## Empirical bar

We need three independent lines of evidence that the primitive uses
prior-session information at read time:

| evidence | answer to | tool | status |
|---|---|---|---|
| Δ_sh-m bootstrap CI excludes 0 | "is memory *this chain's* memory?" | `paper_tools/eval_chain.py` + `paper_tools/bootstrap_ci.py` | done on v2 phaseA softparity_b4 (PG-19 val: +0.0529 [+0.0246, +0.0915]; PG-19 test: +0.0279 [+0.0221, +0.0338]); need same on v3 ckpts |
| α_mem heatmap is non-trivial in some sublayer | "is memory actually being read?" | `paper_tools/routing_trace.py` | not yet written; ~80 LOC |
| ΔNLL on session t responds to perturbation of session (t−k) | "are reads causally used downstream?" | `paper_tools/counterfactual_eval.py` | not yet written; ~100 LOC |

Plus three supporting tables:

| table | what | source |
|---|---|---|
| Init parity | max\|Δ_logit\| at step 0 across modes | `paper_artifacts/eval/init_parity_test.json` (done) |
| Init-parity preservation | max\|Δ_logit\| at steps 0, 1000, 2000, 4400 | re-run `init_parity_test.py` against intermediate ckpts (~30 min) |
| Cost analysis | params, FLOPs/token, wall-clock | `paper_tools/cost_analysis.py` (~50 LOC, not yet written) |

## Headline trajectory (already in hand)

Δ_sh-m vs training step on PG-19 in-trainer eval, n = 92, four cells
at matched compute (PG-19 + TV training corpus, embed extract source,
no auxiliary loss, seed 42, identical hyperparameters):

| mode | best step | Δ_nm-m | Δ_sh-m |
|---|---:|---:|---:|
| `simple_gate` (ReZero scalar gate) | 5200 (terminal plateau) | +0.0090 | +0.0249 |
| `attention_base` (uniform softmax over deltas) | 4400 (still climbing) | +0.0065 | +0.0149 |
| `attention_parity` hard-init (`mem_bias=-32, recent_bias=+32`) | TBD — single ablation cell to fill | — | — |
| `attention_parity` soft-init (`mem_bias=-4, recent_bias=+4`) | 4400 (still climbing) | +0.0131 | **+0.0379** |

Soft-parity at step 4400 has Δ_sh-m of +0.0379, which is **1.52× over
simple_gate's terminal plateau** of +0.0249, and the curve has not
plateaued. Per-step trajectories for both v3 runs are in
[`paper_artifacts/eval/chain_v3_training_summary.md`](../../paper_artifacts/eval/chain_v3_training_summary.md).

## What's intentionally out of scope

Cut from this paper, kept for paper 2 or future work:

- LoCoMo / MSC results.
- The `hidden_<L>` extract source.
- Negative-chain auxiliary loss.
- Source weighting `{pg19:1, tv:4, msc:8}`.
- Comparison to RAG (BM25, Contriever, MiniLM-FT).
- Generation EM/F1 callback metrics.
- 1.7B / 8B scaling.
- Long-context benchmarks (LongBench, RULER long).
- NIAH grid (release in supplementary as a sanity check; not a
  headline metric for a 0.6B / embed-extract setting).

## What's in this folder

```
manuscript.tex          NeurIPS-style paper draft
manuscript.pdf          most-recent compiled PDF
references.bib
neurips_2024.sty
figures/                trajectory.pdf, gate_profile.pdf, horizon_pg19_test.pdf
runs.md                 (TODO) which ckpt + log produced each cell
```

## 6-day calendar

See [`docs/paper1_calendar.md`](../../docs/paper1_calendar.md) for
the day-by-day plan and decision triggers. Summary:

- **Day 0 (today):** reorg done, scope locked.
- **Day 1:** fill the missing hard-init ablation cell (~30 min on a
  v3 step-1000 ckpt with bias still saturated, OR re-run for 4400
  steps on local GPU); rsync v3 best ckpts to cloud and queue
  standalone eval + bootstrap CIs via the watchdog; write
  `routing_trace.py` and `counterfactual_eval.py`; queue both.
- **Day 2:** run cost-analysis script and init-parity-preservation
  eval; rebuild figures; collate tables.
- **Day 3-5:** writing in `manuscript.tex`.
- **Day 6:** submit.
