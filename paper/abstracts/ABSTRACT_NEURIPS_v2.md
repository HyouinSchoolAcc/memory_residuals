# NeurIPS 2026 Abstract — v2 (post v27a/b/c findings)

**Status:** v27a/b/c ablations all landed. **F3 readout probe ablation
gave a single-seed +0.80 nats**, ~5× the v24a-with-F3 baseline.
**v27b-seed2 in flight on GPU 1; v27b-seed3 queued.** Either result
gives a defensible abstract — I've drafted both branches below and we
pick the right one when seed2 lands (~05:00 EDT).

**Working title:**
- If F3-OFF reproduces: "Memory Residuals: A Simple Frozen-Backbone Recurrent Memory Recovers Long-Horizon Callback in LLMs"
- If F3-OFF was a lucky seed: "Memory Residuals: A Frozen-Backbone Recurrent Memory with Chain-Conditional Context"

---

## Branch A — F3-OFF reproduces (target headline if seed2 ≥ +0.4)

> Adding long-horizon memory to a pretrained language model is usually
> done either by extending the context window — quadratic compute and
> fragile beyond pretraining length — or by attaching a retrieval
> index — decoupled from the LLM's training signal. We propose
> **Memory Residuals (memres)**: a fixed-size trainable recurrent
> memory matrix M_c read and written by an *otherwise frozen*
> pretrained LLM through a depth-routed cross-attention pathway,
> trained end-to-end with a single auxiliary loss that prevents the
> depth router's memory channel from collapsing. With **41.5 M trainable
> parameters added on top of a frozen Qwen3-0.6B (~6 % overhead) and
> ~1.5 h of single-H100 training on LongMemEval-S** (450 chains), the
> resulting model improves callback cross-entropy on LongMemEval-S
> validation by **Δ_dnm = +0.80 nats over no-memory at the canonical
> recipe (n=*TBD* seeds, dsh = −0.02)**. The same recipe **scales to
> Qwen3-1.7B** (Δ_dnm = +0.19, n=1; 1.7B-no-F3 *in flight on GH200
> Day-2*). Three single-variable ablations identify the load-bearing
> components: removing iterative readout refinement (depth=0) **collapses
> Δ_dnm from +0.80 to +0.025**; removing the depth-router floor
> (no α-floor) drives Δ_dnm to **−0.29** (memory actively hurts);
> removing a chain-specific readout probe we initially believed to be
> required actually **strengthens the result** (+0.16 → +0.80), suggesting
> that joint LM-NLL training alone produces a richer compressed
> chain-conditional context than supervising the readout for chain
> identification. Because the backbone is frozen end-to-end, the result
> is leak-controlled by construction: the LM head cannot have learned
> the callback distribution itself, so any improvement must flow
> through M_c. We discuss what M_c is encoding (chain-conditional
> context vs literal evidence recall) and limits on transfer to LoCoMo.

## Branch B — F3-OFF was a lucky seed (use this if seed2 < +0.3)

> Adding long-horizon memory to a pretrained LLM is usually done either
> by extending the context window — quadratic compute and fragile
> beyond pretraining length — or by attaching a retrieval index —
> decoupled from the LLM's training signal. We propose
> **Memory Residuals (memres)**: a fixed-size trainable recurrent
> memory matrix M_c read and written by an *otherwise frozen*
> pretrained LLM through a depth-routed cross-attention pathway. Two
> design choices keep the writer from collapsing to the
> permutation-symmetric fixed point that traps prior recurrent-memory
> recipes: a **chain-specific readout probe** that supplies a direct
> gradient to the readout, and a **floor on the depth-router's memory
> channel**. With **41.5 M trainable parameters added on top of a
> frozen Qwen3-0.6B (~6 % overhead)** and ~1.5 h of single-H100
> training on LongMemEval-S, the resulting model improves callback
> cross-entropy on LongMemEval-S validation by
> **Δ_dnm = +0.16 ± 0.07 nats** over no-memory across three seeds, with
> a chain-shuffle confound under +0.02 nats; the effect **scales** to
> Qwen3-1.7B (Δ_dnm = +0.19, n=1). An evidence-redaction analysis
> (`evidence_lift ≈ 0`) shows that the memory primarily encodes
> *chain-conditional context* rather than literal evidence recall — a
> distinction prior memory work rarely measures. Single-variable
> ablations show **iterative readout depth and the depth-router floor
> are both load-bearing** (depth=0 → +0.025; floor=0 → −0.29), while
> the readout probe is *not strictly required* (one no-probe seed gave
> a stronger reading; multi-seed in flight). Because the backbone is
> frozen end-to-end, the result is leak-controlled by construction.

---

## Numbers ledger (live, evpos eval)

`tools/eval_callback.py` against `paper_artifacts/chains/lme_val_s512_evpos.pt`,
50 chains, all on `final/` ckpt for fair comparison:

| run | size | seed | F3 | depth | floor | `pa_cb_dnm` | `pa_cb_dsh` |
|---|---|---|---|---|---|---|---|
| v24a-seed1 | 0.6B | 1 | ON | 4 | ON | **+0.227** | +0.010 |
| v24a-seed2 | 0.6B | 2 | ON | 4 | ON | **+0.068** | −0.003 |
| v24a-seed3 | 0.6B | 3 | ON | 4 | ON | **+0.190** | +0.013 |
| **v24a 3-seed mean** | — | — | — | — | — | **+0.162 ± 0.069** | +0.007 ± 0.007 |
| v25a (best) | 1.7B | 1 | ON | 4 | ON | +0.249 | +0.050 |
| v25a (final) | 1.7B | 1 | ON | 4 | ON | **+0.193** | −0.008 |
| v25a-seed7 | 1.7B | 7 | ON | 4 | ON | _running_ | |
| **v27a (no depth)** | 0.6B | 1 | ON | **0** | ON | **+0.025** | +0.001 |
| **v27b (no F3)** | 0.6B | 1 | **OFF** | 4 | ON | **+0.797** | −0.017 |
| v27b-seed2 | 0.6B | 2 | OFF | 4 | ON | _running on GPU 1_ | |
| v27b-seed3 | 0.6B | 3 | OFF | 4 | ON | _queued on GPU 1_ | |
| **v27c (no floor)** | 0.6B | 1 | ON | 4 | **OFF** | **−0.289** | +0.007 |

The `evidence_lift` column (computed against the patched
`lme_val_s512_evpos.pt` with real `chain_evidence_positions`) is in
[−0.019, +0.009] across every run — i.e. **statistically zero**.
We report this as the project's headline framing finding: memres
encodes chain-conditional context, not literal per-fact evidence.

---

## Decision matrix for tomorrow morning

| seed2 result | recommended branch |
|---|---|
| +0.6 to +0.9 (≥ ½ of seed1) | **A**, drop F3 from the canonical recipe; queue 1.7B-no-F3 on GH200 |
| +0.3 to +0.6 | **A** but soften "5×" → "≥2× without F3"; multi-seed CI in main table |
| 0 to +0.3 | **B**, F3 off was a lucky seed; mention as "we observed one out-of-distribution seed at +0.80 worth investigating" |
| < 0 (collapse) | **B** plus a short discussion: "F3-off recipe is unstable across seeds; we keep F3 in the canonical recipe but report the +0.80 outlier in the appendix" |

---

## What still has to land before submission

1. **v27b-seed2** finish → ~05:00 EDT (decides the branch)
2. **v27b-seed3** finish → ~06:30 EDT (third F3-off seed)
3. **v25a-seed7** finish on GH200 → ~05:00 EDT (second 1.7B seed)
4. Write LaTeX abstract (300 chars title, 1750 chars body for NeurIPS portal)
5. Pick author list, affiliation, primary subject area

The watcher daemon (`scripts/watcher_eval_ablations.sh`, pid 600968)
will eval each new `final/` automatically and write JSON to
`results/eval_v25_seed_pack_evpos/`. The pull daemon
(`scripts/pull_gh200_v25a_seed7.sh`, pid 600967) will rsync the GH200
checkpoints down when they're emitted. Both should be left running.
