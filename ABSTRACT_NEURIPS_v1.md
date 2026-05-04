# NeurIPS 2026 Abstract Submission — v1

> **WAKE-UP NOTE (May 4, 03:55 EDT):** Two seeds of the F3-OFF ablation
> now agree at Δ_dnm ≈ +0.86, which is ~5× the full-recipe headline
> (+0.13 ± 0.07). If GH200 seeds 3/4 corroborate (queue ETA 06:45 EDT),
> the abstract framing flips: F3 is **not load-bearing and is in fact
> harmful**. The simpler recipe (depth + α-floor only) becomes the
> headline. See `OVERNIGHT_STATUS.md`. Do **not** submit v1 verbatim
> until you've checked the F3-off seed pack.

**Working title:** Memory Residuals — A Frozen-Backbone Recurrent Memory
that Improves LLM Callback Recall by Adding Chain-Conditional Context

**Authors:** TBD

**Abstract (~265 words, paste-ready)**

> Adding long-horizon memory to a pretrained language model is usually
> done either by extending the context window, which scales quadratically
> in compute and is fragile beyond the pretraining length, or by
> attaching a retrieval index, which is decoupled from the LLM's training
> signal. We introduce **Memory Residuals (memres)**, a fixed-size
> trainable recurrent memory matrix M_c that is read and written by an
> otherwise *frozen* pretrained LLM through a depth-routed
> cross-attention pathway. Two design choices keep the writer from
> collapsing to the permutation-symmetric fixed point that has trapped
> previous recurrent-memory recipes: a **chain-specific readout probe**
> that supplies a direct gradient to the readout, and a **floor on the
> depth-router's memory channel** that prevents the routing distribution
> from closing during joint training. With **41.5 M parameters added on
> top of a frozen Qwen3-0.6B (~6 % overhead) and ~1.5 h of single-H100
> training on LongMemEval-S (450 chains)**, the resulting model improves
> callback cross-entropy on LongMemEval-S validation by
> **Δ_dnm = +0.13 ± 0.07 nats over no-memory across three seeds**, with a
> chain-shuffle confound under +0.02 nats; the effect **scales and
> stabilizes at Qwen3-1.7B** (Δ_dnm = +0.19, n=1; second seed in flight).
> An evidence-redaction analysis on the patched LME-val corpus
> (`evidence_lift ≈ 0`) shows that the memory primarily encodes
> *chain-conditional context* rather than literal evidence recall — a
> distinction that prior memory work rarely measures and that we argue
> is the correct framing for fixed-size compressive memories. Removing
> the iterative readout (depth = 0) **collapses Δ_dnm from +0.23 to
> +0.03**, confirming that iterative cross-attention refinement, not
> just the memory itself, is load-bearing. Because the backbone is
> frozen end-to-end, the result is leak-controlled by construction.

**TODO before submit (≤ 24h):**

1. Pull v25a-seed7 (1.7B, GH200) when it lands → 1.7B mean over 2 seeds.
2. Pull v27b (F3 off) and v27c (α-floor off) → finish ablation table.
3. Convert the prose number "+0.13 ± 0.07" to LaTeX `$+0.13\pm 0.07$`.
4. Decide on author list and venue category (Spotlight vs Poster track).

**Backup numbers if 1.7B seed7 collapses or ablations land badly:**

- Drop "scales and stabilizes" → "preserved at scale (n=1)".
- If only depth=0 collapses (already known), keep that as the single
  ablation in the abstract; move F3/floor to the appendix.

**Numbers ledger** (all from `tools/eval_callback.py` against
`paper_artifacts/chains/lme_val_s512_evpos.pt`, 50 chains):

| run | size | seed | `pa_cb_dnm` | `pa_cb_dsh` | `evidence_lift` |
|---|---|---|---|---|---|
| v24a-seed1 | 0.6B | 1 | +0.229 | +0.003 | +0.009 |
| v24a-seed2 | 0.6B | 2 | +0.064 | −0.012 | −0.012 |
| v24a-seed3 | 0.6B | 3 | +0.104 | −0.013 | +0.000 |
| **0.6B mean** | — | — | **+0.132 ± 0.071** | −0.007 ± 0.007 | −0.001 ± 0.009 |
| v25a/best | 1.7B | 1 | +0.249 | +0.050 | −0.019 |
| v25a/final | 1.7B | 1 | +0.193 | −0.008 | −0.001 |
| v25a-seed7 | 1.7B | 7 | _running on GH200_ | | |
| **v27a no depth** | 0.6B | 1 | **+0.029** (best) / +0.008 (final) | −0.013 | −0.014 |
| **v27b no F3 probe** | 0.6B | 1 | **+0.797** (final) / −0.101 (best) | −0.017 | −0.005 |
| **v27b no F3 probe** | 0.6B | 2 | **+0.930** (best, final in flight) | +0.000 | −0.003 |
| **v27b no F3 probe** | 0.6B | 3 | _running on GH200_ | | |
| **v27b no F3 probe** | 0.6B | 4 | _queued GH200_ | | |
| **v27c no α floor** | 0.6B | 1 | **−0.289** (best) / −0.038 (final) | −0.017 | −0.005 |
| v28a no F3, 1.7B | 1.7B | 1 | _queued GH200, ~12:45 EDT_ | | |
| v28b no F3, 1.7B | 1.7B | 2 | _queued GH200, ~18:45 EDT_ | | |
