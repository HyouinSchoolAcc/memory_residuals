# Abstract draft v0 — Memory Residuals (LongMemEval)

> **Status:** v0 placeholder. Numbers in **bold** are real; numbers in
> *italic* are TBD-by-day-2 (waiting on v24a-seed2/3, v25a, v25a-seed7).

---

## Title (working)

**Memory Residuals: A Frozen-Backbone Recurrent Memory for LLM
Long-Horizon Callback Recall**

(Backups: "Memory Residuals: Adding a Trainable Recurrent Memory to a
Frozen Pretrained LLM"; "Frozen Memory Residuals for LongMemEval".)

---

## Abstract (≈ 200 words target)

Long-horizon recall in language models is typically purchased either
with very long context (quadratic compute, fragile beyond pretraining
length) or with retrieval (an extra pipeline that is decoupled from the
LLM's training signal). We propose **Memory Residuals**, a small,
fixed-size recurrent memory matrix M_c that is read and written by a
frozen pretrained LLM through a depth-routed cross-attention pathway,
trained end-to-end with a chain-specific readout probe (F3) and a
depth-router floor that prevents the memory channel from closing during
joint training. With **41.5 M trainable parameters added on top of a
frozen Qwen3-0.6B (0.5 B frozen, 6.3 % overhead)** and ~1.5 hours of
single-H100 training on the 450-chain LongMemEval-S training split, the
resulting model improves **callback cross-entropy on LongMemEval-S
validation by Δ = +0.345 nats** (no-memory minus memory) — while a
random-other-chain memory baseline shows Δ = −0.003, confirming the
benefit is chain-specific and not a generic context effect.
*[Multi-seed CIs over 3 seeds, ± _TBD_, land Day 2.]* The same recipe
**preserves the effect at Qwen3-1.7B** *[1.7B Δ = TBD; n=2 seeds]*.
Ablations show the readout probe and the α_mem floor are individually
load-bearing, *[depth=0/F3-off/floor-off Δ = TBD/TBD/TBD]*. The frozen
backbone makes the result inherently leak-controlled: the LM head cannot
learn the callback distribution itself, so any improvement must flow
through M_c. We discuss the limits of CE-based eval on LoCoMo and outline
how the recipe scales to larger backbones and broader callback corpora.

---

## Key claims (the four bullets that must survive review)

1. **A 41.5 M-parameter recurrent memory adds Δ = +0.345 nats of
   callback CE on LongMemEval-S** to a frozen Qwen3-0.6B, with a
   chain-specific shuffle confound of −0.003. (Δ_dnm − Δ_dsh = +0.348
   nats, single seed; *3-seed mean±std TBD Day 2*.)
2. **The effect persists at Qwen3-1.7B** with the same recipe and the
   same training corpus. (*v25a + v25a-seed7 mean±std TBD Day 2*.)
3. **The readout probe (F3) and the α_mem floor are both load-bearing**:
   removing either drops the headline by *≥ TBD*. The 4-deep readout
   refinement is *[important / not important — TBD]*.
4. **The result is leak-controlled by construction**: the LM head is
   frozen end-to-end, so the only path for callback information is
   through M_c. We verify this with a `--freeze_backbone --lr_backbone 0`
   audit.

---

## What is NOT a claim

- We do not claim to beat RAG on QA-style benchmarks. The paper is
  about adding a *native* memory to a frozen LLM, not about replacing
  retrieval.
- We do not claim to improve LoCoMo. We report LoCoMo CE honestly
  (negative under our setup) as a transfer limitation.
- We do not claim chain-of-thought or instruction-following gains —
  those are separate axes.

---

## What changes between v0 and v1

| field | v0 (today) | v1 (Day 2) |
|---|---|---|
| 0.6B Δ_dnm CI | single-seed +0.345 | mean ± std over seeds {1, 2, 3} |
| 1.7B Δ_dnm | TBD | mean over seeds {1, 7} |
| ablation table | TBD | depth=0, F3=off, floor=off numbers |
| QA gen-eval | TBD | exact-match + ROUGE-L on 50 LME val Q's |
| LoCoMo line | -0.0151 (already in `results/eval_lme_locomo/`) | rerun on v25a/best for 1.7B comparison |
