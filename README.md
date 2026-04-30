# Memory Residuals

A fixed-size recurrent memory matrix $M_c \in \mathbb{R}^{K \times d}$
that gets updated end-to-end through the language-modelling loss — no
retrieval index, no separate memory controller, no hand-engineered
gating heuristic. A learned compression of past sessions that reads
into the depth-wise residual stream of a pretrained transformer.

The architectural spec is in
[`memory_residuals.pdf`](memory_residuals.pdf) /
[`memory_residuals.txt`](memory_residuals.txt). What follows is the
honest story of ~45 training cells across ten version epochs — what
broke, why it broke, and what the surviving recipe actually is.

---

## What we learned (v1 → v10)

### TL;DR

1. **Data diversity is the load-bearing axis.** Every positive result
   we have has always come from a diverse corpus (PG-19 + TV, or
   LME + MSC + PG-19 + TV). Every prolonged failure used a narrow,
   single-source distribution (LongMemEval alone). We spent three
   months debugging the architecture when the limit was the data.
2. **The phase-aligned callback-token diagnostic pair
   (Δ_nm-m, Δ_sh-m) is the single highest-value eval we added.**
   Adopting it in v8 exposed architectural failure modes that the
   standard eval read as `Δ_sh-m ≈ 0 "noise"`. The signal to watch
   is **CB Δ_nm-m**, which on v9c grows *monotonically* from −0.03
   to +0.16 nats across training — memory learning to help callback
   tokens more and more. PA CB Δ_sh-m (shuffle-discrimination) is
   weaker and largely drowned in eval noise at our default n=48
   chains; it needs n≥256 to be resolvable.
3. **Routing mode is not the binding constraint — distribution is.**
   On LME-only corpora, `attention_parity +4/-4` collapsed (v9d,
   α_mem ≡ 0). On the *same* architecture + *same* curriculum but
   a diverse corpus, `attention_parity +4/-4` opens α_mem to 4e-4
   and produces PA CB Δ_sh-m = +0.011 inside 200 steps (v10b).
4. **Three classes of "fix" turned out to be treating symptoms:**
   bigger TBPTT windows (v6 w12), heavier regularisers (v5 / v8c),
   and architectural routers (simple_gate vs attention_parity). None
   of them moved the number until the data changed.
5. **The eval-distribution gap is still open.** Standard `Δ_sh-m`
   (eval with 40+ sequential judge updates on held-out chains) is
   at ≈0 even on our best checkpoints. v10 is the first campaign
   with a realistic shot at closing it (~68k chain, ~365k session
   mega corpus + 4B backbone + L_E=10 extraction stack).

### The walls we hit, in order

#### Wall 1 — Training-distribution mismatch (run3 → chain2, early 2026-01)

The pair trainer (`train_phase1.py`, run `run3_qwen3-0.6b-large`)
produces a genuinely positive-result checkpoint on the pair task —
$\Delta_{sh\text{-}m} = +0.029$, callback help ratio 1.77×, beats a
compute-matched RAG baseline. But the chain trainer warm-started from
that checkpoint (`chain2`) explodes on the long-horizon eval:
$\text{CE}_{\text{mem}} = 8.7$ vs $\text{CE}_{\text{nomem}} = 2.5$.
The pair distribution (concatenate 4 sessions, compress once) teaches
the *readout* to be useful; the chain distribution (carry $M_c$
through many judge updates) teaches the *recurrent competition* —
and they are genuinely different problems. **Takeaway: publishable
numbers on the training distribution are not transferable if the
deployment distribution has a different recurrent depth. The pair
paper ships as Experiment 1; the chain recipe is a separate
empirical claim.**

#### Wall 2 — Shortcut learning / style-only memory (chain2 / chain_fresh1, 2026-01)

Once the chain trainer stabilises on PG-19 + TV, $\Delta_{nm\text{-}m}$
turns modestly positive (memory helps aggregate NLL) but
$\Delta_{sh\text{-}m}$ goes *negative*: $-0.014$ on PG-19 val,
$-0.036$ on `chain_fresh1`. Injecting *another* PG-19 book's memory
performs better than injecting nothing at all. The memory channel
has learned a style prior ("this is prose") not episodic content.
This is PITFALLS.md §3 in the flesh. **Takeaway: $\Delta_{sh\text{-}m}$
is load-bearing precisely because aggregate improvements are easy to
fake with genre / style cues. Any recipe that ships must have
$\Delta_{sh\text{-}m} > 0$ with bootstrap CI.**

#### Wall 3 — PG-19 + TV: the "v2 phaseA" positive result (2026-02)

`chain_v2_phaseA_softparity_b4` produces the first real positive chain
result: **$\Delta_{sh\text{-}m} = +0.0529\;[+0.0246,+0.0915]$** on
PG-19 val (bootstrap 95% CI excludes zero), $+0.0279$ on PG-19 test.
This ships as the architectural-primitive result backing Experiment 1.
The recipe: attention_parity with *softened* $\pm 4$ router biases
(vs the original $\pm 32$ bit-exact-parity init), two-stage QKV
competition with soft-init parity, negative-chain contrastive loss at
ramp $\lambda:0.05\to0.5$, PG-19 + TV corpus, `burn_in_max=24` with
resample. **Takeaway: on diverse long-narrative data with the PITFALLS
anti-shortcut regularisers, the architecture works. This result held
up under held-out routing traces and counterfactual horizon sweeps.**

#### Wall 4 — Router collapse on callback-only corpora (v3 → v8, 2026-02 → 2026-04-29)

When we moved to LongMemEval-S (the obvious "corpus with annotated
callbacks" for the recipe paper), the picture collapsed. v3 → v6
cells on LME or LME+MSC all showed the same pattern: $\text{gate}_{\max}
\equiv 0.0000$ across 1.4–4 k steps, $\Delta_{sh\text{-}m} \approx 0$
at every eval. Six consecutive campaigns failed at the same number:

| campaign | single-axis change | result | diagnosed cause |
|---|---|---|---|
| v3 | attention_parity + hard init | $\Delta_{sh\text{-}m} \approx 0$ | router saturated, LME-only |
| v4 | `hidden_14` extract + MSC corpus | $\Delta_{sh\text{-}m} \approx 0$ | router saturated, corpus mix dilutes callback |
| v5 | soft $\pm 4$ init + neg_chain ramp | $\Delta_{sh\text{-}m} \approx 0$ | gate_max stuck at 0, 5 + cells KILLED |
| v6 | LME + gated update + callback loss + callback window bias | gate_max $\equiv 0$ | four-axis writer pivot did not open the router |
| v7 (simple_gate) | routing ablation: scalar gate per sublayer | **gate_max opens** (0.0004 → 0.0074 in 400 steps), but **catastrophic overfit** on P0 curriculum ($\Delta_{nm\text{-}m} = -0.044$) | readout is *exploding* (`‖m^t‖/‖embed‖ = 165`); no scale control |
| v7 (attention_parity) | curriculum + softer bias | $\alpha_{\text{mem}} \equiv 0$, **`‖m^t‖` → 0** under weight decay | readout is *collapsing*; router + optimizer symbiosis |
| v8 (RMSNorm readout) | scale fix on top of simple_gate | gate opens, phase-aligned CB $\Delta_{sh\text{-}m} > 0$ at step 200, but **overfits within 800 steps** on P0-only curriculum | train / eval distribution mismatch (P0 windows ≠ sequential M_c) |
| v8c (+ diverse corpus) | v8 + LME + MSC + PG-19 + TV + over-regularisation | flat everywhere, gate suppressed | regulariser stack (cbw=3, drop=.10, lr=1e-4) prevented *both* overfitting and learning |

The v7 diagnostic-lens change revealed the real dynamics. Both
routing modes had failed, for *opposite* reasons — attention_parity
under weight decay decayed `W_V^{read}` to zero (bit-exact "no-memory"
despite gate firing, read as $\Delta_{sh\text{-}m}=0.0$ *exactly*);
simple_gate got direct gradient on its scalar gate and inflated
`‖W_V^{read}‖` unboundedly until injection overwhelmed the residual
stream (`‖m^t‖/‖embed‖ = 165`). The standard eval read both states
as "noise" and we read "noise" as "keep training." **Takeaway: the
paper's spec assumes depth-pool sources are commensurate in scale;
the code never enforced that. RMSNorm on the readout output
(`MemoryReadout.out_norm`, one line in `modeling_memres.py`) is a
prerequisite for any further training. It's also not sufficient.**

#### Wall 5 — The phase-aligned / standard-eval gap (v8 → v9)

Once v8 added the phase-aligned callback-token $\Delta_{sh\text{-}m}$
evaluation (matches the training distribution: 1 evidence session,
1 judge step, score the callback span), v9 produced the first clean
positive signal: PA CB $\Delta_{sh\text{-}m}$ peak **+0.0266** on v9
baseline, **+0.0310** on v9c diverse. But the standard eval
(40-session sequential $M_c$ on held-out LongMemEval) stayed at
+0.0003. Memory is content-discriminative on the distribution it was
trained on; on the deployment-like distribution the $M_c$ statistics
have drifted to a regime the readout has never seen.

v9 ablation matrix (4 cells, GH200 over 16 h):

| cell | change vs v9 | routing | peak PA CB Δ_sh-m | verdict |
|---|---|---|---:|---|
| v9  | (reference) | simple_gate | +0.0266 | alive |
| v9a | `callback_loss_weight` 3.0 → **1.0** | simple_gate | +0.000 (readout decayed to ‖m^t‖ ≈ 0) | cbw=3.0 is load-bearing; without it `W_V^{read}` decays under weight decay |
| v9b | competition 1.0 → 0.5 + evidence 0.5 + k=8 | simple_gate | +0.000 (readout decayed) | **pure** competition is required; mixing dilutes the judge gradient below the WD floor |
| v9c | LME-only → **diverse v6_lme_msc** | simple_gate | **+0.0310** | diversity is strictly non-harmful on the PA metric and slightly better-calibrated |
| v9d | simple_gate → attention_parity | attention_parity +4/-4 | +0.000 (α_mem ≡ 0) | attention_parity stays architecturally collapsed on LME-only |

**Takeaway: competition curriculum + RMSNorm readout + cbw=3.0 +
simple_gate = first reproducible PA CB signal. Peak value is similar
or slightly better under data diversity. Neither recipe closes the
standard-eval gap.**

#### Wall 6 — Data diversity is not a nice-to-have (v10, 2026-04-30)

The v9 ablation signal and a reread of the historical record
(PG-19-only cells in v2/v3 always had *some* signal regardless of
setup; v9c was the only surviving v9 ablation) led to the v10
hypothesis: **LongMemEval alone is pathologically narrow and the
routing collapses we blamed on architecture are actually
distribution-collapse artefacts.** v10 tests this directly by
constructing a ~68 k-chain / ~365 k-session mega corpus
(LME + MSC + PG-19 + TV + UltraChat + PIPPA + SODA +
OpenAssistant + no_robots + NarrativeQA + WritingPrompts) and
retrying the exact `attention_parity +4/-4` routing that v9d
declared dead.

v10b step-200 eval (0.6 B cheap proxy, same router config as v9d,
only knob different is the corpus):

```
PA-EVAL @ step 200: WS Δnm-m=+0.0040  Δsh-m=+0.0029
                    CB Δnm-m=+0.0113  Δsh-m=+0.0110
ROUTE @ step 200  : mode=attention_parity
                    α_mem_max=0.0004  α_mem_mean=0.0002
READOUT @ step 200: ‖m^t‖/‖embed‖ = 73.53
```

Non-zero α_mem, positive CB $\Delta_{sh\text{-}m}$, readout scale
bounded — all three in the *first eval cycle*. Compare v9d, same
architecture, 4000 steps on LME-only: α_mem exactly zero throughout.
**The routing-mode "failure" that motivated the v7 simple_gate pivot
was a data-distribution failure wearing an architecture mask.**

### What the surviving recipe actually is (as of v10-day-0)

**Architecture.**

- Two-stage QKV competition writer, `--memres_update_mode gated`
  (sigmoid write gate, init bias −1.0 so $g \approx 0.27$).
- Extraction source: `hidden_14` (0.6B, 28-layer backbone) or
  `hidden_18` (4B, 36-layer backbone) — mid-stack contextualised
  token representation, detached from backbone gradient.
- Readout RMSNorm (`MemoryReadout.out_norm`, v8 fix) — prerequisite
  for stability under either routing mode.
- Routing: either `simple_gate` (one scalar gate per sublayer,
  zero-init) or `attention_parity` (block AttnRes depth pool,
  softened $\pm 4$ biases — *works on diverse data, fails on
  LME-only*).

**Training.**

- Diverse corpus, full stop. Minimum viable is v6_lme_msc
  (6 378 chains across LME, MSC, PG-19, TV, RealTalk); the v10
  mega corpus scales this ~10× with UltraChat, PIPPA, SODA, OASST1,
  NoRobots, NarrativeQA, WritingPrompts.
- `curriculum_competition_bias = 1.0` (v9) or 0.5 with 0.3
  evidence-callback (v10 composed). *Pure competition is required
  at 0.6 B; at 4 B we test whether composed curriculum closes the
  eval-distribution gap.*
- `callback_loss_weight = 3.0`. Load-bearing (v9a proved <3.0 lets
  the readout decay under weight decay).
- `window_k = 3` (minimum sessions to enable competition) or 4
  (v10 4B). Larger `window_k` with `carry_state=True` exposes the
  readout to deeper $M_c$ distributions and is the primary lever
  against the standard-eval gap.
- Optimizer: AdamW, lr memres = 3–5e-5, lr backbone = 2–5e-6,
  weight_decay = 0.1, cosine decay with warmup = 200–500.

**Evaluation** (all five at every save cycle):

1. **Phase-aligned callback-token $\Delta_{nm\text{-}m}$** (`pa_cb_dnm`).
   *The primary progress metric.* Measures how much memory helps on
   callback tokens vs bare backbone, on a single-evidence + single-
   judge window that matches training. Grows monotonically when the
   recipe is working (−0.03 → +0.16 on v9c over 4 000 steps).
2. **Phase-aligned callback-token $\Delta_{sh\text{-}m}$** (`pa_cb_dsh`).
   Tests whether the channel is *episodic* vs generic. Must eventually
   be > 0 with a tight CI, but at our default `n=48` chains the per-
   eval std is ~0.014 nats — so individual readings are noisy and
   selecting on this metric alone biases toward noise peaks. Use
   n≥256 or bootstrap CI for gating decisions.
3. **Standard $\Delta_{sh\text{-}m}$ on a 40-sequential-session
   held-out chain.** The deployment-distribution metric; the
   eval-distribution gap is quantified here. Currently ≈ 0 at all
   v9 peaks — the open "Wall 5" problem.
4. **Per-sublayer routing recruitment** (`α_mem_max`, `α_mem_mean`,
   `frac_open`, top sublayers). Mechanistic evidence that *some*
   sublayer × position actually uses memory on held-out data.
   `paper_tools/routing_trace.py`.
5. **Readout magnitude** (`‖m^t‖/‖embed‖`). Sanity pulse — values
   near 0 mean `W_V^{read}` has collapsed, values ≫ 1 mean it's
   exploded. Healthy range is ~50–100 with RMSNorm in place.

Plus causal evidence per promoted checkpoint: counterfactual eval
(perturb session $t-k$, measure CE on $t$ callback) for a horizon
curve (`paper_tools/counterfactual_eval.py`), and bootstrap 95% CI
on the primary $\Delta_{sh\text{-}m}$ against a 1000+-chain sample
(`paper_tools/bootstrap_ci.py`).

### Open problems

1. **Close the eval-distribution gap.** Standard $\Delta_{sh\text{-}m}$
   is still ≈0 on our best checkpoints. v10's 4B mega run is the
   current attempt (window_k=4, carry_state, composed curriculum,
   67 k chains); success threshold is `standard Δ_sh-m > +0.005` at
   step ~20 000.
2. **PA CB Δ_sh-m eval variance is larger than the effect we're
   trying to measure.** The "peak-then-decay" pattern we initially
   read as a mechanistic failure (v9: +0.0266 → −0.0146 in 400
   steps; v9c: +0.0310 → −0.0315 in 1200 steps) is *mostly* eval
   noise + max-selection bias, not a coherent decay. On the v9c
   trajectory the load-bearing diagnostics tell a completely
   different story: `gate_max` saturates at 0.0044 by step 1400 and
   stays *exactly* there for the next 2 200 steps; `frac_open`
   parks at ~0.64; readout magnitude never drifts outside
   73.48 ± 0.02; **CB Δ_nm-m grows *monotonically* from −0.03 to
   +0.16** — memory is doing *more* useful work as training
   progresses, not less. What actually oscillates is only
   Δ_sh-m, and eval-noise math (CB Δ_sh-m std ≈ 0.014 nats at
   n=48 chains × ~20 callback tokens) says the max-of-18-evals
   under zero-mean noise is about +0.032 — which is almost
   exactly the "peak" we were treating as signal. The real
   problem here is (i) we need ~500 chains on the PA eval to
   resolve "Δ_sh-m > 0 in expectation" at p < 0.05, and (ii) the
   `save_best` on phase-aligned PA CB Δ_sh-m probably latches
   onto noise. Workarounds: bump `--phase_aligned_eval_n_chains`
   to 256+, add bootstrap CI to the selection metric, consider
   switching `--save_best_metric` to a composite that also weighs
   CB Δ_nm-m (which is the metric with real monotonic signal).
3. **8B scaling.** `qwen3-8b-xlarge` (L_E=10, ~8.8 B total) exceeds
   a single 96 GB GH200 under full AdamW (~106 GB peak). The v10 run
   is on 4B (fits at ~52 GB peak with gradient checkpointing). 8B
   needs either bitsandbytes AdamW8bit (flag `--use_adam8bit` is in
   the trainer, ready when the wheel is installed) or multi-GPU ZeRO.
4. **Evidence-annotation-free curriculum.** The competition
   curriculum currently picks a random "evidence" session from
   `[0, cb_pos)` — label noise is intentional (matches deployment)
   but we don't know whether cleaner evidence annotations would
   sharpen the judge's gradient further. Needs a synthesis run on
   NarrativeQA-style corpora with ground-truth provenance.
5. **Backbone-free evaluation.** Every eval uses the trained
   backbone. A frozen-backbone ablation (`--freeze_backbone`, added
   in v10) would separate "memory learned episodic recall" from
   "backbone co-adapted to memory injection." Not yet run.

---

## File map

```
memory_residuals.pdf / .txt  position paper (the architectural spec)
atn_residuals.pdf            Block Attention Residuals reference
COMPREHENSIVE.md             full historical ledger, every prior run
                             (long; reference, not reading order)

modeling_memres.py           architecture (config, model, init)
presets.py                   named (backbone, K, L_E, N) tuples
train_chain.py               recurrent chain TBPTT trainer (the active one)
train_phase1.py              pair-based warm-up trainer (Paper 1 recipe)

scripts/                     launchers per cell (v10 cells are active)
paper_tools/                 eval / probes / corpus builders
paper_tools/cloud_watchdog/  remote-survivable job queue + ntfy daemon

paper_artifacts/chains/      pretokenised corpora (lme_*, v6_lme_msc_*,
                             v10 mega_train_s512.pt, realtalk_*)
paper_artifacts/eval/        eval JSONs + bootstrap CIs that paper text cites

experiments/exp1_drop_in_primitive/    Paper 1 (pair recipe, run3) manuscript
experiments/exp2_long_horizon_recipe/  Paper 2 (chain recipe) draft, runs.md

output/                      training checkpoints (gitignored)
logs/                        training & sync logs (gitignored)

archive/                     superseded scripts / datasets / agent notes
```

## Compute

- **Local.** 2 × H100 NVL (94 GB) at the lab box. Currently running
  v10a (simple_gate composed, GPU 0) + v10b (attention_parity +4/-4,
  GPU 1). ~16 h/day usable (residential power-down overnight).
- **Cloud.** 1 × NVIDIA GH200 480 GB at `192.222.50.225`. Currently
  running `chain_v10_4b_mega_attnparity_gh200` (Qwen3-4B + L_E=10
  + mega corpus, 30 000 steps, ~3-day budget). Jobs run in detached
  `tmux` under the cloud watchdog, so they survive SSH drops and
  lab-box power-offs.

## Stop everything

```bash
pkill -f train_chain.py
ssh ubuntu@192.222.50.225 'tmux kill-session -t cwd-daemon; pkill -f heartbeat.sh'
```

## Reading order

1. **This README** — what we learned, what's active, where things
   live.
2. [`memory_residuals.pdf`](memory_residuals.pdf) — the position
   paper. The formal architectural spec (Eqs. 1–10, two-stage QKV
   competition, off-sequence depth-wise injection).
3. [`atn_residuals.pdf`](atn_residuals.pdf) — the Block Attention
   Residuals reference. Read when the routing trace looks weird or
   you're tuning router init biases.
4. [`experiments/exp2_long_horizon_recipe/runs.md`](experiments/exp2_long_horizon_recipe/runs.md)
   — the active run ledger (which cell is doing what right now,
   decision triggers, kill criteria).
5. [`COMPREHENSIVE.md`](COMPREHENSIVE.md) — every prior run, every
   architectural decision, every failure mode in detail (long;
   reference material, not a reading order).
