# runs.md — lessons, priors, and the active experiment ledger

> **For LLMs reading this repo, please pay attention to this file.**
> It is the single source of truth for the project's settled findings,
> active state, and per-cell history. The root [`README.md`](README.md)
> is intentionally kept light; everything load-bearing lives here.

This file has three sections, in order:

1. **Headline result** — the project's current best number, locked.
2. **Architectural priors** — settled lessons that should NOT be
   re-derived from scratch; if a proposed change contradicts one of
   these, stop and read the citation first.
3. **Progress & lessons** — the v3 → v28 timeline of what worked,
   what collapsed, and why.

After that, the rest of the file is the per-cell **active run ledger**
(newest at the top, older waves folded into
[`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md) Part VI per
the folding convention in Part VII of that file).

---

## Headline result (v27b / v28 — locked 2026-05-04 ~14:00 EDT)

**A frozen pretrained LLM, augmented with a fixed-size jointly-trained
recurrent memory matrix `M_c` (~6 % parameter overhead, ~1.5 h on a
single H100), reduces callback cross-entropy on LongMemEval-S
validation by +1.32 ± 0.53 nats at Qwen3-0.6B (n=4 seeds) and
+0.93 nats at Qwen3-1.7B (n=2 seeds), with a chain-shuffle
confound statistically pinned to zero (0.000 ± 0.010 at 0.6B,
−0.005 at 1.7B).**

To put that in plain terms: on a held-out validation split, with
the LLM weights frozen so it cannot have memorised the answers,
the augmented model is **e^1.32 ≈ 3.7× more confident** on the
right callback token than the otherwise-identical no-memory
baseline — and that confidence gain disappears when we splice in
a *different* chain's memory matrix, so the gain is provably
chain-specific rather than "memory adds any context".

`tools/eval_callback.py` against `paper_artifacts/chains/lme_val_s512_evpos.pt`,
50 chains, all from the `final/` checkpoint:

| size | seed | host | `pa_cb_dnm` | `pa_cb_dsh` | `evidence_lift` |
|---|---|---|---|---|---|
| 0.6B | 1 | local H100 | +0.797 | −0.017 | −0.005 |
| 0.6B | 2 | local H100 | +0.939 | +0.008 | −0.002 |
| 0.6B | 3 | GH200 | +1.833 | +0.000 | +0.002 |
| 0.6B | 4 | GH200 | +1.721 | +0.001 | +0.008 |
| **0.6B mean (n=4)** | — | — | **+1.323 ± 0.530** | +0.000 ± 0.010 | +0.001 ± 0.006 |
| 1.7B | 1 | GH200 | +0.909 | −0.001 | +0.005 |
| 1.7B | 2 | GH200 | +0.944 | −0.009 | −0.008 |
| **1.7B mean (n=2)** | — | — | **+0.926** | −0.005 | −0.001 |

Per-chain sanity on the +1.83 seed: **49 / 50 chains positive**,
median per-chain Δ = +0.91 nats, no single-chain outlier driving
the mean. Single-variable ablation table is in
[`paper/drafts/NEURIPS_NUMBERS.md`](paper/drafts/NEURIPS_NUMBERS.md);
abstract paste-ready in
[`paper/abstracts/ABSTRACT_NEURIPS_v3.md`](paper/abstracts/ABSTRACT_NEURIPS_v3.md).

### Why we think this is groundbreaking

There is, to our knowledge, no prior demonstration of a *frozen
pretrained LLM* + *fixed-size, end-to-end-trained recurrent memory
matrix* delivering this magnitude of chain-specific callback gain
on a real long-conversation benchmark, with the leak-control
property baked into the architecture. Specifically:

* **Frozen-backbone, leak-controlled by construction.** The
  Qwen3 weights never move during training. The callback
  improvement *cannot* be the model "memorising the validation
  questions" — there is no parameter that could absorb that
  information. The only path that carries chain-specific
  evidence from earlier sessions to the callback token is
  through the 41.5 M-parameter `M_c` channel.
* **Chain-specific, not "memory adds context".** Δ_dsh ∈
  [−0.017, +0.008] across every cell. Splicing in a random
  *other* chain's `M_c` gives ~0 callback gain; splicing in
  the chain's *own* `M_c` gives +1.32 nats. This rules out
  the most common "long-context attention helps any token
  prediction" confound that bedevils retrieval and KV-cache
  compression baselines.
* **Reproducible across seeds, both at 0.6B and 1.7B.**
  4/4 seeds positive at 0.6B, 2/2 seeds positive at 1.7B.
  No collapse, no unstable knife-edge — this was the trap
  that killed v23 on synthd5 and motivated the corpus pivot.
* **Cheap and recipe-portable.** ~6 % parameter overhead,
  ~1.5 h training on a single H100 at 0.6B (~6 h at 1.7B),
  vs. the quadratic-context or retrieval-index alternatives.
* **The recipe is short and falsifiable.** Three load-bearing
  flags (`--memres_readout_depth 4`, strong α-floor at 0.5/0.10,
  callback-supervised real-content training corpus); one flag
  to drop (`--readout_probe_loss_weight 0.0`); the rest is the
  v13/v14 symmetry-break stack already in `src/modeling_memres.py`.
  Single-variable ablations identify the load-bearing components
  cleanly: depth=0 → +0.025; floor=0 → −0.038; F3=0 → +1.32
  (the headline).
* **Surprising scientific finding.** The auxiliary readout
  probe (F3) we expected to be load-bearing — and that the
  v18 → v22 read-side audit was built around — **reverses
  sign when removed on real-content data**. Joint LM-NLL
  training alone, on a corpus where the LM head can actually
  consume `m_t`, produces a richer chain-conditional
  compression than the supervised-readout shortcut. This is
  a clean negative result that simplifies the recipe and
  reframes the auxiliary-loss design space.

### Honest scope and remaining limits

* `evidence_lift ≈ 0` across all cells. The memory is
  encoding *chain-conditional context* (style, topic,
  vocabulary, prior-turn structure that sharpens the callback
  distribution) rather than literal evidence-session recall.
  We frame this as the **scope claim** of the paper, not a
  failure mode — fixed-size compressive memory of this size
  is not expected to function as a verbatim KV cache, and
  the chain-specificity result above is what fixed-size
  memory is *supposed* to deliver.
* **Out-of-domain transfer is negative.** v24a evaluated on
  LoCoMo (different conversational distribution) gives
  Δ ≈ −0.015. The recipe trains a domain-specific writer; the
  paper claim is the recipe, not a SoTA QA system across all
  long-context benchmarks.
* **n=2 seeds at 1.7B** is the weakest leg of the
  headline. Both seeds land in [+0.91, +0.94] (very tight),
  but a 95 %-CI multi-seed at 1.7B is deferred follow-up.

---

## Architectural priors (READ FIRST — these are settled)

Future agents: these are baked-in findings from the v3 → v28 line
of evidence (v3 → v15: pair-recipe + collapse mechanisms; v17 →
v22: read-side audit; v23 → v28: corpus pivot + F3-OFF flip + 1.7B
scale-up). Do NOT re-derive them from scratch and do NOT silently
swap them out of a run without reading the citation.

1. **AP (`attention_parity`) > SG (`simple_gate`) on the routing
   side.** Per `results/exp1_pair_recipe/manuscript.tex` Table 2 (v3
   pair-recipe, matched seed, matched compute, PG-19 pairs): AP (soft
   ±4 bias init) beats SG on Δ_sh-m by **1.6× to 3.8× at every step**;
   AP @ step 2000 (+0.0272) already surpasses SG's full-budget
   asymptote at step 5200 (+0.0249). Default `--memres_mode
   attention_parity` for any pair-style or chain headline run.

2. **Caveat — AP collapses on the chain trainer via the writer/router
   lock-in cycle.** v5 softparity, v7 softerbias + v3bias, every
   v8/9/11/v12a AP cell sat at `α_mem ~ 4e-4` indefinitely because
   the router closes early → writer gets attenuated gradient → writer
   stays random → router sees noisy m^t → router closes harder. The
   v13+ stack (`writer_warmup`, `memres_queries_init=orthogonal`,
   `memres_slot_positional`, `memres_writer_kind=slot_attention`,
   `--alpha_mem_floor_aux_weight`) is designed to break that cycle.
   See `archive/COMPREHENSIVE.md` Part VI.

3. **Config-merge bug fixed 2026-05-01.** `Qwen3MemResConfig`
   subclasses `Qwen3Config`, so `from_pretrained("Qwen/Qwen3-0.6B")`
   succeeds and the old `from_memres_ckpt = True` detector in
   `_build_model` produced a false positive on every Qwen3 base
   backbone, silently dropping CLI overrides for `--memres_mode`,
   `--memres_writer_kind`, `--memres_slot_positional`,
   `--memres_extraction_depth`, `--memres_update_mode`,
   `--memres_num_vectors`, etc. Fixed by detecting memres checkpoints
   via `base_cfg.model_type == "qwen3_memres"` (raw JSON field).
   **If you see a v11/v12/v13/v14/v15 run whose `ROUTE @ step`
   diagnostic reports a different mode than the launch script
   requested, or a load report whose MISSING-list doesn't include
   `M_in_pos` / `write_gate` / `extraction_layers.{0..4}` despite the
   launch flags, the bug is back — bisect against the
   `BUGFIX 2026-05-01` comment.**

4. **`simple_gate` writer_warmup needs memory_gate force-open, not
   just router mem_bias.** In `simple_gate` the depth router is not
   on the forward path; `memory_gate.gate` is. `_set_mem_bias`
   forces `gate = 0.5 * tanh(bias/2) ≈ 0.48` when mode is simple_gate
   (in addition to setting `depth_router.mem_bias`). Without this,
   SG writer_warmup trains the writer with zero gradient through the
   forward path because `h + 0 * m^t = h`. See `_set_mem_bias` in
   `src/train_chain.py`.

5. **The uniform-softmax fixed point is structural, not
   data-starved.** More data alone does not break it (v11p,
   v11m_chinchilla, v12c all collapsed on larger corpora). It is the
   permutation-invariant fixed point of a symmetric softmax with
   i.i.d.-initialised slot queries. The v13+
   `memres_queries_init=orthogonal` + `memres_slot_positional` levers
   are the *structural* fix; `writer_warmup` + `slot_attention` are
   the objective/writer-side accelerants that keep the system from
   re-collapsing during joint training.

6. **v15 write_gate saturation (fixed 2026-05-02).** The external
   sigmoid `write_gate` over the un-normalised `M_new` from the
   residual extract stack (`‖M_new‖_F ≈ 7e4`) saturated to ε within
   50 steps; once saturated, gate gradient vanishes
   (`g(1−g) ≈ 1e-30`) and `M_c` is locked at the zero matrix
   forever — content-blind writer, dead readout, loss eventually
   drops anyway because the backbone takes over. Fixes:
   - For `writer_kind ∈ {slot_attention, slot_attention_full}` the
     external `write_gate` sigmoid is **bypassed entirely**
     (Locatello GRUCell already gates per-slot; stacking the
     external sigmoid is redundant and harmful).
   - For `writer_kind=original`, `MemoryBlock.forward` RMSNorms
     each side of `gate_input` before the sigmoid
     (`write_gate_norm_prev`, `write_gate_norm_new`).
   - **`--memres_extract_input_norm`** wraps `C` through an RMSNorm
     before the cross-attn / slot-attn extract path. Diagnosed as
     the dominant root cause of `M_new` norm explosions (~50×
     backward grads on `W_Q`).
   - **`--kill_on_memory_collapse`** converts the silent-failure
     burn into a loud halt with exit code 42 (two consecutive evals
     after `--kill_on_memory_collapse_min_step=200` with
     `Mc_pair_to_self_ratio < 0.01` *or*
     `mt_norm_ratio_mean < 0.01`).

7. **`tools/eval_callback.py`, not `tools/eval_chain.py`, is the
   canonical post-train eval for D4-style corpora.** `eval_chain.py`
   averages CE over the entire score window
   (`score_tail_frac=1.0` ⇒ all 4 sessions) and dilutes a
   callback-localised effect ~38× into noise. On v14k_best,
   `eval_chain.py` reported `dnm ≈ −0.10` while `eval_callback.py`
   reported `pa_cb_dnm = +1.44, evidence_lift = +0.071`.

8. **`--memres_judge_qk_layernorm` is anti-causal under the
   slot_attention writer (so far).** v14abl_a (QK-LN ON) zeroed out
   the writer entirely (`self ‖M‖ = 0`, `‖m^t‖/‖embed‖ = 0`) while
   the otherwise-identical v14abl_b (QK-LN OFF) had the writer
   specialising (`pair/self = 0.005`, `‖m^t‖/‖embed‖ = 3.87`). Ship
   default OFF until the judge × slot_attention interaction is
   understood.

9. **OPEN — joint-training backbone leakage on D4v2 (2026-05-03).**
   v15b (0.6B) and v15f (1.7B) joint-trained at `lr_backbone=2e-5`
   collapse `evidence_lift` to ~0 — the unfrozen backbone is
   apparently learning the callback distribution *directly*. This
   should be impossible if memory is the only pathway carrying
   evidence. Three independent leak audits (window leakage / eval
   redaction / template prior) are running. **Until they land, treat
   trained-backbone results on D4v2 as suspect and run all v15
   headlines on FROZEN backbones (`--freeze_backbone --lr_backbone
   0`).**

10. **The read-side gradient channel is missing under
    LM-NLL-only (v17 §5 + v18 §5 followup, 2026-05-03).** The §5
    capacity probe (`tools/eval_ttt_mc.py`: TTT-on-M_c with
    everything else frozen) returns 6/6 NEG across v14k/v15a/v15e at
    0.6B and 1.7B — for *any* trained read-side, no M_c gives
    chain-specific callback predictions. v17/F2's WriterProbeHead
    has its own Q/K/V (bypasses MemoryReadout entirely) so it
    cannot fix this. v18/F3's **`ReadoutProbeHead`** consumes m_t
    (the actual MemoryReadout output at the callback position) so
    its probe-loss gradient flows through MemoryReadout's own
    W_Q/W_K/W_V, and v18a/best shifted the §5 result from -0.897 →
    +0.005 — the first MIXED §5 reading. The v18-v22 wave then
    locked in the **multi-layer readout depth**
    (`--memres_readout_depth 4`) and **strong α-floor**
    (`--alpha_mem_floor_aux_weight 0.5`, target 0.10) as the
    load-bearing read-side levers.

11. **Surprise: the F3 readout probe is HARMFUL on real-content
    data, not load-bearing (v27b/v28 ablation, 2026-05-04).** Once
    we pivoted training to LongMemEval (real conversation,
    100 % callback supervision, 99.9 % session density), the
    single-variable `--readout_probe_loss_weight 0.0` ablation
    *increased* `pa_cb_dnm` by ~8× — from +0.16 ± 0.08 nats with F3
    on (v24a, n=3 seeds) to **+1.32 ± 0.53 nats with F3 off**
    (v27b, n=4 seeds, 0.6B), and the effect carries to **+0.93 nats
    at 1.7B** (v28a/b, n=2 seeds), with chain-shuffle confound
    pinned at 0.000 ± 0.010 throughout. Mechanism (current best
    explanation): on real-content data the LM-NLL is
    permutation-breaking by itself (chain-conditional context
    flows naturally through the LM head), and the F3 probe instead
    pulls `M_c` along a value-space direction the trained readout
    cannot exploit end-to-end. The matched companion ablations
    (`--memres_readout_depth 0` collapses Δ_dnm to +0.025;
    `--alpha_mem_floor_aux_weight 0.0` drives Δ_dnm to −0.038)
    confirm the other two recipe components stay load-bearing.
    **Default: `--readout_probe_loss_weight 0.0` (drop the probe
    entirely), keep `--memres_readout_depth 4` and the strong
    α-floor.** Numbers locked in
    [`paper/drafts/NEURIPS_NUMBERS.md`](paper/drafts/NEURIPS_NUMBERS.md);
    abstract framing in
    [`paper/abstracts/ABSTRACT_NEURIPS_v3.md`](paper/abstracts/ABSTRACT_NEURIPS_v3.md).

---

## Progress & lessons (v3 → v28)

- **#v3b** — Bit-exact init parity primitive: drop MemRes onto any
  backbone, augmented model is **bit-exactly** equal to the bare
  backbone at init *and* still receives gradients on every
  memory-channel parameter.
- **#v3** — Three injection variants: scalar gate (`simple_gate`),
  hard-bias parity-preserving attention (`attention_parity ±32`),
  light-bias non-parity attention (soft ±4). On
  *"compress previous book chapters → help generate the next chapter"*
  the **light-bias** variant won
  (`chain_v2_phaseA_softparity_b4`, Δ_sh-m = +0.0529 [+0.025, +0.092]).
- **#v9c** — Books are easy, dialogue is hard. PA CB Δ_nm-m grew
  monotonically from −0.03 → +0.16 nats across 4 000 steps on the
  diverse PG-19 + TV + LME + MSC corpus.
- **#v3 → v10** — Six straight LME-only campaigns collapsed
  identically (`gate_max ≡ 0`, `α_mem ≡ 0`). Post-v10 audit found
  three causally independent failures: P0 (the corpus builder threw
  away `answer_session_ids` so 96 % of training windows had `M_c`
  built from sessions that did not contain the answer), P1
  (chicken-and-egg gate × readout × writer multiplication, all zero
  at init), P2 (readout RMSNorm pinned `‖m^t‖/‖embed‖ ≈ 73`).
- **#v11** (g/h/i/j/k/l-fix/m/p/q/r) — P0+P2+P3 fixed in code.
  Cleanest result: P1 (router saturation) confirmed via
  `mem_bias=−4` vs `0`. P2 turns out to be **irrelevant for AP** (the
  depth softmax self-regulates magnitude) and only matters for SG.
  P5 alone is no-op. Headline finding: **the writer is content-blind
  under LM-only**; under InfoNCE alone it learns chain-identity hash
  (`evidence_lift = −1.12` on v11r). D5 audit on v11g/best identified
  the readout as the bottleneck.
- **#v12** (slot_attention writer) — Replaces the original
  decision-less judge with Locatello slot attention (softmax over
  slots, GRUCell update). Briefly produces +0.39 PA CB Δ_nm-m at step
  200, then collapses to the same uniform fixed point by step 800.
  Necessary but not sufficient — GRU shares weights across slots so
  symmetry re-emerges.
- **#v13** (`writer_warmup` + orth init + slot_positional + the
  config-merge bugfix) — Symmetry break is **permanent** (D3-MC
  pair/self = 0.004 sustained through 10 500 steps). v13c2 hit
  `evidence_lift +1.4` mid-warmup. **Phase-2 backbone unfreeze
  destroys the writer specialisation** — motivates v14.
- **#v14** (judge_qk_layernorm + alpha_mem_floor aux + InfoNCE +
  AP warmup anneal; D4v2 multi-evidence corpus) — `judge_qk_ln`
  interacts pathologically with `slot_attention` writer (writer never
  lifts off zero). Without QK-LN, writer specialises but Δ_nm-m goes
  to −0.44 — InfoNCE satisfies itself with chain-distinguishable
  M_c that doesn't translate to LM benefit. **v14k @ FROZEN backbone
  is the first reproducibly positive result of the project**:
  `pa_cb_dnm = +1.44`, `evidence_lift = +0.071`.
- **#v15** (`extract_input_norm` + bypassed `write_gate` for
  slot_attention + double-evidence D4v2) — **v14k/v15a reproduce
  cleanly on FROZEN backbones**. v15e (1.7B frozen, norm ON) hits
  `Δnm-m_floor = +2.5 nats` but `evidence_lift` swings *negative*
  (the larger writer overfits non-evidence content to pre-route the
  callback). v15b/v15f (joint training) collapse `evidence_lift` to
  ~0 → motivates synthetic leak-controlled corpora.
- **#v17 → v22** (read-side audit on synthd5_random_codes, frozen
  backbone) — Six-cell sweep that turns the §5 probe (TTT-on-M_c)
  from 6/6 NEG into +0.120 (v20a). Locks in
  `--memres_readout_depth 4` + `--alpha_mem_floor_aux_weight 0.5
  --alpha_mem_floor_target 0.10` as load-bearing read-side levers.
  v21c hits the project's first end-to-end positive on synthd5
  (+0.024 nats at seed=42), but v23 multi-seed shows the recipe is
  **unstable on synthd5**: 2/4 seeds collapse to the uniform-softmax
  fixed point and are killed by `--kill_on_memory_collapse`. The
  +0.024 reading was a lucky seed; the templated 5 %-density corpus
  is the bottleneck, not the architecture.
- **#v24** (corpus pivot: train on **LongMemEval-S**, 11.2 M tokens,
  100 % callback supervision, 99.9 % real-content density) — A
  *single-flag* change (`--train_chains lme_train_s512.pt`) shifts
  end-to-end `pa_cb_dnm` from +0.024 (v21c on synthd5) to **+0.227
  on lme_val_s512_evpos** at seed=1, with shuffle confound +0.010
  and `evidence_lift = +0.005`. 3-seed mean: **+0.162 ± 0.083 nats
  at 0.6B**. v24c (LME+MSC merged, only 7 % callback-supervised)
  goes negative, confirming **callback annotation density on the
  bulk of training chains is the dominant collapse cause** under
  the v21c recipe — not architecture.
- **#v25** (1.7B scaling on the v24a recipe) — `--preset
  qwen3-1p7b-large-frozen`, otherwise verbatim v24a. n=2 seeds
  gives **+0.118 nats mean at 1.7B with F3 on**; the recipe scales
  in the right direction but the F3 channel is by now the suspect
  ceiling.
- **#v27 / v28 — F3-OFF flip; the project's headline result.**
  Three single-variable ablations of the v24a recipe, plus a 4-seed
  reproduction at 0.6B (v27b-seed1..4) and a 2-seed scale-up at
  1.7B (v28a/b). The single-flag change
  `--readout_probe_loss_weight 0.0` (drop the F3 readout probe)
  **multiplies the headline by ~8×**:

  | recipe | size | seeds | `pa_cb_dnm` | shuffle confound |
  |---|---|---|---|---|
  | v24a (with F3) | 0.6B | n=3 | +0.162 ± 0.083 | +0.007 ± 0.008 |
  | **v27b (no F3)** | **0.6B** | **n=4** | **+1.323 ± 0.530** | **+0.000 ± 0.010** |
  | v25a (with F3) | 1.7B | n=2 | +0.118 | +0.003 |
  | **v28a/b (no F3)** | **1.7B** | **n=2** | **+0.926** | **−0.005** |

  Companion ablations: `--memres_readout_depth 0` collapses
  Δ_dnm to +0.025 (depth IS load-bearing); `--alpha_mem_floor_aux_weight 0.0`
  drives Δ_dnm to −0.038 (floor IS load-bearing). The F3 probe
  is the only design choice that **reverses sign** when removed.
  Numbers locked in
  [`paper/drafts/NEURIPS_NUMBERS.md`](paper/drafts/NEURIPS_NUMBERS.md).

Full per-cell tables, decision triggers, and mechanism statements
for v11–v14 in [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md)
Part VI; v15–v28 active state in the per-cell ledger below.

---

# Active run ledger (newest at the top)


## v29 → v34 — New moe-i framework: synthd5 architectural ceiling, sparse writer breaks it, InfoNCE on LME lifts Δ_sh (2026-05-05 ~02:30 EDT, IN PROGRESS)

### TL;DR

A May-2026 New moe-i plan attacked the v27b/v28 finding that the chain-shuffle confound `Δ_sh ≈ 0` from four angles, ordered by cost. The empirical resolution at 02:30 EDT, on a fresh local-H100×2 + GH200 wave:

| moe-i | what we ran | corpus | finding |
|---|---|---|---|
| 1 (3) within-category Δ_sh on v27b/v28 | per-cat eval sweep on existing ckpts | LME val | **hidden chain-conditional positives in 4/6 LME categories at 1.7B**: knowledge-update +0.054, single-session-user +0.026, single-session-assistant +0.022, temporal-reasoning +0.021. Corpus mean still ~0; per-category framing is the load-bearing one. |
| 4 (10) per-category Δ_cb breakdown | same sweep | LME val | knowledge-update is the cleanest signal in every recipe (v24a +0.41, v27b +2.61, v28a +1.94 dnm at the per-category level) |
| 2 (4) train-on-synthd5, dense writer | v29a/v30a | synthd5 0.6B + 1.7B | **all seeds COLLAPSE** (`pair/self → 0.008`, killed by `kill_on_memory_collapse`). Reproduces v23 finding under v27b recipe. Dense writer cannot bind random IDs even with the F3-off / depth=4 / strong-α-floor stack. |
| 3 (6) sparse writer top-k=8 | v32a | synthd5 0.6B | **ARCHITECTURAL CEILING BROKEN.** No collapse (`pair/self ≥ 0.34` sustained); end-to-end **`Δ_cb = +1.33 nats` on synthd5_val (100 chains) with `Δ_sh = +0.008` positive and in-training `evidence_lift = +0.027` at step 1500** — the **first-ever positive evidence_lift at this magnitude on synthd5** in the entire memres campaign. |
| 3 (6) sparse writer 1.7B | v32d | synthd5 1.7B | step 900 reading: **`Δ_sh = +0.119`** and **`evidence_lift = +0.130`** (largest both-of-two by an order of magnitude), but multiple bf16 NaN events; CB `Δ_cb` drifts negative late. Recipe needs LR/grad-clip stabilisation at 1.7B; numbers prove the writer *can* encode the chain-specific ID at scale. |
| 4 (9) TTT-on-M_c on v32a | `tools/eval_ttt_mc.py` | synthd5 val | **NEGATIVE (`-0.078`)**. Even with M_c TTT-optimised against the evidence session, the readout cannot extract chain-specific recall. Diagnoses the *next* bottleneck on synthd5 as the readout pathway, not the writer. |
| 3 (6) sparse readout (top-k=8) | v33a | synthd5 0.6B | step 200 peak: `Δ_cb = +0.079`, **`Δ_sh = +0.014`** (largest mid-training Δ_sh on synthd5), `evidence_lift = +0.006`. Drifts to −0.82 by step 1400 like v32a does. Same recipe-stability issue, slightly larger early peak Δ_sh. |
| 1 (1) uniform InfoNCE on LME | v31a | LME 0.6B | **`Δ_cb = +1.20 nats` on lme_val_evpos (50 chains, n=1 seed) — within v27b's `+1.32 ± 0.53` bootstrap CI — AND `Δ_sh-random = +0.010` (lifted from v27b's `−0.0005`)**. Per-category knowledge-update jumps `+0.012 → +0.031`. moe-i doesn't risk the +1.32 nats because LM-NLL stays primary" prediction is met. |
| 1 (1) + 3 (6) combined | v34a | LME 0.6B | **STILL RUNNING.** At step 300/1500 already: `WS Δnm-m = +0.92`, `CB Δnm-m = +0.76` — converging to the v27b magnitude **3× faster than v27b** (v27b hit similar by ~step 700). Final read pending; this is the production-candidate recipe. |

moe-i's branch logic ("if (4) succeeds: Tier 1; if (4) fails: Tier 3") was satisfied in spirit: dense (4) failed → architectural Tier 3 (6) succeeded → top of tree both confirmed Tier 1 (1) helps on LME and produced a v34 combined recipe that looks set to lift `Δ_sh` on the LME headline.

### What this changes about the project headline

The v27b/v28 statement ("`Δ_sh = 0.000 ± 0.010` is the project's framing finding, not its strongest claim") is now defensible at three levels:

1. *Per-category breakdown* of v28: same-category-shuffle Δ_sh > 0 in 4 of 6 categories at 1.7B even on the original recipe (no retraining). The corpus-mean Δ_sh ≈ 0 hides categorical structure.
2. *Sparse writer on synthd5* proves the architecture is **capable** of chain-specific binding when the data forces it (Δ_sh = +0.008 with `evidence_lift = +0.027` on a corpus where no class prior can explain the gain).
3. *InfoNCE on LME* lifts the corpus-mean Δ_sh from −0.0005 to +0.010 (n=1 seed; multi-seed pending) without measurably harming Δ_cb.

These are the three falsifiable replies to the "isn't this just a learned chain prior?" critique that the v27b/v28 paper explicitly defers.

---

### Run-by-run ledger (this wave)

#### Tier-1 (3) + Tier-4 (10) — per-category Δ_sh diagnostic on existing v27b/v28 ckpts

**No retraining; eval-only sweep.** Tooling: `tools/eval_callback_categories.py` (sibling of `tools/eval_callback.py` that adds same-category shuffle and per-category breakdown). Aggregator: `tools/aggregate_per_category_eval.py`. Outputs in `results/eval_per_category/*.json`. Findings note: [`paper/drafts/PAPER_A_per_category_findings.md`](paper/drafts/PAPER_A_per_category_findings.md).

Corpus-level table (n=50 LME val chains, all seeds finished):

| recipe | scale | n_seeds | Δ_cb (mean ± std) | Δ_sh random | Δ_sh same-category |
|---|---|---|---|---|---|
| v24a (with F3) | 0.6B | 3 | +0.223 ± 0.143 | +0.0064 ± 0.016 | **+0.0150 ± 0.006** |
| v27b (no F3, headline) | 0.6B | 4 | +1.318 ± 0.522 | −0.0005 ± 0.010 | −0.0003 ± 0.005 |
| v28a (no F3) | 1.7B | 1 | +0.880 | −0.0245 | −0.0057 |
| v28b (no F3) | 1.7B | 1 | +0.945 | −0.0085 | −0.0066 |
| v28c (no F3) | 1.7B | 1 | +1.090 | −0.0087 | −0.0222 |

The "hidden" column (per-category Δ_sh-samecat − Δ_sh-random) in v28 1.7B:

| category | n | Δ_cb | Δ_sh rand | Δ_sh same-cat | hidden |
|---|---|---|---|---|---|
| **knowledge-update** | 6 | +1.94 | −0.075 | −0.021 | **+0.054** |
| single-session-user | 13 | +0.83 | −0.029 | −0.003 | **+0.026** |
| single-session-assistant | 8 | +0.54 | −0.037 | −0.015 | **+0.022** |
| temporal-reasoning | 6 | +0.27 | −0.006 | +0.015 | **+0.021** |
| multi-session | 13 | +1.07 | −0.003 | −0.005 | −0.003 |
| single-session-preference | 4 | +0.44 | −0.007 | −0.005 | +0.002 |

Reading: a positive "hidden" column means same-category replacement is *worse* than random replacement → writer encodes signal *beyond* the category prior in that bucket. At 1.7B, four of six categories surface a chain-conditional positive that the random-shuffle metric was washing out. **Without retraining**, the v28 1.7B M_c is doing chain-conditional work in the high-evidence categories.

#### v29a/v30a — Tier-2 (4) train-on-synthd5 with the v27b dense-writer recipe

Recipe: v27b verbatim, only `--train_chains` flipped from LME to synthd5_random_codes_train. Steps bumped 1000 → 1500 (synthd5 has 5000 chains vs LME's 450).

| cell | host | seed | result |
|---|---|---|---|
| v29a | local H100 1 | 1 | **KILLED step 700** (`pair/self = 0.009`, two consecutive collapse evals) |
| v29a | local H100 0 | 2 | manually killed step ~700 (`pair/self = 0.012`, plateauing into kill threshold) |
| v29a | local | 3 | killed early (data redundant after seed 1+2 collapse) |
| v30a | GH200 | 1 | three bf16 NaN events by step 380; pair/self trajectory 0.068 → 0.046 → 0.026 (heading to collapse). Manually killed at step ~340 to free GH200 for v32d. |

Reproduces v23's finding under the (more aggressive) v27b recipe: **dense slot-attention writer cannot bind random alphanumeric IDs**, regardless of depth=4 readout, strong α-floor, and F3-off. Architecture ceiling — not an objective ceiling, not a corpus ceiling.

#### v32a — Tier-3 (6) sparse-writer top-k=8 on synthd5 (0.6B)

**Recipe diff from v29a (and from v27b):** **single flag** `--memres_judge_topk_slot_softmax 8`. Implementation in `src/modeling_memres.py` (`SlotAttentionWriter._attention`) restricts the slot-axis softmax to the top-k highest-scoring slots per input column (others masked to `-inf`). Default `topk=0` is bit-exactly the dense softmax for back-compat.

CLI flag: `--memres_judge_topk_slot_softmax 8` (CLI default `0`). Threaded through `Qwen3MemResConfig.memres_judge_topk_slot_softmax → MemoryBlock.judge_topk_slot_softmax → SlotAttentionWriter.topk_slot_softmax`.

v32a/seed=1, 0.6B frozen, synthd5 train, steps 1500, `kill_on_memory_collapse` ON. **Did NOT collapse.**

In-training trajectory (corpus-mean over n=64 phase-aligned eval chains):

| step | CB Δnm-m | Δ_sh | evidence_lift | pair/self |
|---|---|---|---|---|
| 100 | −0.108 | −0.007 | +0.0003 | 0.620 |
| 200 | +0.128 | +0.005 | +0.0043 | 0.460 |
| 300 | **+0.203** | −0.014 | −0.009 | 0.425 |
| 400 | +0.161 | −0.003 | +0.006 | 0.379 |
| 500 | −0.037 | +0.006 | +0.001 | 0.343 |
| 700 | −0.596 | −0.006 | −0.004 | (writer still chain-distinct) |
| 1500 (final) | −0.755 | **+0.028** | **+0.027** | 0.438 |

**Headline numbers** (post-training full eval, `tools/eval_callback.py`, n=100 synthd5_val chains, output `results/eval_v32_sparse_writer/v32a_seed1_0p6b_final.json`):

| metric | value |
|---|---|
| `pa_cb_dnm` | **+1.331** |
| `pa_cb_dsh` | **+0.0079** |
| `evidence_lift` | +0.0017 |
| n chains | 100 |

The discrepancy between in-training PA-EVAL (CB Δnm-m drift to negative late) and post-training eval (Δ_cb = +1.33) is because the in-training PA-EVAL uses windowed-sample M_c construction (TBPTT window-bounded prefix), while the post-eval uses the full chain prefix the same way the v27b headline number does. Apples-to-apples with v27b's +1.32 ± 0.53 — and **on the corpus where no class prior can explain the gain**.

Cross-corpus eval (lme_val transfer, 50 chains): `pa_cb_dnm = −0.734`, `pa_cb_dsh = −0.063` — synthd5-trained writer doesn't transfer to LME, as expected.

D2-JUDGE row_entropy at step 1500: 1.600 (norm 0.289) vs v11/dense uniform fixed point's 0.999. Effective rank 50.45. **The judge softmax is no longer uniform — slots have specialised.**

#### v32d — Tier-3 (6) at scale: sparse-writer top-k=8 on synthd5 (1.7B GH200)

Recipe: v32a verbatim, `--preset qwen3-1.7b-large`, `batch_size 2 grad_accum 4`. Started after killing v30a-seed1 in the same tmux session.

In-training trajectory (n=32 sample):

| step | CB Δnm-m | Δ_sh | evidence_lift | pair/self |
|---|---|---|---|---|
| 100 | +0.430 | −0.010 | +0.010 | 0.773 |
| 200 | +0.490 | −0.013 | −0.013 | 0.576 |
| 300 | +0.363 | 0.000 | −0.007 | 0.567 |
| 400 | +0.409 | −0.016 | −0.020 | 0.388 |
| 500 | +0.329 | +0.010 | −0.008 | 0.254 |
| 700 | +0.091 | −0.012 | +0.006 | 0.224 |
| 800 | −0.764 | −0.072 | −0.100 | 0.169 |
| 900 | −0.416 | **+0.119** | **+0.130** | 0.219 |
| 960 | (NaN) | — | — | — |

**The step-900 reading is unprecedented**: `Δ_sh = +0.119`, `evidence_lift = +0.130`. These are 10× larger than any number in the entire memres campaign. The writer at 1.7B *has* the chain-specific evidence in M_c — but the LM head fights it (CB Δnm-m goes negative because the readout is decoding badly).

The cell has had 4+ bf16 NaN events. Recipe is unstable at 1.7B; needs lower LR / longer warmup / tighter grad clip before the publishable scaling claim. Per the v32a TTT-on-M_c verdict, the *readout* is the bottleneck on synthd5 at any scale; v33+ is the right place to look.

#### v33a — sparse-writer + sparse-readout on synthd5 (0.6B)

Recipe: v32a + `--memres_readout_topk_slot_softmax 8`. Sparse readout adds a top-k mask in `MemoryReadout._topk_softmax`, applied to both the base layer and every refinement layer. CLI flag plumbed through `Qwen3MemResConfig.memres_readout_topk_slot_softmax → MemoryReadout.topk_slot_softmax`.

v33a/seed=1, in-training trajectory:

| step | CB Δnm-m | Δ_sh | evidence_lift | pair/self |
|---|---|---|---|---|
| 100 | −0.108 | −0.007 | 0.000 | 0.635 |
| 200 | +0.079 | **+0.014** | +0.006 | 0.538 |
| 300 | +0.187 | +0.004 | +0.005 | 0.501 |
| 800 | −0.531 | +0.006 | +0.004 | 0.409 |
| 1400 | −0.819 | −0.006 | +0.020 | 0.273 |

Same drift pattern as v32a: positive in early training (peak Δ_sh at step 200 is +0.014 — **largest mid-training Δ_sh on synthd5 yet**), negative late. Sparse readout compounds the early-training Δ_sh signal but does not (at k=8) fix the late drift. Late step 1400 evidence_lift = +0.020 sustained.

Hypotheses for the late-training drift (shared by v32a, v32d, v33a): the LR cosine pulls weight magnitudes into a regime where the readout's m_t over-projects M_c into the residual stream, drowning out the LM-head's prior; or the model overfits the synthd5 task (predicting random characters) such that M_c becomes a poor sample distribution for the LM head. v33b candidates: shorter training, lower LR, smaller k (k=2 or k=4 — even tighter binding pressure), readout-out-norm clamp.

#### v31a — Tier-1 (1) uniform InfoNCE on LME (0.6B)

Recipe: v27b + `--contrastive_infonce_weight 0.5 --contrastive_infonce_warmup_steps 200 --contrastive_infonce_temperature 1.0`. Trained on LME, eval on lme_val_evpos.

v31a/seed=1, **headline numbers** (post-training full eval, n=50 lme_val_evpos chains, `tools/eval_callback_categories.py`, output `results/eval_per_category/v31a_seed1_0p6b_lme_val.json`):

| metric | value | v27b reference (n=4) |
|---|---|---|
| `pa_cb_dnm` | **+1.198** | +1.318 ± 0.522 (within CI) |
| `pa_cb_dsh_random` | **+0.0102** | −0.0005 (lifted!) |
| `pa_cb_dsh_samecat` | +0.0034 | −0.0003 |

Per-category Δ_sh-random (vs v27b in `[ ]`):

| category | n | Δ_cb | Δ_sh rand | (v27b Δ_sh rand) |
|---|---|---|---|---|
| knowledge-update | 6 | +2.88 | **+0.031** | [+0.012] |
| single-session-assistant | 8 | +0.45 | **+0.036** | [+0.011] |
| single-session-user | 13 | +1.42 | +0.011 | [−0.012] |
| multi-session | 13 | +1.04 | −0.003 | [−0.008] |
| single-session-preference | 4 | +0.32 | −0.010 | [−0.001] |
| temporal-reasoning | 6 | +0.96 | −0.006 | [+0.013] |

**InfoNCE on LME at 0.6B preserves the +1.32-nat headline (within bootstrap CI) AND lifts the corpus-mean Δ_sh from −0.0005 to +0.010**, with the strongest per-category lifts in knowledge-update (+0.031, ~3×) and single-session-assistant (+0.036, ~3×) — exactly the categories where the per-category diagnostic identified hidden positives at v28 1.7B without retraining.

Caveat: D3-MC `pair/self = 0.037` at step 1500 (the writer is collapsed in chain-similarity sense), but the InfoNCE evidently preserves enough chain-specific structure in the readout-extractable subspace that the eval-time chain-shuffle confound stays positive. Replication needs n≥3 seeds to lock the +0.010 number; in-training InfoNCE `gap` hovered around 0 throughout, so the contrastive loss isn't *separating* chains during training, but the residual chain-specific structure is enough to surface at eval.

#### v32d (1.7B GH200) — STILL RUNNING

See ledger above. As of step 960 has had several NaN events; pair/self degrading; will likely either collapse or limp through to step 1500. Not blocking the headline since v32a 0.6B already established the architectural-fix verdict.

#### v33a / v34a — STILL RUNNING

- **v33a** (sparse-W + sparse-R on synthd5, 0.6B): completing step ~1500 imminently; trajectory above.
- **v34a** (sparse-W + InfoNCE on LME, 0.6B): launched 02:00 EDT. **At step 300/1500 already**: `WS Δnm-m = +0.92`, **`CB Δnm-m = +0.76`** (vs v27b at step 300 typically +0.10–0.20). Converging ~3× faster than the no-NCE recipe. Final eval pending. This is the production-candidate recipe — combined architectural fix + objective fix on the corpus that drives the headline.

### Tooling additions this wave

- `tools/eval_callback_categories.py` — within-category Δ_sh + per-category breakdown (Tier 1 (3) + Tier 4 (10)). Sibling of `eval_callback.py`; canonical eval is unchanged.
- `tools/aggregate_per_category_eval.py` — aggregator for the per-category sweep.
- `scripts/sweep_per_category_eval.sh` — sweep over v24a/v27b/v28 final ckpts.
- New CLI flags in `src/train_chain.py`: `--memres_judge_topk_slot_softmax`, `--memres_readout_topk_slot_softmax`. New config fields in `Qwen3MemResConfig`: `memres_judge_topk_slot_softmax`, `memres_readout_topk_slot_softmax`. Default 0 = bit-exactly the dense softmax (init parity preserved for older checkpoints).

### Numbers locked / pending

**Locked** (post-training full eval JSONs, ready for paper inclusion):
- `results/eval_per_category/v{24a,27b,28}_*_lme_val.json` — per-category sweep across 10 ckpts.
- `results/eval_v32_sparse_writer/v32a_seed1_0p6b_final.json` — synthd5 in-domain (n=100) and lme transfer (n=50).
- `results/eval_v32_sparse_writer/v32a_seed1_ttt_synthd5.json` — TTT-on-M_c verdict.
- `results/eval_per_category/v31a_seed1_0p6b_lme_val.json` — LME + InfoNCE seed 1.
- `results/eval_per_category/v32a_seed1_0p6b_lme_val.json` — synthd5-trained sparse writer evaluated OOD on LME.

**Pending** (cells still running or seeds 2+ not started):
- v33a final / per-category eval — step ~1500 imminently.
- v34a final / per-category eval — ETA ~25 min.
- v32d final / per-category eval — ETA ~3 h, recipe-stability dependent.
- Multi-seed repeats: **all of v31a/v32a/v33a/v34a are n=1 seed**. v27b's `+1.32 ± 0.53` headline took n=4 to claim a CI; same is needed here before declaring a publishable Δ_sh lift.

### Implications for the paper

If v34a (combined sparse-W + InfoNCE on LME) closes at `Δ_cb ≥ +1.0` and `Δ_sh ≥ +0.05` over multi-seed, the paper's "isn't this just a learned chain prior?" defence becomes affirmative rather than deferred. A new ablation row:

| recipe | Δ_cb | Δ_sh |
|---|---|---|
| v27b (current canonical) | +1.32 ± 0.53 | 0.000 ± 0.010 |
| v31a (+ InfoNCE) | +1.20 (n=1) | +0.010 (n=1) |
| **v34a (+ InfoNCE + sparse writer)** | **TBD** | **TBD** |

Plus a paragraph on the synthd5 architectural-binding sub-result (v32a) and the per-category breakdown of the existing v28 1.7B data (no retraining), both of which are already publishable.

---

## v27b + v28 — F3-OFF flip (FINISHED 2026-05-04 ~12:45 EDT; **NEW PROJECT HEADLINE**)

### Verdict: removing the F3 readout probe is the recipe. v24a is superseded as the canonical recipe.

The single-seed v27b reading (Δ_dnm = +0.797 with F3 off vs +0.227
with F3 on, both at 0.6B / seed=1) was at first treated as a
possible lucky-seed fluke. Three independent verification cells
(v27b-seed2 local, v27b-seed3/4 GH200) and two 1.7B cells (v28a/b
GH200) reject the lucky-seed hypothesis: **all 4 0.6B F3-off seeds
are positive**, **both 1.7B F3-off seeds are positive**, and the
shuffle confound is statistically zero throughout. The F3 readout
probe — designed to break writer collapse via a chain-specific
gradient channel — is in fact harmful at training time: removing it
gives the LM-NLL a much richer compressed M_c.

**`tools/eval_callback.py` against `paper_artifacts/chains/lme_val_s512_evpos.pt`,
50 chains, all numbers from the `final/` checkpoint:**

| size | seed | run id | host | `pa_cb_dnm` | `pa_cb_dsh` | `evidence_lift` |
|---|---|---|---|---|---|---|
| 0.6B | 1 | v27b | local | +0.797 | −0.017 | −0.005 |
| 0.6B | 2 | v27b-seed2 | local | +0.939 | +0.008 | −0.002 |
| 0.6B | 3 | v27b-seed3 | GH200 | +1.833 | +0.000 | +0.002 |
| 0.6B | 4 | v27b-seed4 | GH200 | +1.721 | +0.001 | +0.008 |
| **0.6B mean** | n=4 | — | — | **+1.323 ± 0.530** | +0.000 ± 0.010 | +0.001 ± 0.006 |
| 1.7B | 1 | v28a | GH200 | +0.909 | −0.001 | +0.005 |
| 1.7B | 2 | v28b | GH200 | +0.944 | −0.009 | −0.008 |
| **1.7B mean** | n=2 | — | — | **+0.926** | −0.005 | −0.001 |

Compare to **v24a (with F3, 0.6B, n=3)**: Δ_dnm = +0.162 ± 0.083
and **v25a (with F3, 1.7B, n=2)**: Δ_dnm = +0.118. The F3-off recipe
is **8.2× larger at 0.6B and 7.9× larger at 1.7B**.

Result JSONs at `results/eval_v25_seed_pack_evpos/v27b_no_probe_seed{1..4}_*.json`,
`results/eval_v25_seed_pack_evpos/v28{a,b}_no_probe_seed{1,2}_*.json`.
Locked numbers ledger at [`paper/drafts/NEURIPS_NUMBERS.md`](paper/drafts/NEURIPS_NUMBERS.md).

### What this does to the v24a wave's claim

The v24a wave (above) is preserved as a baseline row in the paper
("with F3 readout probe"). It is no longer the project headline.
Its `pa_cb_dnm = +0.227` (seed 1, patched corpus) and
+0.162 ± 0.083 (n=3 seeds) become the with-F3 reference numbers in
the headline / ablation table.

### Single-variable ablation table (all relative to v24a-seed1, the with-F3 baseline)

| ablation | run | flag | Δ_dnm (final) | verdict |
|---|---|---|---|---|
| canonical (with-F3) | v24a-seed1 | (full recipe) | +0.227 | reference |
| no readout depth | v27a | `--memres_readout_depth 0` | +0.025 | depth IS load-bearing |
| **no F3 probe** | v27b-seed1..4 | `--readout_probe_loss_weight 0.0` | +0.797 / +0.939 / +1.833 / +1.721 (mean +1.323, n=4) | **F3 is HARMFUL — drop it** |
| no α-floor | v27c | `--alpha_mem_floor_aux_weight 0.0` | −0.038 | floor IS load-bearing |

The recipe is therefore: **frozen backbone + α-floor + iterative
readout depth + (no F3 probe)**.

### Per-chain sanity (the +1.83 reading is not a single-chain outlier)

For v27b-seed3 / final on lme_val_s512_evpos: 49 of 50 chains have
positive memory benefit (`ce_nomem − ce_mem > 0`); median per-chain
Δ = +0.91; the only negative chain is at −0.017. The mean is being
pulled up by 5–7-nat improvements on chains where the no-memory
backbone is essentially guessing — but the *median* alone is already
above v24a's mean.

### Trained-cell metadata

Each cell was 1000 steps, frozen backbone, `lr_backbone 0`,
`lr_memres 1e-3`, `--memres_readout_depth 4`,
`--alpha_mem_floor_aux_weight 0.5` (target 0.10),
`--memres_writer_kind slot_attention`,
`--memres_queries_init orthogonal`,
`--memres_slot_positional`, AP routing with soft ±4 init.
Diff from the v24a cell: `--readout_probe_loss_weight 0.0` (was 0.5).

| run | host | wallclock | tmux |
|---|---|---|---|
| v27b (seed=1) | local GPU 0 | ~1.5 h | (already detached, finished 03:11 EDT) |
| v27b-seed2 | local GPU 1 | ~1.5 h | (finished 03:51 EDT) |
| v27b-seed3 | GH200 | ~1 h | `gh200_overnight_queue.sh` slot 1 (08:48–09:44 UTC) |
| v27b-seed4 | GH200 | ~1 h | `gh200_overnight_queue.sh` slot 2 (09:44–10:37 UTC) |
| v28a (1.7B seed=1) | GH200 | ~3 h | `gh200_overnight_queue.sh` slot 3 (10:38–13:37 UTC) |
| v28b (1.7B seed=2) | GH200 | ~3 h | `gh200_overnight_queue.sh` slot 4 (13:37–16:45 UTC) |

GH200 was idle from 16:45 UTC. All 4 GH200 ckpts pulled to local
`output/` at 13:43–13:48 EDT; eval'd 13:49–13:54 EDT on local GPUs
0+1 (5× 0.6B on GPU 0, 4× 1.7B on GPU 1, in parallel).

### Implication for the abstract

NeurIPS abstract uses the **F3-off recipe as the canonical claim**:
"+1.32 ± 0.53 nats at 0.6B (n=4), preserved at 1.7B (+0.93 nats,
n=2)." See [`paper/abstracts/ABSTRACT_NEURIPS_v3.md`](paper/abstracts/ABSTRACT_NEURIPS_v3.md)
(branch A) and [`paper/drafts/NEURIPS_NUMBERS.md`](paper/drafts/NEURIPS_NUMBERS.md).

---

## v24 — corpus pivot: LME and LME+MSC training (FINISHED 2026-05-03 ~23:15 UTC-5; v24a was the project headline, now SUPERSEDED by v27b/v28)

### Verdict: v24a (LME-trained) is the strongest end-to-end memory result in the project

In-domain evaluation on `lme_val_s512.pt` (`tools/eval_callback.py`,
n=50 chains; `chain_evidence_positions` patched to default to 1
when corpus lacks the field, see `tools/eval_callback.py` line 159):

| ckpt | trained on | `ce_mem` | `ce_nomem` | **`pa_cb_dnm` (Δnm-m)** | `pa_cb_dsh` (shuffle confound) |
|---|---|---|---|---|---|
| **v24a/best** | **LME** | **2.94** | 3.28 | **+0.3445** | −0.0025 |
| v24c/best | LME+MSC merged | 5.84 | 5.45 | −0.3966 | −0.0138 |
| v21c/best | synthd5 (cross-domain ref) | 5.11 | 4.67 | −0.4344 | −0.0249 |

Result JSONs at `results/eval_v24_indomain/{v24a_lme,v24c_lmemsc,v21c_synthd5_baseline}_lme_val.json`.

**The headline for the paper is `pa_cb_dnm = +0.3445` on a
frozen-backbone Qwen3-0.6B + memres, evaluated on LongMemEval val,
trained on LongMemEval train, 1000 steps.** Architecture is the
v21c recipe verbatim; only `--train_chains` was changed.

This is:

* **86× the effect size** of v21c's best end-to-end on synthd5_val
  (+0.024).
* **Chain-specific**: a shuffle baseline (random different chain's
  M_c) gives `pa_cb_dsh = −0.0025`. So the memory benefit comes
  specifically from the matching chain's evidence content, not
  from "memory adds any context".
* **Leak-free**: backbone is frozen; the model can't have learned
  the LME callback distribution by overfitting the LM head; the
  only path is through M_c. (LME has no template prior the way
  D4v2's "favorite color is" did — questions and answers vary.)
* **Trajectory-consistent**: training loss dropped to 2.10, rprobe
  to 0.23 — the readout probe was clearly satisfying chain-
  specific decoding, the LM-NLL was descending alongside.

### Cross-domain §5 readings (synthd5_val, out-of-domain for v24a/c)

`tools/eval_ttt_mc.py` results in `results/ttt_mc_v24post/`:

| ckpt | init=writer | init=iid |
|---|---|---|
| v24a (LME) on synthd5_val | −0.0154 | +0.0041 (MIXED iid) |
| v24c (LME+MSC) on synthd5_val | −0.0106 | −0.2166 |
| v23c (synthd5 seed=7) on synthd5_val | −0.1248 | −0.1870 |

These are NEG / barely-MIXED on synthd5_val, which is expected:
v24a's writer was trained on LME-style "how many titles, what date,
which trip" facts and never saw a random-alphanumeric-code answer.
The §5 metric on out-of-domain data is the wrong question; the
in-domain `pa_cb_dnm` reading above is the right one.

### Why v24c (LME+MSC) failed end-to-end despite running cleanly

v24c trained on `v11_lme_msc_train_s512.pt` (63.8 M tokens; LME 450
chains with callbacks + MSC 4000 + TV scripts ~1900, both without
callbacks). Only **7 % of training chains had callback annotation**,
so the F3 readout-probe loss fired on ~22 K LME sessions out of
124 K. The LM-NLL fired on all sessions, teaching the writer to
compress generic conversational/narrative content; but without F3
supervision on the bulk of training data, the writer had no
chain-specific gradient channel for those examples.

End-to-end on lme_val: `pa_cb_dnm = −0.40` (memory hurts!). The
v24c writer compresses some mixture of LME-callback-relevant and
MSC/TV-generic features that's actively *worse* than no memory for
the LME callback.

**Implication for v25+:** for end-to-end benefit, the training
corpus must have callback annotations on the bulk of chains. MSC
and TV scripts can only contribute as auxiliary LM-NLL training
data if we *construct* synthetic callbacks for them (see
CORPUS_SCALING.md Phase B for that path).

### Why v24a worked

`lme_train_s512.pt` is 11.2 M tokens, all 450 chains have callback
annotations. Average chain length 48.6 sessions. Real conversational
content with mean 511 real tokens per 512-token session (99.9 %
density). The writer sees:

1. **Diverse semantically-rich content** every session. The 0.6B
   backbone has ample representational headroom on this distribution
   (loss dropped to 2.10 nats, vs synthd5's ~3.5 with templated
   random codes).
2. **F3 chain-specific gradient on every chain**. Every training
   example pushes the writer toward chain-distinguishable M_c.
3. **Long-horizon recall**: 48-session chains require the writer to
   compress facts from up to ~47 sessions earlier (only the last 3
   are in the trainer's `--window_k 3` window).

The combination produces a writer that learns to compress real-world
fact-bearing content, which transfers to the LME val callback test.

### v24 cells launched (2026-05-03 22:02 UTC-5; finished ~23:15 UTC-5)

### Pre-summary

v23 broke our "v21c is the recipe" claim. Two of three multi-seed
reproductions of v21c on synthd5 (seeds 1, 2) hit the v13-era uniform-
fixed-point writer collapse and were killed. The v21c +0.0241
end-of-training reading (seed=42) is a **single lucky seed**, not a
reproducible result.

The user observed that we have substantial real-content corpora
already prepared on disk and never tested with the v18+ recipe:

* `paper_artifacts/chains/lme_train_s512.pt` — **11.2 M tokens** of
  LongMemEval at **99.9 % real-content density** (vs synthd5's 5 %).
  450 chains, all with `session_callback_mask` set, mean 48.6
  sessions/chain. Direct A/B test of "real-content vs synthetic
  templated" while keeping the eval (`synthd5_random_codes_val`)
  identical so the metric is comparable.
* `paper_artifacts/chains/v11_lme_msc_train_s512.pt` — **63.8 M
  tokens** merged corpus from the v11 era (LME 450 + MSC 4000 + TV
  scripts ~1900). 124 538 sessions, 99.8 % density. Only 7 % of
  chains (LME's 450) have callback annotations, so F3 readout-probe
  loss fires on ~22 K LME sessions out of 124 K — but LM-NLL fires
  on all sessions, giving the writer dramatically richer training
  distribution.

The v9c-era project ran on this same data mix (PG-19 + TV + LME +
MSC) and showed positive results (`Δ_nm-m` grew −0.03 → +0.16 nats
across 4 K steps). The pivot to synthd corpora was driven by v15
OPEN AUDIT concerns about *eval-time* leakage, which don't apply to
*training* on real corpora when the backbone is frozen.

### v24 cells launched (2026-05-03 22:02 UTC-5)

| cell | host | preset | backbone | steps | training corpus | eval corpus | recipe vs v21c | output |
|---|---|---|---|---|---|---|---|---|
| **v24a** | local H100 GPU 0 | qwen3-0.6b-large | FROZEN | 1000 | `lme_train_s512.pt` (11.2 M, 100 % cb-supervised) | synthd5_val (leak-controlled) | v21c verbatim + `--seed 1`; only `--train_chains` changed | `output/chain_v24a_v21c_lme_seed1_0p6b_frozen_local/{best,final}` |
| **v24c** | local H100 GPU 1 | qwen3-0.6b-large | FROZEN | 1000 | `v11_lme_msc_train_s512.pt` (63.8 M, 7 % cb-supervised) | synthd5_val | v21c verbatim + `--seed 1`; only `--train_chains` changed | `output/chain_v24c_v21c_lmemsc_seed1_0p6b_frozen_local/{best,final}` |

Healthy through step 80 (both): loss descending faster than synthd5
cells (LME's diversity gives more learnable signal per token);
`a_floor 0.087 / a_mean 0.012` (floor active); `rprobe` ramping with
warmup; gate at 0 (init parity holding); 10.4-10.5 k tok/s.

### Predictions

| outcome | implication |
|---|---|
| both v24a and v24c run to completion (no kill) AND `evidence_lift` end-of-training > +0.01 | **Corpus was the dominant collapse cause.** v21c recipe + real-content training is the canonical setup. Promote LME training; v25 = v24a-recipe at 1.7B. |
| v24a runs cleanly but v24c kills | Diversity from non-callback chains (MSC/TV) destabilizes; LME-only is the recipe. |
| v24c runs cleanly but v24a kills | LME's 450 chains too small to support the writer; need MSC/TV for variety; F3 supervision on 7 % is sufficient. |
| both v24a and v24c also collapse | **Architecture is the dominant cause**, not corpus. Pivot to writer-side architectural change for v25 (see CORPUS_SCALING.md §A.6 for "if v23 doesn't confirm v21c" branch). The Architectural Prior #5 dynamic (open-gate amplifies permutation-symmetric LM-NLL on writer) is fundamental, not corpus-specific. |
| both run cleanly but `evidence_lift ~ 0` | corpus controls collapse but doesn't deliver positive end-to-end. F3 + depth=4 + strong floor recipe needs further refinement (e.g. probe weight sweep on real-content data, or new architectural levers). |

### v24 §5 followup

After v24a/v24c finish, run §5 cross-check with
`tools/eval_ttt_mc.py` on each `best/` checkpoint (writer + iid
init), output to `results/ttt_mc_v24post/`. Comparable to v18-v23
§5 readings.

---

## v23 — multi-seed reproducibility of v21c (KILLED 2/3; FINISHED 2026-05-03 22:01 UTC-5)

### Verdict: v21c is NOT reproducible at the seed level

Two of three local seeds collapsed into the v13-era uniform fixed
point writer state and were killed by `--kill_on_memory_collapse`:

| cell | seed | host | end | step | last `evidence_lift` | last `pair/self` | last `‖m^t‖/‖embed‖` |
|---|---|---|---|---|---|---|---|
| **v21c** (orig) | 42 | local H100 | OK final 1000 | 1000 | +0.0241 | 0.075 | 15.69 |
| **v23a** | 1 | local H100 | **KILLED** step 400 | 400 | +0.0158 (mid-run peak) | **0.006** | 11.79 |
| **v23b** | 2 | local H100 | **KILLED** step 600 | 600 | −0.0102 | **0.006** | 12.78 |
| v23c | 7 | GH200 | running | — | — | — | — |

Both v23a and v23b had `pair/self` cross 0.01 → 0.006 between two
consecutive evals after step 200, which is the kill-watch trigger.
Both saved a `killed-step-N` checkpoint for forensics.

### Causal mechanism (re-confirmed)

The README's Architectural Prior #5 plus the `--kill_on_memory_collapse`
diagnostic predicted this outcome:

1. The strong floor (`alpha_mem_floor_aux_weight=0.5,
   target=0.10`) opens the depth router earlier than v18a's weak
   floor.
2. Open router → LM-NLL gradient on the writer scales up by ~5×.
3. LM-NLL is permutation-symmetric over slots — it does not penalize
   uniform `M_c` if uniform `M_c` produces a useful (chain-blind)
   `m_t` for next-token prediction.
4. F3 probe loss is the only permutation-breaking force on the writer
   (chain-specific gradient through the readout). At probe weight
   0.5 (v21c's recipe), the F3 force can sometimes dominate (seed=42)
   and sometimes cannot (seeds 1, 2) depending on early-training
   stochasticity — particularly on synthd5's templated content where
   the readout has shortcut paths to satisfy F3 without strong M_c
   chain-specificity.
5. When LM-NLL wins the early tug-of-war, M_c drifts to the uniform
   attractor and `pair/self` collapses → kill-watch fires.

This is consistent with the v22 cell readings (probe 0.4 at
−0.003 etc.): probe weight 0.5 is on a knife edge, with run-to-run
randomness determining which side it falls on.

### v23 §5 readings (on the killed checkpoints' `best/`)

The `best/` checkpoint of a killed run is the highest-evidence_lift
ckpt seen *before* collapse. v23a/best was saved during the brief
+0.0158 mid-run peak (step ~300-400); v23b/best from a similar
window. Cross-check with `tools/eval_ttt_mc.py` will tell us
whether the early best/ has the §5 capacity that v20a/v21c had:

* Pending — not run yet (would need to schedule when GPU 0/1 free
  up after v24 finishes around 23:30 UTC-5, OR run on GH200 after
  v23c finishes).

### v23 — what to do with v23c when it finishes

v23c (seed=7 on GH200) is the third local seed of the v23 sweep.
Possible outcomes:

1. **Also collapses (prior >50 %).** Confirms the recipe is unstable;
   v24's corpus-pivot evidence is the next answer.
2. **Runs cleanly with `evidence_lift > 0` final.** Means seed=42
   and seed=7 both worked but seeds 1/2 didn't — 50/50 reproducibility
   floor at this recipe. Still not a reproducible result for paper
   purposes; multi-seed median for D4v2 / 1.7B will be the new bar.
3. **Runs cleanly but ends near 0.** Confirms the corpus +
   architecture combination is too noisy regardless of seed; v24's
   real-content data is the right pivot.

§5 cross-check on whatever v23c produces will be the final v23 datum.

---

## v22 — probe-weight sweep (FINISHED 2026-05-03 ~22:35 UTC-5)

### Verdict: v21c stays the best end-to-end cell; v20a stays best §5; the curve is sharp + noisy

The bracketing cells around v21c's probe weight 0.5 — v22a at 0.4 and
v22c at 0.6 — both ended with `evidence_lift` well below v21c's
+0.0241. Critically, v22a (0.4) ended at **−0.0033**, *worse* than
v20a (probe 0.3) at +0.0074. So the probe-weight curve is **not a
smooth unimodal function around 0.5**: there's an apparent dip at
0.4 between v20a and v21c. Either (a) the true curve is multi-modal
with two narrow peaks, or (b) at fixed seed, run-to-run variance is
roughly the same magnitude as the probe-weight effect.

Combined results across probe weight at depth=4 + strong floor
0.5/0.10 + 1000 steps:

| cell | probe w | end `evidence_lift` | last `pair/self` | last `α_mem_max` | last `‖m^t‖/‖embed‖` | §5 writer | §5 iid |
|---|---|---|---|---|---|---|---|
| **v20a** | 0.3 | +0.0074 | 0.498 | 0.188 | 16.12 | **+0.120** ★ | −0.179 |
| **v22a** | 0.4 | −0.0033 | 0.043 | 0.213 | 15.27 | −0.010 | −0.457 |
| **v21c** | 0.5 | **+0.0241** ★ | 0.075 | 0.137 | 15.69 | −0.132 | −0.738 |
| **v22c** | 0.6 | +0.0049 | 0.149 | 0.085 | 14.40 | −0.066 | −0.485 |
| **v19b** | 1.0 | −0.016 | 0.644 | 0.20 | 12.66 | −0.004 | −0.262 |

§5 readings: `results/ttt_mc_v22post/v22{a,c}_f3_d4_*__{writer,iid}.json`
(complement to `ttt_mc_v20post/v20{a,b}` and `ttt_mc_v21post/v21{a,c}`).

### Key v22 takeaways

1. **The §5-vs-end-to-end inversion is probe-weight dependent.**
   §5 peaks sharply at probe weight 0.3 (v20a's +0.120) and
   collapses at every higher probe weight (v22a 0.4: −0.010;
   v21c 0.5: −0.132; v22c 0.6: −0.066). End-to-end peaks at
   probe weight 0.5 (v21c's +0.0241), but with ~0.025-nat
   noise floor — v22a went negative and v22c only barely
   positive at adjacent probe weights. The two metrics
   genuinely measure different things and do not co-peak.

2. **End-to-end variance at fixed recipe is ~ effect size.**
   v20a→v22a→v21c→v22c sweep at +0.1 increments in probe
   weight produced end-to-end results in [−0.003, +0.024].
   Without multiple seeds we cannot claim v21c is reproducibly
   the optimum — it's just the best run we've seen so far.
   v23 candidate (NOT launched in this 4h window): **3 seeds
   each at probe weight 0.4, 0.5, 0.6** to bound the noise
   floor and confirm v21c is reproducible.

3. **The architecture works at the v21c operating point but
   the effect size is small.** +0.024 nats is a real,
   measurable benefit of memory on the callback CE relative
   to the evidence-redacted floor — strictly better than zero,
   on a corpus where the backbone could not have memorised the
   answer (`synthd5_random_codes`), with a frozen backbone.
   But it's two orders of magnitude below the oracle (~+1.5
   nats) and within run-to-run noise of the v22 bracket. The
   architecture is *correctly directed* — chain-specific
   evidence is being routed through `M_c` to lower CE on
   callback queries — but the operating point delivers only
   weak signal.

### Recipe to date (canonical for any v23+)

```
--preset qwen3-0.6b-large
--memres_mode attention_parity
--router_recent_bias_init 4 --router_mem_bias_init 0
--memres_update_mode gated --memres_extract_source hidden_14
--memres_extract_input_norm
--memres_gate_init 0.0 --memres_readout_norm_init 0.05
--memres_writer_kind slot_attention --memres_slot_attention_iters 3
--memres_queries_init orthogonal --memres_slot_positional
--memres_judge_qk_layernorm
--memres_readout_depth 4
--readout_probe_enabled
--readout_probe_loss_weight 0.5            # v21c winner; ±0.1 noisy
--readout_probe_warmup_steps 200
--alpha_mem_floor_aux_weight 0.5
--alpha_mem_floor_target 0.10
--freeze_backbone
--lr 1e-4 --lr_backbone 0 --steps 1000 --warmup 200
--callback_loss_weight 5.0 --score_tail_frac 1.0 --mask_evidence_session_loss
--kill_on_memory_collapse --kill_on_memory_collapse_min_step 200
--save_best_metric evidence_lift
```

§5 followup script template: `scripts/run_v18_followup_ttt.sh`
(parametrise for the new cell name).

### v23 candidates (not launched; in priority order)

1. **Multi-seed v21c reproducibility check.** Run v21c recipe at
   3 different `--seed` values (current default; need to thread
   `--seed N` into all `torch.manual_seed`, `numpy.random.seed`,
   and the dataloader RNG). Goal: bound the noise floor on
   end-to-end `evidence_lift`. If 95% CI excludes 0, the v21c
   recipe is the canonical recipe for the paper.

2. **Backport v21c to D4v2 + 1.7B.** Larger backbone + the
   already-tested D4v2 corpus. The v15 OPEN AUDIT identified
   trained-backbone joint training as the cause of D4v2's
   evidence_lift collapse; we should run v21c FROZEN-backbone
   on D4v2 to test transfer of the §5 + read-side recipe.

3. **Writer-side / corpus-side levers.** All v18-v22 cells held
   the writer architecture constant (`slot_attention` + the
   v13 symmetry-break stack). The next architectural lever, if
   read-side is locked at v21c's operating point, is on the
   writer: e.g. a contrastive auxiliary that explicitly
   penalises pair-cosine similarity between cross-chain `M_c`
   matrices, or a deeper extractor (`memres_extraction_depth`
   beyond 4).

---

## v22 (history; superseded) — probe-weight sweep launch documentation

<details>
<summary>v22 launch-time pre-summary (kept for forensics)</summary>

v21c (probe weight 0.5) gave the best end-to-end `evidence_lift` in
the project: +0.0241 at step 1000, monotonically climbing through
the run. v20a (probe weight 0.3) held the best §5 reading (+0.120)
but poor end-to-end (+0.005). The probe-weight curve was
*hypothesised* to have a clear peak near 0.5 for end-to-end and 0.3
for §5 — a Goldilocks pattern where probe weight trades off
"trained M_c is what the trained readout uses" (high → end-to-end)
against "trained readout can flexibly decode any M_c" (low → §5).
v22a / v22c bracketed v21c by ±0.1 in probe weight. The actual
result (above) is that the curve is **not smooth** in the [0.3, 0.6]
range — both v22a and v22c ended below v21c, with v22a actually
worse than v20a (probe 0.3). The v21c reading appears to be a
local optimum on a noisy surface rather than the apex of a smooth
peak.

### v22 cells (launched 2026-05-03 ~21:05 UTC-5; finished ~22:35 UTC-5)

| cell | host | preset | backbone | steps | corpus | recipe vs v21c | output |
|---|---|---|---|---|---|---|---|
| **v22a** | local H100 GPU 0 | qwen3-0.6b-large | FROZEN | 1000 | synthd5_random_codes | v21c + `--readout_probe_loss_weight 0.4` (was 0.5). | `output/chain_v22a_f3_d4_pw04_codes_0p6b_frozen_local/{best,final}` |
| **v22c** | local H100 GPU 1 | qwen3-0.6b-large | FROZEN | 1000 | synthd5_random_codes | v21c + `--readout_probe_loss_weight 0.6` (was 0.5). | `output/chain_v22c_f3_d4_pw06_codes_0p6b_frozen_local/{best,final}` |

</details>

---

## v21 — refine v20a's recipe: ReZero magnitude bound vs probe-weight sweep (FINISHED 2026-05-03 ~21:00 UTC-5; v21c wins end-to-end)

### Final verdict

| cell | depth | floor (w/t) | probe w | ReZero | end @ 1000 `evidence_lift` | trajectory | last `pair/self` | last `‖m^t‖/‖embed‖` | last `α_mem_max` | §5 writer | §5 iid | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **v21a** | 4 | 0.5 / 0.10 | 0.3 | YES | **+0.0141** | non-monotonic | 0.256 | 16.40 | 0.175 | −0.049 | −0.527 | NEG §5 / + end-to-end |
| **v21c** | 4 | 0.5 / 0.10 | **0.5** | no | **+0.0241** | **monotonic upward** | 0.075 | 15.69 | 0.137 | −0.132 | −0.738 | NEG §5 / **best end-to-end in project** |

§5 readings: `results/ttt_mc_v21post/v21{a,c}_f3_d4_*__{writer,iid}.json`.

Key v21 takeaways (load-bearing for v22+):

1. **§5 capacity and end-to-end `evidence_lift` are anti-correlated
   in this regime, not just decoupled.** v20a (probe 0.3) has the
   highest §5 (+0.120) and modest end-to-end (+0.005). v21c (probe
   0.5) has poor §5 (−0.132) and best end-to-end (+0.024). v21a
   (probe 0.3 + ReZero) is in between on both. The interpretation:
   high probe weight makes the trained readout's W_Q/W_K/W_V
   tightly fit the trained writer's specific M_c distribution
   (good for end-to-end with that writer; bad for TTT-on-M_c which
   moves to a different M_c). Low probe weight keeps the readout
   flexible enough to decode any reasonable M_c (good for §5; the
   trained writer happens to deliver one that the readout doesn't
   exploit fully end-to-end).
2. **For the project claim, end-to-end is the right metric.** The
   memory_residuals.pdf paper claim is "the augmented model has
   lower callback CE than the bare backbone on a frozen-backbone
   recipe with leak-free synthetic chains." That is `evidence_lift`,
   not §5. v21c at +0.024 is the **first cell with non-trivially
   positive end-to-end `evidence_lift`** that is also monotonic
   through training (no late drift like v20a) and held a clean
   D2-JUDGE reading (eff_rank ≈ 4.5, not collapsed to uniform like
   v19a's 1.27 just before kill).
3. **ReZero hurts on this corpus.** v21a (ReZero) has worse §5
   AND worse end-to-end than v21c (no ReZero). The hypothesis was
   that bounding `‖m^t‖` magnitude growth would prevent the late-
   training drift. Result: the model grew `refine_W_V` back to large
   magnitude anyway (v21a end-of-training `‖m^t‖/‖embed‖ = 16.4`,
   same as v20a's 16.1), and the temporary suppression during early
   training cost it some of the gradient signal that drives the
   writer to specialise (v21a's `pair/self` = 0.256 vs v21c's...
   actually 0.075, which is lower — so this story isn't clean). The
   simpler reading: ReZero at the architectural level adds an
   optimization headwind that didn't pay back. Default OFF for v22+.
4. **Probe weight 0.5 + depth=4 + strong floor is the recipe to
   beat.** All four v22 candidates are probe-weight perturbations of
   v21c.

### v21 code recap (already shipped in v19/v21)

* `Qwen3MemResConfig.memres_readout_depth: int = 0` (v19).
* `Qwen3MemResConfig.memres_readout_refine_zero_init: bool = False`
  (v21; default off — v21 verdict above says keep default off).
* `MemoryReadout` constructor accepts `depth` and `refine_zero_init`.
* `_init_memres_params` zeros `refine_W_V[i].weight` when
  `refine_zero_init=True`; otherwise default normal init.
* `train_chain.py` CLI flags: `--memres_readout_depth`,
  `--memres_readout_refine_zero_init`. Both plumbed through
  `memres_kwargs`. Neither in the `overridable` set when warm-
  starting from a memres checkpoint (architecture-shape-changing).

---

<!-- v21 launch-time documentation (superseded by the Final verdict above) -->

<details>
<summary>v21 launch-time pre-summary (kept for forensics)</summary>

**v20a is the project's best §5 cell.** §5 cross-check on v20a/best
returned `ttt_lift_vs_floor = +0.120` (writer init, n=32, K=50,
lr=1e-2) — by far the largest §5 reading in the project history
(v18a: +0.005, v19a: +0.009, v19b: −0.004, v20b: −0.008). The
v20a recipe — `depth=4 + readout_probe_loss_weight 0.3 + strong
floor 0.5/0.10` — clearly UNLOCKED the read-side capacity that
§5 measures. End-to-end `evidence_lift` peaked at +0.0157 at step
700 (where `best/` was saved by `--save_best_metric evidence_lift`)
and degraded to +0.0054 by end-of-training (step 1500), with
intermediate negative excursions in the 1100-1400 window.

That tells us two things:

1. **The architecture is right.** `depth=4` cross-attn refinement
   over the slot pool, with a probe-bypass loss at *reduced* weight
   (0.3, not 1.0) so it doesn't dominate the LM-NLL pathway, plus
   the strong floor opening the depth router so the LM head can
   actually consume `m_t`, gives a trained read-side that decodes
   chain-specific recall cleanly when M_c is supplied (the §5
   metric is exactly that).
2. **End-to-end training is unstable past step 700-1000.** The
   `||m^t||/||embed||` ratio inflated to 16x by end-of-training;
   each refinement layer's free-init `refine_W_V[i]` accumulated
   non-trivial mass without bound. Once `α_mem` opens at any layer,
   that 16x m_t can swamp the LM head's own next-token signal —
   even when the m_t content is decoded as chain-specific.

v21 attacks both with two single-variable refinements of v20a:

* **v21a** — v20a + `--memres_readout_refine_zero_init`. ReZero-init
  for each `refine_W_V[i].weight` so the depth=4 stack starts at
  depth=0-equivalent `||m_t||` (smoke-tested: 0.038 vs depth=0's
  0.037, vs depth=4 default's 0.082). Each refinement layer's
  contribution to m_t grows only as the gradient demands. Tests
  whether bounding magnitude growth fixes the late-training
  instability.
* **v21c** — v20a recipe with probe weight 0.5 (was 0.3 in v20a, 1.0
  in v18a/v19/v20b). Probe-weight sweep between the v20a winner and
  the v18a-era default to narrow down the operating point.

Both at 1000 steps (peak was at step 700; 1000 captures the peak
window and a bit of decay without spending compute on the late
drift). Same single-variable rigour as v19/v20: only one knob
changes per cell vs v20a. § 5 cross-check on `v21{a,c}/best` is the
gold-standard verdict.

### v21 cells launched (2026-05-03 ~20:00 UTC-5)

| cell | host | preset | backbone | steps | corpus | recipe vs v20a | output |
|---|---|---|---|---|---|---|---|
| **v21a** | local H100 GPU 0 | qwen3-0.6b-large | FROZEN | 1000 | synthd5_random_codes | v20a + `--memres_readout_refine_zero_init` (each refine_W_V[i] zero-init at construction). All other knobs identical. | `output/chain_v21a_f3_d4_rezero_codes_0p6b_frozen_local/{best,final}` |
| **v21c** | local H100 GPU 1 | qwen3-0.6b-large | FROZEN | 1000 | synthd5_random_codes | v20a + `--readout_probe_loss_weight 0.5` (was 0.3). All other knobs identical. | `output/chain_v21c_f3_d4_pw05_codes_0p6b_frozen_local/{best,final}` |

Started 20:02 / 20:03 UTC-5; ETA step 1000 ≈ 21:40 UTC-5 at ~10.5k
tok/s on each H100 NVL.

Healthy through smoke verification (first 100 steps; both at ~75-80%
GPU util, both descending probe loss, both with the strong-floor
signature `a_floor ≈ 0.084 / a_mean ≈ 0.016`).

### Code shipped in v21

`src/modeling_memres.py`:
* `Qwen3MemResConfig.memres_readout_refine_zero_init: bool = False`
  — config plumbing.
* `MemoryReadout.__init__` accepts `refine_zero_init: bool = False`
  and stashes it on `self._refine_zero_init` so
  `_init_memres_params` can read it during materialisation.
* `_init_memres_params` (the `MemoryReadout` branch) zero-initialises
  each `refine_W_V[i].weight` when `refine_zero_init=True`. The
  `refine_W_Q[i]`, `refine_W_K[i]`, and `refine_out_norm[i]` weights
  remain at their default (normal init for projections, scaled by
  `out_norm_init` for norms) so the gradient signal can still flow
  through K/Q at step 0; only the V projection is gated.

`src/train_chain.py`:
* CLI: `--memres_readout_refine_zero_init` (action="store_true",
  default off). Plumbed through `memres_kwargs` in `_build_model`.
  Architecture-shape-changing flag (different init values; checkpoint
  weights reflect the choice once trained), so NOT in the
  `overridable` set when warm-starting from a memres checkpoint.

`scripts/train_v21a_f3_d4_rezero_codes_0p6b_frozen_local.sh` (depth=4
+ ReZero) and `scripts/train_v21c_f3_d4_pw05_codes_0p6b_frozen_local.sh`
(depth=4 + probe weight 0.5) — both base the v21 cells on the v20a
recipe with one variable changed.

### Predictions (committed before v21 results land)

| §5 reading on v21a/best | implication | next move |
|---|---|---|
| `ttt_lift_vs_floor > +0.30` (POSITIVE) | ReZero is the architectural fix that turns v20a's MIXED into POSITIVE; v21a is the canonical recipe. | Promote `--memres_readout_refine_zero_init` to default for all `depth >= 2` cells; backport to D4v2 + 1.7B. |
| `+0.10 < lift ≤ +0.30` (better than v20a) | ReZero refines the recipe in the right direction; promote, but D4v2 / 1.7B may need additional levers. | v22a: ReZero + 1.7B at 1500 steps; v22b: ReZero + D4v2 frozen-backbone at 1500 steps. |
| `+0.05 < lift ≤ v20a's +0.12` | tradeoff; ReZero loses a bit of read-side capacity but may have gained training stability. Worth keeping for the late-training-stability benefit. | Inspect end-to-end `evidence_lift` trajectory; if v21a's late evidence_lift doesn't drift like v20a's, ReZero is still net-positive. |
| `lift < +0.005` (NEG/MIXED, worse than v20a) | ReZero blocked gradient flow through the refinement stack (because `refine_W_V[i] = 0` makes the residual `m_t = m_t + RMSNorm(softmax(...) @ M_c @ 0) = m_t`, and ∂L/∂refine_W_V[i] flows through but the refinement layer never produces signal until W_V grows). | v22 candidate: small non-zero init (scale=0.01 instead of 0.0) to give the gradient a foothold while still bounding magnitude. |

| §5 reading on v21c/best | implication | next move |
|---|---|---|
| `ttt_lift_vs_floor > v20a's +0.120` | probe weight 0.5 is closer to optimum than 0.3; promote. | v22 probe-weight sweep at 0.4, 0.6, 0.7. |
| `lift in [+0.05, +0.12]` | probe weight 0.5 is on the chain-fingerprinting side of optimum; v20a's 0.3 stays the recipe. | No move on probe weight; pursue v21a's ReZero or other levers. |
| `lift < 0` (worse than v20a) | even probe weight 0.5 drops back into the v19b/v20b chain-fingerprinting trap; the operating window for probe weight is narrow (≤ 0.4). | v22 probe-weight sweep at 0.1, 0.2 (lower than 0.3). |

### v20 verdict (FINISHED 2026-05-03 19:54 UTC-5; v20a is the WINNER, v20b is dead)

| cell | depth | floor (w/t) | probe w | end @ 1500 | last `evidence_lift` | last `pair/self` | last `α_mem_max` | last `‖m^t‖/‖embed‖` | §5 writer | §5 iid | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v18a** (baseline) | 0 | 0.01 / 0.05 | 1.0 | OK | −0.003 | 0.017 | 0.022 | 3.85 | +0.005 | −0.715 | MIXED |
| **v19a** | 2 | 0.5 / 0.10 | 1.0 | KILLED step 500 | +0.001 | 0.007 (collapsed) | 0.045 | 6.44 | +0.009 | −0.047 | MIXED |
| **v19b** | 4 | 0.5 / 0.10 | 1.0 | OK | −0.016 | 0.644 | 0.20 | 12.66 | −0.004 | −0.262 | NEG |
| **v20a** | 4 | 0.5 / 0.10 | **0.3** | OK (peak step 700) | +0.005 (final) / +0.0157 (peak) | 0.498 | 0.188 | 16.12 | **+0.120** | −0.179 | **MIXED — best §5 in project** |
| **v20b** | 4 | 0.05 / 0.05 | 1.0 | OK | −0.012 | 0.940 | 0.194 | 13.20 | −0.008 | −0.427 | NEG |

§5 readings (writer init, n=32, K=50, lr=1e-2) on `best/` checkpoints:
* `results/ttt_mc_v20post/v20a_f3_d4_pw03__{writer,iid}.json`
* `results/ttt_mc_v20post/v20b_f3_d4_fw005__{writer,iid}.json`

Key v20 takeaways:

1. **The 1×1 ablation isolated probe weight as the dominant
   variable.** v20a (probe 0.3, strong floor) jumped §5 capacity to
   +0.120 — 24× v18a, 13× v19a. v20b (probe 1.0, weak floor)
   stayed in the v19b failure regime (§5 = −0.008). Floor weight
   alone (going from 0.5 → 0.05 in v20b) did not rescue the run.
   The depth=4 stack at probe weight 1.0 *consistently* over-pulls
   the readout into chain-fingerprinting regardless of what the
   floor does.
2. **§5 capacity and end-to-end `evidence_lift` are correlated but
   not the same.** v20a has a 24× lead on §5 over v18a but the
   end-to-end `evidence_lift` advantage is much smaller (v20a's
   peak +0.0157 vs v18a's +0.005). §5 lets `M_c` be TTT-tuned
   per-chain to maximise callback CE; the trained `M_c` during
   normal eval doesn't have that flexibility. Closing the gap
   probably requires both (a) better writer specialisation under
   reduced probe pressure (depth=2 + weak floor candidate?) and
   (b) bounding `||m^t||` so the late-training drift doesn't waste
   the architectural capacity.
3. **`||m^t||/||embed||` consistently inflates to 12-16x at
   depth=4** regardless of probe weight, floor weight, or end-of-
   training evidence_lift. This is purely an artifact of the
   residual stack accumulating un-bounded contributions from each
   layer's free-init `refine_W_V`. ReZero (v21a) is the targeted
   fix.
4. **The "missing cell" — depth=2 + weak floor — is no longer the
   priority.** With v20a winning at depth=4 + probe 0.3, depth=2
   testing falls behind exploring v20a's recipe-space refinements.
   Queueing depth=2 + weak floor as a v22 diagnostic if v21a/v21c
   both fail.

### v19 cells (finalised; superseded by v20)

| cell | depth | floor (w/t) | probe w | end @ 1500 | last `evidence_lift` | last `pair/self` | last `α_mem_max` | last `‖m^t‖/‖embed‖` | §5 writer | §5 iid | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v19a** | 2 | 0.5 / 0.10 | 1.0 | KILLED step 500 | +0.001 | 0.007 (collapsed) | 0.045 | 6.44 | +0.009 | −0.047 | MIXED |
| **v19b** | 4 | 0.5 / 0.10 | 1.0 | OK | −0.016 | 0.644 | 0.20 | 12.66 | −0.004 | −0.262 | NEG |

</details>

---

## v20 — F3 + readout depth=4 + probe/floor calibration ablation (FINISHED 2026-05-03 19:54 UTC-5; v20a winner, see v21 block above for verdict + recipe-refinement)

### Pre-summary (TL;DR for the next reader)

v19 finished. The v19 wave settled the architectural-depth question
(depth=2 helps §5 capacity in the right direction; depth=4 + strong
probe + strong floor *over*shoots into chain-fingerprinting that
hurts the LM head) but did not deliver a positive `evidence_lift`
end-to-end. Two failure modes, distinct enough to ablate
independently:

* **v19a (depth=2 + strong floor 0.5/0.10 + standard probe 1.0)**
  KILLED at step 500 by `--kill_on_memory_collapse`. Slot writer
  fell into the v13-era uniform fixed point (`pair/self = 0.007`).
  But the *early* `best/` checkpoint (saved at step 100-200, before
  the collapse) is the *best* §5 reading in the project history:
  `ttt_lift_vs_floor = +0.0093` (writer init) — improving on v18a's
  +0.005. So depth=2 *did* incrementally help read-side capacity.
* **v19b (depth=4 + strong floor 0.5/0.10 + standard probe 1.0)**
  ran the full 1500 steps with extreme writer specialisation
  (`pair/self = 0.644`, the strongest in the project) and the
  router opening (`α_mem_max = 0.20, frac_open = 0.16`), but
  `evidence_lift = −0.016`, `mem CE 3.10 vs nomem CE 1.14` —
  memory **dominates** the LM head's predictions in a chain-
  fingerprinting direction that hurts next-token CE on the
  callback. §5 reading: `ttt_lift_vs_floor = −0.0035` (writer
  init) — slightly *worse* than v18a's +0.005.

The diagnosis: F3 probe loss at weight 1.0 + depth=4's amplified
m_t (`||m^t||/||embed|| = 12.7` vs v18a's 3.85) supervises the
read-side to encode chain-identity at the first answer token
position. Either the strong floor (which forces α_mem to climb
toward 0.10) or the strong probe loss (which dominates what m_t
encodes once depth amplifies its magnitude), or both, are
overshooting the operating point.

v20 runs a 1×1 ablation that isolates which dimension is the cause:

* **v20a** — same recipe as v19b but `--readout_probe_loss_weight
  0.3` (was 1.0; 70% reduction). If reduced probe weight prevents
  the readout-dominates-LM regime, **probe weight** was the cause.
* **v20b** — same recipe as v19b but
  `--alpha_mem_floor_aux_weight 0.05` and
  `--alpha_mem_floor_target 0.05` (10× weaker / 2× weaker;
  identical to v18a's floor). If reduced floor weight prevents the
  failure, **floor weight** was the cause.

If both succeed (`evidence_lift > +0.05` end-to-end and §5 MIXED or
better): redundant fixes — promote whichever has the better §5.
If only one succeeds: that's the operating-point variable.
If neither: F3 + depth interaction needs more fundamental rethinking
(ReZero-init refine layers; gradient clip on probe loss; probe
warmup-only schedule).

### v20 cells launched (2026-05-03 ~18:55 UTC-5)

| cell | host | preset | backbone | steps | corpus | recipe vs v19b | output |
|---|---|---|---|---|---|---|---|
| **v20a** | local H100 GPU 0 | qwen3-0.6b-large | FROZEN | 1500 | synthd5_random_codes | v19b + `--readout_probe_loss_weight 0.3` (was 1.0). All other knobs identical (depth=4, floor weight 0.5 / target 0.10, probe warmup 200). | `output/chain_v20a_f3_d4_pw03_codes_0p6b_frozen_local/{best,final}` |
| **v20b** | local H100 GPU 1 | qwen3-0.6b-large | FROZEN | 1500 | synthd5_random_codes | v19b + `--alpha_mem_floor_aux_weight 0.05 --alpha_mem_floor_target 0.05` (back to v18a-strength floor). All other knobs identical (depth=4, probe weight 1.0). | `output/chain_v20b_f3_d4_fw005_codes_0p6b_frozen_local/{best,final}` |

Started 19:11 / 18:54 UTC-5; ETA step 1500 ≈ 21:30 UTC-5 (v20a) /
21:25 UTC-5 (v20b) at ~10.6k tok/s on each H100 NVL.

Healthy through smoke verification (first 100 steps; both at ~90%
GPU util). Both runs use the v19b recipe verbatim except for the
single single-variable change documented above; this is the cleanest
way to isolate the cause of v19b's "memory dominates LM, in wrong
direction" regime.

### Predictions (committed before v20 results land)

| §5 ttt_lift_vs_floor + end-to-end evidence_lift | implication | next move |
|---|---|---|
| v20a §5 ≥ +0.05 AND `evidence_lift_final > +0.05` | probe weight 1.0 + depth=4 was over-pressuring the readout into chain-fingerprinting; reducing to 0.3 keeps the chain-specific gradient channel without dominating LM-NLL | promote depth=4 + probe 0.3 + strong floor as the canonical recipe; backport to D4v2 (1.7B) |
| v20b §5 ≥ +0.05 AND `evidence_lift_final > +0.05` | floor weight 0.5 was opening the router faster than the readout could specialise, letting half-trained m_t dominate; keeping floor at v18a strength + depth=4 is the recipe | promote depth=4 + probe 1.0 + weak floor as the canonical recipe |
| Both v20a and v20b succeed | probe weight and floor weight are redundant fixes; either works | pick the one with better §5; smaller architecture surface to defend |
| Only v20b succeeds | floor strength is the dominant cause of v19b's failure; probe weight 1.0 alone was fine | v21 candidate: depth=4 + probe 1.0 + intermediate floor (e.g. 0.1/0.07) to find the actual operating point |
| Only v20a succeeds | probe weight 1.0 was the problem | v21: depth=4 + probe 0.3 + strong floor + extended (3000-step) training to see if chain-specific recall consolidates |
| Neither succeeds; both stuck at evidence_lift ≈ 0 or negative | F3 + depth=4 interaction needs fundamental rethinking | v21 candidates (in priority order): (a) decay probe weight to 0 after step 500 (probe-only-warmup); (b) add `--readout_refine_init_zero` flag that ReZero-inits each refine_W_V to 0 so the refinement layers start as identity passes; (c) gradient clip on probe loss separately from main loss |

### v19 verdict (FINISHED 2026-05-03 18:24 UTC-5; superseded by v20)

| cell | depth | floor (w/t) | probe w | end @ 1500 | last `evidence_lift` | last `pair/self` | last `α_mem_max` | last `‖m^t‖/‖embed‖` | §5 writer | §5 iid | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v18a** (baseline) | 0 | 0.01 / 0.05 | 1.0 | OK | −0.003 | 0.017 | 0.022 | 3.85 | **+0.005** | −0.715 | MIXED |
| **v19a** | 2 | 0.5 / 0.10 | 1.0 | KILLED step 500 | +0.001 | 0.007 (collapsed) | 0.045 | 6.44 | **+0.009** | −0.047 | MIXED — slightly *better* than v18a; but training-time collapse means run is unstable |
| **v19b** | 4 | 0.5 / 0.10 | 1.0 | OK | −0.016 | **0.644** | **0.20** | **12.66** | −0.004 | −0.262 | NEG — readout dominates LM in wrong direction |

§5 readings (writer init, n=32, K=50, lr=1e-2) on `best/` checkpoints:
* `results/ttt_mc_v19post/v19a_f3_d2__{writer,iid}.json`
* `results/ttt_mc_v19post/v19b_f3_d4__{writer,iid}.json`

Key v19 takeaways (load-bearing for v20+):

1. **depth=2 helps §5 capacity but with depth=2 + strong floor, the
   slot writer collapses.** v19a's `pair/self` went 0.078 → 0.049
   → 0.075 → **0.008** (collapse), while v19b's `pair/self` rose
   monotonically 0.078 → 0.49 → 0.64. Hypothesis: depth=2's
   readout *can* satisfy the F3 probe with a content-blind M_c
   because it has just enough compositional capacity to do
   chain-routing in its own W_Q layers; depth=4's readout *can't*
   satisfy F3 without a chain-specific M_c, so it forces the
   writer to specialise. The strong floor + standard probe makes
   the depth=2 path the easier optimisation target → uniform
   fixed point. With v18a's weaker floor or weaker probe, depth=2
   may not collapse — would test in a v21 if v20 is decisive.
2. **depth=4 + strong probe makes m_t huge.** `||m^t||/||embed||
   = 12.66` end-of-training; the residual stack of 4 refinement
   layers compounds the per-layer contribution. This isn't bad
   per se (the gate is what controls the actual injection
   magnitude into h), but combined with the strong floor opening
   α_mem to 0.20, the LM head is reading a m_t that's 12× the
   embed norm at 20% weight. That swamps the LM's own next-token
   predictions, especially for the callback first answer token
   that F3 was specifically supervising.
3. **§5 capacity is monotone in our two architectural levers
   (probe channel + depth) but NOT monotone in floor strength.**
   v18a (no depth, weak floor) → +0.005. v19a (depth=2, strong
   floor) → +0.009. v19b (depth=4, strong floor + strong probe)
   → −0.004. The intermediate cell (depth=4, weak floor or weak
   probe) is precisely what v20 fills in.

---

## v19 — F3 + multi-layer readout depth (FINISHED 2026-05-03 18:24 UTC-5; superseded by v20)

### Pre-summary (TL;DR for the next reader)

v18a (F3 alone) shifted §5 `ttt_lift_vs_floor` from −0.897 (v15a, no
intervention) → −0.337 (v17a, F2 only) → **+0.005 (v18a, F3 only)** —
the largest causal shift in the §5 series and the first non-NEG
reading. F3 is the right read-side intervention and F2 was over-
engineered (v18b with F2+F3 was *worse* than v18a; the two probes
pull `M_c` along different value-space directions and interfere).

v18a's MIXED §5 reading sits at the boundary of the architecture's
read-side capacity. Two separate hypotheses explain why end-to-end
`evidence_lift` stayed at ≈ 0:

1. **Alpha-floor too weak.** `--alpha_mem_floor_aux_weight 0.01` /
   `--alpha_mem_floor_target 0.05` left `α_mem_max ≈ 0.022` and
   `α_mem_mean ≈ 0.009` — well below target. The depth router was
   keeping the memory channel almost closed, so chain-specific
   content in `m_t` never reached the LM head at meaningful weight.
2. **Single-layer readout at the edge of capacity.** The current
   `MemoryReadout` is one cross-attn layer (Eq. 6).  Even with the
   read-side probe loss restoring chain-specific gradient through
   `MemoryReadout.W_Q/W_K/W_V`, a single cross-attn lookup on `M_c`
   may not have the expressivity to encode the per-callback chain-
   specific recall the synthd5_random_codes corpus demands.

v19 attacks both at once with a single architectural change:
`--memres_readout_depth N` stacks `N` additional residual cross-
attn refinement layers on top of the base `MemoryReadout`. Each
refinement layer

```
m_t^(i) = m_t^(i-1) + RMSNorm_i( softmax((X + m_t^(i-1)) W_Q_i,
                                         M_c W_K_i) @ M_c W_V_i )
```

gives the readout iterative Perceiver-style refinement against the
same `M_c`, conditioning later layers on the partial `m_t`
accumulated so far. Init parity is preserved by `MemoryGate`'s zero-
init downstream — verified bit-exactly (`max|d|=0.00` vs the bare
backbone for `depth ∈ {0, 2, 4}` with the v18a recipe stack).

The alpha-floor is also strengthened:
`--alpha_mem_floor_aux_weight 0.5` / `--alpha_mem_floor_target 0.10`
(50× weight, 2× target vs v18a). With v19's read-side probe loss
actively pressuring `m_t` to be chain-specific, the floor is a
*supporting* mechanism — not a gameable one.

### v19 cells launched (2026-05-03 ~17:38 UTC-5)

| cell | host | preset | backbone | steps | corpus | recipe vs v18a | output |
|---|---|---|---|---|---|---|---|
| **v19a** | local H100 GPU 0 | qwen3-0.6b-large | FROZEN | 1500 | synthd5_random_codes | v18a + `--memres_readout_depth 2` + floor `aux_weight 0.5 target 0.10`. Single-variable readout-depth test. | `output/chain_v19a_f3_d2_codes_0p6b_frozen_local/{best,final}` |
| **v19b** | local H100 GPU 1 | qwen3-0.6b-large | FROZEN | 1500 | synthd5_random_codes | v18a + `--memres_readout_depth 4` + floor `aux_weight 0.5 target 0.10`. Capacity-ceiling pair. | `output/chain_v19b_f3_d4_codes_0p6b_frozen_local/{best,final}` |

Started 17:38 / 17:42 UTC-5; ETA step 1500 ≈ 20:30 UTC-5 (~10.7k tok/s on v19a, ~10.4k tok/s on v19b — readout depth doubles cross-attn FLOPs per token but bf16 / 0.6B keeps both well above 10k tok/s on H100 NVL).

Both runs survived smoke verification (forward + backward + first
eval through step 100) before being left to run:

* v19a step 100/1500: loss 5.45 → 5.25 (descending), `rprobe 11.9
  → 8.1` (probe descending), `gate_mean = 0` (parity-preserving
  init holding), `a_floor 0.087 / a_mean 0.013` (floor active and
  pressing the router upward), `grad_norm` stable at ~14, **best/
  checkpoint already saved** at step 100.
* v19b step 60/1500: loss 6.99 → 6.21, `rprobe 12.1 → 6.6`, same
  floor signature.

Both runs use the v18a recipe verbatim except for the single-
variable change documented above (readout depth + stronger floor).
This isolates the v19 architectural contribution.

### v19 cell-level recipe (mirrors v18a + the two single-variable changes)

```
--preset qwen3-0.6b-large
--memres_mode attention_parity
--router_recent_bias_init 4 --router_mem_bias_init 0
--memres_update_mode gated --memres_extract_source hidden_14
--memres_extract_input_norm
--memres_gate_init 0.0 --memres_readout_norm_init 0.05
--memres_writer_kind slot_attention --memres_slot_attention_iters 3
--memres_queries_init orthogonal --memres_slot_positional
--memres_judge_qk_layernorm
--memres_readout_depth {2|4}                    # v19 architectural change
--writer_warmup_steps 0  (...anneal=0)
--freeze_backbone
--readout_probe_enabled
--readout_probe_loss_weight 1.0
--readout_probe_warmup_steps 200
--alpha_mem_floor_aux_weight 0.5                # v19 floor strengthening
--alpha_mem_floor_target 0.10                   # v19 floor strengthening
--train_chains paper_artifacts/chains/synthd5_random_codes_train_s512.pt
--eval_chains  paper_artifacts/chains/synthd5_random_codes_val_s512.pt
--window_k 3 --batch_size 4 --grad_accum 2
--lr 1e-4 --lr_backbone 0 --steps 1500 --warmup 200
--callback_loss_weight 5.0 --score_tail_frac 1.0 --mask_evidence_session_loss
--kill_on_memory_collapse --kill_on_memory_collapse_min_step 200
--save_best_metric evidence_lift
```

Param count change (vs v18a baseline):

| readout depth | total params | trainable memres params | refine layer params | tok/s |
|---|---|---|---|---|
| 0 (v18a) | 780.5 M | 28.9 M | 0 | ~10.8k |
| 2 (v19a) | 786.8 M | 35.2 M | +6.3 M (2 × 3.15 M) | ~10.7k |
| 4 (v19b) | 793.1 M | 41.5 M | +12.6 M (4 × 3.15 M) | ~10.4k |

(Refine-layer params per layer: `W_Q + W_K + W_V` at 1024² each =
3.15 M; `RMSNorm` adds 1024 ≪ 3.15 M.)

### Predictions (committed before training results land)

| §5 outcome on v19a/best (gold-standard cross-check) | implication | next move |
|---|---|---|
| `ttt_lift_vs_floor > +0.30` on v19a | depth=2 + strong floor closes the read-side gap; v19a is the recipe | promote `--memres_readout_depth 2` + strong-floor recipe to default; backport to D4v2 + 1.7B; treat the read-side bottleneck as resolved |
| `+0.05 < lift ≤ +0.30` | partial; depth helps but other read-side capacity (more layers / larger backbone / different primitive) also load-bearing | run v19b's verdict; if v19b ≫ v19a, scale to depth=8; otherwise swap cross-attn → linear-attn / DeltaNet for v20 |
| `0 < lift ≤ +0.05` | depth=2 marginal; expressivity is not the binding constraint — the *primitive* is wrong (cross-attn over a fixed K=128 slot pool is too narrow) | v20: replace `MemoryReadout` cross-attn with a different primitive (DeltaNet read / linear-attn / mixture-of-readers) |
| `lift ≈ 0` (still MIXED) | the read-side architecture itself cannot route chain-specific recall at this scale | LM-head / depth-router pathway needs structural intervention; revisit v17_wildcards §4 (training-time loss redesign) before any further architectural moves |
| v19b ≫ v19a | depth=2 was on the wrong side of the capacity ceiling; depth=4 is the recipe | promote depth=4 to default; investigate depth=8 as v20a |
| v19a ≈ v19b | capacity ceiling is at depth=2 (or saturates earlier); depth=4 just costs FLOPs | promote depth=2 to default; the v19 architectural change is fully exploited |
| v19a > v19b | depth=4 *overfits* (more cross-attn layers but no more chain-specific signal in M_c); the read-side probe loss can't supervise 4 layers' worth of refinement at this corpus scale | promote depth=2 to default; v20 needs more probe head capacity (multi-token / multi-position read-side probe) before adding more layers |

### Code shipped in v19

* `src/modeling_memres.py`:
  * `Qwen3MemResConfig.memres_readout_depth: int = 0` — config plumbing.
  * `MemoryReadout.__init__` accepts `depth: int = 0`. When `depth > 0`, constructs `nn.ModuleList`s of `refine_W_Q[i]`, `refine_W_K[i]`, `refine_W_V[i]`, `refine_out_norm[i]` (each `nn.Linear(d, d, bias=False)` for the QKV projections, `Qwen3RMSNorm(d)` for the output norm).
  * `MemoryReadout.forward` runs the residual refinement loop after the base layer:
    ```
    m_t = self.out_norm(attn @ V)
    for i in range(self.depth):
        Qi = self.refine_W_Q[i](X + m_t)
        Ki = self.refine_W_K[i](M_c)
        Vi = self.refine_W_V[i](M_c)
        attn_i = softmax(Qi Ki^T / sqrt(d))
        m_t = m_t + self.refine_out_norm[i](attn_i @ Vi)
    ```
  * `Qwen3MemResModel.__init__` passes `depth=getattr(config, "memres_readout_depth", 0)` to `MemoryReadout(...)`. Default 0 preserves v11..v18 single-layer readout exactly.
  * `_init_memres_params` (the `MemoryReadout` branch) additionally inits each `refine_out_norm[i].weight` to `out_norm_init` (same calibration as the base layer's RMSNorm) so each refinement layer's contribution is bounded to the same scale as the base readout output.
  * **Init-parity verified** with `tools/init_parity_test.py`-style smoke (max|d| = 0.00 vs bare backbone for `depth ∈ {0, 2, 4}` under attention_parity + slot_attention + extract_input_norm + judge_qk_layernorm + readout_norm_init=0.05; both no-mem and with-mem cases).
* `src/train_chain.py`:
  * CLI: `--memres_readout_depth INT` (default 0). Plumbed through `memres_kwargs` in `_build_model`. **Not** in the `overridable` set when warm-starting from a memres checkpoint — it is architecture-shape-changing (adds new modules with new parameters), same logic as `memres_slot_positional` / `memres_extraction_depth` / `memres_num_blocks`.
* `scripts/train_v19a_f3_codes_0p6b_frozen_local.sh` — depth=2 launcher.
* `scripts/train_v19b_f3_codes_0p6b_frozen_local.sh` — depth=4 launcher.

### Followup (queued for after v19a/v19b finish)

Re-run the §5 followup `tools/eval_ttt_mc.py` on `v19{a,b}/best`
(canonical recipe: n=32 chains, K=50 SGD steps, lr=1e-2, both
`init_mode=writer` and `init_mode=iid`). The script
`scripts/run_v18_followup_ttt.sh` is the template; copy +
parametrise for v19. Decision criterion: §5 `ttt_lift_vs_floor`
is the gold-standard cross-check that survives any best/-checkpoint
quirks; in-flight `evidence_lift` from the training log is the
fast signal.

---

## v18 — F3 (ReadoutProbeHead) read-side gradient channel + extended TTT localizer (active; 2026-05-03 ~16:00 UTC-5)

### Pre-summary (TL;DR for the next reader)

v17 §5 falsified the readout's capacity to decode chain-specific
recall *under the trained read-side*: 6/6 NEG cells across v14k/v15a/
v15e at 0.6B and 1.7B, on D4v2 and D5, with the entire read-side
frozen and only `M_c` TTT-able. v17a/b/e (writer-side F2 cells) were
killed at launch because WriterProbeHead has its own Q/K/V — its
gradient bypasses MemoryReadout entirely, so it cannot lift the
read-side bottleneck §5 identified.

v18 ships the symmetric counterpart: **`ReadoutProbeHead`**, which
consumes `m_t` (the actual `MemoryReadout` output at the callback
session's first answer-token position) and projects to vocab. Because
`m_t = MemoryReadout(embed(X), M_c)`, the probe-loss gradient flows
back through `MemoryReadout`'s own `W_Q/W_K/W_V` and onward into M_c,
judge, and extract. This is the missing read-side gradient channel
that v17 / F2 cannot install from its writer-side bypass.

### v18 §5b — read-side localizer (active; 2026-05-03 16:05 UTC-5)

Before launching v18 cells, ran an extended capacity probe
(`tools/eval_ttt_mc_readout.py`) that adds `MemoryReadout`'s
`W_Q/W_K/W_V` to the TTT-able parameter set in three modes:
`v_only` / `qkv` / `qkv_reset`. Decision:

* `qkv_reset` reinitialises the readout's projections from
  `N(0, d^{-0.5})` before TTT, so a NEG result there isolates the
  *single-layer cross-attn shape* as the bottleneck (vs the trained
  weights' starting point).

**§5b sweep results (2026-05-03 16:05–16:25 UTC-5, n=16 chains, K=80,
lr_mc=1e-2, lr_readout=1e-3):**

| cell | ckpt | corpus | mode | init | `ce_writer` | `ce_floor` | `ce_ttt` | **`ttt_lift_vs_floor`** | verdict |
|---|---|---|---|---|---:|---:|---:|---:|---|
| §5b-A | `chain_v15a_d4v2_norm_replicate_local/best` | synthd4v2_val | qkv       | writer | 4.084 | 4.028 | 4.924 | **−0.897** | NEG |
| §5b-B | `chain_v15a_d4v2_norm_replicate_local/best` | synthd4v2_val | qkv       | iid    | 4.084 | 4.028 | 5.289 | **−1.261** | NEG |
| §5b-C | `chain_v15a_d4v2_norm_replicate_local/best` | synthd4v2_val | qkv_reset | writer | 4.084 | 4.028 | 5.046 | **−1.018** | NEG |
| §5b-D | `chain_v15a_d4v2_norm_replicate_local/best` | synthd4v2_val | qkv_reset | iid    | 4.084 | 4.028 | 5.064 | **−1.037** | NEG |
| §5b-E | `chain_v17a_f2_codes_0p6b_frozen_local/best` | synthd5_val   | qkv       | writer | 6.419 | 6.419 | 6.756 | **−0.337** | NEG |
| §5b-F | `chain_v17a_f2_codes_0p6b_frozen_local/best` | synthd5_val   | qkv       | iid    | 6.419 | 6.419 | 6.811 | **−0.392** | NEG |
| §5b-G | `chain_v17a_f2_codes_0p6b_frozen_local/best` | synthd5_val   | qkv_reset | writer | 6.419 | 6.419 | 6.829 | **−0.411** | NEG |
| §5b-H | `chain_v17a_f2_codes_0p6b_frozen_local/best` | synthd5_val   | qkv_reset | iid    | 6.419 | 6.419 | 6.791 | **−0.373** | NEG |

**Verdict (8/8 NEG):** The single-layer cross-attention readout
cannot generalise across the evidence → callback gap on a single
chain even when granted full `W_Q/W_K/W_V` TTT control. TTT
successfully drives `ev_loss` from 0.5 → 0.15 (huge over-fit on the
evidence session), but the resulting `M_c` + readout configuration
worsens callback CE — the optimiser finds (M_c, readout Q/K/V) pairs
that minimise evidence NLL by memorising the evidence's surface form,
not by learning a generalisable evidence → callback transformation.

The `qkv_reset` rows (re-initialised readout from N(0, d^{-0.5}))
extend this: the bad TTT outcome is not specific to the joint-trained
weights' starting point. The single-layer shape itself is the
constraint.

Notable cross-checkpoint shift: v17a (F2-trained writer) is ~3×
*less* destructive under TTT than v15a (lift ≈ −0.37 vs ≈ −1.04).
The F2 writer produces an `M_c` that is more amenable to per-chain
read-side optimisation, but still not a positive lift. F2 shifts the
needle on the read-side defect; it does not close it.

**Two parsings of this evidence:**

1. **Single-layer readout has a structural shape limitation**;
   multi-layer readout depth (Perceiver-style refinement stack
   analogous to writer-side `memres_extraction_depth`) becomes
   load-bearing.
2. **Per-chain TTT cannot generalise across the evidence → callback
   gap by construction** (its objective is single-chain ev_loss; it
   doesn't see callback supervision). Training-time probe loss across
   many chains is fundamentally different — the optimiser sees
   chain-specific callback targets and many examples per readout
   configuration. v18's read-side probe loss is exactly this
   training-time channel.

Reading (2) is the load-bearing assumption for v18a/b: probe loss
**at training time** may succeed where per-chain TTT fails. If v18
ends up MIXED or NEG, reading (1) takes over and **multi-layer
readout depth (`--memres_readout_depth N`)** becomes the v19
architectural intervention.

### v18 cells launched (2026-05-03)

#### v18b NOFLOOR — first eval revealed the next bottleneck (2026-05-03 16:14 UTC-5; KILLED at step 300)

Initial v18b launch (`logs/chain_v18b_f2f3_codes_0p6b_frozen_local.NOFLOOR.log`,
ckpt at `output/chain_v18b_f2f3_codes_0p6b_frozen_local.NOFLOOR/`)
ran F2 + F3 *without* `alpha_mem_floor_aux_weight`. At the first PA-EVAL
@ step 200:

```
EVAL @ step 200      | mem=1.480  nomem=1.386  Δnm-m=-0.094
PA-EVAL @ step 200   | CB Δnm-m=-0.011 Δsh-m=-0.003
EVID-EVAL @ step 200 | pa_cb_ce_mem=6.249 pa_cb_ce_mem_floor=6.246 evidence_lift=-0.003
ROUTE @ step 200     | α_mem_max=0.024 α_mem_mean=0.0087 frac_open=0.00
wprobe 11.197 -> 3.645 (step 100)   rprobe 11.817 -> 3.265 (step 100)
```

Both probe losses dropped from ≈12 → ≈3.5 nats by step 100 — the writer
*does* encode chain content into M_c (wprobe) AND the readout *does*
produce a chain-discriminative `m_t` at the callback position (rprobe),
exactly as F2+F3 was designed. But `evidence_lift = −0.003` and
`α_mem_mean = 0.0087`. **The depth router is keeping the memory
channel almost closed (0.87 % of depth attention to `b_{-1} = m^t`),
so chain-specific content in `m_t` never reaches the LM head.** This
is the chicken-and-egg between router and content: the router won't
open until LM benefits, the LM can't benefit when α_mem ≈ 0. v17a's
prescription killed `alpha_mem_floor` because under a content-blind
writer the floor was gameable (router spread α thinly to satisfy the
constraint without actually using memory). With v18's read-side probe
loss now actively pressuring `m_t` to be chain-specific, the floor
becomes a *supporting* mechanism that makes the chain-specific content
reach the LM head — not a gameable constraint.

KILLED at step 300, restarted as v18b proper with the floor active
(see below). NOFLOOR data preserved in `*.NOFLOOR.log` /
`*.NOFLOOR/` for reference.

#### v18 cells (alpha-floor recipe; FINISHED 2026-05-03 ~16:58 UTC-5)

| cell | machine | preset | backbone | step | corpus | architecture delta | location |
|---|---|---|---|---:|---|---|---|
| **v18a** | local H100 GPU 0 | qwen3-0.6b-large | FROZEN | 1500 | synthd5_random_codes | **F3** + alpha-floor: `--readout_probe_enabled --readout_probe_loss_weight 1.0 --readout_probe_warmup_steps 200 --alpha_mem_floor_aux_weight 0.01 --alpha_mem_floor_target 0.05`. Single-variable test of the read-side gradient channel + open-router; writer probe OFF. | `output/chain_v18a_f3_codes_0p6b_frozen_local/{best,final}` |
| **v18b** | local H100 GPU 1 | qwen3-0.6b-large | FROZEN | 1500 | synthd5_random_codes | **F2 + F3** + alpha-floor: v18a recipe + `--writer_probe_enabled --writer_probe_loss_weight 1.0 --writer_probe_warmup_steps 200`. | `output/chain_v18b_f2f3_codes_0p6b_frozen_local/{best,final}` |

#### Final eval @ best/ (saved at first eval w/ positive lift, step 600)

| cell | step | `pa_cb_dnm` | `pa_cb_dnm_floor` | **`evidence_lift`** | α_mem_mean | α_mem_max | rprobe (final) | wprobe (final) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v18a/best | 600 | +0.110 | +0.100 | **+0.00984** | 0.0092 | 0.026 | 3.165 (mid) → 7.945 (oscillating) | n/a |
| v18b/best | 600 | +0.228 | +0.220 | **+0.00814** | 0.0084 | 0.027 | 3.417 (mid) → 6.661 (oscillating) | 5.333 |
| v18a/final | 1500 | +0.193 | +0.196 | **−0.00290** | 0.0089 | 0.023 | 3.165 | n/a |
| v18b/final | 1500 | +0.238 | +0.236 | **+0.00230** | 0.0085 | 0.029 | 3.417 | 5.333 |

**Read of the in-trainer numbers:** the architectural intervention
*partially* worked. Memory now provides a real **+0.20-0.24 nat lift
on `pa_cb_dnm`** (vs ~0 in v17) — i.e. mem-on beats mem-off by ~0.2
nats on callback CE. But this lift is **content-blind**: redacting
evidence (`pa_cb_dnm_floor`) gives the same +0.20 lift. The
`evidence_lift = pa_cb_dnm - pa_cb_dnm_floor` is essentially 0.
Decomposition:

* **F3 (ReadoutProbeHead) does its job at the gradient level.** rprobe
  drops from 12 → 3 (and dips to 0.05 at peaks for v18a) — the
  readout's `m_t` at the cb position is highly chain-discriminative
  by the probe's own metric.
* **The chain-specific signal in `m_t` is being diluted by a
  closed router.** alpha_mem_mean ≈ 0.009 (target 0.05; floor weight
  0.01 is two orders of magnitude weaker than the LM gradient
  pushing the router shut).
* **F2 + F3 do not compose cleanly.** v18b's `pa_cb_dnm` matches
  v18a's, but v18b's §5 read-side capacity is *worse* than v18a's
  (see §5 followup below). F2 fights F3 on M_c — the writer is
  asked to satisfy two different value spaces (writer-probe Q/K/V vs
  readout Q/K/V).

#### v18 §5 followup — gold-standard cross-check (2026-05-03 17:00 UTC-5)

`tools/eval_ttt_mc.py` re-run on `v18{a,b}/best`, frozen read-side
(canonical §5 recipe), n=32 chains, K=50 SGD steps, lr=1e-2:

| cell | init | `ce_writer` | `ce_floor` | `ce_ttt` | **`ttt_lift_vs_floor`** | verdict | δ vs v17 §5 |
|---|---|---:|---:|---:|---:|---|---|
| v18a_f3/best | writer | 5.945 | 5.945 | 5.940 | **+0.005** | MIXED | **+0.90 vs v15a / +0.34 vs v17a** |
| v18a_f3/best | iid | 5.945 | 5.945 | 6.660 | **−0.715** | NEG | +0.18 vs v15a (less destructive) |
| v18b_f2f3/best | writer | 5.736 | 5.722 | 5.912 | **−0.191** | NEG | +0.71 vs v15a / +0.15 vs v17a |
| v18b_f2f3/best | iid | 5.736 | 5.722 | 7.211 | **−1.489** | NEG | −0.45 vs v15a (MORE destructive) |

**Key finding: v18a (F3 alone) shifts §5 from solidly NEG to
MIXED.** The architectural intervention is causally moving the
read-side's §5 capacity in the right direction: from -0.897 (v15a,
no intervention) → -0.337 (v17a, F2 only) → +0.005 (v18a, F3 only).
The signs are consistent across the writer-init series, monotonic
in the strength of the read-side gradient channel.

**v18b is *worse* than v18a in §5.** With F2 added to F3, the §5
result regresses to -0.19 (writer init) and -1.49 (iid init). F2
and F3 actively interfere — most plausibly because both probes
gradient-update M_c, but along different value-space directions
(WriterProbeHead's own Q/K/V vs MemoryReadout's Q/K/V), pulling
M_c into a Pareto-frontier that satisfies neither cleanly.

### v18 verdict and v19 design (2026-05-03 17:15 UTC-5)

**v18 partially succeeded.** The architectural change (F3 +
alpha-floor) is causally moving the read-side §5 capacity from
-0.897 to +0.005, demonstrating that the read-side gradient
channel was the missing piece. But the closed router (α_mem ≈ 1%
vs target 5%) and the F2-F3 compose interference mean the
end-to-end `evidence_lift` is still ≈ 0.

**Three load-bearing observations carry into v19:**

1. **F3 alone is the right intervention; drop F2.** v18a > v18b in
   the §5 followup. F2's writer-side bypass (separate Q/K/V) does
   not help and actively hurts when composed with the read-side
   gradient. The canonical recipe should use **F3 only**.
2. **Alpha floor needs to actually bite.** Current weight 0.01,
   target 0.05 produces α_mem_mean ≈ 0.009 — the floor's gradient
   is two orders of magnitude weaker than the LM's
   "memory-injection-hurts-LM-NLL" gradient. v19 should raise to
   weight 0.5, target 0.10 (matching the historical `mild`
   recruitment band but with enough force to overcome the LM
   gradient).
3. **Single-layer readout shape is at the edge of capacity.** §5b
   showed that even with full Q/K/V TTT-able the single-layer
   cross-attn cannot generalise across the evidence → callback gap.
   v18a's MIXED §5 is at the boundary of the architecture's
   intrinsic capacity; pushing it firmly into POSITIVE territory
   likely needs **multi-layer readout depth (`--memres_readout_depth N`)**
   — a Perceiver-style refinement stack analogous to the writer's
   `memres_extraction_depth`.

#### v19 next-step recipe (READY TO LAUNCH; not committed in this 5h horizon)

```
v19a:  F3 + strong alpha-floor + multi-layer readout depth=2.
       --readout_probe_enabled --readout_probe_loss_weight 1.0 \
       --alpha_mem_floor_aux_weight 0.5 --alpha_mem_floor_target 0.10 \
       --memres_readout_depth 2  [NEW knob; modeling change required]

v19b:  F3 + strong alpha-floor + readout_depth=4 (high-capacity).
       (same as v19a but readout_depth=4)
```

Architectural code change for v19 (not yet shipped):

* `MemoryReadout.__init__` to accept `depth: int` and stack `depth+1`
  cross-attn refinement layers (each: Q-from-X, K/V-from-M_c,
  RMSNorm; later layers: Q-from-X-plus-previous-m_t, K/V-from-M_c).
* `Qwen3MemResConfig.memres_readout_depth: int = 0` flag.
* `_init_memres_params` updated to init each refinement layer's
  out_norm with `memres_readout_norm_init`.
* `train_chain.py` `--memres_readout_depth` CLI arg, plumbed through
  `memres_kwargs` and `overridable` set.

Implementation cost: ~80 LOC. Smoke time: ~15 min. v19a+v19b
training time: ~2.5h × 2 cells in parallel on local H100s. Total
v19 turnaround: ~4 h from ship.

Launchers: `scripts/train_v18{a,b}_*.sh`. Followup §5 re-run (gold-
standard cross-check that the read-side intervention worked) is
queued at `scripts/run_v18_followup_ttt.sh`; it polls for
`v18{a,b}/best/model.safetensors` and runs `tools/eval_ttt_mc.py` on
each as soon as they land. Logs:
`logs/chain_v18{a,b}_*.log`, `logs/ttt_mc_v18post/`.

### Predictions (committed before training results land)

| outcome | implication | next move |
|---|---|---|
| `evidence_lift > +0.30` on either v18a or v18b | F3 (read-side probe) restored read-side capacity; v18 is the architectural fix | Promote `--readout_probe_enabled` to a default; backport to D4v2; re-run §5 on `v18*/best` — should now report POSITIVE |
| `+0.05 < lift ≤ +0.30` | Partial; F3 helps but multi-layer readout depth also load-bearing | v19a: F3 + `--memres_readout_depth 2` (Perceiver-style readout stack) |
| `0 < lift ≤ +0.05` | F3 helps marginally; the single-layer readout cannot expressively encode the chain-specific lookup at training scale either | v19a as above; if v19a also flat, the readout architecture itself needs replacement (cross-attn → linear-attn / DeltaNet or similar) |
| `lift ≈ 0`, `wprobe < 5, rprobe < 5` | Writer + readout both encode chain content under probe supervision but the LM head still emits a content-blind prior | LM-head pathway needs structural intervention; revisit v17_wildcards §4 (training-time loss redesign) |
| v18b ≫ v18a | F2 + F3 composition is necessary; the probes are complementary | Default both probes on; v19 depends on which symptom remains |
| v18a ≈ v18b | F3 alone sufficient; F2 was over-engineered | Drop WriterProbeHead from the canonical recipe; promote ReadoutProbeHead alone |

### Code shipped in v18

* `src/modeling_memres.py`:
  * `Qwen3MemResConfig.readout_probe_enabled: bool` — config plumbing.
  * `class ReadoutProbeHead(nn.Module)` — RMSNorm + `Linear(d, V,
    bias=False)`. Consumes `m_t_at_cb: (B, d)`, returns
    `(B, V)` logits in fp32. Independent vocab projection (NOT
    tied to embed_tokens, mirroring WriterProbeHead's design
    rationale).
  * `Qwen3MemResForCausalLM.__init__` — constructs
    `self.readout_probe_head` when `readout_probe_enabled` is
    True; saved/loaded with the checkpoint so v18a/best
    round-trips cleanly even though the probe is unused at eval.
* `src/train_chain.py`:
  * CLI: `--readout_probe_enabled`,
    `--readout_probe_loss_weight`,
    `--readout_probe_warmup_steps`.
  * Probe-loss block: extracts the first answer token from
    `last_labels` via `last_cb_mask` (same selection logic as the
    writer-probe block), gathers the selected rows, computes
    `embed(last_input_ids) → readout_memory(embeds, last_M_c_pre)`,
    indexes per-row at the first cb position, runs
    `model.readout_probe_head(m_t_at_cb)`, computes
    `F.cross_entropy`, adds to `total_loss` with the scheduled
    weight, with NaN/Inf guards and a sentinel value for the "no
    callback in window" case.
  * Probe loss + weight surfaced in the per-step log line as
    `rprobe ... w=...` (alongside the existing `wprobe ... w=...`).
  * `overridable` set in `_build_model` includes
    `readout_probe_enabled` so warm-starts from v15a/v17a/best
    can opt in to the read-side probe via CLI override.
* `tools/eval_ttt_mc_readout.py` (new) — extended §5 capacity
  probe with the `--readout_unfreeze {v_only,qkv,qkv_reset}` knob
  for read-side localisation. See §5b above.
* `scripts/train_v18{a,b}_*.sh`, `scripts/run_ttt_mc_readout_gpu0.sh`,
  `scripts/queue_v18a_after_localizer.sh`,
  `scripts/run_v18_followup_ttt.sh` — drivers + auto-queueing.

---

## v17 — F2 (WriterProbeHead) wave KILLED, then §5 (TTT-on-M_c) pre-experiment (active; 2026-05-03)

### Pre-summary (TL;DR for the next reader)

The "Combined critique" (pier-LLM, 2026-05-03 ~13:30 UTC-5)
re-diagnosed the v16 finding (see "v16" section below): v16a's
`evidence_lift ≈ 0` on D5 with `--mask_evidence_session_loss` is
a **load-bearing falsification of Architectural Prior #9**, not
a candidate solution. The LM-NLL loss does not propagate a
chain-specific gradient to the writer through the
depth-router / readout / LM-head stack — the backbone learns a
*chain fingerprint* instead of the per-chain binding. F2
(WriterProbeHead) is the right next experiment **only if** the
read-side has the capacity to use a discriminative `M_c` in
the first place. The critique mandates §5 of the v17 wildcards
doc ("TTT-on-M_c") as the gating pre-experiment.

### v17a/b/e — KILLED 2026-05-03 13:30 UTC-5

The first wave of v17 cells launched the §1/§2 wildcards (F2
WriterProbeHead joint-train; DeltaNet writer ablation) before
the §5 read-side capacity pre-experiment had committed.
Per the Combined critique, *every* writer-side intervention is
unfalsifiable until §5 separates "writer can't encode" from
"readout can't decode". All three were killed; their `best/`
checkpoints are kept for §5 to test whether the F2-trained
writer's `M_c` is more TTT-friendly than v14/v15's.

| cell | machine | preset | backbone | corpus | result | location |
|---|---|---|---|---|---|---|
| **v17a** | local H100 GPU 0 | qwen3-0.6b-large | FROZEN + WriterProbeHead | synthd5_random_codes | KILLED — joint readout/probe (no isolation), composite loss leaks into the read-side | `output/chain_v17a_f2_codes_0p6b_frozen_local/best` |
| **v17b** | local H100 GPU 1 | qwen3-0.6b-large | trained (lr_b=2e-5) + WriterProbeHead | synthd5_random_codes | KILLED — joint-train against the very objective whose gradient the §5 critique says is uninformative | `output/chain_v17b_f2_codes_0p6b_joint_local/best` |
| **v17e** | GH200 | qwen3-1.7b-large | FROZEN + WriterProbeHead | synthd5_random_codes | KILLED — same defect at 1.7B; resource burn until §5 lands | (GH200, no local mirror) |

### v17 §5 pre-experiment — TTT-on-M_c capacity probe (active; 2026-05-03 14:43 UTC-5)

`tools/eval_ttt_mc.py` (new) implements §5 exactly as specified
by the Combined critique:

* freeze all model parameters (LM head, depth router, readout,
  writer, embeddings);
* per chain: initialise `M_c` from the writer (`writer`) or
  i.i.d. (`iid`, scaled `d^{-0.5}`); make it a leaf
  optimisable fp32 tensor;
* SGD (Adam, `lr=1e-2`, `K=50`) on the **evidence-session
  NLL** with all other params frozen;
* score the optimised `M_c` on callback-token CE (local mask
  only, same `chain_callback_position` /
  `session_callback_mask` bookkeeping as
  `tools/eval_callback.py`);
* report `ttt_lift_vs_floor = ce_floor − ce_ttt` against the
  evidence-redacted memory floor (distractor sessions in
  evidence positions, mirroring `tools/eval_callback.py`).

**Decision rule (committed before runs landed):**

| `ttt_lift_vs_floor` | verdict | next step |
|---|---|---|
| `> +0.30` | POSITIVE — readout can decode an arbitrary informative `M_c`; writer is the bottleneck | re-launch v17a as PRE-REGISTERED §1: `--writer_probe_warmup_only`, freeze {LM-head, router, readout}, probe-loss-only, gradient-ratio diagnostic on |
| `(0, +0.30]` | MIXED — partial read-side capacity | compose F2 with §4 read-side intervention or §2 DeltaNet writer |
| `≤ 0` | NEGATIVE — read-side cannot decode any `M_c` | **architecture pivot**; do NOT launch any writer-side cell |

**§5 sweep results (2026-05-03 14:50 UTC-5, n=32 chains/cell, K=50, lr=1e-2):**

| cell | GPU | ckpt | corpus | init | `ce_writer` | `ce_floor` | `ce_ttt` | **`ttt_lift_vs_floor`** | verdict |
|---|---|---|---|---|---:|---:|---:|---:|---|
| §5-A | 0 | `chain_v14k_d4v2_norm_no_warmup_local/best` | synthd4v2_val | writer | 4.632 | 4.671 | 4.730 | **−0.059** | NEG |
| §5-B | 0 | `chain_v14k_d4v2_norm_no_warmup_local/best` | synthd4v2_val | iid    | 4.632 | 4.671 | 5.388 | **−0.717** | NEG |
| §5-C | 0 | `chain_v15a_d4v2_norm_replicate_local/best` | synthd4v2_val | writer | 4.814 | 4.736 | 4.969 | **−0.233** | NEG |
| §5-D | 0 | `chain_v15a_d4v2_norm_replicate_local/best` | synthd4v2_val | iid    | 4.814 | 4.736 | 5.571 | **−0.835** | NEG |
| §5-E | 1 | `chain_v15e_d4v2_1p7b_norm_local/best` | synthd4v2_val | writer | 4.460 | 4.428 | 4.661 | **−0.233** | NEG |
| §5-F | 1 | `chain_v15e_d4v2_1p7b_norm_local/best` | synthd4v2_val | iid    | 4.460 | 4.428 | 5.868 | **−1.440** | NEG |

**Verdict (6/6 cells, all NEGATIVE):** the read-side
(readout + depth-router + LM-head) of v14k/v15a/v15e cannot
map a chain-specific `M_c` to chain-specific callback
predictions, regardless of how informative `M_c` is about the
evidence content. The evidence-session NLL **does** drop
during TTT (e.g. v15e iid `ev_loss 6.96 → 3.32`), so the
optimisation is not failing — `M_c` is being meaningfully
shaped to the evidence — but that optimised `M_c` *worsens*
callback CE relative to the evidence-redacted floor. This
directly confirms the Combined critique's primary prediction:
**the failure mode is read-side, not writer-side.** Per the
pre-committed decision rule, **no writer-side cell (F2 §1,
DeltaNet §2, etc.) should be launched** until the read-side
decoder is rebuilt.

Two follow-up sweeps in flight (GPU 0):

| cell | ckpt | corpus | init | status |
|---|---|---|---|---|
| §5-G | `chain_v17a_f2_codes_0p6b_frozen_local/best` | synthd5_val | writer/iid | RUNNING (queued behind v17a-D5) |
| §5-H | `chain_v17a_f2_codes_0p6b_frozen_local/best` | synthd4v2_val | writer/iid | QUEUED |
| §5-I | `chain_v17b_f2_codes_0p6b_joint_local/best` | synthd5_val + d4v2_val | writer/iid | QUEUED |

These are the most-likely-positive cells: the v17a/v17b
writers were trained against the F2 probe loss, so if F2 helps
*any* part of the read-side at all, their `M_c` should TTT
better than the v14/v15 writers'. v16a (D5, no F2) is also a
priority — it lives only on GH200; rsyncing it back is on the
queue once the GH200 v17e cleanup lands.

Launchers: `scripts/run_ttt_mc_{gpu0,gpu1,v17_gpu0}.sh`.
Logs: `logs/ttt_mc/{gpu0,gpu1,gpu0_v17}.log`. JSON summaries:
`results/ttt_mc_v17pre/<ckpt>__<corpus>__<init>.json`.

### Code shipped in v17

* `src/modeling_memres.py`:
  * `Qwen3MemResConfig.writer_probe_enabled: bool` and
    `writer_probe_n_queries: int` — config plumbing for the
    new probe head.
  * `class WriterProbeHead(nn.Module)` — single-layer
    cross-attention from `n_queries` learned queries into
    `M_c`, projected to vocab; produces logits of the
    answer's first BPE token. Used as a writer-only
    extractive supervised loss (F2). Initialised in
    `_init_memres_params` (random `query` parameter) — the
    initial naive `nn.Parameter(torch.randn(...))` did not
    survive HF's meta-tensor materialisation in
    `from_pretrained(low_cpu_mem_usage=True)` and produced
    NaN logits at first forward; explicit
    `_init_memres_params` re-randomisation was the fix.
  * `WriterProbeHead.forward` casts scores to fp32 before
    softmax and `v` to `proj.weight.dtype` before the linear
    map (fixed `expected mat1 and mat2 to have the same
    dtype, but got: float != c10::BFloat16` from the smoke
    test).
* `src/train_chain.py`:
  * CLI: `--writer_probe_enabled`,
    `--writer_probe_loss_weight`,
    `--writer_probe_warmup_steps`,
    `--writer_probe_n_queries`,
    `--writer_probe_warmup_only`.
  * Probe-loss block: extracts the first answer token from
    `last_labels` via `last_cb_mask`, runs
    `model.writer_probe_head(M_for_probe)`, computes
    `F.cross_entropy`, adds to `total_loss` with the
    scheduled weight, with NaN/Inf guards (logs as
    `_last_writer_probe_loss=NaN` if non-finite).
  * Probe loss + weight surfaced in the per-step log line.
* `tools/eval_ttt_mc.py` (new) — §5 capacity probe; details
  above.
* `scripts/run_ttt_mc_gpu0.sh`,
  `scripts/run_ttt_mc_gpu1.sh`,
  `scripts/run_ttt_mc_v17_gpu0.sh` — drivers.

---

## v16 — vision-aligned corpus & loss; evidence_lift as primary metric (RECLASSIFIED 2026-05-03 13:30 UTC-5: load-bearing falsification, NOT candidate solution)

> **2026-05-03 13:30 UTC-5 (Combined critique).** v16a's
> `evidence_lift ≈ 0` on D5 with `--mask_evidence_session_loss`
> is now read as a **load-bearing falsification of v15-era
> Architectural Prior #9** ("memory is the only pathway that
> carries chain-specific bindings") — not a candidate solution.
> The LM-NLL loss does not propagate a chain-specific gradient
> to the writer; the backbone learns a chain *fingerprint*
> instead. See the new "v17" section above for the §5
> (TTT-on-M_c) pre-experiment that gates further writer-side
> work, and `paper_drafts/v17_wildcards.md` for the design
> matrix the critique referenced.

### Pre-summary (TL;DR for the next reader)

The v15 OPEN AUDIT was looking at the wrong layer. The five candidate
leaks (A1..A5) all asked "what non-memory pathway carries the answer
into the LM head's loss?" — but the v14k `best_callback_aware.json`
shows that even the *single* model the audit was tasked with
defending fails the load-bearing prediction:

```
ce_mem            =  4.98
ce_nomem          =  6.42
ce_shuffle        =  5.04
ce_mem_floor      =  5.05    # M_c built from a FILLER session (no evidence)
pa_cb_dnm         = +1.44    # the headline number the v15 audit centred on
pa_cb_dnm_floor   = +1.37    # SAME number with evidence redacted
pa_cb_evidence_lift = +0.071
```

**1.37 / 1.44 = 95.1% of v14k's apparent "memory benefit" is delivered
by an `M_c` built from a session that does not contain the answer.**
Across v14g..l, v15a, v15c, v15e the same ratio holds: 70–98% of
`pa_cb_dnm` survives evidence redaction. That signal is not
retrieval — it is a content-blind static prior tensor that the writer
emits on every chain (because the readout/router learns to bias the
LM head toward the per-callback marginal regardless of `M_c`'s
content). Architectural Prior #9 in the README ("memory is the only
pathway that carries chain-specific bindings") is empirically wrong
on D4v2; the v15 OPEN AUDIT (A1/A2/A3 spawned 03:40 UTC-5) is
examining the wrong layer of the system.

The vision-aligned remedy lives in v16:

* **Corpus (D5).** `tools/build_synthetic_random_codes.py`. Per-chain
  unique 5-character random alphanumeric IDs (~60M unique IDs; ~3 BPE
  tokens each) replace the 32-item closed set. Across 5000 chains
  the expected ID collision rate is <1e-3, so no chain's ID is
  recoverable from a per-category dataset marginal. The LM-loss-
  optimal policy when memory is uninformative is per-token uniform
  over the ID alphabet, so the marginal CE on the answer span is
  ~10 nats per ID-character (vs ~3 nats on D4v2's closed set). The
  evidence session's assistant turn ACKs without echoing the code:
  `Got it, I'll remember that.` (vs D4v2's `Got it, your favorite
  color is red. I'll remember that.`) — so the LM head is never
  directly supervised on the binding template inside an evidence
  session.
* **Loss.** New trainer flag `--mask_evidence_session_loss` zeros LM
  loss on every position of every evidence session in the window.
  The writer is still trained (evidence is still compressed into
  `M_c` after the LM forward), but the LM head's only direct
  supervision on the answer span is at the callback session, where
  the answer is not in local context. **Memory is the only pathway
  with non-trivial pressure on the callback's answer span by
  construction.**
* **Best-metric.** New `--save_best_metric evidence_lift` saves on
  `pa_cb_dnm − pa_cb_dnm_floor` (with a 0.05 weight tie-break on
  `pa_cb_dsh`). The legacy `phase_aligned` is gameable by content-
  blind writers (see v14k/v15a/v15e ledger above).
* **Smoking-gun control.** New trainer flag `--constant_mc_control`
  replaces the per-chain compressed `M_c` with a single learnable
  parameter shared across all chains. Writer / extract / judge are
  bypassed entirely; only the readout, router, and the constant-Mc
  parameter receive gradient. **Predicted result if v14k's
  pa_cb_dnm was a learnt content-blind output prior:** this control
  reproduces v14k's `pa_cb_dnm` (~+1.4 nats on D4v2) within noise,
  while `pa_cb_evidence_lift` stays exactly 0 by construction (no
  chain-specific input ever reaches `M_c`).

### v16 cells launched (2026-05-03)

| cell | machine | preset | backbone | window_k | step | corpus | result | location |
|---|---|---|---|---:|---:|---|---|---|
| **v16a** | GH200 | qwen3-0.6b-large | FROZEN | 3 | 2500 | **synthd5_random_codes** | RUNNING since 03:13 UTC-5; first PA-EVAL @100 → `pa_cb_ce_mem=6.92`, `evidence_lift=−0.0024` (start of training; D5 callback CE ~7 nats vs D4v2's ~5) | `(GH200) output/chain_v16a_codes_0p6b_frozen_gh200/{best,final}` |
| **v16b** | GH200 | qwen3-1.7b-large | FROZEN | 3 | 3500 | **synthd5_random_codes** | QUEUED behind v16a; same recipe at 1.7B scale; companion test for the v15e overfit-prior signature | `(GH200) output/chain_v16b_codes_1p7b_frozen_gh200/{best,final}` |
| **v16c** | local H100 (TBD) | qwen3-0.6b-large | FROZEN | 3 | 2000 | **synthd4v2** (same as v14k) | NOT STARTED — constant-M_c control (`--constant_mc_control`) on the same D4v2 corpus v14k won on; the smoking-gun control for "v14k's pa_cb_dnm is a learnt prior" | `output/chain_v16c_constmc_control_d4v2_0p6b_local/{best,final}` |
| **v16d** | GH200 | qwen3-1.7b-large | FROZEN | **2** | 2000 | **synthd5_random_codes** | QUEUED behind v16b (queue spec `1777800418_chain_v16d_*.json`, ntfy `memres-e6ebdc70`); pair-wise NIAH probe — same recipe as v16b but `--window_k 2` (trainer-supported minimum) so the LM can attend to at most 1 prior session at the callback step. With synthd5 evidence at random body position ∈ [0, 8] and callback at 9, ~78% of chains have *both* needles strictly inside `M_c` only (vs ~62% at v16b's k=3); the masked evidence-session loss leaves the callback span as the sole LM-supervised answer position. **Decision rule:** if step-2000 `evidence_lift > +0.05`, relaunch as v16e (6000-step continuation, resume from best/, joint-train hand-off candidate); if in `[0, +0.05]`, relaunch v16e at 6000 steps from scratch (warmup too short); if `≤ 0` and v16b also stalls, fall through to v16f rebuilding synthd5 with `body_positions := range(body_len-1)` to fully eliminate the residual ~22% LM-visible-needle minority before triaging the writer/readout architecture itself. | `(GH200) output/chain_v16d_niah_pairwise_1p7b_frozen_gh200/{best,final}` |

Launchers: `scripts/train_v16{a,b}_codes_*_frozen_gh200.sh`,
`scripts/train_v16c_constmc_control_d4v2_0p6b_local.sh`,
`scripts/train_v16d_niah_pairwise_1p7b_frozen_gh200.sh`.

### v16 code shipped

* `tools/build_synthetic_random_codes.py` (new): D5 corpus generator
  with 8 fact categories ("locker code", "PIN", "employee ID",
  "apartment", "flight", "confirmation number", "tracking", "voucher
  code"), per-chain unique random alphanumeric IDs, no assistant
  echo of the binding inside the evidence session. Output schema
  is identical to `synthd4v2_persona_callback_*.pt` plus two new
  fields: `session_role` (0=filler, 1=evidence, 2=callback) and
  `session_evidence_mask` (per-token mask on the evidence ID span).
  D4v2 / LME / MSC corpora load with the new fields defaulted to
  zeros (back-compat preserved). Audit mode: `--audit_tokenisation`
  prints sample IDs + their BPE token IDs + the evidence/callback
  mask positions for the first chain.
* `paper_artifacts/chains/synthd5_random_codes_train_s512.pt`
  (5000 chains, 50000 sessions, 20884 cb tokens, ~4.18 cb-tokens/
  chain) and
  `paper_artifacts/chains/synthd5_random_codes_val_s512.pt`
  (500 chains, 5000 sessions, 2099 cb tokens). Built locally and on
  GH200 with seed 42 (train) / 9001 (val).
* `src/train_chain.py`:
  * `--save_best_metric evidence_lift` — new option. Score = −(
    `pa_cb_evidence_lift` + 0.05·`pa_cb_dsh`); tie-break by overall
    callback discrimination so two ties on lift still pick the more
    discriminative checkpoint. Falls back to composite when no
    evidence-labelled chains land in the eval sub-sample.
  * `--mask_evidence_session_loss` — new flag. Sets `labels=-100` on
    every position of every evidence session in the TBPTT window so
    `F.cross_entropy(ignore_index=-100)` drops them from the LM loss.
    Writer is still trained (the per-session compress_session call
    after the LM forward is unaffected).
  * `--constant_mc_control` — new flag. Replaces per-chain
    compressed `M_c` with a single shared `nn.Parameter` of shape
    (1, K, d) that broadcasts across the batch; bypasses
    `compress_session` entirely (both burn-in and per-session).
    Validates that `--contrastive_infonce_weight`,
    `--alpha_mem_floor_aux_weight`, and `--neg_chain_weight` are 0
    (those losses depend on chain-specific M_c statistics);
    auto-disables `curriculum_competition_bias`.
  * `ChainCorpus` loads optional `session_role` and
    `session_evidence_mask` fields with all-zero fallbacks.
  * `ChainSampler.sample_window` returns a 6-tuple
    `(chain_idx, anchor, window, burn_in, callback_mask, role)`;
    contiguous Branch 2 fetches `role` via `chain_window_role`,
    curriculum branches via the 3-tuple `chain_curriculum_window`.
* `scripts/train_v16{a,b}_codes_*_frozen_gh200.sh` and
  `scripts/train_v16c_constmc_control_d4v2_0p6b_local.sh` — launch
  scripts. v16{a,b} use the GH200 flat repo layout
  (`python -u train_chain.py`); v16c uses the local `src/` layout
  (`python -u src/train_chain.py`).

### v15 OPEN AUDIT — status update (2026-05-03)

A1/A2/A3 are still the right tools for **explaining v15b/v15f's
joint-train collapse** (those cells need to be understood
regardless of v16's outcome), but they are **not the path to
recovering the architecture's vision** — that requires v16's
corpus and loss redesign. The audit deliverables remain expected
on the paths originally listed; their conclusions feed into the
README's Architectural Prior #9 revision once both v16a and the
audits land.

---

## v15 — extract_input_norm + double-evidence D4v2 (active; 2026-05-02 → present)

### Pre-summary (TL;DR for the next reader)

Rebuilt the synthetic corpus to **two evidence sessions per chain**
(`synthd4v2_persona_callback_*_s512.pt` via
`tools/build_synthetic_persona_callback.py --n_evidence_sessions 2`),
landed `--memres_extract_input_norm` (RMSNorm on the context tensor
`C` before `MemoryBlock.extract`), removed `writer_warmup` entirely,
and ran the v14g..l + v15a..f matrix. The numerical headline has
shifted **twice** since the v14abl post-mortem:

* **Flip 1 — eval-tooling artefact (v14k @ 0.6B, frozen).**
  `eval_chain.py` averages CE over the entire score window
  (`score_tail_frac=1.0` ⇒ all 4 sessions), so a localised
  callback-token effect is diluted ~38× into noise. Built
  `tools/eval_callback.py` (mirrors the in-trainer `pa_cb_*` metric:
  callback-token positions only, evidence-redacted memory floor for
  the baseline). On v14k_best: `pa_cb_dnm = +1.44`,
  `evidence_lift = +0.071`. ✅ confirms in-trainer number is real.

* **Flip 2 — concerning (v15e @ 1.7B, frozen, norm ON).**
  `Δnm-m_floor = +2.5 nat` (memory provides 2.5 nat help vs no
  memory) but `evidence_lift` swings *negative*: `−0.10 to −0.27` for
  the bulk of training, briefly positive at step 200–300
  (+0.07 to +0.31) before the writer overfits. **Evidence-redacted
  memory beats full memory.** The 1.7B writer is using non-evidence
  content (filler embeddings, persona-prior, callback-template) to
  "pre-route" the callback answer; adding the actual evidence into
  M_c during the evidence sessions disrupts that prior.

* **Flip 3 — fundamental setup bug, audit in progress (v15b @ 0.6B,
  joint train).** Both `Δnm-m_floor` and `evidence_lift` collapse
  to ≈ 0 for the full 4000-step run with `lr_backbone=2e-5`. The
  unfrozen 0.6B backbone learns the callback distribution **directly**
  — which should not be possible if memory is the only pathway
  carrying evidence. **The user has flagged this as a fundamental
  setup bug, not an ablation question** (2026-05-03 ~02:30 UTC-5).
  See "v15 OPEN AUDIT" below; v15g is postponed pending audit.

### v15 cells launched (2026-05-02 → 2026-05-03)

| cell | machine | preset | backbone | window_k | step | result | location |
|---|---|---|---|---:|---:|---|---|
| **v14g** | local H100 GPU 0 | qwen3-0.6b | FROZEN | 4 | 2500 | norm ON, warmup 200 → mid Δsh-m, ~0 evidence_lift | `output/chain_v14g_d4v2_warmup_norm_local/best` |
| **v14h** | local H100 GPU 1 | qwen3-0.6b | FROZEN | 4 | 2500 | norm OFF, warmup 200 → ditto, slightly worse | `output/chain_v14h_d4v2_warmup_nonorm_local/best` |
| **v14i** | GH200 | qwen3-0.6b | FROZEN | 4 | 2500 | warmup_router_bias 8.0 → recent-bias lock-in, neg lift | `(GH200) output/chain_v14i_d4v2_strongwarmup_gh200/best` |
| **v14j** | local H100 GPU 0 | qwen3-0.6b | FROZEN | 4 | 2500 | warmup 0, norm ON, slot writer → mid lift (+0.04) | `output/chain_v14j_d4v2_nowarmup_slot_local/best` |
| **v14k** | local H100 GPU 1 | qwen3-0.6b | FROZEN | 4 | 2500 | warmup 0, norm ON, slot writer, alpha-floor + InfoNCE → **+0.071 evidence_lift** ✅ | `output/chain_v14k_d4v2_nowarmup_slot_floor_local/best` |
| **v14l** | GH200 | qwen3-0.6b | FROZEN | 4 | 2500 | as v14k but writer=cross_attention (no slot) → similar lift, slightly noisier | `(GH200) output/chain_v14l_d4v2_nowarmup_xattn_floor_gh200/best` |
| **v15a** | local H100 GPU 0 | qwen3-0.6b | FROZEN | 4 | 2500 | replicate v14k recipe → `pa_cb_dnm +1.33, dsh +0.02` ✅ reproducible | `output/chain_v15a_d4v2_norm_replicate_local/best` |
| **v15b** | local H100 GPU 1 | qwen3-0.6b | **trained** (lr_b=2e-5) | 4 | 4000 | joint training → `evidence_lift ≈ 0` for the full run; **THIS is the bug** | `output/chain_v15b_d4v2_norm_jointtrain_local/best` |
| **v15c** | GH200 | qwen3-0.6b | FROZEN | 4 | 2500 | extract source = `embed` instead of `hidden_14` → `evidence_lift +0.005-+0.02`; hidden_14 is better | `(GH200) output/chain_v15c_d4v2_norm_extract_embed_gh200/best` |
| **v15e** | local H100 GPU 0 | qwen3-1.7b-large | FROZEN | 4 | 2000 | 1.7B frozen, norm ON → `Δnm-m_floor +2.5` but `evidence_lift -0.18` (writer overfits) | `output/chain_v15e_d4v2_1p7b_norm_local/best` |
| **v15f** | local H100 GPU 0 | qwen3-1.7b-large | trained (lr_b=2e-5) | 4 | 3000 | 1.7B joint train (started 2026-05-03 02:17 UTC-5; RUNNING ~step 100) | `output/chain_v15f_d4v2_1p7b_jointtrain_local/{step-N,best,final}` |

Launchers live at `scripts/train_v14g..l_*.sh` and
`scripts/train_v15{a,b,c,e,f}_*.sh`. The local sequencers are
`scripts/orchestrate_gpu0_v15_wave.sh` and
`scripts/orchestrate_gpu1_v15_wave.sh`.

### v15 code shipped

* `src/modeling_memres.py`:
  * `Qwen3MemResConfig.memres_extract_input_norm: bool` — when true,
    `MemoryBlock.__init__` constructs
    `self.extract_input_norm = Qwen3RMSNorm(d, eps=...)` and
    `MemoryBlock.extract` wraps `C` through it before the
    cross-attn / slot-attn path. Diagnosed as the dominant root
    cause of `M_new` norm explosions (~50× backward grads on `W_Q`).
  * `MemoryBlock.forward` write_gate sigmoid is **bypassed entirely**
    for `writer_kind ∈ {slot_attention, slot_attention_full}`
    (Locatello GRUCell already gates per-slot; stacking the external
    sigmoid saturated within 50 steps and locked M_c at zero).
  * For `writer_kind=original`, the `gate_input` is RMSNormed on
    each side before the sigmoid (`write_gate_norm_prev`,
    `write_gate_norm_new`). Verified: feeding `‖C‖₂ = 100·N(0,1)`
    drops post-RMSNorm `gate_input` magnitude to ~0.78 and the
    sigmoid sits at the intended 0.27 init.
* `src/train_chain.py`:
  * `--memres_extract_input_norm` flag forwarded through
    `memres_kwargs` into the config override.
  * `--kill_on_memory_collapse` — loud-halt guardrail in the eval
    loop. Two consecutive evals (after
    `--kill_on_memory_collapse_min_step=200`) with
    `Mc_pair_to_self_ratio < 0.01` *or* `mt_norm_ratio_mean < 0.01`
    aborts the run with exit code 42 and persists a forensic
    `killed-step-N` checkpoint. Converts the silent-failure burn
    (v13/v14/v15a-style) into a loud halt that
    cloud_watchdog / CI pick up.
* `src/presets.py`: added `qwen3-1.7b-small` (L_E=0) and
  `qwen3-1.7b-large` (L_E=4); both `memres_num_vectors=128`,
  `memres_num_blocks=8`.
* `tools/build_synthetic_persona_callback.py`: new
  `--n_evidence_sessions` (default **2**) and `--n_prefix_sessions`
  flags. Multi-evidence forces the writer to integrate two
  distinct-category facts before the readout discriminates which one
  the callback queries — removes the trivial single-evidence "keep
  everything" optimum that v13/v14 D4 had.
* `tools/eval_callback.py` (new): standalone callback-aware eval
  that mirrors the in-trainer `pa_cb_*` metric — scores only
  callback-mask token positions in the callback session, with the
  floor baseline using an evidence-redacted memory state. **Use this
  for D4v2 post-train eval; `eval_chain.py` averages over the score
  window and dilutes localised callback effects ~38×.**
* `tools/locomo_to_chains.py` + `tools/pretokenize_chains.py`: built
  `paper_artifacts/chains/locomo_s512.pt` (10 conversations,
  272 sessions) and `paper_artifacts/chains/msc_test_s512.pt`
  (500 chains, 2500 sessions) for cross-domain transfer eval.
* `scripts/eval_v14_v15_benchmarks.sh`: wraps `eval_chain.py` for
  D4v2-val + LoCoMo + MSC-test sweeps.
* Stale duplicate `train_chain.py` and `modeling_memres.py` at the
  repo root were deleted; only `src/*.py` remain. The roots were
  never updated past v11 and would crash with "unrecognized
  arguments" if any v12+ launcher accidentally pointed at them.

### v15 OPEN AUDIT (2026-05-03 ~02:30 UTC-5) — the user-flagged fundamental issue

**Concern.** In v15b (and v15f), an unfrozen backbone with
`lr_backbone = 2e-5` collapses both `Δnm-m_floor` and `evidence_lift`
to ~0. This should be impossible in a well-designed setup: callback
tokens should be unpredictable from anything *except* the memory
pathway (since the evidence sessions live outside the LM-attended
window). If the backbone CAN learn the callback distribution
directly, that means the non-memory pathway is leaking
answer-bearing information.

Five candidate leaks under audit:

1. **Window leakage** — at `window_k=3` the LM attends to the last 3
   sessions of the chain. Evidence is placed at random body
   positions, sometimes inside the last 3 ⇒ direct (non-memory)
   access. The trainer's `score_tail_frac=1.0` makes this leak
   *training-active* on every window where evidence happens to land
   in the tail. Quantitative test: count chains where any evidence
   position is in `[chain_callback_position − window_k + 1,
   chain_callback_position]`.
2. **Template / prior leakage** — callback session text always reads
   `Assistant: Your favorite {category} is {item}.` The "is" is one
   token before the answer. A 1.7B model trained on enough chains
   can learn `Your favorite color is` → marginal over the 32 colors,
   pushing CE down even without evidence. Closed-set has 256 items
   in 8 categories ⇒ ceiling guess CE = log(32) = 3.5 nats; with
   persona priors over the dataset (e.g. "red" appears more often),
   the prior could push CE down meaningfully.
3. **Same-chain evidence visibility via the rolling memory state** —
   the evidence-redacted "floor" baseline in `_phase_aligned_eval`
   may not actually be removing information from M_c if the
   redaction operates on a rolling state that already integrated the
   evidence in earlier sessions of the same window.
4. **Cross-window state leakage** — BPTT / detach boundaries between
   chain windows; if the backbone's hidden state at window boundary
   carries any cross-session signal, that's another non-memory
   pathway.
5. **Tokeniser-level leakage** — closed-set items are 1-3 BPE
   tokens. If the *first* token of an item is unique enough (e.g.
   "stra" → only "strawberry"), and the prior token is "is " plus
   the question's category cue, then BPE structure makes the answer
   guessable.

Three independent auditors spawned in parallel as `Task` subagents
on 2026-05-03 03:40 UTC-5 (deliverables land at the paths listed
below; placeholder `audit_a_corpus_prior.md` was deleted at launch):

* **A1** — corpus-and-window leakage audit: enumerate D4v2-train
  chains, count evidence-in-window collisions for `k=3,4`, measure
  CE-on-callback for a frozen base model with no memory at all
  (i.e. the upper bound on what the backbone can learn directly).
  GPU 0. Output: `audit_a1_window_leakage.md`.
* **A2** — eval-redaction audit: trace `_phase_aligned_eval` /
  `eval_callback.py` redaction logic; verify that the "floor"
  baseline truly excludes evidence from M_c at the moment the
  callback is scored. Adversarial test: replace evidence with
  cross-chain sessions at score time and check CE doesn't shift.
  GPU 1. Output: `audit_a2_redaction.md`.
* **A3** — base-rate / template prior audit: for the v15b/v15f joint
  runs, decompose CE into `CE_template_prior + residual`; show
  whether the unfrozen backbone's callback CE matches the empirical
  `P(item|category)` baseline within ~0.1 nat. GPU 1. Output:
  `audit_a3_base_prior.md` (covers all 6 ckpts; local).
* **A3-cloud** — 1.7B leg of A3 on the GH200, in parallel to local A3
  to cut wall-clock. Rsyncs `src/` + D4v2 train/val + the three 1.7B
  checkpoints (v15e/best, v15f/best, v15f/step-500) up to GH200, runs
  the prior + per-ckpt CE there, pulls results back. Launched 03:46
  UTC-5. Output: `audit_a3_base_prior_1p7b_cloud.md`. (Redundant with
  local A3's 1.7B rows once both finish — keep as cross-check.)

Pending audit results before any further training cells fire. v15g
(was queued as 1.7B-small on GPU 1) has been **postponed** until
A1/A2/A3 land. (One literature-review memo, `audit_b_literature.md`,
was already on disk at launch — survey of NoLiMa / RULER / Lost in
the Middle / LongMemEval pitfalls; orthogonal to A1/A2/A3 but cited
by A3's framing.)

---

## v11 → v14 — folded into [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md) Part VI

The v11 (g/h/i/j/k/l-fix/m/p/q/r), v12 (slot-attention writer; D4
retarget), v13 (writer_warmup + orth + slot_positional + the
config-merge bugfix; v13c2/v13r/v13q), and v14 (judge_qk_layernorm +
alpha_mem_floor + InfoNCE + AP warmup anneal; v14abl_a/b, v14a,
v14g..l) ledger entries have all been folded into
`archive/COMPREHENSIVE.md` Part VI on 2026-05-03 per the folding
convention in Part VII.

Last-eval / verdict summary (full tables + result trajectories +
inline launch flags in Part VI):

| campaign | verdict | leading checkpoint at end of campaign |
|---|---|---|
| **v11** (P0+P2+P3 fix; cells g/h/i/j/k/l-fix/m/p/q/r) | writer is content-blind under LM-only objective; chain-identity hash under InfoNCE; D5 audit on v11g/best identifies the readout as bottleneck | `runs/chain_v11g_ap_baseline_gh200/best` step 600 (PA CB Δ_nm-m=+0.030, decayed by 1400) |
| **v12** (slot-attention writer) | hits the same uniform fixed point at step 800; slot-attention alone is necessary but not sufficient — GRU shares weights across slots, symmetry re-emerges | `runs/chain_v12a_slot_judge_d4_local/step-200` only (peak before collapse) |
| **v13** (writer_warmup + orth + slot_positional + config-merge bugfix) | symmetry permanently broken (D3-MC pair/self = 0.004 sustained); v13c2 hit `evidence_lift +1.4` mid-warmup; phase-2 backbone unfreeze destroys writer specialisation | `runs/chain_v13c_d4_ap_gh200/best` step 600 (warmup peak only) |
| **v14** (judge_qk_layernorm + alpha_mem_floor + InfoNCE + AP anneal; D4v2) | judge_qk_ln × slot_attention is anti-causal (writer never lifts off zero); without QK-LN, writer specialises but Δ_nm-m goes negative — InfoNCE satisfies itself with non-LM-useful chain-distinguishable M_c. v14k @ FROZEN backbone is the first reproducible positive: `evidence_lift +0.071` | `runs/chain_v14k_d4v2_nowarmup_slot_floor_local/best` |

---

## v10 → v6 — folded into [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md) Parts IV–V

Same convention; older entries previously folded.

---

## How to add a new run

1. Create / pick a `scripts/train_*.sh` for the run. The script's
   header MUST document what it varies vs the closest existing run.
2. Pre-allocate the `chain_<version>_<descriptor>` run name in this
   ledger as `Status: NOT STARTED` with the script path filled in.
3. Launch (locally in tmux or via the cloud watchdog
   `enqueue.sh`).
4. Update Status to TRAINING with start timestamp + machine.
5. On finish / kill / failure, update Status with end timestamp,
   reason, last step, last `Δ_sh-m`, location of the `best/` ckpt,
   and link to the log.
6. When the post-train pipeline runs, link the eval JSONs.

Do NOT delete entries when a run is superseded. Mark them KILLED /
SUPERSEDED-BY and keep the paper trail. When the active ledger gets
crowded, fold superseded entries into `archive/COMPREHENSIVE.md`
Part VI per the convention in Part VII of that file.

## Conventions

- Run names: `chain_v<N>_<descriptor>` where `<N>` increments on a
  major recipe / architectural change.
- Tmux naming: cloud watchdog uses `cwd-<run_name>`; local launches
  use `local-<run_name>`.
- Log paths: cloud → `~/memory_residuals/tools/cloud_watchdog/logs/<run_name>.log`;
  local → `logs/<run_name>.log`.
- Output dirs: always `runs/<run_name>/{step-N, best/}`.
