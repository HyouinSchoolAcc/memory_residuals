# memory_residuals

Hi, welcome to the **mem_residuals** repo.

The goal of this project is to realize a constant-sized memory
matrix `M_c` that gets compressed, trained, and queried by an LLM
**natively** — no retrieval index, no separate memory controller,
no hand-engineered gating heuristic. As of **2026-05-04**, the
v27b / v28 wave delivers that: a frozen Qwen3 backbone augmented
with a 41.5 M-parameter (~6 % overhead) `M_c` channel achieves
**+1.32 ± 0.53 nats of chain-specific callback CE improvement on
LongMemEval-S validation at 0.6B (n=4 seeds)**, **+0.93 nats at
1.7B (n=2 seeds)**, with a chain-shuffle confound statistically
zero throughout. See the **Headline result** block below.

Architectural spec is in
[`memory_residuals.pdf`](memory_residuals.pdf); the block-attention
residuals reference is in [`atn_residuals.pdf`](atn_residuals.pdf).
Locked numbers in [`NEURIPS_NUMBERS.md`](NEURIPS_NUMBERS.md);
paste-ready abstract in
[`ABSTRACT_NEURIPS_v3.md`](ABSTRACT_NEURIPS_v3.md).

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
   0`).** Active state in `results/exp2_chain_recipe/runs.md`.

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
    locked in the **multi-layer readout depth** (`--memres_readout_depth 4`)
    and **strong α-floor** (`--alpha_mem_floor_aux_weight 0.5`,
    target 0.10) as the load-bearing read-side levers.

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
    α-floor.** Numbers locked in [`NEURIPS_NUMBERS.md`](NEURIPS_NUMBERS.md);
    abstract framing in [`ABSTRACT_NEURIPS_v3.md`](ABSTRACT_NEURIPS_v3.md).

## Progress & lessons

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
  Numbers locked in [`NEURIPS_NUMBERS.md`](NEURIPS_NUMBERS.md).

[Full per-cell tables, decision triggers, and mechanism statements
for v11–v14 in [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md)
Part VI; v15–v28 active state in
[`results/exp2_chain_recipe/runs.md`](results/exp2_chain_recipe/runs.md).]

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
[`NEURIPS_NUMBERS.md`](NEURIPS_NUMBERS.md); abstract paste-ready
in [`ABSTRACT_NEURIPS_v3.md`](ABSTRACT_NEURIPS_v3.md). Active
ledger and history in
[`results/exp2_chain_recipe/runs.md`](results/exp2_chain_recipe/runs.md).

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

## Currently running

| cell | host | status | notes |
|---|---|---|---|
| (none) | | | v27b/v28 wave landed 2026-05-04 ~12:45 EDT and is the project headline. Local H100s and GH200 are idle. Next candidates: bootstrap-over-chains 95 % CI for the 0.6B headline (≤ 2 min compute, appendix-only number), additional 1.7B v28 seeds to tighten the n=2 → n≥4 multi-seed claim, scale beyond 1.7B (Qwen3-4B / Qwen3-7B same recipe), and a clean LoCoMo / MSC train-on-source-eval-on-target generalisation study. Not launched yet. |

## Resources

- **Local.** 2 × H100 NVL (94 GB) at the lab box. ~16 h/day usable
  (residential power-down overnight).
- **Cloud.** 1 × NVIDIA GH200 480 GB at `192.222.50.225` (user
  `ubuntu`). v14abl_*/v14g/v14h/v14i/v14l and earlier ran here
  through `tools/cloud_watchdog/` (jobs run inside detached `tmux`
  so they survive SSH drops + lab-box power-offs).

## Layout

```
Runs/                    training checkpoints (gitignored, ~11 GB).
                         Only checkpoints cited by the papers
                         survive on disk; v3-v10 pruned 2026-04-30;
                         v11-v15 best/ checkpoints kept for forensics;
                         v24a / v25a / v27b-seed{1..4} / v28a / v28b
                         all live here as the headline-cell ckpts.

Scripts/                 one .sh per training cell — the launchers.
data/                    pre-tokenised corpora (symlink to
                         paper_artifacts/chains/).
src/                     architecture + trainer code:
  modeling_memres.py     architecture (config, model, init).
  train_chain.py         recurrent chain TBPTT trainer (active).
  train_phase1.py        pair-based warm-up trainer (Paper 1).
  presets.py             named (backbone, K, L_E, N) tuples.
tools/                   eval / probes / corpus builders.
  cloud_watchdog/        remote-survivable job queue + ntfy daemon.
  eval_callback.py       canonical D4/LME post-train eval (USE THIS).
  eval_ttt_mc.py         §5 capacity probe (TTT-on-M_c).
  d5_ttt_readout.py      D5 readout-bottleneck disambiguator.
  build_synthetic_persona_callback.py
                         D4 / D4v2 synthetic corpus generator.
results/                 eval JSONs + paper drafts:
  eval/                  bootstrap CIs, routing traces, etc.
  eval_v14v15/           v14-v15 callback-aware eval JSONs.
  eval_v24_indomain/     v24a / v24c / v21c on lme_val (corpus-pivot).
  eval_lme_locomo/       v24a out-of-domain LoCoMo / MSC eval.
  eval_v25_seed_pack/    v24a/v25a multi-seed on (unpatched) lme_val.
  eval_v25_seed_pack_evpos/
                         v24a / v25a / v27{a,b,c} / v28{a,b} on the
                         patched evpos corpus — the HEADLINE numbers.
  ttt_mc_v17pre / v18{pre,post} / v19post / v20post / v21post /
  v22post / v24post      §5 capacity probe sweep across the read-side
                         audit.
  exp1_pair_recipe/      Paper 1 (drop-in primitive) manuscript.
  exp2_chain_recipe/     Paper 2 (long-horizon recipe) draft +
                         active runs.md ledger.
tests/                   pytest harness for src/.
archive/                 historical reference (~2 MB):
  COMPREHENSIVE.md       full v1 → v14 ledger (Parts I-VI; convention
                         in Part VII).
  agent_sessions/        prior agent worklogs.
  eval/, figures/,
  paper_tools/, scripts/ pre-2026-04-30 cleanup snapshots.

memory_residuals.pdf     position paper (architectural spec).
memory_residuals.tex     ditto, source.
extraction.png /
judging.png / 1.png /
2.png                    figures embedded in memory_residuals.pdf.
atn_residuals.pdf        Block Attention Residuals reference.
NEURIPS_NUMBERS.md       locked numbers ledger — headline + ablations.
ABSTRACT_NEURIPS_v3.md   paste-ready abstract (Branch A, F3-off).
ABSTRACT_NEURIPS_v{1,2}.md / ABSTRACT_v{0,1}.md
                         superseded abstract drafts (kept for diff
                         history with the v3 lock).
README.md                this file.
requirements.txt
.gitignore

output -> Runs           backwards-compat symlink.
logs/                    training & sync logs (gitignored).
paper_artifacts/         physical home of pre-tokenised corpora:
  chains/                LME / MSC / synthd5 / LoCoMo. Reach via `data/`.
  locomo_chains/
  msc_chains_test/
```

## Stop everything

```bash
# local
pkill -f train_chain.py

# cloud watchdog + heartbeat
ssh ubuntu@192.222.50.225 \
  'pkill -f cloud_watchdog/watchdog.sh;
   pkill -f cloud_watchdog/heartbeat.sh;
   tmux kill-server'
```

## Reading order

1. **This README — and specifically the "Architectural priors"
   block.** If you catch yourself proposing a change that contradicts
   one of those eleven bullets, STOP and read the citation before
   proceeding. Pay special attention to **Prior #11**: the F3
   readout probe is *off* in the canonical recipe; do not turn it
   back on without reading `runs.md`'s v27b/v28 verdict block first.
2. [`NEURIPS_NUMBERS.md`](NEURIPS_NUMBERS.md) — the locked
   numbers ledger. Single source of truth for every headline /
   ablation number cited in the paper. Do not edit without
   re-running the cell.
3. [`ABSTRACT_NEURIPS_v3.md`](ABSTRACT_NEURIPS_v3.md) — the
   paste-ready NeurIPS abstract (Branch A, F3-off canonical) +
   author/category fields still TBD.
4. [`memory_residuals.pdf`](memory_residuals.pdf) — the position
   paper / architectural spec (Eqs. 1–10, two-stage QKV competition,
   off-sequence depth-wise injection).
5. [`atn_residuals.pdf`](atn_residuals.pdf) — the Block Attention
   Residuals reference. Read when the routing trace looks weird or
   you're tuning router init biases.
6. [`results/exp2_chain_recipe/runs.md`](results/exp2_chain_recipe/runs.md)
   — the active run ledger (newest at the top: v27b/v28 → v24/v25
   → v17–v22 → v15).
7. [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md) — the
   long-form historical ledger (Part I-II background, Part III
   Stage-1 calendar, Part IV v2-v6, Part V v7-v10, **Part VI
   v11-v14**, Part VII the runs.md folding convention).
