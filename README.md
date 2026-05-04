# memory_residuals

Hi, welcome to the **mem_residuals** repo.

We're trying to realize a constant-sized memory matrix `M_c` that gets
compressed, trained, and queried by an LLM **natively** — no retrieval
index, no separate memory controller, no hand-engineered gating
heuristic. Architectural spec is in
[`memory_residuals.pdf`](memory_residuals.pdf); the block-attention
residuals reference is in [`atn_residuals.pdf`](atn_residuals.pdf).

## Architectural priors (READ FIRST — these are settled)

Future agents: these are baked-in findings from the v3-v15 line of
evidence. Do NOT re-derive them from scratch and do NOT silently swap
them out of a run without reading the citation.

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
    W_Q/W_K/W_V. v18a/best (F3 + alpha-floor, no F2) shifts the
    §5 result from -0.897 → +0.005 (writer-init), the largest
    causal shift in the §5 series and the first MIXED reading.
    F2 + F3 compose *worse* than F3 alone (the two probes pull M_c
    along different value-space directions). End-to-end
    `evidence_lift` is still ≈ 0 because alpha_mem_floor at
    weight 0.01 is too weak to keep the depth router open against
    the LM gradient. **Use `--readout_probe_enabled` going
    forward; do NOT compose with `--writer_probe_enabled`. The
    next-generation alpha-floor weight should be ~0.5 with target
    0.10, and multi-layer readout depth (`--memres_readout_depth
    N`, v19 candidate) is likely needed for fully POSITIVE §5.**

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
  ~0; **OPEN AUDIT** in flight on three suspected non-memory
  pathways (window leakage / eval redaction / template prior).

[Full per-cell tables, decision triggers, and mechanism statements
for v11–v14 in [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md)
Part VI. Active v15 state in
[`results/exp2_chain_recipe/runs.md`](results/exp2_chain_recipe/runs.md).]

## Currently running

| cell | host | status | notes |
|---|---|---|---|
| (none — v24 wave finished 2026-05-03 23:15 UTC-5) | | | next cell is **v25 1.7B-large frozen + LME** (scale up the v24a winning recipe to a bigger backbone). Recipe + design notes in `results/exp2_chain_recipe/runs.md` v24 verdict block. Not launched yet. |

**v24 wave verdict (2026-05-03 23:15 UTC-5) — HEADLINE: v24a/LME is the project's strongest end-to-end result.**

The corpus pivot from synthd5 (templated random-codes, 5 % real-
content density) to LongMemEval (real conversational content, 99.9 %
density) **completely changes the picture**. End-to-end in-domain
on `lme_val_s512.pt`:

| ckpt | trained on | `ce_mem` | `ce_nomem` | **`pa_cb_dnm` (Δnm-m)** | `pa_cb_dsh` (shuffle) |
|---|---|---|---|---|---|
| **v24a/best** | **LME** | **2.94** | 3.28 | **+0.3445** | −0.003 |
| v24c/best | LME+MSC merged | 5.84 | 5.45 | −0.40 | −0.014 |
| v21c/best | synthd5 | 5.11 | 4.67 | −0.43 | −0.025 |

`pa_cb_dnm = +0.3445` on lme_val means **memory reduces callback CE
by 0.34 nats** vs the no-memory baseline. That's ~86× the magnitude
of v21c's best end-to-end on synthd5_val (+0.024) and is **chain-
specific** (shuffle is near zero, ruling out a "memory adds general
context" confound).

This is the **first cell in the project with a non-trivial,
chain-specific, leak-free end-to-end memory benefit**. The memory
residuals architecture works on real long-context conversation data
where:

* Training corpus has 100 % callback supervision (every chain has
  a Q&A pair the F3 probe can target).
* Sessions are dense (~99.9 % real content per 512-token session,
  vs synthd5's ~5 %).
* Content is naturalistic conversation that the backbone has
  representational headroom to compress meaningfully.

§5 readings on synthd5_val (out-of-domain for v24a/c) are NEG (as
expected — v24a's writer has never seen random-code style facts).
For v21c on synthd5_val (in-domain) the §5 was +0.005 / +0.120 on
v20a; we should not over-interpret the cross-domain §5.

**Multi-seed v23 verdict on synthd5 (FINISHED 2026-05-03 22:30 UTC-5):**
The v21c recipe is **unstable** on synthd5_random_codes:

| seed | training corpus | end | last `pair/self` | end `evidence_lift` |
|---|---|---|---|---|
| 1 (v23a) | synthd5 | KILLED step 400 | 0.006 (collapse) | (n/a) |
| 2 (v23b) | synthd5 | KILLED step 600 | 0.006 (collapse) | (n/a) |
| 7 (v23c) | synthd5 | OK final 1000 | 0.032 | −0.0006 |
| 42 (v21c orig) | synthd5 | OK final 1000 | 0.075 | +0.0241 |

Only 1 of 4 seeds (the original seed=42) produced a non-trivial
end-to-end reading on synthd5; 2 of 4 collapsed; 1 of 4 ran cleanly
but ended near 0. The +0.024 reading was a lucky seed, not a
reproducible result. **For synthd5, v21c is not a recipe; for LME,
v24a IS a recipe.**

GH200 idle. Local H100s idle.

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
                         Only checkpoints cited by the two papers
                         survive on disk; v3-v10 were pruned
                         2026-04-30; v11-v14 best/ checkpoints live
                         here while v15 audit is in progress.

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
  eval_callback.py       canonical D4 post-train eval (USE THIS).
  d5_ttt_readout.py      D5 readout-bottleneck disambiguator.
  build_synthetic_persona_callback.py
                         D4 / D4v2 synthetic corpus generator.
results/                 eval JSONs + paper drafts:
  eval/                  bootstrap CIs, routing traces, etc.
  eval_v14v15/           v14-v15 callback-aware eval JSONs.
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
README.md                this file.
requirements.txt
.gitignore

output -> Runs           backwards-compat symlink.
logs/                    training & sync logs (gitignored).
paper_artifacts/         physical home of pre-tokenised corpora and
  chains/                LoCoMo / MSC test sets. Reach via `data/`.
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
   one of those nine bullets, STOP and read the citation before
   proceeding.
2. [`memory_residuals.pdf`](memory_residuals.pdf) — the position
   paper / architectural spec (Eqs. 1–10, two-stage QKV competition,
   off-sequence depth-wise injection).
3. [`atn_residuals.pdf`](atn_residuals.pdf) — the Block Attention
   Residuals reference. Read when the routing trace looks weird or
   you're tuning router init biases.
4. [`results/exp2_chain_recipe/runs.md`](results/exp2_chain_recipe/runs.md)
   — the active run ledger (currently: v15 + the open audit).
5. [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md) — the
   long-form historical ledger (Part I-II background, Part III
   Stage-1 calendar, Part IV v2-v6, Part V v7-v10, **Part VI
   v11-v14**, Part VII the runs.md folding convention).
