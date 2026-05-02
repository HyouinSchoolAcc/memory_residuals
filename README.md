# memory_residuals

Hi, welcome to the **mem_residuals** repo.

We're trying to realize a constant-sized memory matrix `M_c` that gets
compressed, trained, and queried by an LLM **natively** — no retrieval
index, no separate memory controller, no hand-engineered gating
heuristic. Architectural spec is in
[`memory_residuals.pdf`](memory_residuals.pdf); the block-attention
residuals reference is in [`atn_residuals.pdf`](atn_residuals.pdf).

## Architectural priors (READ FIRST — these are settled)

Future agents: these are baked-in findings from the paper line of
evidence. Do NOT re-derive them from scratch and do NOT silently swap
them out of a run without reading the citation.

- **AP (`attention_parity`) > SG (`simple_gate`) on the routing side.**
  Per `results/exp1_pair_recipe/manuscript.tex` Table 2 (v3 pair-recipe
  headline, matched seed, matched compute, PG-19 pairs): AP (soft ±4
  bias init) beats SG on Δ_sh-m by **1.6× to 3.8× at every step** of
  the head-to-head trajectory, with AP @ step 2000 (+0.0272) already
  **surpassing SG's full-budget asymptote at step 5200 (+0.0249)**.
  The manuscript's exact phrasing: *"the routing-pool variant achieves
  the scalar-gate's asymptote in 2/5 the steps and is still improving
  when stopped."* Standalone PG-19 val agrees: AP Δ_nm-m = +0.0113
  vs SG = +0.0098; capture ratio 0.15 vs 0.13. This is the strongest
  routing-side prior we have. The headline paper run (v13p and
  successors) should default to `--memres_mode attention_parity` unless
  a chain-trainer cell explicitly demonstrates AP collapse the v13+
  interventions can't break.

- **Caveat — AP has historically collapsed on the CHAIN trainer.**
  v5 softparity, v7 softerbias + v3bias, and every v8/9/11/v12a AP
  cell sat at `alpha_mem ~ 4e-4` indefinitely because the router
  closes early → writer gets attenuated gradient → writer stays
  random → router sees noisy m^t → router closes harder. This is the
  "collapse cycle" COMPREHENSIVE.md §v7 diagnosed as *"causally
  equivalent to no memory regardless of M_c."* The v13 stack
  (`writer_warmup`, `memres_queries_init=orthogonal`,
  `memres_slot_positional`, `memres_writer_kind=slot_attention`) is
  designed specifically to break that cycle so AP's pair-trainer
  advantage can transfer. If you're staring at an AP cell that
  collapsed at step ~500, check whether the full v13 stack was
  actually active (see next point).

- **Config-merge bug (fixed 2026-05-01).** `Qwen3MemResConfig` is a
  subclass of `Qwen3Config`, so `from_pretrained("Qwen/Qwen3-0.6B")`
  succeeds and the old `from_memres_ckpt = True` detector in
  `train_chain.py::_build_model` produced a false positive on every
  Qwen3 base backbone. The subsequent `overridable`-subset merge
  silently DROPPED CLI overrides for `--memres_mode`,
  `--memres_writer_kind`, `--memres_slot_positional`,
  `--memres_extraction_depth`, `--memres_update_mode`,
  `--memres_num_vectors`, etc. Fixed by detecting memres checkpoints
  via `base_cfg.model_type == "qwen3_memres"` (the raw JSON field).
  **If you see a v11 / v12 / v13 run whose `ROUTE @ step` diagnostic
  reports a mode different from what the launch script requested, or
  a load report whose MISSING-list doesn't include `M_in_pos` /
  `write_gate` / `extraction_layers.{0..4}` despite the launch flags,
  the bug is back — bisect against the `BUGFIX 2026-05-01` comment.**

- **`simple_gate` needs memory_gate force-open during writer_warmup,
  not router mem_bias.** In `simple_gate` the depth router is not on
  the forward path; `memory_gate.gate` is. `_set_mem_bias` therefore
  forces `gate = 0.5 * tanh(bias/2) ≈ 0.48` when mode is simple_gate,
  in addition to setting `depth_router.mem_bias`. Without this, SG
  writer_warmup trains the writer with zero gradient through the
  forward path because `h + 0 * m^t = h` regardless of router bias.
  See `_set_mem_bias` in `src/train_chain.py`.

- **The uniform-softmax fixed point is structural, not data-starved.**
  More data alone does not break it (v11p, v11m_chinchilla, v12c all
  collapsed on larger corpora). It is the permutation-invariant fixed
  point of a symmetric softmax with i.i.d.-initialised slot queries.
  The v13 `memres_queries_init=orthogonal` + `memres_slot_positional`
  levers are the structural fix; `writer_warmup` + `slot_attention`
  are the objective/writer-side accelerants that keep the system from
  re-collapsing during joint training.

## Progress & lessons

- **#v3b** — We can drop MemRes onto any backbone with the augmented
  model **bit-exactly** equal to the bare backbone at init *and* still
  receive gradients on every memory-channel parameter. This is the
  load-bearing primitive for everything below.
- **#v3** — We tried three ways to inject memory into the backbone:
  1. a scalar gate per layer (`simple_gate`),
  2. heavily-biased attention at every layer that *does* preserve
     bit-wise parity (`attention_parity`, hard ±32),
  3. lightly-biased attention at every layer (soft ±4 — does **not**
     preserve bit-wise parity at init).

  On the task *"compress previous book chapters → help generate the
  next chapter"*, the **light-bias** variant crushed the others
  (`chain_v2_phaseA_softparity_b4`, Δ_sh-m = +0.0529 [+0.025, +0.092],
  bootstrap-CI excludes zero on PG-19 val).

- **#v9c** — Books produce meaningful compression gradients easily and
  are a **much easier** task than dialogue compression: books have
  temporal dependency, continuity, and many more tokens per chain.
  Phase-aligned CB Δ_nm-m grew monotonically from −0.03 → +0.16 nats
  across 4 000 steps on the diverse PG-19 + TV + LME + MSC corpus.

- **#v3 → v10** — Dialogue datasets (LongMemEval, MSC, RealTalk) need a
  **custom memory compression learning stage** or the backbone keeps
  throwing away every memory state we hand it. Six straight LME-only
  campaigns collapsed the same way (gate_max ≡ 0, α_mem ≡ 0). We
  initially blamed the architecture; the post-v10 audit found three
  *causally independent* failures, each sufficient on its own:
  - **P0 (data, ~100× leverage)** — the corpus builder threw away the
    `answer_session_ids` annotations LongMemEval-S ships with, so 96 %
    of training windows had `M_c` built from sessions that
    demonstrably did *not* contain the answer. The LM-loss-optimal
    policy under that distribution is "ignore memory".
  - **P1 (chicken-and-egg)** — gate, readout, and writer all multiply
    each other in the forward path (`h += g * m^t`). With `g = 0` and
    `W_V^read = randn(d⁻¹ᐟ²)` at init, no parameter sees gradient at
    step 0 and they all stay at zero forever.
  - **P2 (magnitude)** — the readout RMSNorm pinned
    `‖m^t‖/‖embed‖ ≈ 73`, so the useful gate range was `[0, ~0.014]`
    — too narrow for AdamW's natural step size to land in stably.

## Currently running: v11

v11 fixes P0/P1/P2 simultaneously and runs as an 11-cell ablation
matrix (1 local + 10 GH200). The decision triggers (step-200 /
step-500 / step-1000 / KILL) for every cell are inlined in the
launcher comments under `Scripts/train_v11*.sh`.

| cell | host | routing | knob change vs v9c | what it tests |
|---|---|---|---|---|
| **`chain_v11_evidence_aware_local`** | local H100 (active) | `simple_gate` | gate_init = 0.005, readout_norm_init = 0.05, evidence-aware curriculum, `--burn_in_max 0 --window_k 3` | does the P0+P1+P2 stack give PA CB Δ_nm-m > +0.02 by step 500 on the cheap proxy? |
| `train_v11g_ap_baseline_gh200` | GH200 | `attention_parity` +4 / 0 | softer mem-bias (0 vs −4) at init; same P0+P2 fixes | does softer bias open α_mem ~50× faster than legacy v3 init? |
| `train_v11h_ap_norm1_gh200` | GH200 | `attention_parity` +4 / 0 | drops P2 (`readout_norm_init = 1.0`) | does the depth softmax self-regulate magnitude on its own, or is P2 mandatory for AP too? |
| `train_v11i_ap_pm4_gh200` | GH200 | `attention_parity` +4 / **−4** | reverts to legacy v3/v10b bias | with P0 + P2 in place, can the v3 default still recover? |
| `train_v11j_ap_carry_depth_gh200` | GH200 | `attention_parity` +4 / 0 | `window_k = 4`, `--carry_state`, `--burn_in_max 12 --burn_in_resample` | closes the train/eval depth gap (P5) — primary lever against the standard Δ_sh-m ≈ 0 problem |
| `train_v11k_ap_no_evidence_gh200` | GH200 | `attention_parity` +4 / 0 | reverts P0 only (legacy `v6_lme_msc` corpus, uniform "evidence") | clean A/B isolating P0's contribution |
| **`train_v11r_ap_readout_warmup_gh200`** | GH200 | `attention_parity` +4 / 0 | **`--readout_warmup_steps 500 --readout_warmup_router_bias 4.0` + `--contrastive_infonce_weight 0.5`** | architectural fix A: targets the readout-router lock-in surfaced by D5 audit on v11g/best (writer encodes content; readout never converges because router closes early). Phase 1 trains readout in isolation under forced-open routing; phase 2 unfreezes and anneals. **Highest-leverage v11 cell — slot 3 in the queue.** |
| `train_v11l_ap_frozen_backbone_gh200` | GH200 | `attention_parity` +4 / 0 | `--freeze_backbone`; otherwise IDENTICAL to v11g | single-knob ablation: is v11g's grow-then-decay caused by backbone co-evolution (block summaries crowding `m^t` out of the AP softmax)? |
| `train_v11m_ap_chinchilla_gh200` | GH200 | `attention_parity` +4 / 0 | 16 000 steps, `window_k=4`, `--carry_state`, `--burn_in_max 12 --burn_in_resample`; ~262 M tokens | applies the Chinchilla budget to the from-scratch ~9.7 M-param memres subsystem (was at ~25 % of Chinchilla in v11g) |
| `train_v11p_ap_frozen_chinchilla_mega_gh200` | GH200 | `attention_parity` +4 / 0 | `--freeze_backbone` + `v11_mega` corpus (67 745 chains) + 25 000 steps + `--lr 1e-4` + `window_k=4 --carry_state` | v11 HEADLINE replacement for the killed 4B mega: cleanest from-scratch memres training, stable backbone target, 2.1× Chinchilla token budget, 10× larger corpus |
| `train_v11q_ap_contrastive_gh200` | GH200 | `attention_parity` +4 / 0 | `--contrastive_infonce_weight 0.5` (warmup 0.05 → 0.5 over 500 steps, callback-only scoring); otherwise IDENTICAL to v11g | dense supervision on Δ_sh-m: B-way InfoNCE over in-batch negatives directly pressures M_c to be chain-specific. Now also serves as the **InfoNCE-without-warmup ablation for v11r** (v11r combines warmup + InfoNCE; q isolates the supervision contribution). |

### Diagnostic toolkit (D1-D5; 2026-05-01)

Five mechanism-level audits that ship with the trainer (`--diagnose_grad_groups`,
`--diagnose_memory_dynamics`) plus standalone scripts:

- **D1** per-module gradient L2 norms (ratios to backbone). Surfaces gradient
  starvation in the writer subsystem.
- **D2** judge attention decisiveness (row-entropy, keep/write mass split,
  effective rank). Detects decision-less judge softmax.
- **D3** M_c stability + chain-distinguishability (per-step Frobenius drift,
  pairwise distance between distinct chains). Detects content-blind writer.
- **D4** synthetic gold-standard persona-callback corpus
  (`tools/build_synthetic_persona_callback.py`; 5000 chains × 9 sessions,
  256-item closed set; ground-truth callable CE → 0 only if memory works).
- **D5** TTT-on-readout disambiguator
  (`tools/d5_ttt_readout.py`; freeze writer + router + LM head, train ONLY
  the readout for 300 steps). Distinguishes "writer broken" vs "readout
  broken" vs "router locked".

Audit on `v11g/best` (results in `v11g_diag_synth.json`, `v11g_d5_ttt.json`):
the writer encodes chain-content, the **readout is the bottleneck**, and the
router locked the memory pathway closed before the readout converged. v11r is
the architectural response.

The active run ledger (which cell is doing what right now) is in
[`results/exp2_chain_recipe/runs.md`](results/exp2_chain_recipe/runs.md).

## Resources

- **Local.** 2 × H100 NVL (94 GB) at the lab box. Currently busy with
  `chain_v11_evidence_aware_local` on GPU 0. ~16 h/day usable
  (residential power-down overnight).
- **Cloud.** 1 × NVIDIA GH200 480 GB at `192.222.50.225` (user
  `ubuntu`). v11 GH200 cells are queued through the cloud watchdog
  (`tools/cloud_watchdog/`); jobs run inside detached `tmux` so they
  survive SSH drops + lab-box power-offs.

## Layout

```
Runs/                         training checkpoints (gitignored, ~11 GB).
                              Only runs cited by the two papers survive
                              on disk; v3-v10 were pruned 2026-04-30.

Scripts/                      one .sh per training cell — the launchers.
                              v10 launchers were pruned; superseded ones
                              live under archive/scripts/.

data/                         pre-tokenised corpora.
                              (symlink → paper_artifacts/chains/ while
                               the active v11 cell still has the old
                               literal path baked into its launch
                               command; rename to a real dir once v11
                               finishes.)

tools/                        eval / probes / corpus builders.
  cloud_watchdog/             remote-survivable job queue + ntfy daemon.

results/                      eval JSONs + paper drafts.
  eval/                       bootstrap CIs, routing traces,
                              counterfactual horizon sweeps.
  exp1_pair_recipe/           Paper 1 (drop-in primitive, run3 result)
                              manuscript + figures.
  exp2_chain_recipe/          Paper 2 (long-horizon recipe) draft +
                              the active runs.md ledger.

src/                          all the architecture + trainer code.
  modeling_memres.py          architecture (config, model, init).
  train_chain.py              recurrent chain TBPTT trainer (active).
  train_phase1.py             pair-based warm-up trainer (Paper 1 recipe).
  presets.py                  named (backbone, K, L_E, N) tuples.

requirements.txt

memory_residuals.pdf          position paper (architectural spec).
atn_residuals.pdf             Block Attention Residuals reference.
README.md                     this file.
.gitignore

archive/                      historical reference (~1 MB):
                              prior scripts, tools, eval JSONs, agent
                              session notes, and COMPREHENSIVE.md (the
                              full v1→v10 ledger every prior decision
                              is in).

output -> Runs                backwards-compat symlink (the active v11
                              cell's launch command bakes in
                              `output/...`).
logs/                         training & sync logs (gitignored).
paper_artifacts/chains/       physical home of the pre-tokenised
                              corpora; gitignored. Reach via `data/`.
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

1. **This README — and specifically the "Architectural priors" block
   at the top.** If you catch yourself proposing a change that
   contradicts one of those bullets, STOP and read the citation
   before proceeding.
2. [`memory_residuals.pdf`](memory_residuals.pdf) — the position
   paper / architectural spec (Eqs. 1–10, two-stage QKV competition,
   off-sequence depth-wise injection).
3. [`atn_residuals.pdf`](atn_residuals.pdf) — the Block Attention
   Residuals reference. Read when the routing trace looks weird or
   you're tuning router init biases.
4. [`results/exp2_chain_recipe/runs.md`](results/exp2_chain_recipe/runs.md)
   — the active run ledger.
5. [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md) — the
   long-form historical ledger (every prior run, every architectural
   decision, every failure mode in detail; reference material, not a
   reading order).
