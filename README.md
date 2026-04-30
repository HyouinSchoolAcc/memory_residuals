# memory_residuals

Hi, welcome to the **mem_residuals** repo.

We're trying to realize a constant-sized memory matrix `M_c` that gets
compressed, trained, and queried by an LLM **natively** — no retrieval
index, no separate memory controller, no hand-engineered gating
heuristic. Architectural spec is in
[`memory_residuals.pdf`](memory_residuals.pdf); the block-attention
residuals reference is in [`atn_residuals.pdf`](atn_residuals.pdf).

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

v11 fixes P0/P1/P2 simultaneously and runs as a 7-cell ablation
matrix. The decision triggers (step-200 / step-500 / step-1000 /
KILL) for every cell are inlined in the launcher comments under
`Scripts/train_v11*.sh`.

| cell | host | routing | knob change vs v9c | what it tests |
|---|---|---|---|---|
| **`chain_v11_evidence_aware_local`** | local H100 (active) | `simple_gate` | gate_init = 0.005, readout_norm_init = 0.05, evidence-aware curriculum, `--burn_in_max 0 --window_k 3` | does the P0+P1+P2 stack give PA CB Δ_nm-m > +0.02 by step 500 on the cheap proxy? |
| `train_v11g_ap_baseline_gh200` | GH200 | `attention_parity` +4 / 0 | softer mem-bias (0 vs −4) at init; same P0+P2 fixes | does softer bias open α_mem ~50× faster than legacy v3 init? |
| `train_v11h_ap_norm1_gh200` | GH200 | `attention_parity` +4 / 0 | drops P2 (`readout_norm_init = 1.0`) | does the depth softmax self-regulate magnitude on its own, or is P2 mandatory for AP too? |
| `train_v11i_ap_pm4_gh200` | GH200 | `attention_parity` +4 / **−4** | reverts to legacy v3/v10b bias | with P0 + P2 in place, can the v3 default still recover? |
| `train_v11j_ap_carry_depth_gh200` | GH200 | `attention_parity` +4 / 0 | `window_k = 4`, `--carry_state`, `--burn_in_max 12 --burn_in_resample` | closes the train/eval depth gap (P5) — primary lever against the standard Δ_sh-m ≈ 0 problem |
| `train_v11k_ap_no_evidence_gh200` | GH200 | `attention_parity` +4 / 0 | reverts P0 only (legacy `v6_lme_msc` corpus, uniform "evidence") | clean A/B isolating P0's contribution |
| `train_v11_4b_mega_gh200` | GH200 | `attention_parity` +4 / 0 | 4 B backbone, `hidden_18`, L_E = 10, ~365 k-session mega-corpus | v11 headline run; queued *after* v11{g,h,i,j,k} validate the recipe |

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

modeling_memres.py            architecture (config, model, init).
train_chain.py                recurrent chain TBPTT trainer (active).
train_phase1.py               pair-based warm-up trainer (Paper 1 recipe).
presets.py                    named (backbone, K, L_E, N) tuples.
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

1. **This README.**
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
