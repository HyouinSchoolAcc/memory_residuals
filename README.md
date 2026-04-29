# Memory Residuals

## Introduction

Hi! Welcome to the memory residuals repo.

We want to create the **first usable backpropagating memory system for
conversational AI**. The core idea is a fixed-size recurrent memory matrix
$M_c \in \mathbb{R}^{K \times d}$ that gets updated end-to-end through the
language-modelling loss — no retrieval index, no separate memory controller,
no hand-engineered gating heuristic. Just a learned compression of past
sessions that reads into the depth-wise residual stream of a pretrained
transformer.

The architecture lives in [`memory_residuals.pdf`](memory_residuals.pdf)
(position paper) and builds on
[`atn_residuals.pdf`](atn_residuals.pdf) (the Block Attention Residuals
routing primitive we modify).

## Preliminary work

We have:

- **Architecture.** Three routing modes (`simple_gate`, `attention_base`,
  `attention_parity`) on top of a Qwen3 backbone, all in
  `modeling_memres.py`. The default `attention_parity` mode produces
  *bit-identical* logits to bare Qwen3 at step 0 (verified in
  `paper_artifacts/eval/init_parity_test.json`), so every "memory" gain
  is additive on top of the pretrained behaviour.
- **Trainer.** Recurrent chain TBPTT trainer (`train_chain.py`) that
  unrolls $M_c$ across windows of $k$ consecutive sessions, plus an
  optional negative-chain contrastive loss for chain-specificity.
- **Eval pipeline.** `paper_tools/eval_chain.py` scores `mem` /
  `nomem` / `shuffled-history` / `oracle-concatenation` / RAG variants
  on the same checkpoint. Companion probes in `paper_tools/`:
  `callback_probe.py` (NLL on named-entity callbacks),
  `horizon_analysis.py` (Δ bucketed by chain length),
  `init_parity_test.py`, RAG baselines (BM25, dense, fine-tuned dense).
- **PG-19 ablation matrix (v2/v3).** Four 0.6B chain-trainer runs at
  matched compute on PG-19 + TV chains:

  | run | mode | steps reached | $\Delta_{nm-m}$ | $\Delta_{sh-m}$ | status |
  |---|---|---:|---:|---:|---|
  | `chain_v2_abl_residual_mode` | `simple_gate` | 5275 / 6000 | +0.0090 | +0.0249 | terminal plateau |
  | `chain_v2_phaseA_softparity_b4` | `attention_parity` (soft) | 2125 / 6000 | +0.0107 | +0.0272 | killed mid-climb |
  | `chain_v3_softparity_full` (live) | `attention_parity` (soft) | in flight, ~step 2000 | +0.0078 | +0.0177 | extending the soft-parity curve to step 6000 |
  | `chain_v3_attentionbase_full` (live) | `attention_base` | in flight, ~step 2000 | +0.0033 | +0.0080 | the missing third row of the ablation |

  Headline so far: at matched compute, soft `attention_parity` reaches
  `simple_gate`'s terminal plateau (+0.0249) in ~2/5 the steps and then
  *exceeds* it (+0.0272 at step 2000, still climbing). `attention_base`
  is currently tracking soft-parity's curve at a ~1000-step lag, which
  isolates the parity-init contribution from the routing-pool
  contribution.

- **Init-parity diagnostic** (`paper_artifacts/eval/init_parity_test.json`,
  Qwen3-0.6B, bf16):

  | mode | max\|Δ_logit\| vs bare Qwen3 |
  |---|---:|
  | `simple_gate` (gate $g_\ell{=}0$), no memory and with memory | **0.000** |
  | `attention_base` (uniform softmax over delta sources) | 34.5 |
  | `attention_parity` (cumulative pool, `mem_bias=-32`, `recent_bias=+32`) | **0.000** |

For the full historical record (every prior run, every failure mode,
every architectural decision and ADR) see
[`COMPREHENSIVE.md`](COMPREHENSIVE.md).

## Papers we plan to extract

1. **"Residual memory wins in roleplay recall vs scalar-gate memory."**
   Headline architectural claim. Soft-parity Block AttnRes routing pool
   beats scalar-gate residual memory injection on multi-session
   dialogue benchmarks (LoCoMo, MSC, NIAH-conversational, RULER).
   **This is the paper we're shipping in 6 days — see "Paper 1 plan"
   below.**

2. **"We eliminated the need to query at all — Memory Residuals beat
   RAG without retrieval."** End-to-end differentiable memory replaces
   top-k retrieval entirely. Same architecture as paper 1, different
   framing: vs BM25, vs Contriever, vs fine-tuned dense retrieval, all
   at matched inference compute, with latency / cost tables. The
   systems-and-comparison paper that follows paper 1.

3. **"Create-recall: humanoid memory — rationalities, emotional shifts,
   intimacy deltas."** Probes whether the learned memory tracks
   character-level state (preferences, beliefs, emotional valence,
   relationship distance) across sessions, not just topical or factual
   recall. The longer-term, more speculative paper.

## Paper 1 plan (6-day sprint)

### Reframed claim

> *On multi-session dialogue benchmarks (LoCoMo, MSC test, NIAH,
> RULER), at matched compute, Memory Residuals with the soft-parity
> Block AttnRes routing pool reach lower NLL and higher callback-EM
> than (a) the no-memory baseline, (b) BM25 retrieval, (c) dense
> MiniLM retrieval, and (d) a fine-tuned dense retriever.*

### Experiments

| # | experiment | dataset | priority |
|---:|---|---|---|
| E1 | Mixed-corpus softparity training (PG-19 40% + TV 20% + MSC 40%) | mixed | MUST |
| E2 | Mixed-corpus simple_gate baseline (same recipe) | mixed | MUST |
| E3 | Mixed-corpus `attention_base` (no parity init) | mixed | SHOULD |
| E4 | Hidden-state extraction ablation (compress layer-14 hidden states, not raw embeddings) | mixed | SHOULD |

### Evaluation matrix

| # | metric / benchmark | what it tests | priority |
|---:|---|---|---|
| M1 | `eval_chain.py` Δ on LoCoMo + MSC + PG-19 + TV held-out | next-token NLL deltas | MUST |
| M2 | NIAH at depths {1, 5, 10, 20, 30} sessions × {start, middle, end} positions | depth-of-recall | MUST |
| M3 | Generation quality on LoCoMo callbacks (greedy + nucleus, EM / F1 / ROUGE-L) | quality, not just perplexity | MUST |
| M4 | RAG baselines: BM25, Contriever, MiniLM-finetuned, all at matched FLOPs | the "no querying" claim | MUST |
| M5 | RULER at S=4k and 8k (conversational subsets: niah-mk, niah-mv, vt, qa-1, qa-2) | standardized comparability | MUST |
| M6 | Bootstrap CI (1k resamples) on every Δ in the main table | statistical rigour | MUST |
| M7 | Counterfactual sensitivity probe (alter session-$t-k$, measure ΔNLL on $t$) | causal use of prefix | SHOULD |
| M8 | Per-depth routing trace ($\alpha_{\text{mem}}$ averaged across LoCoMo, by sublayer) | mechanistic figure | SHOULD |
| M9 | LongBench-en (NarrativeQA, MultiFieldQA, Qasper, GovReport) | published benchmarks | SHOULD |
| M10 | Callback probe + horizon analysis on every checkpoint | already implemented | MUST |

### Datasets

| corpus | role | tokens | path |
|---|---|---:|---|
| PG-19 (5000-book sample) | training (40%) | ~470 M | `paper_artifacts/chains/stage1_train_s512_full.pt` |
| TV continuity chains (30 shows) | training (20%) | ~15 M | `paper_artifacts/chains/tv_train_s512.pt` |
| **MSC train (NEW for paper 1)** | training (40%) | ~18 M | `memory_residuals_data/hf_corpora/msc/data/train-*.parquet` (chain-format pending — see `paper_tools/build_msc_chains.py`) |
| LoCoMo | eval only | ~250 k | `paper_artifacts/chains/locomo_s512.pt` |
| MSC test | eval only | ~2.7 MB | `memory_residuals_data/hf_corpora/msc/data/test-*.parquet` |
| TV held-out | eval only | small | shows outside the `CONTINUITY_SHOWS` allowlist |

The MSC training-split inclusion is the change from prior practice:
[`docs/STAGE_PLAN.md`](docs/STAGE_PLAN.md) (the original plan)
explicitly excluded MSC train as "synthetic-templatey". A reviewer
flagged that books don't structurally test cross-session conversational
recall, and the deadline made the trade-off worth it. Synthetic
benchmarks now reward synthetic training data; we're transparent about
this in §7 (Limitations) of the paper.

## Compute

Our lab has **2× H100 NVL** (96 GB HBM each) at the local box, with
**16 h/day** usable (residential power-down for ~8 h overnight). The
trainer doesn't checkpoint optimizer state, so each training run must
fit inside one ~14 h window — at current throughput (~10.8 k tok/s) a
6 000-step run completes in ~10 h on a single H100.

H100 SXM5 cloud quota was unavailable, so we rented **1× GH200 480GB**:

```bash
ssh ubuntu@192.222.50.225
```

GH200 has 96 GB HBM3 (compute-equivalent to an H100 SXM5 for our
workloads) plus 480 GB unified Grace memory — useful as a head-room
buffer if we ever spill activations during long-context eval.
Rented for ~$2.49/h × ~96 h ≈ $240, with $260 of the $500 budget held
in reserve.

The cloud GPU runs **24 h / day** and is the canonical evaluation host
plus job-queue runner — work continues there when the local box is
asleep.

### 6-day calendar

| day | local GPU 0 | local GPU 1 | cloud GH200 | human |
|---:|---|---|---|---|
| 0 (today) | `chain_v3_softparity_full` finishing | `chain_v3_attentionbase_full` finishing | env up; sync repo + ckpts; queue NIAH harness | repo cleanup, README rewrite |
| 1 | mixed-corpus softparity (E1), 10 h | mixed-corpus simple_gate (E2), 10 h | NIAH on v3 ckpts + bootstrap-CI util + gen-quality on LoCoMo | start §3 (data) |
| 2 | eval E1 ckpt | eval E2 ckpt + NIAH | RAG baselines (BM25 + Contriever + MiniLM-FT) + RULER setup | start §5 (results) |
| 3 | counterfactual + routing trace | mixed-corpus `attention_base` (E3), 10 h | RULER + LongBench-en | §5 + §6 drafts |
| 4 | hidden-state extraction (E4), 10 h | spare / re-eval | callback EM eval, fill remaining cells | §1, §2, §6 |
| 5 | eval E4 ckpt; ablation table fill | spare | release cloud GPU after final eval | §7, §8, full read |
| 6 | submit | submit | (off, save remaining $260) | submit |

The full task breakdown with priorities, decision triggers, and
fall-backs lives in
[`docs/paper1_calendar.md`](docs/paper1_calendar.md).

## Watchdogs and remote-survivability

The cloud GH200 is the **canonical job runner** — anything queued there
survives the local box being powered off. The system lives at
[`paper_tools/cloud_watchdog/`](paper_tools/cloud_watchdog/):

```
paper_tools/cloud_watchdog/
├── watchdog.sh              # daemon: polls queue/, runs jobs, archives
├── notify.sh                # ntfy.sh phone-notification helper
├── enqueue.sh               # convenience wrapper: enqueue.sh <name> <cmd>
├── queue/                   # pending jobs (one JSON per job)
├── running/                 # in-flight jobs (with PID + tmux session id)
├── done/                    # finished jobs + result manifests
├── failed/                  # crashed jobs + stderr captures
├── logs/                    # per-job stdout/stderr
└── README.md                # operator manual
```

To queue a job from anywhere with SSH access:

```bash
ssh ubuntu@192.222.50.225 \
  '~/memory_residuals/paper_tools/cloud_watchdog/enqueue.sh niah_v3 \
     "python paper_tools/niah_eval.py --model output/chain_v3_softparity_full/best ..."'
```

The daemon picks the job up within 30 s, runs it inside a `tmux`
session named `cwd-<job_name>` (so SSH drops never kill it), and pushes
a phone notification when it completes or fails. See
[`paper_tools/cloud_watchdog/README.md`](paper_tools/cloud_watchdog/README.md)
for the operator manual.

## How to train (chain TBPTT, single GPU)

```bash
python -u train_chain.py \
  --preset qwen3-0.6b-large --memres_mode attention_parity \
  --router_mem_bias_init -4.0 --router_recent_bias_init 4.0 \
  --window_k 8 --session_len 512 \
  --steps 6000 --batch_size 4 --grad_accum 4 \
  --warmup 200 --lr 3e-4 --lr_backbone 3e-5 \
  --train_chains paper_artifacts/chains/stage1_train_s512_full.pt \
  --eval_chains paper_artifacts/chains/stage1_validation_s512.pt \
  --gradient_checkpointing --eval_every 200 --save_every 1000 --log_every 25 \
  --eval_n_chains 24 --eval_window 4 --save_best_metric composite \
  --out_dir output/<run_name> --seed 42
```

## How to evaluate

```bash
python paper_tools/eval_chain.py \
  --model_path output/<run_name>/best \
  --corpora paper_artifacts/chains/stage1_validation_s512.pt \
            paper_artifacts/chains/stage1_test_s512.pt \
            paper_artifacts/chains/locomo_s512.pt \
  --names pg19_val pg19_test locomo \
  --score_window 4 --oracle_window 4 \
  --output paper_artifacts/eval/<run_name>_eval.json
```

## Repository layout

```
modeling_memres.py              model + Block AttnRes routing
presets.py                      named (backbone, K, L_E, N) tuples
train_chain.py                  recurrent chain TBPTT trainer
train_phase1.py                 pair-based warm-up trainer (Stage 0)

paper_tools/                    eval, probes, RAG baselines, parity test
paper_tools/cloud_watchdog/     remote-survivable job queue + ntfy daemon

paper_artifacts/eval/           eval JSONs, plots
paper_artifacts/chains/         pre-tokenized chain corpora
paper/                          NeurIPS-style draft (.tex / .pdf)
docs/                           ADRs and per-paper decision logs

output/                         training checkpoints (gitignored, ~30 GB)
logs/                           training logs (gitignored)

COMPREHENSIVE.md                full historical record + design rationale
memory_residuals.{pdf,txt}      position paper
atn_residuals.pdf               Block Attention Residuals reference paper
```

## Stop everything

```bash
# local
pkill -f 'train_chain.py'
pkill -f 'paper_tools/cloud_watchdog'

# cloud
ssh ubuntu@192.222.50.225 \
  'pkill -f train_chain.py; pkill -f cloud_watchdog/watchdog.sh; tmux kill-server'
```
