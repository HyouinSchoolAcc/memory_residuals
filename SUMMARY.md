# Memory Residuals — Project Summary

A faithful implementation, training pipeline, and empirical study of the
*Memory Residuals* architecture from `memory_residuals.pdf`: a Transformer
modification that maintains a fixed-size recurrent memory matrix
$M_c \in \mathbb{R}^{K \times d}$ and injects a per-position memory readout
$m^t$ into every sublayer through a learned ReZero-style gate, without
breaking the pretrained backbone's residual stream.

This is the working repository.  See `paper/memory_residuals_empirical.tex`
for the in-progress write-up.

## File map (post-cleanup)

```
memory_residuals/
├── SUMMARY.md                       <- this file
├── README.md                        <- original architecture-only README
├── memory_residuals.pdf / .txt      <- the source paper
├── modeling_memres.py               <- model: MemoryBlock, MemoryReadout,
│                                       MemoryGate, Qwen3MemRes{Model,ForCausalLM}
├── presets.py                       <- {qwen3-0.6b-small, qwen3-0.6b-large,
│                                       qwen3-8b-small, qwen3-8b-large}
├── train_phase1.py                  <- pair-based "warm-up" trainer
│                                       (single-step compress + score)
├── train_chain.py                   <- recurrent chain TBPTT trainer
│                                       (the real one — exercises M_c
│                                       across consecutive sessions)
├── paper_tools/
│   ├── pretokenize_chains.py        <- chain JSONL -> packed (N, S) tensor
│   ├── pretokenize_pairs.py         <- pair JSONL -> packed history/current
│   ├── locomo_to_chains.py          <- LoCoMo JSON -> chain JSONL
│   ├── eval_chain.py                <- chain-aware eval: mem/nomem/shuffle/oracle
│   ├── eval_suite.py                <- pair eval (history/current pairs)
│   ├── rag_baseline.py              <- bare-backbone + dense retrieval
│   ├── rag_baseline_finetuned.py    <- finetuned-backbone + dense retrieval
│   ├── callback_probe.py            <- per-callback NLL probe (PITFALLS §3)
│   ├── audit_data.py                <- corpus token audit
│   ├── prepare_pairs.py             <- chain -> pair window converter
│   └── aggregate_results.py         <- collect eval JSONs into a CSV/MD table
├── paper/
│   └── memory_residuals_empirical.{tex,pdf}
├── paper_artifacts/
│   ├── chains/                      <- pre-tokenised chain corpora
│   │   ├── stage1_train_s512.pt     <- PG-19 + 30 TV shows, ~89K sessions
│   │   ├── stage1_validation_s512.pt
│   │   ├── stage1_test_s512.pt
│   │   ├── tv_train_s512.pt         <- TV-only (30 chains)
│   │   └── locomo_s512.pt           <- LoCoMo (10 chains, eval-only)
│   ├── eval/                        <- per-checkpoint eval JSONs
│   ├── locomo_chains/               <- LoCoMo session-format JSONLs
│   └── *.png                        <- gate-profile and probe plots
├── output/
│   ├── run3_qwen3-0.6b-large/       <- pair "warm-up" baseline (8000 steps)
│   ├── chain_fresh1/                <- fresh chain TBPTT (5000 steps,
│   │                                   no contrastive)
│   ├── chain_tv1/, chain_tv2/       <- TV-only chain training (overfit
│   │                                   then plateau, kept for ablation)
│   └── chain_neg1/                  <- chain TBPTT + negative-chain
│                                       contrastive loss (currently the
│                                       most promising recipe)
└── legacy/                          <- earlier reference implementations,
                                       not used by the current pipeline
```

## What's been tried, in order

| Run | Trainer | Data | Steps | Memory eval (rigorous) | Notes |
|---|---|---|---:|---|---|
| `run3_qwen3-0.6b-large` | `train_phase1.py` (pair) | PG-19+TV pairs | 8000 | $\Delta_{nm-m}=+0.026$, $\Delta_{sh-m}=+0.029$ on **pair** eval (n=256); explodes to mem CE=8.7 on long-horizon **chain** eval | Pair-trained warm-up.  Two architectural fixes landed here: zero-init $g_\ell$ (gate) and `memres_mode="residual"` to keep the backbone residual stream intact |
| `chain2_qwen3-0.6b-large` | `train_chain.py` warm-started from `run3` | PG-19+TV chains, k=4 | 3000 | $\Delta_{sh-m}=-0.014$ PG-19, $-0.012$ LoCoMo | First chain run with `judge_norm` RMSNorm.  Stable but PITFALLS §3 shortcut-learning failure — memory had become style-only |
| `chain_fresh1` | fresh init, no warm-start | PG-19+TV chains, k=8 | 5000 | $\Delta_{sh-m}=-0.036$ PG-19, $-0.016$ LoCoMo (standalone); $+0.020$ PG-19 in-trainer eval | Stable at 30+ session unrolls.  In-trainer Δ_sh-m drifted positive but the rigorous standalone eval revealed the in-trainer protocol was overstating specificity |
| `chain_tv1`, `chain_tv2` | TV-only training | TV chains, k=8 | 6000 | LoCoMo eval positive at step 500 ($\Delta_{sh-m}=+0.011$, $+0.002$), then overfit | Useful as a domain-shift ablation; not a final result |
| `chain_neg1` (currently training) | chain TBPTT + **negative-chain contrastive** loss ($\lambda=0.5$, margin 0.05) | PG-19+TV chains, k=8 | 5000 (in progress) | step 1000 in-trainer $\Delta_{sh-m}=+0.0114$ | Implements the PITFALLS §3 prescription: explicit gradient against shortcut learning |

## Architecture invariants (verified in code)

1. **Init = bare backbone.**  All `MemoryGate.gate[i]=0` at construction
   and after `_init_weights`, so $h^{\mathrm{pre}}_\ell = h^{\mathrm{post}}_{\ell-1}$
   and the augmented model produces bit-identical logits to a vanilla
   Qwen3-0.6B forward pass.  Verified: `paper_artifacts/eval/init_parity_test.json`.
2. **Two-stage write.**  `MemoryBlock.extract` (Eq. 1, 3-4) and
   `MemoryBlock.judge` (Eq. 2) have separate parameters, separate
   queries ($M_{\mathrm{in}}$, $M_{\mathrm{judge}}$), and separate
   cross-attention heads.
3. **Zero-sum forgetting.**  The judge softmax is across the $2K$-row
   pool $[M_c^{t-1}; M_{\mathrm{new}}]$, not within slot.
4. **Stable recurrence at long horizons.**  `MemoryBlock.judge_norm` is
   an RMSNorm applied after the judge cross-attention; without it
   $\|M_c\|_F$ drifts after ~10 unrolls and eval CE explodes.
5. **Off-sequence routing.**  $M_c$ never enters the token sequence.
   Cost is $O(SK)$ for the readout + $O(L)$ for the depth-wise gates per
   token, vs $O((S+K)^2)$ for a sequence-prepended baseline.

## Reproducing the headline experiment

On a 2× H100 node with the existing data:

```bash
# 1. Pre-tokenise (already done; outputs land in paper_artifacts/chains/)
python paper_tools/pretokenize_chains.py \
  --in_dir ../memory_residuals_data/stage1/pg19/train \
  --in_dir ../memory_residuals_data/stage1/tv \
  --out_path paper_artifacts/chains/stage1_train_s512.pt \
  --session_len 512 --min_tokens 64 --min_sessions_per_chain 4 \
  --max_chains 2000 --workers 32

python paper_tools/pretokenize_chains.py \
  --in_dir ../memory_residuals_data/stage1/pg19/validation \
  --out_path paper_artifacts/chains/stage1_validation_s512.pt

python paper_tools/locomo_to_chains.py
python paper_tools/pretokenize_chains.py \
  --in_dir paper_artifacts/locomo_chains \
  --out_path paper_artifacts/chains/locomo_s512.pt

# 2. Train (single H100 is enough for Qwen3-0.6B-large)
NCCL_SOCKET_IFNAME=lo CUDA_VISIBLE_DEVICES=0 \
python -u train_chain.py \
  --preset qwen3-0.6b-large \
  --window_k 8 --session_len 512 \
  --steps 5000 --batch_size 2 --grad_accum 2 \
  --warmup 200 --lr 5e-4 --lr_backbone 1e-5 \
  --memory_dropout 0.10 --context_dropout 0.30 \
  --neg_chain_weight 0.5 --neg_chain_margin 0.05 \
  --gradient_checkpointing \
  --out_dir output/chain_neg_repro

# 3. Rigorous eval on PG-19 validation + LoCoMo
python paper_tools/eval_chain.py \
  --model_path output/chain_neg_repro/best \
  --corpora paper_artifacts/chains/stage1_validation_s512.pt \
            paper_artifacts/chains/locomo_s512.pt \
  --names pg19_validation locomo \
  --score_window 4 --oracle_window 4 \
  --output paper_artifacts/eval/chain_neg_repro_eval.json
```

A 5000-step single-GPU run takes ~2 hours, eval takes ~1 minute.

## Open questions / future steps

In rough priority order for the paper:

1. **Land positive $\Delta_{sh-m}$ on rigorous standalone eval** for
   PG-19 *and* LoCoMo.  `chain_neg1` is the active attempt; in-trainer
   numbers are positive but standalone numbers are pending.

2. **Run the callback probe on the chain-trained checkpoints.**  We
   have probe results for `run3` (pair-trained) only; need them on
   `chain_neg1/best` once it converges.  Target: callback help ratio
   $> 1.5\times$ on PG-19 *and* on LoCoMo.

3. **8B-class run.**  `train_chain.py` accepts `--shard_strategy
   fsdp_full --gradient_checkpointing` and `--preset qwen3-8b-large`.
   Untried; expect ~1 GPU-day on 2× H100 with FSDP.

4. **Burn-in stability fix.**  The current `--burn_in_max=0`
   default skips burn-in because the gradient-tracked $k=8$ window
   already hits memory budgets in bf16 + gradient checkpointing.  The
   no-grad burn-in path is implemented but it produces $M_c$ states
   that are out-of-distribution for downstream readout (causes loss
   spikes at step 5).  Either keep burn-in gradient-tracked (i.e.
   bigger $k$) or warm-start from a chain-trained checkpoint.

5. **Long-horizon test.**  Eval at chain length $\geq 30$ sessions (LoCoMo
   conv-41 has 32) and report whether memory benefit decays vs
   degrades.  This is the LRMT failure mode — flagged in PITFALLS §5.

6. **Mix in MSC v2 / Persona-Chat** for training only if user re-permits
   crowd-sourced human-written multi-session chat.  Currently training
   uses **only** PG-19 (human-written prose) + 30 TV-show transcripts
   (human-written dialogue); LoCoMo and pair-MSC are eval-only.

## Things I'd ask reviewers about before submission

- The in-trainer-vs-standalone $\Delta_{sh-m}$ discrepancy (16-chain
  subset with possibly-shorter shuffle prefix vs 47-chain rigorous
  full-length) is a methodology subtlety that needs to be transparent
  in the paper.  We standardise on the standalone protocol everywhere
  in published numbers.
- The fundamental ceiling: oracle is only ~0.20 nats below no-memory on
  PG-19, so any compressed-memory scheme has at most that much to win.
  The interesting metric is the *capture ratio* (memory $\Delta$ as a
  fraction of oracle $\Delta$), not the raw $\Delta$.

## Removed from the project

To keep the working tree small, the following have been deleted (the
information they captured is preserved in eval JSONs / paper tables):

- `output/{_smoke*,_pilot*,smoke_*,memres-d512-*,run1*,run2*,chain1*,
  chain2*,chain_fresh_small}` — superseded scratch runs.
- `paper_artifacts/runs/qwen3-{0.6b,8b}-{small,large}` — old pair-trained
  pilot checkpoints; numbers retained in `paper_artifacts/eval/`.
- `paper_artifacts/pairs/` — old pair JSONLs and pre-tokenised tensors,
  superseded by chain corpora.
- `wandb/`, `__pycache__/`, `data/friends_scripts.jsonl` — debug
  artifacts and a legacy single-show JSONL.
- Legacy training/eval scripts moved to `legacy/`:
  `eval_base.py`, `eval_memres.py`, `train_memres.py`,
  `train_memres_chain.py`, `modeling_memory_residuals.py`,
  `probe_memres.py`, `visualize_memres.py`, `atn_.pdf`.
