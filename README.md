# Memory Residuals

We are training **Memory Residuals** on top of a Qwen3 backbone. The
default routing mode is **`block_attnres` with `parity_init=True`** — at
step 0 the augmented model produces bit-identical logits to bare Qwen3
(`paper_artifacts/eval/init_parity_test.json`), so all "memory" learning
is additive on top of the pretrained behaviour.

The architecture is described in [`memory_residuals.pdf`](memory_residuals.pdf)
(position paper) and [`atn_residuals.pdf`](atn_residuals.pdf) (the Block
Attention Residuals routing primitive we build on).

For everything else — full design rationale, every prior run and how it
failed, compute-targeted next-step plans, paper-to-code map, eval
methodology — see [`COMPREHENSIVE.md`](COMPREHENSIVE.md).

## Idea

Maintain a fixed-size recurrent memory matrix
$M_c \in \mathbb{R}^{K \times d}$ (one per chain, not per token).
Update it once per session via two-stage QKV competition (extract a
candidate $E_t$ from the new session, then judge it against the prior
$M_c^{t-1}$ in a zero-sum softmax over a $2K$-row pool). At read
time, every position in the next session cross-attends $M_c$ once to
produce a per-position memory readout $m^t$, which is injected into
the depth-wise residual stream via the Block AttnRes router as
source $b_{-1}$. End-to-end differentiable; the persistent state
$M_c$ is shaped by the language-modelling loss alone.

## Dataset

Pre-tokenized chain corpora live in `paper_artifacts/chains/`
(produced from the sibling `memory_residuals_data/` folder; not
re-built here):

| split | file | source | sessions |
|---|---|---|---|
| train | `stage1_train_s512.pt` (gitignored, 176 MB) | PG-19 books + 30 TV transcripts | ~89K |
| val | `stage1_validation_s512.pt` | PG-19 validation | small |
| test | `stage1_test_s512.pt` | PG-19 test | small |
| TV-only ablation | `tv_train_s512.pt` | 30 TV transcripts | small |
| eval | `locomo_s512.pt` | LoCoMo (10 conversation chains) | eval-only |

All tensors are packed to `session_len=512`. Pair-format windows for
warm-up training are produced on the fly by
`paper_tools/prepare_pairs.py`.

## How to train

Single-GPU recurrent chain TBPTT (the experiment that exercises $M_c$
end-to-end across consecutive sessions):

```bash
python -u train_chain.py \
  --preset qwen3-0.6b-large \
  --window_k 8 --session_len 512 \
  --steps 5000 --batch_size 2 --grad_accum 2 \
  --warmup 200 --lr 5e-4 --lr_backbone 1e-5 \
  --memory_dropout 0.10 --context_dropout 0.30 \
  --neg_chain_weight 0.5 --neg_chain_margin 0.05 \
  --gradient_checkpointing \
  --out_dir output/chain_neg_repro
```

Eval (memory vs no-memory vs shuffled-memory vs oracle, on a held-out chain corpus):

```bash
python paper_tools/eval_chain.py \
  --model_path output/chain_neg_repro/best \
  --corpora paper_artifacts/chains/stage1_validation_s512.pt \
            paper_artifacts/chains/locomo_s512.pt \
  --names pg19_validation locomo \
  --score_window 4 --oracle_window 4 \
  --output paper_artifacts/eval/chain_neg_repro_eval.json
```

## Experiments — results so far

The headline metrics are
$\Delta_{nm-m} = \mathrm{CE}_{\text{nomem}} - \mathrm{CE}_{\text{mem}}$
(memory help) and
$\Delta_{sh-m} = \mathrm{CE}_{\text{shuffle}} - \mathrm{CE}_{\text{mem}}$
(history specificity — positive means the model uses *this* chain's
memory, not a generic style prior).

| run | trainer | data | steps | $\Delta_{nm-m}$ | $\Delta_{sh-m}$ | verdict |
|---|---|---|---:|---:|---:|---|
| `run3_qwen3-0.6b-large` | `train_phase1.py` (pair) | PG-19 + TV pairs | 8 000 | +0.026 | +0.029 | works on pair eval; explodes on chain eval (CE 8.7) |
| `chain2_qwen3-0.6b-large` | `train_chain.py`, warm-started from run3 | PG-19 + TV chains, k=4 | 3 000 | +0.063 (PG-19) | −0.014 (PG-19) | shortcut learning — memory became style-only |
| `chain_fresh1` | fresh init | PG-19 + TV chains, k=8 | 5 000 | +0.008 | −0.036 (PG-19) | stable to 30+ sessions, still shortcut-learns |
| `chain_tv1`, `chain_tv2` | TV-only | TV chains, k=8 | 6 000 | small + | +0.011, +0.002 (LoCoMo @ step 500) | brief positive then overfits |
| `chain_neg1` | chain TBPTT + negative-chain contrastive ($\lambda=0.5$, m=0.05) | PG-19 + TV chains, k=8 | 5 000 (in progress) | −0.0003 | +0.014 (in-trainer step 1000) | currently the most promising recipe |

Init-parity diagnostic (`paper_artifacts/eval/init_parity_test.json`,
Qwen3-0.6B, bf16, 30 prompt + 27 history tokens):

| mode | max\|Δ_logit\| vs bare Qwen3 |
|---|---|
| `residual` (ReZero gate $g_\ell{=}0$), no memory | **0.000** |
| `residual` (ReZero gate $g_\ell{=}0$), memory | **0.000** |
| `block_attnres` default (uniform softmax over deltas) | 34.5 |
| `block_attnres_parity_init` (cumulative pool + recent_bias = +32) | **0.000** |

## Future TODO

In rough priority order — see `COMPREHENSIVE.md` Part I §3–4 for the
full compute-targeted plans (2× H100, 24 h plan and 20× A100, 168 GPU-h plan).

- [ ] Land positive $\Delta_{sh-m}$ on the **rigorous standalone eval** for both PG-19 and LoCoMo. `chain_neg1` is the active attempt; in-trainer numbers are positive but standalone numbers are pending.
- [ ] Run `paper_tools/callback_probe.py` on `chain_neg1/best` and verify the per-callback help ratio is > 1.5× on PG-19 and LoCoMo (we have it for `run3` only).
- [ ] Long-horizon test at chain length ≥ 30 sessions (LoCoMo conv-41 has 32). Report whether memory benefit decays gracefully or collapses (the LRMT failure mode).
- [ ] 8B-class run with `--preset qwen3-8b-large --shard_strategy fsdp_full --gradient_checkpointing`. Untried; ~1 GPU-day on 2× H100 expected.
- [ ] Burn-in stability: the no-grad burn-in path produces $M_c$ states out-of-distribution for the readout (loss spikes at step 5). Either widen the gradient-tracked window or warm-start from a chain-trained checkpoint.
- [ ] Ablate the new `block_attnres_parity_init` mode against `residual` end-to-end (init-parity is matched; need to compare downstream training dynamics).
- [ ] Decide on MSC v2 / Persona-Chat inclusion for *training* (currently eval-only) once the data licence question is settled.

## Layout

```
modeling_memres.py          model + Block AttnRes routing (parity_init)
presets.py                  named (backbone, K, L_E, N) tuples
train_phase1.py             pair-based warm-up trainer
train_chain.py              recurrent chain TBPTT trainer
paper_tools/                eval, probes, RAG baselines, audit, parity test
paper_artifacts/            eval JSONs, plots, pre-tokenized chain caches
output/                     training checkpoints  (gitignored)
memory_residuals.{pdf,txt}  position paper
atn_residuals.pdf           Block Attention Residuals reference paper
COMPREHENSIVE.md            everything that used to be in BRIEFING + SUMMARY
```
