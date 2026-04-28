# Memory Residuals

We are training **Memory Residuals** on top of a Qwen3 backbone. The
default routing mode is **`attention_parity`** (the full Block AttnRes
routing pool with cumulative-state sources and a parity-preserving
init) — at step 0 the augmented model produces bit-identical logits to
bare Qwen3 (`paper_artifacts/eval/init_parity_test.json`), so all
"memory" learning is additive on top of the pretrained behaviour.

## Pick up here tomorrow (snapshot taken Apr 28, ~02:15 local)

**State on disk** — both runs were live when the server was shut down;
their best-so-far checkpoints persisted on disk. The exact step they
died at depends on when shutdown landed, but the `output/<run>/best/`
dirs always reflect the highest-composite-metric eval seen up to that
point.

| run | mode | output dir | last EVAL captured | $\Delta_{nm-m}$ | $\Delta_{sh-m}$ |
|---|---|---|---:|---:|---:|
| A — head-to-head pool | soft `attention_parity` (`mem_bias=-4`, `recent_bias=+4`) | `output/chain_v2_phaseA_softparity_b4` | step 1800 (still climbing) | **+0.0102** | **+0.0240** |
| B — head-to-head gate | `simple_gate` (legacy `--memres_mode residual`) | `output/chain_v2_abl_residual_mode` | step 4800 (plateau ~3500) | +0.0088 | +0.0243 |

The matched-step head-to-head was the point of these runs. At every
checkpoint up to where they overlap (steps 200–1800), soft
`attention_parity` recruited memory **1.6×–3.8× faster** than
`simple_gate` on identical data, lr, and seed. By step 1800,
`attention_parity` had already matched what `simple_gate` reached at
step 4800 (Δ_sh-m ≈ +0.024) — i.e. **~2.7× more sample-efficient**.
This is the routing-pool-vs-scalar-gate result the paper hinges on.

The full per-step trajectory is in `logs/chain_v2_phaseA_softparity_b4.log`
and `logs/chain_v2_abl_residual_mode.log` (logs/ is gitignored, lives
on disk only).

### First 30 min tomorrow — turn the in-trainer numbers into rigorous standalone eval

The `EVAL @ step N` lines above are computed inside the trainer with
n=92 sessions and `eval_window=4`. Re-run on the same `best/` ckpts
through `paper_tools/eval_chain.py` to get the paper-grade numbers
on PG-19 val + LoCoMo (and PG-19 test, which has been held out):

```bash
cd ~/Desktop/fine-tune/memory_residuals

for run in chain_v2_phaseA_softparity_b4 chain_v2_abl_residual_mode; do
  python paper_tools/eval_chain.py \
    --model_path output/$run/best \
    --corpora paper_artifacts/chains/stage1_validation_s512.pt \
              paper_artifacts/chains/stage1_test_s512.pt \
              paper_artifacts/chains/locomo_s512.pt \
    --names pg19_val pg19_test locomo \
    --score_window 4 --oracle_window 4 \
    --output paper_artifacts/eval/${run}_eval.json
done
```

Then `paper_tools/callback_probe.py` for the per-callback help ratio
on each (we want > 1.5× on PG-19 and LoCoMo) and
`paper_tools/horizon_analysis.py` to bucket help by chain length.

### Then choose A, B, or C for the day's compute

**A — Extend the soft-parity run to step 6000** (deterministic).
The trainer doesn't checkpoint optimizer/scheduler state, so a true
"resume" isn't possible. With `--seed 42` the trajectory is reproducible
from step 0, so re-running the exact same command will retrace
steps 1–~1800 in ~3 h and then continue past where it died. Total
~10 h on the H100.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train_chain.py \
  --preset qwen3-0.6b-large --memres_mode attention_parity \
  --router_mem_bias_init -4.0 --router_recent_bias_init 4.0 \
  --window_k 8 --session_len 512 \
  --steps 6000 --batch_size 4 --grad_accum 4 \
  --warmup 200 --lr 3e-4 --lr_backbone 3e-5 \
  --memory_dropout 0.0 --context_dropout 0.0 \
  --neg_chain_weight 0.0 --burn_in_max 0 \
  --train_chains paper_artifacts/chains/stage1_train_s512_full.pt \
  --eval_chains paper_artifacts/chains/stage1_validation_s512.pt \
  --gradient_checkpointing --eval_every 200 --save_every 1000 --log_every 25 \
  --eval_n_chains 24 --eval_window 4 --save_best_metric composite \
  --out_dir output/chain_v2_phaseA_softparity_b4 --seed 42 \
  > logs/chain_v2_phaseA_softparity_b4.log 2>&1 &
```

(The simple_gate run is essentially plateaued — extending it is low
value. If you want a clean step-6000 number for the table, run it
the same way against `output/chain_v2_abl_residual_mode` on GPU 1.)

**B — Skip extension, start Phase B (contrastive curriculum) warm
from soft-parity best.** Phase B layers `--neg_chain_weight 0.5
--neg_chain_margin 0.05 --memory_dropout 0.10 --context_dropout 0.30`
on top, ramping the contrastive weight over the first 1000 steps.
Cheaper headline number than A, harder to attribute the gain.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u train_chain.py \
  --preset qwen3-0.6b-large --memres_mode attention_parity \
  --router_mem_bias_init -4.0 --router_recent_bias_init 4.0 \
  --init_from output/chain_v2_phaseA_softparity_b4/best \
  --window_k 8 --session_len 512 \
  --steps 12000 --batch_size 4 --grad_accum 4 \
  --warmup 0 --lr 1e-4 --lr_backbone 1e-5 \
  --memory_dropout 0.10 --context_dropout 0.30 \
  --neg_chain_weight 0.5 --neg_chain_initial_weight 0.05 --neg_chain_warmup_steps 1000 \
  --neg_chain_margin 0.05 --burn_in_max 0 \
  --train_chains paper_artifacts/chains/stage1_train_s512_full.pt \
  --eval_chains paper_artifacts/chains/stage1_validation_s512.pt \
  --gradient_checkpointing --eval_every 200 --save_every 1000 --log_every 25 \
  --eval_n_chains 24 --eval_window 4 --save_best_metric composite \
  --out_dir output/chain_v2_phaseB_softparity --seed 42 \
  > logs/chain_v2_phaseB_softparity.log 2>&1 &
```

**C — Add the missing third row of the ablation table:
`attention_base`** (delta-pool, no init parity, the original AttnRes
paper). The table needs all three modes for the paper. Same data and
hyperparams as the two existing runs, on GPU 1 in parallel.

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u train_chain.py \
  --preset qwen3-0.6b-large --memres_mode attention_base \
  --window_k 8 --session_len 512 \
  --steps 6000 --batch_size 4 --grad_accum 4 \
  --warmup 200 --lr 3e-4 --lr_backbone 3e-5 \
  --memory_dropout 0.0 --context_dropout 0.0 \
  --neg_chain_weight 0.0 --burn_in_max 0 \
  --train_chains paper_artifacts/chains/stage1_train_s512_full.pt \
  --eval_chains paper_artifacts/chains/stage1_validation_s512.pt \
  --gradient_checkpointing --eval_every 200 --save_every 1000 --log_every 25 \
  --eval_n_chains 24 --eval_window 4 --save_best_metric composite \
  --out_dir output/chain_v2_phaseA_attentionbase --seed 42 \
  > logs/chain_v2_phaseA_attentionbase.log 2>&1 &
```

Recommendation: **run A on GPU 0 + C on GPU 1 in parallel.** That
gives the paper a clean step-6000 number for soft `attention_parity`
*and* fills in the missing `attention_base` row of the ablation
table, in one ~10 h day. Phase B (option B) becomes the next day's
work from A's step-6000 best ckpt.

### Phone notifications while you're away

`paper_tools/notify_eval.sh` watches log files and pushes every
`EVAL @ step …` and `Saved checkpoint …` line to ntfy.sh as a phone
notification. One-time setup:

```bash
TOPIC="memres-exx-$(openssl rand -hex 4)"
echo "subscribe to https://ntfy.sh/$TOPIC on your phone (ntfy app)"

cd ~/Desktop/fine-tune/memory_residuals
nohup paper_tools/notify_eval.sh "$TOPIC" \
    logs/chain_v2_phaseA_softparity_b4.log \
    logs/chain_v2_abl_residual_mode.log \
    logs/chain_v2_phaseA_attentionbase.log \
    > logs/notify.log 2>&1 &
```

### Caveats

- `train_chain.py` saves model + config but **not** optimiser /
  scheduler / RNG state, so any "resume" loses Adam momentum and
  cosine-decay phase. If you need an exact resume, that's a one-day
  feature add (save `optim.state_dict()`, `lr_scheduler.state_dict()`,
  `torch.get_rng_state()`, dataloader sampler position; load on init).
  Until then, deterministic re-run from step 0 with the same `--seed`
  is the cleanest "extend" path.
- The on-disk config of in-flight checkpoints uses the **legacy**
  field names (`memres_mode: "residual"` / `"block_attnres"`,
  `block_attnres_parity_init: true/false`). The back-compat shim in
  `modeling_memres._normalise_memres_mode` translates these on
  reload, so eval scripts work unchanged.


Three routing modes exist, exposed via `--memres_mode`:

| mode | what it is | step-0 parity vs bare Qwen3 |
|---|---|---|
| `simple_gate` | Toy / baseline. Standard pretrained residual flow, plus a per-sublayer ReZero-style scalar gate $g_\ell$ (init 0) that adds the memory readout $m^t$ at each sublayer input. | bit-exact (gate = 0) |
| `attention_base` | Full AttnRes pool, *delta sources*: each slot is one sublayer's $\delta$. Closest to the original AttnRes paper. | broken (~34) — softmax cannot reconstruct $b_0 + \sum \delta$ |
| `attention_parity` | Full AttnRes pool, *cumulative sources*: each slot is a hidden-state checkpoint $h_k$. Router init `mem_bias=-32`, `recent_bias=+32` makes the softmax one-hot on the most-recent slot at step 0. | bit-exact |

Legacy strings `residual` (= `simple_gate`) and `block_attnres`
(= `attention_parity` by default, or `attention_base` with
`--no-block_attnres_parity_init`) are still accepted everywhere for
back-compat with old shell scripts and saved checkpoints.

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
| `simple_gate` (ReZero gate $g_\ell{=}0$), no memory | **0.000** |
| `simple_gate` (ReZero gate $g_\ell{=}0$), memory | **0.000** |
| `attention_base` (uniform softmax over deltas) | 34.5 |
| `attention_parity` (cumulative pool + `recent_bias = +32`) | **0.000** |

## Future TODO

In rough priority order — see `COMPREHENSIVE.md` Part I §3–4 for the
full compute-targeted plans (2× H100, 24 h plan and 20× A100, 168 GPU-h plan).

- [ ] Land positive $\Delta_{sh-m}$ on the **rigorous standalone eval** for both PG-19 and LoCoMo. `chain_neg1` is the active attempt; in-trainer numbers are positive but standalone numbers are pending.
- [ ] Run `paper_tools/callback_probe.py` on `chain_neg1/best` and verify the per-callback help ratio is > 1.5× on PG-19 and LoCoMo (we have it for `run3` only).
- [ ] Long-horizon test at chain length ≥ 30 sessions (LoCoMo conv-41 has 32). Report whether memory benefit decays gracefully or collapses (the LRMT failure mode).
- [ ] 8B-class run with `--preset qwen3-8b-large --shard_strategy fsdp_full --gradient_checkpointing`. Untried; ~1 GPU-day on 2× H100 expected.
- [ ] Burn-in stability: the no-grad burn-in path produces $M_c$ states out-of-distribution for the readout (loss spikes at step 5). Either widen the gradient-tracked window or warm-start from a chain-trained checkpoint.
- [ ] Ablate `attention_parity` vs `simple_gate` end-to-end (init-parity is matched in both; need to compare downstream training dynamics — the `attention_parity` mode trades off a saturated softmax at init, the `simple_gate` mode trades off being a strict simplification of the routing pool).
- [ ] Decide on MSC v2 / Persona-Chat inclusion for *training* (currently eval-only) once the data licence question is settled.

## Layout

```
modeling_memres.py          model + Block AttnRes routing (simple_gate / attention_base / attention_parity)
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
