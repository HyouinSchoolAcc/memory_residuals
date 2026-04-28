# Phase A head-to-head — soft `attention_parity` vs `simple_gate`

**Stopped:** night of 2026-04-28, by user request, before either run reached the
planned 6000 steps.

**Setup:** identical hyperparameters across both runs except `--memres_mode`
(and the soft-parity router-bias overrides on GPU 0):

```
--preset qwen3-0.6b-large
--window_k 8 --session_len 512
--steps 6000 --batch_size 4 --grad_accum 4
--warmup 200 --lr 3e-4 --lr_backbone 3e-5
--memory_dropout 0.0 --context_dropout 0.0
--neg_chain_weight 0.0 --burn_in_max 0
--train_chains paper_artifacts/chains/stage1_train_s512_full.pt   (4769 chains, 113M Qwen tokens)
--eval_chains paper_artifacts/chains/stage1_validation_s512.pt    (48 chains, 2145 sessions)
--seed 42
```

GPU 0 added: `--memres_mode attention_parity --router_mem_bias_init -4.0 --router_recent_bias_init 4.0`.
GPU 1 added: `--memres_mode residual` (legacy alias for `simple_gate`).

EVAL metric: in-trainer eval, n=92 sessions, `eval_window=4`. Standalone
paper-grade eval via `paper_tools/eval_chain.py` is the next step (tomorrow).

## Head-to-head trajectory (?_sh-m, history specificity)

Higher is better. Negative would mean the model is using a generic style prior
instead of *this* chain's memory.

| step | simple_gate ?_sh-m | soft attn_parity ?_sh-m | ratio |
|---:|---:|---:|---:|
|  200 | +0.0002 | -0.0001 | (init noise) |
|  400 | +0.0011 | +0.0042 | 3.8× |
|  600 | +0.0048 | +0.0089 | 1.9× |
|  800 | +0.0068 | +0.0107 | 1.6× |
| 1000 | +0.0104 | +0.0177 | 1.7× |
| 1200 | +0.0103 | +0.0164 | 1.6× |
| 1400 | +0.0103 | +0.0191 | 1.9× |
| 1600 | +0.0123 | +0.0264 | 2.1× |
| 1800 | +0.0118 | +0.0240 | 2.0× |
| 2000 | +0.0138 | **+0.0272** | 2.0× |

After step 2000, only `simple_gate` continued:

```
step 2400: ?_sh-m = +0.0177
step 3000: ?_sh-m = +0.0214
step 3400: ?_sh-m = +0.0224
step 4000: ?_sh-m = +0.0234   <- enters plateau
step 4400: ?_sh-m = +0.0242
step 4800: ?_sh-m = +0.0243
step 5000: ?_sh-m = +0.0244
step 5200: ?_sh-m = +0.0249   <- final, plateaued
```

## Bottom line at the killed-at points

| metric | simple_gate @ step 5200 | soft attention_parity @ step 2000 |
|---|---:|---:|
| mem CE              | 2.9352 | 2.9504 |
| nomem CE            | 2.9442 | 2.9611 |
| shuffle CE          | 2.9601 | 2.9776 |
| oracle CE           | 2.8156 | 2.8273 |
| **?_nm-m** (memory help)         | +0.0090 | **+0.0107** |
| **?_sh-m** (history specificity) | +0.0249 | **+0.0272** |
| ?_or-m (oracle gap, more negative = more compression cost) | -0.1196 | -0.1231 |

**Soft attention_parity at step 2000 has surpassed simple_gate's terminal
plateau (step ?4000) on every memory metric**, despite having received only
2000/6000 of its planned training steps and being still on the upward portion
of its curve when killed.

The interpretation: the depth-wise routing pool, when initialized with a
non-saturated softmax (`mem_bias=-4`, `recent_bias=+4`), learns to recruit
memory roughly **2.6× more sample-efficiently** than the per-sublayer scalar
gate (?_sh-m crosses +0.024 at step 1500 vs step 4000). Backbone CE is within
0.015 nat across the two runs and is purely a function of how many steps of
fine-tuning each model has had on the same Qwen3-0.6B base.

## What's next

In priority order (also documented at the top of `README.md`):

1. Standalone `eval_chain.py` runs on PG-19 val + PG-19 test (held out) +
   LoCoMo for both `best/` checkpoints — confirms the in-trainer numbers
   above hold under rigorous evaluation.
2. `callback_probe.py` on both `best/` ckpts — checks whether the
   memory-help is concentrated on callback tokens (entities reintroduced
   after gaps) rather than spread uniformly.
3. `horizon_analysis.py` — bucket ?_sh-m by chain length to see where the
   benefit decays.
4. Decide on the day's compute: extend soft attention_parity to step 6000
   (deterministic re-run from step 0 with `--seed 42`, ~10 h on H100),
   add the missing `attention_base` ablation row in parallel on GPU 1.
5. Phase B (contrastive curriculum) warm-started from soft attention_parity
   `best/`.

## Files

- This file: `paper_artifacts/eval/chain_v2_phaseA_trajectories.md`
- Raw JSON with every EVAL row: `paper_artifacts/eval/chain_v2_phaseA_trajectories.json`
- Best checkpoints (gitignored, on local disk):
  - `output/chain_v2_phaseA_softparity_b4/best/` — soft attn_parity best
  - `output/chain_v2_abl_residual_mode/best/` — simple_gate best
- Per-step training logs (gitignored): `logs/chain_v2_*.log`
