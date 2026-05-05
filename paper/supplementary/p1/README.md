# Memory Residuals — Supplementary Material

This bundle contains the source code, training scripts, evaluation
scripts, locked numbers ledger, and per-checkpoint evaluation JSONs
needed to reproduce the headline result of the paper:

> A frozen Qwen3 backbone augmented with a fixed-size, jointly-trained
> Memory Residuals matrix M_c improves callback cross-entropy on
> LongMemEval-S validation by +1.32 ± 0.53 nats at Qwen3-0.6B
> (n=4 seeds) and +0.93 nats at Qwen3-1.7B (n=2 seeds), with a
> chain-shuffle confound of 0.000 ± 0.010 throughout.

## Layout

```
code/                     architecture, trainer, evaluator
  src/                    modeling_memres.py, train_chain.py, presets.py, ...
  tools/eval_callback.py  canonical post-train evaluator (the script the
                          paper's headline numbers come from)
  tests/                  pytest harness
recipes/                  launcher scripts for the headline + ablation cells
  v27b_*.sh               headline 0.6B (4 seeds, F3-OFF, depth=4, floor ON)
  v28*.sh                 scaling 1.7B (2 seeds, same recipe)
  v27a_*.sh               ablation: --memres_readout_depth 0 (depth-off)
  v27c_*.sh               ablation: --alpha_mem_floor_aux_weight 0.0 (floor-off)
  v24a_*.sh               reference: F3-ON recipe (the original v24a)
numbers/
  NEURIPS_NUMBERS.md      single source of truth for every paper number
evals/                    per-checkpoint evaluator output (v24a / v25a /
                          v27a / v27b{seed1..4} / v27c / v28a / v28b on the
                          patched lme_val_s512_evpos corpus)
```

## Reproducing the headline (v27b-seed1, ~1.5 h on 1× H100)

1. Acquire LongMemEval-S from the upstream release and pre-tokenise the
   training and validation splits to `paper_artifacts/chains/lme_train_s512.pt`
   and `paper_artifacts/chains/lme_val_s512_evpos.pt` (the validation
   corpus must include the `chain_evidence_positions` field).
2. Launch `recipes/v27b_v24a_no_probe_seed1_0p6b_frozen_local.sh`.
3. After ~1.5 h, evaluate with:
   ```
   python tools/eval_callback.py \
     --ckpt Runs/<run_dir>/final \
     --chain_pt paper_artifacts/chains/lme_val_s512_evpos.pt
   ```
   Expected `pa_cb_dnm = +0.797` for this seed; the four-seed mean is
   +1.323 ± 0.530 nats.

## Notes

- Backbone weights are frozen throughout training (`--freeze_backbone
  --lr_backbone 0`); the leak-control claim follows directly.
- The corpus filenames assume the layout described above; any
  reproducer is free to symlink or repath as long as the eval script
  finds `chain_evidence_positions`.
- The `v25a_seed5` checkpoint referenced by some ledger rows is the
  third 1.7B seed; including it would tighten the 1.7B mean if you
  budget the additional ~6 h of compute.
