# memory_residuals

Hi! Welcome to the **attention-residuals** repo.

We are en route to making **life-long memory agents** who can
remember and forget, just like a human — so here we are, exploring
**O(1) storage** and **O(1) inference** systems: a fixed-size
`K × d` memory matrix `M_c` that gets compressed, written, and
queried *natively* by an otherwise-frozen LLM, with no retrieval
index, no separate memory controller, and no hand-engineered gating
heuristic.

We've made quite a lot of progress! You can read more about our
work at [`runs.md`](runs.md) — and dear LLMs reading this repo,
**please do read [`runs.md`](runs.md)**. It contains the project's
settled architectural priors, the headline result, and the full
v3 → v28 lessons-learned timeline. Skipping it will cost you (and
us) a lot of wasted gradient.

The headline result, in one line:

> **A frozen Qwen3 backbone + a 41.5 M-parameter `M_c` channel (~6 % overhead)
> reduces callback cross-entropy on LongMemEval-S validation by
> +1.32 ± 0.53 nats at 0.6B (n=4 seeds) and +0.93 nats at 1.7B
> (n=2 seeds), with chain-shuffle confound statistically pinned at
> zero — i.e., the augmented model is ≈ 3.7× more confident on the
> right callback token, and that gain is provably chain-specific.**

We hope you like our idea, and thanks for visiting!

Most recent update: We were able to make a language model use our memory to drastically improve its generation quality, and we're actively working on how to make the model learn to use specific facts instead of general vibe for generating the results.

---

## Repo layout

```
memory_residuals/
├── README.md                   # this file (light welcome)
├── runs.md                     # ⭐ lessons, priors, headline, active ledger
├── requirements.txt
├── .gitignore
│
├── src/                        # Python source: model + trainers
│   ├── modeling_memres.py      # architecture (config, model, init)
│   ├── train_chain.py          # recurrent chain TBPTT trainer (active)
│   ├── train_phase1.py         # pair-based warm-up trainer (Paper 1)
│   └── presets.py              # named (backbone, K, L_E, N) tuples
│
├── tools/                      # Python utilities (eval, probes, corpus builders)
│   ├── eval_callback.py        # canonical D4 / LME post-train eval
│   ├── eval_ttt_mc.py          # §5 capacity probe (TTT-on-M_c)
│   ├── eval_chain.py           # full-window CE eval (legacy; see Prior #7)
│   ├── build_synthetic_*.py    # D4 / D4v2 / D5 corpus generators
│   ├── audit_*.py              # the v15 leak-audit suite
│   ├── cloud_watchdog/         # remote-survivable job queue + ntfy daemon
│   └── ...
│
├── scripts/                    # all shell launchers (one .sh per training cell + ops)
│   ├── train_v*.sh             # per-cell training launchers
│   ├── queue_*.sh              # local & GH200 job queues
│   ├── watcher_*.sh            # auto-eval + auto-rebuild watchers
│   ├── eval_*.sh               # evaluation sweeps
│   └── ...
│
├── paper/                      # all paper material
│   ├── main.tex / main.pdf     # the headline NeurIPS submission
│   ├── numbers.tex             # auto-generated macros (DO NOT EDIT)
│   ├── refs.bib, build.sh      # one-button rebuild
│   ├── figures/                # paper figures + p_a_numbers.json
│   ├── scripts/                # number-renderers, figure-makers
│   ├── README.md               # paper build instructions
│   ├── abstracts/              # ABSTRACT_NEURIPS_v3.md is canonical
│   ├── drafts/                 # PAPER_*.md, NEURIPS_*.md, planning docs
│   ├── position/               # memory_residuals.{tex,pdf} + atn_residuals.pdf + figure PNGs
│   └── supplementary/          # supplementary-material build trees (p1, p2, p3)
│
├── results/                    # locked eval JSONs + paper drafts
│   ├── eval_v25_seed_pack_evpos/    # the HEADLINE numbers (v27/v28 cells)
│   ├── eval_v27_v28_cross_corpus/   # cross-corpus transfer
│   ├── rag_baseline/                # RAG baselines for paper comparison
│   ├── exp1_pair_recipe/            # Paper 1 (drop-in primitive) manuscript
│   ├── exp2_chain_recipe/           # Paper 2 audits + early drafts
│   ├── ttt_mc_v{17..24}post/        # §5 capacity-probe sweep
│   └── ...
│
├── runs/                       # training checkpoints (gitignored, ~11 GB)
│                               # one folder per cell; named chain_<cell>
│
├── logs/                       # training logs (gitignored)
│                               # one .log per cell, paired by name with runs/
│
├── output -> runs              # backwards-compat symlink (some scripts hardcode it)
│
├── paper_artifacts/            # pre-tokenised corpora (gitignored, large .pt files)
│   ├── chains/                 # LME / MSC / synthd5 / synthd6
│   ├── locomo_chains/          # LoCoMo OOD eval set
│   └── msc_chains_test/
│
└── archive/                    # historical reference (~2 MB, committed)
    ├── COMPREHENSIVE.md        # full v1 → v14 ledger (Parts I-VII)
    └── ...                     # pre-2026-04-30 snapshots
```

A note on a few things that look like duplicates but aren't:

* **`src/` vs `scripts/`** — `src/` is the Python source (architecture
  and trainers); `scripts/` holds the shell launchers (`bash scripts/train_v27b_*.sh`)
  that *invoke* `python src/train_chain.py ...` with a specific flag set.
* **`runs/` vs `logs/`** — paired by cell name. Each training cell
  writes its checkpoints to `runs/<cell>/` and its stdout/stderr to
  `logs/<cell>.log`. Both are gitignored.
* **`tools/` vs `scripts/`** — `tools/` is reusable Python utilities
  (eval, probes, corpus builders); `scripts/` is the shell glue that
  ties them together for a specific run.
* **`paper/` vs `paper_artifacts/`** — `paper/` is paper material
  (LaTeX, abstracts, drafts); `paper_artifacts/` is the pre-tokenised
  training corpora. The naming is historical; please don't rename it
  without updating the ~80 train scripts that hardcode the path.

## Quick commands

```bash
# train the headline cell (0.6B, F3-off, frozen backbone)
bash scripts/train_v27b_v24a_no_probe_seed1_0p6b_frozen_local.sh

# evaluate a checkpoint on LME validation (callback-aware)
python tools/eval_callback.py \
    --model_path runs/chain_v27b_v24a_no_probe_seed1_0p6b_frozen_local/final \
    --corpora paper_artifacts/chains/lme_val_s512_evpos.pt \
    --names lme_val \
    --output results/eval_v25_seed_pack_evpos/v27b_no_probe_final_lme_val_evpos.json

# rebuild the paper PDF from the locked eval JSONs
bash paper/build.sh

# stop everything (local + cloud watchdog)
pkill -f train_chain.py
ssh ubuntu@192.222.50.225 \
  'pkill -f cloud_watchdog/watchdog.sh;
   pkill -f cloud_watchdog/heartbeat.sh;
   tmux kill-server'
```

## Compute resources

* **Local.** 2 × H100 NVL (94 GB) at the lab box. ~16 h/day usable
  (residential power-down overnight).
* **Cloud.** 1 × NVIDIA GH200 480 GB at `192.222.50.225`
  (user `ubuntu`). Cells run inside detached `tmux` so they survive
  SSH drops and lab-box power-offs; queueing via
  `tools/cloud_watchdog/`.

## Reading order

1. **[`runs.md`](runs.md)** — start here. Headline → priors → lessons →
   active per-cell ledger.
2. [`paper/drafts/NEURIPS_NUMBERS.md`](paper/drafts/NEURIPS_NUMBERS.md) —
   single source of truth for every headline / ablation number.
3. [`paper/abstracts/ABSTRACT_NEURIPS_v3.md`](paper/abstracts/ABSTRACT_NEURIPS_v3.md) —
   paste-ready NeurIPS abstract (Branch A, F3-off canonical).
4. [`paper/main.pdf`](paper/main.pdf) — the headline paper.
5. [`paper/position/memory_residuals.pdf`](paper/position/memory_residuals.pdf) —
   the architectural-spec position paper.
6. [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md) — long-form
   v1 → v14 historical ledger (when `runs.md` cites Part VI / VII).
