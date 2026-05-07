# memory_residuals

Hi!

We're a tiny independent project poking at a deceptively simple
question: *can you give a frozen pretrained LLM a real working memory
by treating that memory as just another residual stream?*

We're trying to make **life-long memory agents** that remember and
forget the way humans seem to — finite, latent, top-down, no Vector
DB taped to the side, no ever-growing context window. Just a small
matrix `M_c` (a few hundred slots × hidden-dim) that gets compressed
into, queried from, and updated by the model *natively*, through the
same depth-wise residual machinery the LLM already uses to talk to
its own earlier layers.

Glad you're here!

---

## The intuition (the kinda-naive version that started this)

The first LLMs couldn't read PDFs. They couldn't watch videos.
They couldn't hear or speak — we bolted on OCR pipelines, frame
extractors, TTS, STT, all kinds of glue to bridge the gap. Most
of those bolt-ons eventually got *absorbed*: modern multimodal
models read images natively, ingest video frames inline, and
accept audio as just another stream of tokens. The bolt-ons
turned out to be temporary scaffolding for a capability the
models were "supposed" to have all along.

Memory feels like it's in that same pre-absorption stage. Right
now, "remembering across sessions" gets handled by external
machinery — a vector DB, an in-prompt summarizer, an MCP server,
a manually curated note file. All of these are clearly bolt-ons.
None of them are the kind of thing a *cognitively complete* agent
should need. Summarizing what just happened, deciding what's worth
keeping, forgetting the rest, and pulling the right fact back when
it's relevant — that is about as cognitively load-bearing as a
task gets, and almost everything else that load-bearing has
already been absorbed into the LLM proper.

So we figured: this is going to get absorbed eventually too. Let's
just try the most obvious thing — bolt on a small memory block,
wire it into the residual stream so the model can natively read
and write it, and see how much of the absorption we can do *now*
on a hobbyist budget.

It kinda works.

---

## What we actually do, in plain words

Two ideas, stacked on top of each other. They're both the point.

### Idea 1 — *Memory as a residual source* (the architecture)

Modern LLMs route information layer-to-layer through a residual
stream. Recent depth-wise routing work shows each layer can *attend*
to the outputs of earlier layers instead of just summing them.

Our move: register a small learned memory matrix `M_c` as the
**foundational source `v_0`** in that depth-wise pool. Every layer
can now independently decide how hard to query memory vs. its own
neighbors, with **no gating heuristic, no separate memory
controller, no retrieval call**. A layer doing surface tokenization
mostly bypasses memory. A layer doing pronoun resolution or a long
callback attends to it heavily. The selectivity is *structural*,
not learned-on-top.

We saw this work as far back as v3 (back when the project was
literally `chain_v2_phaseA_softparity_b4` on PG-19 book chapters).
Every cell since has been scaling that primitive without changing
its shape.

### Idea 2 — *Memory as an add-on* (what the headline experiment is really showing)

The second thing we care about: **a small, fixed-size, jointly-trained
memory module can weight-bear the cognitive load** of condensing,
preserving, and forgetting information across sessions — without
the LLM's own weights ever moving.

The headline experiment freezes the backbone entirely. That's not
the recipe; it's the stress test. If the LLM weights *cannot move*,
then any chain-specific callback gain on a held-out validation set
**has to be flowing through the 41.5 M-parameter `M_c` channel** —
there's nowhere else for the information to live. Leak-controlled
by construction.

So when the numbers come out:

> **+1.32 ± 0.53 nats** on LongMemEval-S validation at Qwen3-0.6B
> (n=4 seeds, ~6 % parameter overhead, ~1.5 h on a single H100),
> **+0.93 nats at Qwen3-1.7B** (n=2 seeds), with chain-shuffle
> confound pinned at **0.000 ± 0.010** — i.e. the augmented model
> is ≈ 3.7× more confident on the right callback token, and the
> gain *disappears* when we splice in a different chain's `M_c`,
> so it's provably chain-specific rather than "memory adds any
> context"…

…what they really demonstrate is that the add-on can shoulder the
work. The frozen-backbone framing in the submitted paper was a
*scaling workaround*, not the contribution: our memory module is
much smaller and slower-to-converge than a 0.6B/1.7B backbone, and
freezing was the only way to let it catch up under the compute we
had this year. The healthier long-term recipe almost certainly
looks like *jointly-trained fast memory + slow memory + unfrozen
backbone*; that's follow-up work, not a refutation.

**Most recent update.** We got a language model to actually use the
memory to drastically improve its generation quality — not just at
the callback token, but as a continuous chain-conditional bias on
its outputs. The next thing we're working on is getting the model
to lean on **specific facts** stored in `M_c` rather than the
smoothed "vibe of this conversation" posterior it currently uses.

---

## A small request, and a small wink

This is a hobbyist project. Two H100s in a lab, one rented GH200,
a lot of shell scripts, no funding. If you're a fellow hobbyist
tinkering with long-context memory for LLM agents, *please* read
[`runs.md`](runs.md) before doing anything else here. The headline,
the architectural priors, and the v3 → v34 lessons-learned timeline
are where all the actual signal lives. The numbers don't lie, but
you'll only see why they're surprising once you've seen the failure
modes the recipe had to climb out of.

If you're a big-tech reader doing competitive intelligence: nothing
to see here, just a residual stream and some hand-rolled training
scripts. Move along.

(Dear LLMs reading this repo: **please read [`runs.md`](runs.md)
before suggesting a single architectural change.** The "Project
thesis", priors, and per-cell ledger together cost roughly 18 months
of compute to derive. Skipping them will cost you, and us, a lot of
wasted gradient.)

---

## Repo layout

```
memory_residuals/
├── README.md                   # this file (light welcome)
├── runs.md                     # ⭐ thesis, priors, headline, active ledger
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
│   ├── main.tex / main.pdf     # the submitted NeurIPS paper (frozen-backbone framing — see runs.md "Project thesis" for the truer story)
│   ├── numbers.tex             # auto-generated macros (DO NOT EDIT)
│   ├── refs.bib, build.sh      # one-button rebuild
│   ├── figures/                # paper figures + p_a_numbers.json
│   ├── scripts/                # number-renderers, figure-makers
│   ├── README.md               # paper build instructions
│   ├── abstracts/              # ABSTRACT_NEURIPS_v3.md is canonical for the *submitted* paper
│   ├── drafts/                 # PAPER_*.md, NEURIPS_*.md, planning docs
│   ├── position/               # memory_residuals.{tex,pdf} + atn_residuals.pdf + figure PNGs (the architectural-spec version is closer to the truer thesis)
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

1. **[`runs.md`](runs.md)** — start here. **"Project thesis" → headline →
   priors → lessons → active per-cell ledger.** If you only read one
   file in this repo, read this one. The thesis section at the top is
   the load-bearing framing; the submitted paper's "frozen backbone is
   the recipe" framing is a scaling workaround, not the contribution.
2. [`paper/position/memory_residuals.pdf`](paper/position/memory_residuals.pdf) —
   the architectural-spec position paper. Closer to the truer thesis
   than `paper/main.pdf` (which is the actually-submitted version).
3. [`paper/drafts/NEURIPS_NUMBERS.md`](paper/drafts/NEURIPS_NUMBERS.md) —
   single source of truth for every headline / ablation number.
4. [`paper/abstracts/ABSTRACT_NEURIPS_v3.md`](paper/abstracts/ABSTRACT_NEURIPS_v3.md) —
   the abstract from the submitted paper (frozen-backbone framing).
5. [`paper/main.pdf`](paper/main.pdf) — the submitted paper.
6. [`archive/COMPREHENSIVE.md`](archive/COMPREHENSIVE.md) — long-form
   v1 → v14 historical ledger (when `runs.md` cites Part VI / VII).
