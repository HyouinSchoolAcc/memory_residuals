# Paper — build folder

This folder contains the LaTeX source, figure-generation scripts, and
auto-rebuild plumbing for the headline paper:

> **Memory Residuals at Scale: A Frozen-Backbone Recurrent Memory
> That Beats RAG on Long-Horizon Callback Recall**
> (NeurIPS 2026 Main Track submission)

It also holds everything else paper-shaped that used to clutter the
repo root: abstracts, paper drafts, the position-paper PDF, and the
supplementary-material build trees.

## Layout

```
paper/
  main.tex                     # the headline paper; uses \input{numbers}
  main.pdf                     # the compiled paper (output)
  numbers.tex                  # auto-generated; DO NOT EDIT
  xcorpus_numbers.tex          # auto-generated cross-corpus macros
  refs.bib                     # references
  neurips_2024.sty             # NeurIPS style file (symlinked)
  build.sh                     # one-button rebuild
  README.md                    # this file
  figures/
    p_a_numbers.json           # machine-readable summary (input to LaTeX macros)
    p_a_headline.pdf           # 2-panel bar chart memres vs RAG (Fig. 1)
    p_a_per_chain.pdf          # per-chain scatter memres vs oracle-RAG (Fig. 2)
    p_a_mechanism.pdf          # Δ_shuffle + evidence_lift comparison (Fig. 3)
  scripts/
    compute_numbers.py         # reads eval JSONs → p_a_numbers.json
    compute_xcorpus_numbers.py # cross-corpus eval → xcorpus_numbers.tex
    render_numbers_tex.py      # p_a_numbers.json → numbers.tex
    make_p_a_headline.py       # bar chart
    make_p_a_per_chain.py      # scatter
    make_p_a_mechanism.py      # mechanism comparison

  abstracts/                   # paste-ready abstract drafts (v3 is canonical)
    ABSTRACT_NEURIPS_v3.md     # current; matches main.tex
    ABSTRACT_NEURIPS_v2.md
    ABSTRACT_NEURIPS_v1.md
    ABSTRACT_v0.md / ABSTRACT_v1.md  # earlier framings, kept for diff history

  drafts/                      # long-form prose, planning, locked numbers
    NEURIPS_NUMBERS.md         # locked numbers ledger (single source of truth)
    NEURIPS_SUBMISSION_HOWTO.md
    NEURIPS_SUBMISSIONS.md
    PAPER_PLAN.md
    PAPER_A_v28_RAG.md
    PAPER_B_attention_parity.md
    PAPER_C_negative_results.md
    V20_ARCHITECTURE.md
    OVERNIGHT_STATUS.md
    CORPUS_SCALING.md

  position/                    # the architectural-spec position paper
    memory_residuals.tex
    memory_residuals.pdf
    atn_residuals.pdf          # Block Attention Residuals reference
    figures/                   # PNGs embedded in memory_residuals.pdf

  supplementary/               # supplementary-material build trees
    p1/  p2/  p3/              # per-paper supplementary zips' source
```

## Build

```bash
bash build.sh
# -> writes paper/main.pdf (~340 KB, 12 pages: 7 main + 2 refs + 2 appendix + 1 checklist)
```

`build.sh` is idempotent. It:

1. Re-reads the locked eval JSONs in
   `../results/eval_v25_seed_pack_evpos/` and `../results/rag_baseline/`.
2. Recomputes summary statistics (`figures/p_a_numbers.json`).
3. Renders LaTeX macros (`numbers.tex`, `xcorpus_numbers.tex`).
4. Re-generates the three PDF figures.
5. Runs `pdflatex` + `bibtex` + 2× `pdflatex`.

If you re-train a seed and the new eval JSON lands, just re-run
`build.sh`; the paper updates automatically (no LaTeX edits needed).

## Numbers source of truth

Every number in the paper traces back to a JSON in `../results/`:

| paper element                       | source JSON(s) |
|---|---|
| 0.6B memres headline (n=4)          | `eval_v25_seed_pack_evpos/v27b_no_probe[_seed{2,3,4}]_final_lme_val_evpos.json` |
| 1.7B memres scaling                 | `eval_v25_seed_pack_evpos/v28{a,b,c,d}_no_probe_seed{1,2,3,4}_final_lme_val_evpos.json` |
| RAG cells (0.6B)                    | `rag_baseline/qwen3_0p6b_{nomem,bm25_top{1,3},dense_top3,oracle_top3}.json` |
| RAG cells (1.7B)                    | `rag_baseline/qwen3_1p7b_{nomem,bm25_top3,dense_top3,oracle_top3}.json` |
| no-depth ablation (n=4)             | `eval_v25_seed_pack_evpos/v27a_*` |
| no-floor ablation (n=4)             | `eval_v25_seed_pack_evpos/v27c_*` |
| 95 % bootstrap CI                   | computed in `compute_numbers.py` over the per-chain residuals |

## Auto-rebuild watcher

`../scripts/watcher_paper_a_autorebuild.sh` polls the eval JSONs every
2 minutes and runs `build.sh` whenever a new one lands. Designed to
sit in `tmux` while v28c / v28d (and any future seeds) finish on the
GH200. Launch with:

```bash
tmux new-session -d -s paper_build "cd memory_residuals && bash scripts/watcher_paper_a_autorebuild.sh"
```

## Style file caveat

The `neurips_2024.sty` symlink in this folder is a stand-in for the
official `neurips_2026.sty` (which has not yet been downloaded from
<https://media.neurips.cc/Conferences/NeurIPS2026/>). The two are
structurally identical for layout purposes; replace the symlink and
the `\usepackage[preprint]{neurips_2024}` line in `main.tex` before
the final upload.

## Anonymity

`main.tex` uses `\author{Anonymous Authors}` and the bibliography
contains no self-cites. Verify after each rebuild via:

```bash
pdftotext main.pdf - | grep -iE "yueze|liu|kumdam|illinois|exx|sn4622|ubuntu|home/exx|192\.222|S;G|@gmail|@illinois"
# (only matches should be from the [12] Liu et al. "Lost in the middle" citation)
```
