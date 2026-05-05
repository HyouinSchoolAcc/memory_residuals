#!/usr/bin/env bash
# Build Paper A from scratch.
#
#   1. Compute summary stats (paper_a/figures/p_a_numbers.json)
#   2. Render numbers.tex from those stats
#   3. Generate the 3 figures (PDFs in paper_a/figures/)
#   4. Compile main.tex (-> main.pdf)
#
# Re-run this whenever a new eval JSON lands.
set -euo pipefail
cd "$(dirname "$0")"

python scripts/compute_numbers.py
python scripts/render_numbers_tex.py
python scripts/make_p_a_headline.py
python scripts/make_p_a_per_chain.py
python scripts/make_p_a_mechanism.py

# LaTeX compile (pdflatex + bibtex + 2x pdflatex for refs/tables)
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

echo "================================================================"
echo "Paper A built. PDF at: $(pwd)/main.pdf"
ls -la main.pdf 2>/dev/null
echo "Page count:" $(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/ {print $2}')
echo "================================================================"
