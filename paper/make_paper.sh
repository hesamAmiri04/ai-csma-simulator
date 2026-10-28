#!/bin/sh
# Rebuild every table, figure and number in the paper from csma_rl/results_*/raw.csv.
# Needs: python3 with numpy, scipy, matplotlib; pdflatex with IEEEtran.
set -e
cd "$(dirname "$0")"
python3 build_data.py
python3 build_tex.py > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
echo "Built paper/main.pdf"
