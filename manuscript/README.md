# Manuscript (draft v0.1)

All numbers, tables and the figure are generated; do not edit them by hand:

    python analysis/paper_tables.py results    # writes numbers.tex, tables/*.tex, figures/recency_gain.*

Requires the eval logits in `results/*/eval/` (git-ignored; extracted from the Kaggle archives).

Compile: no local LaTeX needed. Zip this folder (main.tex, refs.bib, numbers.tex, tables/, figures/)
and upload it to Overleaf (free plan) as a new project, then press Recompile. Engine: pdfLaTeX + BibTeX.
