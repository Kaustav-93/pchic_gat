"""Assemble the standalone exploratory notebook."""
import json, sys, ast
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib
cells = importlib.import_module("_nb_eda_cells").CELLS

errors = []
for i, (kind, src) in enumerate(cells):
    if kind != "code":
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        line = src.splitlines()[e.lineno - 1] if e.lineno and e.lineno <= len(src.splitlines()) else ""
        errors.append(f"cell {i} line {e.lineno}: {e.msg}\n    {line}")
if errors:
    print("SYNTAX ERRORS"); [print(" ", e) for e in errors]; raise SystemExit(1)

nb = {
    "cells": [{"cell_type": k, "metadata": {}, "source": s.splitlines(keepends=True),
               **({"execution_count": None, "outputs": []} if k == "code" else {})}
              for k, s in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python (pangenomic-gat)",
                       "language": "python", "name": "pangenomic-gat"},
        "language_info": {"name": "python", "version": "3.10",
                          "mimetype": "text/x-python", "file_extension": ".py",
                          "pygments_lexer": "ipython3",
                          "nbconvert_exporter": "python",
                          "codemirror_mode": {"name": "ipython", "version": 3}},
        "title": "Exploratory analysis - PCA, UMAP, clustering",
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).parent / "pangenomic_exploratory_analysis.ipynb"
out.write_text(json.dumps(nb, indent=1))
n_md = sum(1 for k, _ in cells if k == "markdown")
n_code = sum(1 for k, _ in cells if k == "code")
print(f"wrote {out.name}")
print(f"  {len(cells)} cells: {n_md} markdown, {n_code} code")
print(f"  {sum(len(s.splitlines()) for k, s in cells if k=='code')} lines of code, "
      f"{out.stat().st_size/1024:.0f} KB")
print("  all code cells parse cleanly")
