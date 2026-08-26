"""Assemble the two cell-definition modules into a single .ipynb."""
import json, sys, ast
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import importlib
p1 = importlib.import_module("_nb_cells_part1")
p2 = importlib.import_module("_nb_cells_part2")
p3 = importlib.import_module("_nb_cells_part3")

cells = p1.CELLS + p2.CELLS + p3.CELLS

# ---- syntax check every code cell before we write anything -------------------
errors = []
for i, (kind, src) in enumerate(cells):
    if kind != "code":
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append((i, e.lineno, e.msg,
                       src.splitlines()[e.lineno - 1] if e.lineno and
                       e.lineno <= len(src.splitlines()) else ""))
if errors:
    print("SYNTAX ERRORS")
    for i, ln, msg, line in errors:
        print(f"  cell {i}  line {ln}: {msg}\n    {line}")
    raise SystemExit(1)

nb = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if kind == "code" else {}),
        }
        for kind, src in cells
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python (pangenomic-gat)",
            "language": "python",
            "name": "pangenomic-gat",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "codemirror_mode": {"name": "ipython", "version": 3},
        },
        "title": "Pangenomic AI - promoter-centred graph attention network",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent.parent / "notebooks" / "pangenomic_gat_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1))

n_md = sum(1 for k, _ in cells if k == "markdown")
n_code = sum(1 for k, _ in cells if k == "code")
n_lines = sum(len(s.splitlines()) for k, s in cells if k == "code")
print(f"wrote {out.name}")
print(f"  {len(cells)} cells: {n_md} markdown, {n_code} code")
print(f"  {n_lines} lines of code, {out.stat().st_size/1024:.0f} KB")
print("  all code cells parse cleanly")
