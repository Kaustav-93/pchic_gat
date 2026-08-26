"""Shared helpers for assembling the derivative notebooks from
_nb_base_cells.py (a verbatim dump of pangenomic_gat_pipeline_pchic_only.ipynb).

Mirrors the existing _build_notebook.py convention: cells are (kind, source)
tuples, every code cell is ast-parsed before writing, output is a plain
nbformat-4 .ipynb.
"""
import ast
import json
from pathlib import Path

import _nb_base_cells as _base

ROOT = Path(__file__).parent


def load_base():
    """Fresh mutable copy of the 69 base cells as a list of [kind, src]."""
    return [[k, s] for k, s in _base.CELLS]


def replace_cell(cells, idx, old, new, expect=1):
    """In-place substring replace inside cells[idx]'s source. Asserts `old`
    occurs exactly `expect` times, so a silent no-op (source drifted) or an
    unexpected multi-match is caught immediately rather than shipping a
    notebook that doesn't actually contain the edit."""
    kind, src = cells[idx]
    n = src.count(old)
    if n != expect:
        raise ValueError(
            f"cell {idx}: expected {expect} occurrence(s) of the anchor text, "
            f"found {n}.\n--- anchor ---\n{old[:300]}"
        )
    cells[idx][1] = src.replace(old, new)


def insert_after(cells, idx, kind, src):
    cells.insert(idx + 1, [kind, src])


def find_cell_idx(cells, substr, expect=1):
    """Locate a cell by a unique substring of its source, instead of a
    hardcoded position -- robust to earlier inserts/drops shifting indices."""
    hits = [i for i, (_, s) in enumerate(cells) if substr in s]
    if len(hits) != expect:
        raise ValueError(f"expected {expect} cell(s) containing {substr!r}, found {len(hits)}: {hits}")
    return hits[0] if expect == 1 else hits


def drop_cells(cells, indices):
    for i in sorted(indices, reverse=True):
        cells.pop(i)


def append_cells(cells, new_cells):
    cells.extend([list(c) for c in new_cells])


def build_ipynb(cells, out_path, title, kernel_display="Python (pangenomic-gat)"):
    errors = []
    for i, (kind, src) in enumerate(cells):
        if kind != "code":
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad_line = src.splitlines()[e.lineno - 1] if e.lineno and e.lineno <= len(src.splitlines()) else ""
            errors.append((i, e.lineno, e.msg, bad_line))
    if errors:
        print(f"SYNTAX ERRORS in {out_path.name}")
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
                "display_name": kernel_display,
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
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(nb, indent=1))

    n_md = sum(1 for k, _ in cells if k == "markdown")
    n_code = sum(1 for k, _ in cells if k == "code")
    n_lines = sum(len(s.splitlines()) for k, s in cells if k == "code")
    print(f"wrote {out_path}")
    print(f"  {len(cells)} cells: {n_md} markdown, {n_code} code")
    print(f"  {n_lines} lines of code, {out_path.stat().st_size/1024:.0f} KB")
    print("  all code cells parse cleanly")
