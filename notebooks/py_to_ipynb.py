"""Convert a `# %%`-delimited .py script into a Jupyter notebook.

    python py_to_ipynb.py SCRIPT.py [OUT.ipynb] [--metadata-from OTHER.ipynb]

`# %%` starts a code cell, `# %% [markdown]` a markdown cell (its lines lose
their leading "# "). The .py stays the source of truth; re-run after editing.
"""
import argparse
import json
import re
from pathlib import Path

import nbformat


def convert(py_path, out_path, metadata_from=None):
    text = Path(py_path).read_text()
    parts = re.split(r"^(# %%(?: \[markdown\])?)[ \t]*$", text, flags=re.M)
    nb = nbformat.v4.new_notebook()
    if metadata_from:
        nb.metadata = json.loads(Path(metadata_from).read_text()).get("metadata", {})
    # parts = [preamble, marker, body, marker, body, ...]
    for marker, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if marker.endswith("[markdown]"):
            lines = [re.sub(r"^# ?", "", ln) for ln in body.split("\n")]
            nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(lines).strip("\n")))
        elif body.strip():
            nb.cells.append(nbformat.v4.new_code_cell(body))
    nbformat.validate(nb)
    nbformat.write(nb, out_path)
    return len(nb.cells)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--metadata-from")
    a = ap.parse_args()
    out = a.out or str(Path(a.script).with_suffix(".ipynb"))
    print(f"{convert(a.script, out, a.metadata_from)} cells -> {out}")
