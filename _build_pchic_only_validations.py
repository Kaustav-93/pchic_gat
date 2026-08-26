"""Build pangenomic_gat_pipeline_pchic_only_validations.ipynb:

Identical pipeline to pangenomic_gat_pipeline_pchic_only.ipynb (16,384 bp
promoter window, same graphs, same model) PLUS:
  - a disk cache for the built graphs (skip the ~350k-graph rebuild on rerun)
  - Part III: chromosome-block CV, leave-one-tissue-out CV, blood-as-test

Results go to results_pchic_only_validations/ (original results/ untouched).
Reuses the SAME cache/ dir for promoter one-hot sequences (identical window
and gene set means an identical cache key, so it loads instantly).
"""
from pathlib import Path
from _nb_tools import load_base, replace_cell, drop_cells, append_cells, build_ipynb
import _nb_edits as E

ROOT = Path(__file__).parent
cells = load_base()

BANNER = '''# Variant note — this notebook adds validation strategies

This notebook is `pangenomic_gat_pipeline_pchic_only.ipynb`, unchanged in
every modelling respect (same 16,384 bp promoter window, same graphs, same
model, same 5-fold gene-grouped CV as the primary result), **plus**:

1. **A graph disk cache** (new cell right after graph construction): the
   built `GRAPHS`/`META` are saved to `cache/graphs_<config-hash>.pt` and
   reloaded from there on any rerun with the same settings, so the
   expensive graph-construction step only ever runs once.
2. **Part III**, appended after the original notebook's final cell:
   chromosome-block CV, leave-one-tissue-out CV, and a blood-tissues-as-test
   holdout — see that section's own introduction for why each matters and
   what each does and does not establish.

Everything through the original "What changed, and what to check first"
section is byte-for-byte the same pipeline as the base notebook. Results
from *this* notebook are written to `results_pchic_only_validations/` so
nothing already in `results/` is touched or overwritten.'''

cells = [["markdown", BANNER]] + cells

# index shift: everything below is +1 relative to the raw base-cell indices
# since we prepended the banner. Recompute the ones we need to touch.
IDX = {"config": 3, "build_graphs": 19, "cell36_equiv": 37}
# sanity: base cell 2 (config) -> index 3 after prepend; base cell 18 (build
# graphs) -> index 19; base cell 36 -> index 37. Verify before editing.
assert cells[IDX["config"]][0] == "code" and "RESULTS_DIR" in cells[IDX["config"]][1]
assert cells[IDX["build_graphs"]][0] == "code" and "build_graphs_parallel" in cells[IDX["build_graphs"]][1]
assert cells[IDX["cell36_equiv"]][1].strip() == E.CELL36_OLD.strip()

replace_cell(cells, IDX["config"],
            'RESULTS_DIR    = PROJECT_ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True)',
            'RESULTS_DIR    = PROJECT_ROOT / "results_pchic_only_validations"; RESULTS_DIR.mkdir(exist_ok=True)')

replace_cell(cells, IDX["build_graphs"], E.CELL18_OLD_BUILD_CALL, E.CELL18_NEW_BUILD_CALL)

replace_cell(cells, IDX["cell36_equiv"], E.CELL36_OLD, E.CELL36_NEW)

# drop the two broken interactive-debug cells (37, 38 pre-prepend -> 38, 39)
drop_cells(cells, [38, 39])
assert "FIND THE 9 GRAPHS" not in "".join(s for _, s in cells)
assert "DIAGNOSTIC" not in "".join(s for k, s in cells if k == "code" and "non-finite content" in s)

append_cells(cells, [
    ("markdown", E.PART3_MD_INTRO),
    ("code", E.PART3_CODE_SETUP),
    ("markdown", E.PART3_MD_CHROM),
    ("code", E.PART3_CODE_CHROM),
    ("code", E.PART3_CODE_CHROM_TRAIN),
    ("markdown", E.PART3_MD_LOTO),
    ("code", E.PART3_CODE_LOTO),
    ("markdown", E.PART3_MD_BLOOD),
    ("code", E.PART3_CODE_BLOOD),
    ("markdown", E.PART3_MD_COMPARE),
    ("code", E.PART3_CODE_COMPARE),
])

build_ipynb(cells, ROOT / "pangenomic_gat_pipeline_pchic_only_validations.ipynb",
           title="Pangenomic AI - PCHi-C only + validation strategies")
