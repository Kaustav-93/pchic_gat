# Four new notebooks, derived from `pangenomic_gat_pipeline_pchic_only.ipynb`

All four are complete, standalone, syntax-checked (`ast.parse`'d, not
executed) `.ipynb` files. None have been run — you run them. Each writes to
its own results folder so nothing collides with the original `results/` or
with each other.

| # | Notebook | What's different | Output folder |
|---|---|---|---|
| 1 | `pangenomic_gat_pipeline_pchic_only_validations.ipynb` | Same 16,384 bp window/graphs/model. Adds graph disk-caching + Part III (chromosome-block CV, leave-one-tissue-out CV, blood-as-test). | `results_pchic_only_validations/` |
| 2 | `pangenomic_gat_pipeline_1mb_window_fast.ipynb` | 1,000,000 bp promoter window. `fast_mode=True`, 8,000 genes, 5 tissues. Same Part III validations as #1. | `results_1mb_window_fast/` |
| 3 | `pangenomic_gat_pipeline_1mb_window_full.ipynb` | Same as #2 but `fast_mode=False` — every gene/tissue. **Very expensive, read its banner cell first.** | `results_1mb_window_full/` |
| 4 | `pangenomic_gat_pipeline_trimodal.ipynb` | Epigenomic state (histone/CTCF) becomes a third input modality with its own encoder (`EpiEncoder`) and embedding (`z_epi`), 3-way InfoNCE alignment, and a variant-count-agnostic `GatedFusion` replacing the two-tower FiLM. Same 16,384 bp window. | `results_trimodal_fast/` or `results_trimodal_full/` (auto-picked from `CFG.fast_mode`) |

**Suggested run order:** #1 first — it reuses the *existing* `cache/` promoter
sequences (identical window/gene set → identical cache key), so graph
construction is the only new cost. Then #2 (1 Mb fast) to sanity-check the
new window size cheaply. #3 (1 Mb full) and #4 (trimodal) can run
independently after that, in either order.

## Read each notebook's own banner cell (cell 1) first

Every notebook opens with a "Variant note" markdown cell explaining exactly
what changed and why, including the specific engineering tradeoffs made
(sequence-bank memory handling at 1 Mb, the fusion redesign for 3
modalities, etc.) — that is the authoritative description, not this file.

## Things worth knowing before you hit Run All

- **Graph caching.** All four notebooks save the built `GRAPHS`/`META` to
  `cache/graphs_<config-hash>.pt` and reload from there on any rerun with
  the same settings — the ~350k-graph build only happens once per config.
- **1 Mb window memory.** The sequence bank is kept `uint8` end-to-end and
  cast to `float32` only per training batch (not once for the whole gene
  set) — this is a real change from the base notebook, not just a config
  tweak, and is necessary at this window size. A preflight cell prints the
  expected RAM footprint before extraction starts. Full mode is on the
  order of 80 GB for the one-hot array; fast mode (8,000 genes) is roughly
  a third of that.
- **`pangenomic_gat_pipeline_1mb_window_full.ipynb` is the most expensive
  file in this set** — a ~4x longer effective sequence than the 16 kb
  baseline, 3 ablation arms x 5 folds, then chromosome-block CV, up to 46
  leave-one-tissue-out runs, and the blood holdout. Its banner cell says
  which two config knobs (`CFG.variants`, `LOTO_TISSUES`) to trim first if
  that's more compute than you want in one run.
- **Leave-one-tissue-out (all three of #1/#2/#3)** loops over every tissue
  in `TISSUE_LIST` by default — one full training run per tissue. Edit the
  `LOTO_TISSUES` list at the top of that cell to a subset if needed; it's
  called out again right there.
- **H4 (tissue classification) is not meaningful under leave-one-tissue-out**
  — the held-out class was never a training target. It's still reported
  (for completeness) but should not be read as a result.
- **H3 (τ, tissue-restriction) under any tissue-holdout scheme** (LOTO,
  blood-as-test) is a per-gene label, so a gene can appear in training via
  its non-held-out tissues — read those H3 numbers as an upper bound, not
  a clean generalisation test. This is explained in each notebook's Part
  III intro, not hidden.
- **Trimodal fusion is not the original FiLM formula.** `GatedFusion`
  generalises to 1–3 active towers; for the seq+str-only arm it is a
  different (not worse, not validated against, just different) mechanism
  than the base notebook's `FiLMFusion`. Documented in the trimodal
  notebook's banner and in `GatedFusion`'s own docstring-comment.
- **Trimodal Tower C overlaps Tower B's information.** Tower B (structure)
  still sees histone/CTCF signal on every node; Tower C (epigenomic) sees
  the same signal, anchor-only, summarised, as its own modality. That's a
  deliberate scope decision (keeps the existing graph cache valid) — the
  harder ablation (strip histone from Tower B entirely) is described as a
  one-line follow-up in the banner cell but not implemented here.
- Two broken, one-off interactive debugging cells that were left in the
  original notebook (they reference variables that only exist if a
  previous, failed run is still in kernel memory — they raise `NameError`
  on a clean restart) were **not** carried into any of these four. Nothing
  else from the original pipeline was removed.

## How these were built

`_nb_base_cells.py` is a verbatim cell-by-cell dump of
`pangenomic_gat_pipeline_pchic_only.ipynb`. `_nb_tools.py` and
`_nb_edits.py` hold shared helpers and the reusable Part III validation
cells. `_build_pchic_only_validations.py`, `_build_1mb_window.py`, and
`_build_trimodal.py` each assemble one (or two) of the notebooks above from
that base plus targeted, anchor-verified edits, and `ast`-check every code
cell before writing the `.ipynb` — same convention the original
`_build_notebook.py` / `_nb_cells_part*.py` already used in this repo.
Re-run any of the three build scripts after editing its edits to regenerate
that notebook.
