# Archive bundle -- results_trimodal_1mb_h2signal_fast

Generated 2026-09-19 17:04:10 by
notebooks/pangenomic_gat_pipeline_trimodal_1mb_manuscript_completeness.ipynb

## Contents
- config.json -- the exact Config this run used (see asdict(CFG))
- checkpoints/model_fold*.pt -- the primary (all3) arm's trained fold weights
- results/*.csv -- every headline, ablation, stratification, SOTA-comparison,
  and new-analysis (Parts V-X) CSV this run produced
- MANIFEST.json -- sha256 + byte size of every file above, for integrity
  verification after transfer/download

## To archive with a DOI
1. Zip this directory: `zip -r archive_bundle.zip archive_bundle/`
2. Upload to https://zenodo.org/deposit/new (or use their API) -- Zenodo
   mints a DOI on publish and versions the deposit on any later re-upload.
3. Cite the resulting DOI (not the GitHub URL alone) in the manuscript's
   Data Availability section.

## To verify integrity after transfer
```python
import json, hashlib
m = json.load(open("MANIFEST.json"))
for f in m["files"]:
    h = hashlib.sha256(open(f["path"], "rb").read()).hexdigest()
    assert h == f["sha256"], f"MISMATCH: {f['path']}"
print("all files verified")
```
