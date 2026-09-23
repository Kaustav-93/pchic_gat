# pchic_saikat Data Integration Guide

**Date:** September 7, 2026  
**Status:** ✅ Data mapped, symlinks created, ready to run

---

## Summary

The `pchic_saikat/` folder contains **46 tissue-specific PCHi-C loop files** and **RNA-seq expression matrices** that are the exact dataset the notebooks in this repository were designed for. With minimal path configuration (symlinks created), the notebooks can now run using this data.

---

## Data Inventory

### 1. PCHi-C Loop Files
**Location:** `pchic_saikat/Gene_Annotated_PCHIC_Loops_Selected BEST TO USE/`

- **46 files** covering diverse tissues and cell types
- **Format:** Tab-separated, 12 columns per file
- **Naming:** `{4-digit ID}_{Tissue Name}_merged_loop.txt`
- **Total interactions:** 4,927,322 loops
  - Cis (same chromosome): 4,897,198 (99.4%)
  - Trans (different chromosomes): 30,124 (0.6%)

**Columns:**
```
bait_frag, other_frag, N_reads, score, 
bait_gene, bait_gene_strand, oe_gene, oe_gene_strand,
bait_closest_gene, bait_closest_gene_strand, 
oe_closest_gene, oe_closest_gene_strand
```

**Top tissues by loop count:**
- Blood - Non-activated CD4+ Cells: 215,687
- Blood - Total CD4+ Cells: 210,318
- Cells - Embryonic Stem Cell: 208,886
- Blood - Macrophages: 205,582
- Blood - Naive CD8+ Cells: 205,030

**Smallest datasets:**
- Ovary: 7,215
- Muscle - Skeletal: 8,791
- Small Intestine: 12,843

### 2. RNA-seq Expression Data
**Location:** `pchic_saikat/rnaseq_results/`

Three matrices with identical dimensions:
- `rnaseq.rsem_genes_tpm.txt` (TPM normalized)
- `rnaseq.rsem_genes_expected_count.txt` (raw counts)
- `rnaseq.rsem_genes_fpkm.txt` (FPKM normalized)

**Dimensions:**
- **Rows:** 57,420 genes (ENSEMBL IDs + gene symbols)
- **Columns:** 196 total (gene_id, gene_name, + 194 samples)

**Sample sources:**
- GSE86189: 110 samples
- GSE128234: 36 samples (Uterus)
- EGAD00001002268: 30 samples (Blood cell types)
- GSE113483: 10 samples (Brain)
- GSE133126: 6 samples (Embryonic stem cells)
- GSE108316: 2 samples (Bone marrow)

**Sample naming:** `{ID}_{GEO accession}_{Tissue}_{replicate}`

---

## Configuration Changes Made

### Symlinks Created
```bash
cd /home/subhadip/pchic_gat
ln -sfn "pchic_saikat/Gene_Annotated_PCHIC_Loops_Selected BEST TO USE" interection_data
ln -sfn pchic_saikat/rnaseq_results rna_seq
```

These symlinks point the notebooks at your pchic_saikat data without modifying any notebook code.

**Verification:**
```bash
ls interection_data/*.txt | wc -l  # Should show 46
ls rna_seq/*.txt                    # Should show 3 matrices
```

---

## Notebooks Ready to Run

### ✅ Can Run NOW (PCHi-C + RNA-seq only)

1. **`pangenomic_exploratory_analysis.ipynb`** ⭐ **BEST STARTING POINT**
   - Requires: Loop files + TPM matrix only
   - Optional: Histone bigWigs (skipped if missing), sequence cache
   - No genome or Hi-C required
   - Outputs: PCA, UMAP, t-SNE, clustering analysis

2. **`pangenomic_gat_pipeline_pchic_only.ipynb`**
   - The main GAT model for PCHi-C-only graphs
   - Requires: Loop files + TPM + `reference/hg19.2bit` + `refGene_hg19.txt` ✅ (already present)
   - Histone masked if missing (safe to run)
   - No Hi-C required

3. **`pangenomic_gat_pipeline_pchic_only_validations.ipynb`**
   - Same as above + extra validation (chromosome-block CV, leave-one-tissue-out, blood holdout)

4. **`pangenomic_gat_pipeline_1mb_window_fast.ipynb`**
   - Fast mode: 8,000 genes, 5 tissues, 1 Mb promoter windows

5. **`pangenomic_gat_pipeline_1mb_window_full.ipynb`**
   - Full mode: all genes/tissues, 1 Mb windows (~80 GB RAM required)

### ⚠️ Can Run But Missing Modalities

6. **`pangenomic_gat_pipeline.ipynb`**
   - Requires Hi-C matrices (`hic/*.hic`)
   - Currently only manifest exists; edges will be masked
   - Still functional but misses Hi-C features

7. **`pangenomic_gat_pipeline_trimodal.ipynb`**
   - Requires histone bigWigs as a third modality (Tower C)
   - Currently only manifests exist
   - Will start but epigenomic tower is empty/masked

---

## Downloading Histone & Hi-C (Optional)

### Why Selective Download?
Downloading all public Hi-C and histone data would be several TB. The provided scripts **already filter automatically** to only the 46 tissues you have in `interection_data/`.

### Estimated Sizes
- **Histone bigWigs:** ~2-10 GB (many tissues lack public ChIP-seq)
- **Hi-C matrices:** ~5-15 GB (most tissues lack public Hi-C)
- **Total realistic:** 10-25 GB

### Quick Download Script
```bash
conda activate pangenomic-gat
bash download_subset_data.sh
```

This script:
1. Queries ENCODE for histone ChIP-seq (7 marks × your 46 tissues)
2. Queries ENCODE + 4DN for Hi-C matrices (where available)
3. Downloads only matches for your tissues
4. Skips tissues without public data (expected and handled)

### Manual Download (more control)
```bash
conda activate pangenomic-gat

# Histone only
python codes/fetch_histone_encode.py \
    --interactions interection_data \
    --out histone \
    --assemblies hg19 GRCh38

# Hi-C only
python codes/fetch_hic.py \
    --interactions interection_data \
    --out hic \
    --sources encode 4dn \
    --max-gb 10
```

Check manifests after download:
- `histone/DOWNLOAD_manifest.csv` — shows which tissues matched, assembly, proxy vs exact
- `hic/HIC_manifest.csv` — shows which Hi-C files downloaded

**Missing data is normal:** Many tissues lack public Hi-C or histone tracks. The notebooks mask missing features rather than failing.

---

## Quick Start

### 1. Activate Environment
```bash
conda activate pangenomic-gat
```

If the environment doesn't exist:
```bash
bash setup_env.sh  # Creates env, installs dependencies, verifies
```

### 2. Launch Jupyter
```bash
jupyter lab notebooks/pangenomic_exploratory_analysis.ipynb
```

**In Jupyter:** Kernel → Change Kernel → "Python (pangenomic-gat)"

### 3. Run the Notebook
The exploratory analysis notebook:
- Loads topology from PCHi-C loops
- Loads expression labels from TPM
- Builds multiple representations (topology, k-mer, expression, histone if available)
- Runs PCA, UMAP, t-SNE, clustering
- Compares class separability across representations

**Fast mode is enabled by default** — adjust `N_TISSUES` and `N_SUBSAMPLE` in the notebook config cell if you want full-scale analysis.

---

## Tissue Coverage

All 46 tissues in your dataset:

**Blood/Immune (16 tissues):**
- Blood - Erythroblast, Neutrophils, Monocytes, Macrophages, Megakaryocytes
- Blood - Naive/Total CD4+ T Cells, Naive/Total CD8+ T Cells, Activated/Non-activated CD4+
- Blood - Naive/Total B Cells, Lymphoblast
- Thymus, Spleen, Bone Marrow

**Brain/Neural (4 tissues):**
- Brain - Frontal Cortex (BA9), Hippocampus
- Cells - Astrocyte, Neuron, Neuroectodermal Cell

**Stem/Progenitor (3 tissues):**
- Cells - Embryonic Stem Cell, Mesenchymal Stem Cell
- Embryo - Trophoblast, Mesendoderm

**Major Organs (12 tissues):**
- Heart - Left Ventricle, Lung, Liver, Pancreas, Kidney
- Stomach, Small Intestine, Uterus, Ovary, Bladder
- Adrenal Gland, Nerve - Tibial

**Tissue/Skin (5 tissues):**
- Muscle - Skeletal, Adipose - Subcutaneous
- Skin - Not Sun Exposed, Cervix - Ectocervix
- Artery - Aorta

**Cell Lines (6 tissues):**
- Cells - Cultured fibroblasts, Lymphoblastoid Cell, Leukocyte, Endothelial Cell

---

## Data Quality Notes

### PCHi-C Loops
- **Gene annotation coverage:** 85-100% (most tissues >99% for bait gene)
- **Interaction distances (median cis):**
  - Short range (<100kb): Neurons, Ovary, Muscle
  - Medium (100-300kb): Most blood/immune
  - Long (>300kb): Embryonic stem cells, Thymus
- **Score ranges:** 2.0 to ~2000, median 2.5-8.5 depending on tissue
- **Assembly:** hg19 (verified by notebooks)

### RNA-seq
- **43 unique tissue IDs** map to the PCHi-C loop files
- **30 unique tissue names** (some IDs are biological replicates)
- Good overlap for blood, embryonic tissues, major organs
- Some PCHi-C tissues lack RNA-seq (e.g., some brain subtypes)

---

## Troubleshooting

### "No *_merged_loop.txt found"
```bash
# Verify symlink
ls -l interection_data
# Should point to: pchic_saikat/Gene_Annotated_PCHIC_Loops_Selected BEST TO USE
```

### "rnaseq.rsem_genes_tpm.txt missing"
```bash
# Verify symlink
ls -l rna_seq
# Should point to: pchic_saikat/rnaseq_results
```

### "Must run in pangenomic-gat environment"
```bash
conda activate pangenomic-gat
jupyter lab  # Restart Jupyter from activated env
```

### Notebook refuses to run
- Cell 2 checks the conda environment — it must be `pangenomic-gat`
- Change kernel: Kernel → Change Kernel → Python (pangenomic-gat)

---

## Next Steps

1. **Start with exploratory analysis** to understand the data structure
2. **Run pchic_only pipeline** for the full GAT model (sequence + structure towers)
3. **Optionally download histone/Hi-C** for richer features (but models run without them)
4. **Try validation notebooks** for chromosome-block CV and leave-one-tissue-out

---

## References

- **Project docs:** `README.md`, `PROJECT_SUMMARY.txt`, `MODEL_ARCHITECTURE.md`
- **Notebook variants:** `NEW_NOTEBOOKS_README.md`
- **Data download:** `download_data.sh` (full), `download_subset_data.sh` (selective)
- **Fetch scripts:** `codes/fetch_histone_encode.py`, `codes/fetch_hic.py`

---

**Status:** ✅ Ready to run. Data is integrated, symlinks are set, notebooks can access pchic_saikat files.
