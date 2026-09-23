#!/bin/bash
# download_subset_data.sh
# Selective download: only histone and Hi-C for the 46 tissues in pchic_saikat

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "========================================="
echo "Selective Data Download for pchic_saikat"
echo "========================================="
echo ""
echo "This script downloads ONLY:"
echo "  - Histone ChIP-seq bigWigs for the 46 tissues in interection_data/"
echo "  - Hi-C matrices for those same tissues (where available)"
echo ""
echo "Estimated download: 10-25 GB (many tissues lack public Hi-C/histone)"
echo ""

# Verify symlinks exist
if [ ! -L "interection_data" ]; then
    echo "ERROR: interection_data symlink not found"
    echo "Run: ln -sfn 'pchic_saikat/Gene_Annotated_PCHIC_Loops_Selected BEST TO USE' interection_data"
    exit 1
fi

if [ ! -L "rna_seq" ]; then
    echo "ERROR: rna_seq symlink not found"
    echo "Run: ln -sfn pchic_saikat/rnaseq_results rna_seq"
    exit 1
fi

# Verify conda env
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "pangenomic-gat" ]; then
    echo "ERROR: Must run in pangenomic-gat conda environment"
    echo "Run: conda activate pangenomic-gat"
    exit 1
fi

# Count tissues
N_TISSUES=$(ls interection_data/*_merged_loop.txt 2>/dev/null | wc -l)
echo "Found $N_TISSUES tissue loop files in interection_data/"
echo ""

# ============================================
# 1. Download Histone bigWigs
# ============================================
echo "Step 1: Downloading histone ChIP-seq bigWigs from ENCODE..."
echo "------------------------------------------------------------"
echo "This will:"
echo "  - Query ENCODE for H3K4me1, H3K4me3, H3K27ac, H3K27me3, H3K36me3, H3K9me3, CTCF"
echo "  - Download only tissues with PCHi-C loops (auto-detected)"
echo "  - Prefer hg19; fall back to GRCh38 when hg19 unavailable"
echo "  - Skip tissues where ENCODE has no matching biosample"
echo ""

read -p "Download histone bigWigs? (y/n) [y]: " -n 1 -r HIST_REPLY
echo ""
HIST_REPLY=${HIST_REPLY:-y}

if [[ $HIST_REPLY =~ ^[Yy]$ ]]; then
    python codes/fetch_histone_encode.py \
        --interactions interection_data \
        --out histone \
        --assemblies hg19 GRCh38
    echo ""
    echo "Histone download complete. Check histone/DOWNLOAD_manifest.csv for details."
else
    echo "Skipping histone download."
fi

echo ""
echo ""

# ============================================
# 2. Download Hi-C matrices
# ============================================
echo "Step 2: Downloading Hi-C matrices from ENCODE and 4DN..."
echo "--------------------------------------------------------"
echo "This will:"
echo "  - Query ENCODE and 4D Nucleome portals"
echo "  - Download .hic files for tissues with available data"
echo "  - Keep stub placeholders for tissues without Hi-C"
echo "  - Skip files larger than 10 GB (configurable with --max-gb)"
echo ""
echo "Note: Many tissues lack public Hi-C. This is normal and expected."
echo "      The notebooks mask missing Hi-C edges (has_hic=0) rather than failing."
echo ""

read -p "Download Hi-C matrices? (y/n) [y]: " -n 1 -r HIC_REPLY
echo ""
HIC_REPLY=${HIC_REPLY:-y}

if [[ $HIC_REPLY =~ ^[Yy]$ ]]; then
    python codes/fetch_hic.py \
        --interactions interection_data \
        --out hic \
        --sources encode 4dn \
        --max-gb 10
    echo ""
    echo "Hi-C download complete. Check hic/HIC_manifest.csv for details."
else
    echo "Skipping Hi-C download."
fi

echo ""
echo ""

# ============================================
# Summary
# ============================================
echo "========================================="
echo "Download Summary"
echo "========================================="
echo ""
echo "Tissues scanned: $N_TISSUES"
echo ""
echo "Histone bigWigs:"
ls histone/*.bw 2>/dev/null | wc -l | xargs echo "  Downloaded:"
echo "  Manifest: histone/DOWNLOAD_manifest.csv"
echo ""
echo "Hi-C matrices:"
ls hic/*.hic 2>/dev/null | wc -l | xargs echo "  Downloaded:"
echo "  Manifest: hic/HIC_manifest.csv"
echo ""
echo "Next steps:"
echo "  1. conda activate pangenomic-gat"
echo "  2. jupyter lab notebooks/pangenomic_exploratory_analysis.ipynb"
echo "     (or any other notebook — data is ready)"
echo ""
echo "Notebooks that can run NOW with pchic_saikat data:"
echo "  - pangenomic_exploratory_analysis.ipynb (best for exploration)"
echo "  - pangenomic_gat_pipeline_pchic_only.ipynb"
echo "  - pangenomic_gat_pipeline_pchic_only_validations.ipynb"
echo "  - pangenomic_gat_pipeline_1mb_window_fast.ipynb"
echo "  - pangenomic_gat_pipeline_1mb_window_full.ipynb"
echo ""
echo "Missing histone/Hi-C for some tissues is EXPECTED and handled by the code."
echo ""
