# Cells 0-13 of the pipeline notebook: config -> data -> graphs -> labels -> splits
CELLS = []
def MD(s):   CELLS.append(("markdown", s.strip("\n")))
def CODE(s): CELLS.append(("code",     s.strip("\n")))

# =============================================================================
MD(r"""
# Pangenomic AI — promoter-centred graph attention network

This notebook implements, end to end, the architecture from the pipeline slide:

```
tissue-specific PCHi-C / Hi-C subgraphs        promoter DNA sequence
                |                                        |
        STRUCTURE TOWER (GATv2, L layers x K heads)   SEQUENCE TOWER (CNN)
                |                                        |
             z_str  (per gene x tissue)              z_seq  (per gene)
                \                                        /
                 \------  contrastive alignment  -------/
                                  |
                        FiLM fusion  ->  z(g,t)
                                  |
    +--------------+--------------+--------------+--------------+
    | H1 link pred | H2 expression| H3 tissue cls| H4 embedding  |
    +--------------+--------------+--------------+--------------+
```

**Data this notebook consumes** (all paths configured in the next cell):

| folder | contents | role |
|---|---|---|
| `interection_data/` | 46 `*_merged_loop.txt` PCHi-C loop files | graph edges |
| `histone/` | `{sample}.{mark}.bw` bigWigs | node features |
| `rna_seq/` | RSEM TPM / FPKM / counts matrices | expression labels |
| `hic/` | `{sample}.hic` (placeholders for now) | extra edge feature |
| `reference/` | `hg19.2bit`, `refGene_hg19.txt` | promoter sequence + TSS |

**Four supervised heads**, all evaluated with 5-fold cross-validation:

1. **H1 link prediction** — is this promoter–fragment contact real? (binary)
2. **H2 expression** — is this gene expressed in this tissue? (binary, TPM > 1)
3. **H3 tissue-specificity** — is this gene tissue-restricted or housekeeping? (binary, tau index)
4. **H4 tissue classification** — which tissue is this graph from? (multiclass)

---

### Before you run anything

```bash
bash setup_env.sh          # builds the conda env and verifies every dependency
bash download_data.sh      # hg19.2bit, refGene, ENCODE histone, dummy hic
conda activate pangenomic-gat
jupyter lab pangenomic_gat_pipeline.ipynb
```

Then **Kernel → Change Kernel → `Python (pangenomic-gat)`**. Cell 2 refuses to
continue if you are in any other environment, because a half-satisfied
dependency set produces silently wrong features rather than a clean error.

### Runtime

`FAST_MODE = True` (the default in the next cell) subsamples to a few tissues
and a few thousand genes so the whole notebook runs in ~10–20 min on a laptop
CPU. Set it to `False` for the full 46-tissue run, which wants a GPU and a few
hours. Every cell prints its outputs and shapes so you can see what changed.
""")

# =============================================================================
MD(r"""
## 1 · Configuration

Everything tunable lives here so that no magic numbers are buried further down.
A few choices worth flagging:

- **`GENOME_BUILD = "hg19"`** — the fragment coordinates in your loop files are
  GRCh37. Cell 6 *verifies* this rather than trusting it: it checks that
  annotated genes actually fall inside their own bait fragments, and aborts if
  the hg38 hypothesis fits better.
- **`PROMOTER_WINDOW = 2000`** — ±1 kb around the canonical TSS. Long enough to
  carry core-promoter and proximal-element signal, short enough that the CNN
  tower stays cheap.
- **`MAX_NEIGHBOURS = 32`** — promoter subgraphs are truncated to the 32
  highest-CHiCAGO-scoring contacts. Degree is extremely heavy-tailed (a few
  baits have hundreds of partners) and unbounded degree makes batching
  pathological. The cut is by score, so we keep the most confident contacts.
- **`TPM_THRESHOLD = 1.0`** — the standard "expressed" cut for RSEM TPM.
- **`TAU_THRESHOLD = 0.85`** — Yanai's tau above this is conventionally called
  tissue-restricted.
- **`GROUP_SPLIT_BY = "gene"`** — folds are split by *gene*, not by row. If the
  same gene appeared in train and test (in different tissues) the model could
  memorise gene identity and the metrics would be inflated. This is the single
  most important line in the notebook for making the numbers mean anything.
""")

CODE(r"""
from pathlib import Path
import os, sys, json, math, random, warnings, time
from dataclasses import dataclass, asdict, field

# --------------------------------------------------------------- paths
def _find_project_root(start: Path) -> Path:
    # Notebooks live in notebooks/, one level below the repo root, so
    # Path.cwd() (Jupyter sets it to the notebook's own directory) is no
    # longer the project root by itself. Walk upward for the marker that
    # identifies the root instead of assuming a fixed number of parent hops.
    for p in (start, *start.parents):
        if (p / "setup_env.sh").exists():
            return p
    return start

PROJECT_ROOT   = Path(os.environ.get("PANGENOMIC_ROOT", _find_project_root(Path.cwd()))).resolve()
INTERACT_DIR   = PROJECT_ROOT / "interection_data"
HISTONE_DIR    = PROJECT_ROOT / "histone"
RNASEQ_DIR     = PROJECT_ROOT / "rna_seq"
HIC_DIR        = PROJECT_ROOT / "hic"
REF_DIR        = PROJECT_ROOT / "reference"
CACHE_DIR      = PROJECT_ROOT / "cache";   CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR    = PROJECT_ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR        = RESULTS_DIR / "figures";  FIG_DIR.mkdir(exist_ok=True)

GENOME_2BIT    = REF_DIR / "hg19.2bit"
REFGENE_TXT    = REF_DIR / "refGene_hg19.txt"
TPM_FILE       = RNASEQ_DIR / "rnaseq.rsem_genes_tpm.txt"


@dataclass
class Config:
    # ---- reproducibility
    seed: int = 42

    # ---- scale
    fast_mode: bool = True            # subsample so the notebook runs quickly
    fast_n_tissues: int = 8           # tissues kept when fast_mode
    fast_n_genes: int = 2500          # genes kept when fast_mode

    # ---- genome / promoter
    genome_build: str = "hg19"
    # Long context is the point of the Hyena tower: a dilated CNN needs
    # log2(L) layers to reach 16 kb, a Hyena operator reaches it in one.
    promoter_window: int = 16384      # bp, centred on the canonical TSS
    seq_len_fast: int = 4096          # shorter window in fast_mode

    # ---- graph construction
    max_neighbours: int = 32          # cap on promoter degree (by CHiCAGO score)
    min_chicago: float = 5.0          # standard CHiCAGO significance cut
    add_backbone_edges: bool = True   # GrapHi-C style linear-adjacency edges
    hic_resolution: int = 10000       # bp, for the .hic lookup

    # ---- histone
    histone_marks: tuple = ("H3K4me1", "H3K4me3", "H3K27ac",
                            "H3K27me3", "H3K36me3", "H3K9me3", "CTCF")
    histone_stat: str = "mean"        # bigWig summary over each fragment

    # ---- labels
    tpm_threshold: float = 1.0        # expressed if TPM > this
    tau_threshold: float = 0.85       # tissue-restricted if tau > this
    link_holdout_frac: float = 0.15   # edges held out for the link-prediction head

    # ---- model : structure tower (GATv2)
    d_hidden: int = 128               # d_h, GAT hidden width
    d_embed: int = 128                # d, both embedding spaces and the fusion
    gat_layers: int = 3               # L
    gat_heads: int = 4                # K
    dropout: float = 0.2
    attn_dropout: float = 0.1

    # ---- model : sequence tower (HyenaDNA-style long convolution)
    hyena_d_model: int = 128          # channel width inside the Hyena stack
    hyena_blocks: int = 4             # number of HyenaOperator + SwiGLU blocks
    hyena_order: int = 2              # depth of the gating/convolution recurrence
    hyena_filter_order: int = 64      # width of the implicit filter MLP
    stem_kernel: int = 15             # single-nucleotide stem conv
    stem_stride: int = 4              # >1 only to bound FFT cost; 1 = base resolution
    rc_equivariant: bool = True       # average forward and reverse-complement

    # ---- ablation
    variants: tuple = ("both", "gat", "seq")   # which arms to train
    run_hybrid: bool = True           # DHL-style classical heads on the embeddings

    # ---- loss weights  (H1 link, H2 expression, H3 tau, H4 tissue, alignment)
    w_link: float = 1.0
    w_expr: float = 1.0
    w_tau: float = 0.5
    w_tissue: float = 0.5
    w_align: float = 0.3
    temperature: float = 0.1          # InfoNCE temperature

    # ---- training
    n_folds: int = 5
    group_split_by: str = "gene"      # 'gene' | 'chrom' | 'none'
    epochs: int = 40
    epochs_fast: int = 12
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8                 # early stopping on validation loss
    num_workers: int = 0

    @property
    def seq_len(self) -> int:
        return self.seq_len_fast if self.fast_mode else self.promoter_window

    @property
    def n_epochs(self) -> int:
        return self.epochs_fast if self.fast_mode else self.epochs


CFG = Config()

# --------------------------------------------------------------- seeding
def set_all_seeds(seed: int):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as _np; _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _t
        _t.manual_seed(seed); _t.cuda.manual_seed_all(seed)
        _t.backends.cudnn.deterministic = True
        _t.backends.cudnn.benchmark = False
    except ImportError:
        pass

set_all_seeds(CFG.seed)
warnings.filterwarnings("ignore", category=FutureWarning)

print("PROJECT_ROOT :", PROJECT_ROOT)
print("mode         :", "FAST (subsampled)" if CFG.fast_mode else "FULL")
print("promoter win :", CFG.seq_len, "bp")
print("epochs       :", CFG.n_epochs)
print()
for k, v in asdict(CFG).items():
    print(f"  {k:<20} {v}")
""")

# =============================================================================
MD(r"""
## 2 · Environment guard and imports

This cell does three things and will stop the notebook if any of them fails.

1. **Confirms the kernel is the `pangenomic-gat` conda env.** A notebook run in
   the wrong interpreter is the most common cause of "it worked yesterday":
   you get a *different* `torch_geometric`, or `pyBigWig` is missing and the
   histone features silently become zeros. Better to fail loudly on cell 2.
2. **Imports everything up front** and prints versions, so the run is
   self-documenting — the printed output is a record of exactly what produced
   the figures below.
3. **Picks a device**, preferring CUDA, then Apple MPS, then CPU.

Optional packages (`hicstraw`, `umap`, `torch_scatter`) are imported inside
`try` blocks and set an availability flag; the pipeline degrades in a way that
is announced rather than hidden.
""")

CODE(r"""
import importlib, platform, subprocess

EXPECTED_ENV = "pangenomic-gat"
active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
print(f"python      : {platform.python_version()}")
print(f"executable  : {sys.executable}")
print(f"conda env   : {active_env or '(none detected)'}")

if EXPECTED_ENV not in sys.executable and active_env != EXPECTED_ENV:
    raise RuntimeError(
        f"\n\nThis notebook must run inside the '{EXPECTED_ENV}' conda env.\n"
        f"  Currently: {sys.executable}\n\n"
        f"  Fix:  bash setup_env.sh\n"
        f"        conda activate {EXPECTED_ENV}\n"
        f"        jupyter lab   # then Kernel > Change Kernel > Python ({EXPECTED_ENV})\n"
    )

# ---------------------------------------------------------------- hard deps
import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_networkx, softmax as pyg_softmax

import pyBigWig
import py2bit

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score, jaccard_score,
    brier_score_loss, log_loss, classification_report,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

HARD = {
    "numpy": np.__version__, "pandas": pd.__version__,
    "scipy": sstats.__name__ and __import__("scipy").__version__,
    "matplotlib": matplotlib.__version__, "seaborn": sns.__version__,
    "networkx": nx.__version__, "torch": torch.__version__,
    "torch_geometric": torch_geometric.__version__,
    "pyBigWig": getattr(pyBigWig, "__version__", "n/a"),
    "py2bit": getattr(py2bit, "__version__", "n/a"),
    "scikit-learn": __import__("sklearn").__version__,
}
print("\nrequired packages")
for k, v in HARD.items():
    print(f"  OK    {k:<18} {v}")

# ---------------------------------------------------------------- soft deps
OPTIONAL = {}
def _try(name, attr="__version__"):
    try:
        m = importlib.import_module(name)
        OPTIONAL[name] = getattr(m, attr, "present")
        return m
    except Exception as e:
        OPTIONAL[name] = f"UNAVAILABLE ({type(e).__name__})"
        return None

hicstraw   = _try("hicstraw")
pyliftover = _try("pyliftover")
umap_mod   = _try("umap")
_          = _try("torch_scatter")
HAS_HICSTRAW   = hicstraw is not None
HAS_PYLIFTOVER = pyliftover is not None
HAS_UMAP     = umap_mod is not None

print("\noptional packages")
for k, v in OPTIONAL.items():
    print(f"  {'OK   ' if 'UNAVAILABLE' not in str(v) else 'SKIP '} {k:<18} {v}")

# ---------------------------------------------------------------- device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(0)
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps"); dev_name = "Apple MPS"
else:
    DEVICE = torch.device("cpu"); dev_name = platform.processor() or "cpu"
print(f"\ndevice      : {DEVICE}  ({dev_name})")

# ---------------------------------------------------------------- plotting
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})
PALETTE = ["#00A38C", "#E08A2E", "#D14A73", "#5B85D6", "#7C6BAD", "#4C9A5A"]
sns.set_palette(PALETTE)

def savefig(fig, name):
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"  figure -> {p.relative_to(PROJECT_ROOT)}")
    return p

print("\nenvironment ready")
""")

# =============================================================================
MD(r"""
## 3 · Data inventory

Before parsing anything we take stock of what is actually on disk. This cell is
pure reconnaissance — it opens no large files — but it determines the shape of
everything downstream:

- **Which tissues exist**, parsed from the loop file names
  (`0077_Liver_merged_loop.txt` → id `0077`, tissue `Liver`).
- **Histone coverage matrix** — a tissue × mark grid of what is present. This
  is the honest picture of your feature completeness. Any tissue/mark cell that
  is empty here becomes a masked feature later, never an imputed one.
- **RNA-seq sample columns** grouped back to tissue ids, so we know which
  tissues have expression labels at all.
- **`.hic` files** — expected to be placeholders at this stage.

Note the duplicate-mark files in `histone/` (`... (1).bw`). Those are treated as
replicates of the same mark and averaged, not as separate features.
""")

CODE(r"""
import re
from collections import defaultdict

# ------------------------------------------------- tissues from loop filenames
loop_files = sorted(INTERACT_DIR.glob("*_merged_loop.txt"))
if not loop_files:
    raise FileNotFoundError(f"No *_merged_loop.txt in {INTERACT_DIR}")

TISSUES = []            # list of (sample_id, tissue_name, path)
for p in loop_files:
    stem = p.name.replace("_merged_loop.txt", "")
    sid, _, tname = stem.partition("_")
    TISSUES.append((sid, tname, p))

tissue_df = pd.DataFrame(TISSUES, columns=["sample_id", "tissue", "path"])
tissue_df["file_mb"] = [round(p.stat().st_size / 1e6, 1) for p in tissue_df["path"]]
print(f"{len(tissue_df)} tissues with PCHi-C loops\n")
print(tissue_df[["sample_id", "tissue", "file_mb"]].to_string(index=False))

# ------------------------------------------------- histone coverage
hist_files = sorted(HISTONE_DIR.glob("*.bw"))
# sid -> mark -> [(path, assembly)]
# fetch_histone_encode.py writes "{sid}.{mark}.bw" for hg19 and
# "{sid}.{mark}.GRCh38.bw" when only GRCh38 existed on ENCODE. Replicate files
# carry a " (1)" suffix and are averaged, not treated as separate marks.
hist_map = defaultdict(lambda: defaultdict(list))
mark_re = re.compile(
    r"^(?P<sid>\d+)\.(?P<mark>.+?)(?:\s*\(\d+\))?(?:\.(?P<asm>GRCh38|hg38|hg19))?\.bw$")
for p in hist_files:
    m = mark_re.match(p.name)
    if not m:
        print(f"  ! unparsed histone filename: {p.name}")
        continue
    asm = m.group("asm") or "hg19"
    asm = "GRCh38" if asm in ("GRCh38", "hg38") else "hg19"
    hist_map[m.group("sid")][m.group("mark")].append((p, asm))

n_grch38 = sum(1 for sid in hist_map for mk in hist_map[sid]
               for _, a in hist_map[sid][mk] if a == "GRCh38")
cover = pd.DataFrame(
    [[len(hist_map.get(sid, {}).get(mk, [])) for mk in CFG.histone_marks]
     for sid in tissue_df["sample_id"]],
    index=pd.MultiIndex.from_frame(tissue_df[["sample_id", "tissue"]]),
    columns=list(CFG.histone_marks),
)
print(f"\n\n{len(hist_files)} histone bigWigs covering "
      f"{len(hist_map)} sample ids\n")
print("histone coverage (files per tissue x mark; 0 = will be masked)")
print(cover.loc[(cover.sum(axis=1) > 0)].to_string())
n_cells = cover.size
n_filled = int((cover > 0).sum().sum())
print(f"\ncoverage: {n_filled}/{n_cells} tissue x mark cells "
      f"({100*n_filled/n_cells:.1f}%)")
if n_grch38:
    print(f"  {n_grch38} of these tracks are GRCh38 (ENCODE has retired much of")
    print("  its hg19 output). Their queries are lifted hg19 -> GRCh38 in cell 8;")
    print("  the tracks themselves are left alone.")
if n_filled < n_cells:
    print("  -> run  python fetch_histone_encode.py  to pull more from ENCODE;")
    print("     whatever remains missing is masked, not imputed.")

man = HISTONE_DIR / "DOWNLOAD_manifest.csv"
if man.exists():
    mdf = pd.read_csv(man)
    print(f"\ndownload manifest: {len(mdf)} rows")
    if "match_quality" in mdf.columns:
        print(mdf["match_quality"].value_counts().to_string())
        prox = mdf[(mdf.get("status") == "downloaded")
                   & (mdf["match_quality"] == "proxy")]
        if len(prox):
            print(f"  WARNING: {len(prox)} tracks come from a PROXY biosample "
                  "(related but not the same tissue).")
            print("  Treat those marks as noisy; see histone/DOWNLOAD_manifest.csv")

# ------------------------------------------------- rna-seq
tpm_header = pd.read_csv(TPM_FILE, sep="\t", nrows=0)
rna_cols = [c for c in tpm_header.columns if c not in ("gene_id", "gene_name")]
rna_by_tissue = defaultdict(list)
for c in rna_cols:
    rna_by_tissue[c.split("_")[0]].append(c)
print(f"\n\nRNA-seq: {len(rna_cols)} sample columns over "
      f"{len(rna_by_tissue)} tissue ids")
have_rna = tissue_df["sample_id"].isin(rna_by_tissue).sum()
print(f"  {have_rna}/{len(tissue_df)} loop tissues have matching RNA-seq columns")

# ------------------------------------------------- hic
hic_files = sorted(HIC_DIR.glob("*.hic"))
print(f"\n.hic files: {len(hic_files)}")
if hic_files:
    head = hic_files[0].read_bytes()[:24]
    is_placeholder = head.startswith(b"DUMMY_HIC_PLACEHOLDER")
    print(f"  first file looks like a "
          f"{'PLACEHOLDER (expected for now)' if is_placeholder else 'real .hic archive'}")

# ------------------------------------------------- reference
for f, why in [(GENOME_2BIT, "promoter sequence"), (REFGENE_TXT, "canonical TSS")]:
    print(f"\n{f.name:<22} {'present' if f.exists() else 'MISSING'}  ({why})")
    if not f.exists():
        print("   -> bash download_data.sh")
""")

# =============================================================================
MD(r"""
## 4 · Parse the PCHi-C loop files

Each `*_merged_loop.txt` has 12 tab-separated columns:

| column | example | meaning |
|---|---|---|
| `bait_frag` | `chr1,927395,936954` | the promoter-capture bait fragment |
| `other_frag` | `chr1,1000704,1005126` | the distal ("other end") fragment |
| `N_reads` | `8` | supporting read pairs |
| `score` | `5.16` | CHiCAGO score |
| `bait_gene` | `RP11-54O7.17,HES4,ISG15` | genes annotated to the bait |
| `oe_gene` | `RP11-465B22.3` | genes at the other end |
| ...`_strand`, `_closest_gene` | | strand and nearest-gene fallbacks |

Two things are worth being explicit about:

**A bait fragment can carry several genes.** `HindIII`/`MboI` fragments are
kilobases long and often span more than one TSS. We `explode` the comma-list so
each (gene, bait, other-end) triple becomes one row. That means a single
physical contact can support several gene-level graphs — correct, but it also
means rows are *not* independent, which is exactly why the cross-validation
splits by gene later on.

**We filter on CHiCAGO score ≥ 5**, the threshold the CHiCAGO authors recommend
for calling a significant interaction. Everything below is dropped rather than
kept as a weak edge, because sub-threshold contacts are dominated by the
distance-decay background and would teach the attention layer to attend to
proximity rather than regulation.
""")

CODE(r"""
def parse_frag(s: str):
    # 'chr1,927395,936954' -> ('chr1', 927395, 936954)
    c, a, b = s.split(",")
    return c, int(a), int(b)


def load_loops(path: Path, sample_id: str, tissue: str,
               min_score: float) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["score"]   = pd.to_numeric(df["score"], errors="coerce")
    df["N_reads"] = pd.to_numeric(df["N_reads"], errors="coerce")
    n0 = len(df)
    df = df[df["score"] >= min_score].copy()

    for side in ("bait", "other"):
        col = "bait_frag" if side == "bait" else "other_frag"
        parts = df[col].str.split(",", expand=True)
        df[f"{side}_chr"]   = parts[0]
        df[f"{side}_start"] = pd.to_numeric(parts[1], errors="coerce")
        df[f"{side}_end"]   = pd.to_numeric(parts[2], errors="coerce")

    # genes annotated to the bait; fall back to the closest gene when absent
    genes = df["bait_gene"].fillna("").where(
        df["bait_gene"].fillna("").str.len() > 0,
        df["bait_closest_gene"].fillna(""))
    df["gene"] = genes.str.split(",")
    df = df.explode("gene")
    df["gene"] = df["gene"].str.strip()
    df = df[(df["gene"] != "") & df["gene"].notna()]

    df["sample_id"] = sample_id
    df["tissue"]    = tissue
    df["distance"]  = (
        (df["other_start"] + df["other_end"]) / 2
        - (df["bait_start"] + df["bait_end"]) / 2
    ).abs()
    df["same_chrom"] = (df["bait_chr"] == df["other_chr"]).astype(int)
    df.loc[df["same_chrom"] == 0, "distance"] = np.nan   # trans: distance undefined

    keep = ["sample_id", "tissue", "gene",
            "bait_chr", "bait_start", "bait_end",
            "other_chr", "other_start", "other_end",
            "N_reads", "score", "distance", "same_chrom"]
    return df[keep].reset_index(drop=True), n0


# ---- decide which tissues to load -----------------------------------------
sel = tissue_df.copy()
if CFG.fast_mode:
    # prefer tissues that have BOTH histone marks and RNA-seq, then top up
    has_hist = sel["sample_id"].isin(hist_map.keys())
    has_rna  = sel["sample_id"].isin(rna_by_tissue.keys())
    sel = pd.concat([sel[has_hist & has_rna], sel[~(has_hist & has_rna)]])
    sel = sel.head(CFG.fast_n_tissues)
    print(f"FAST_MODE: using {len(sel)} of {len(tissue_df)} tissues\n")

frames, raw_counts = [], {}
for _, r in tqdm(list(sel.iterrows()), desc="parsing loop files"):
    d, n0 = load_loops(r["path"], r["sample_id"], r["tissue"], CFG.min_chicago)
    frames.append(d)
    raw_counts[r["tissue"]] = (n0, len(d))

loops = pd.concat(frames, ignore_index=True)
del frames

print(f"\nparsed {len(loops):,} (gene, bait, other-end) rows "
      f"across {loops['tissue'].nunique()} tissues")
print(f"unique genes           : {loops['gene'].nunique():,}")
print(f"unique bait fragments  : {loops.groupby(['bait_chr','bait_start']).ngroup().nunique():,}")
print(f"cis / trans            : {int(loops['same_chrom'].sum()):,} / "
      f"{int((1-loops['same_chrom']).sum()):,}")
print()
print(loops.head(8).to_string(index=False))

print("\nper-tissue: rows before score filter -> gene-exploded rows after")
summary = pd.DataFrame(
    [(t, a, b) for t, (a, b) in raw_counts.items()],
    columns=["tissue", "raw_loops", "rows_after_filter_and_explode"]
).sort_values("raw_loops", ascending=False)
print(summary.to_string(index=False))

print("\ndistance (cis only) / CHiCAGO score")
print(loops[["distance", "score", "N_reads"]].describe().to_string())
""")

# =============================================================================
MD(r"""
## 5 · Interaction QC plots

Four diagnostics, each of which can invalidate the run if it looks wrong:

1. **Rows per tissue** — wildly uneven depth means the tissue-classification
   head can cheat by learning graph size. We check the spread and, if it is
   extreme, the `MAX_NEIGHBOURS` cap in cell 10 partially equalises it.
2. **CHiCAGO score distribution** — should be a long right tail starting at our
   cut of 5. A spike exactly at the threshold means the upstream caller already
   filtered, and our filter is a no-op.
3. **Contact distance** — the classic PCHi-C profile peaks in the tens-to-
   hundreds of kb. A distribution peaked at the resolution limit suggests the
   file contains raw proximity rather than called loops.
4. **Promoter degree** — how many partners each gene's bait has. This is the
   distribution `MAX_NEIGHBOURS` truncates, so we look at it before choosing.
""")

CODE(r"""
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

# 1. rows per tissue
cnt = loops["tissue"].value_counts().sort_values()
axes[0, 0].barh(cnt.index, cnt.values, color=PALETTE[0])
axes[0, 0].set_title("Gene-exploded interaction rows per tissue")
axes[0, 0].set_xlabel("rows")
axes[0, 0].tick_params(axis="y", labelsize=7)

# 2. CHiCAGO score
axes[0, 1].hist(loops["score"].clip(upper=40), bins=80,
                color=PALETTE[1], log=True)
axes[0, 1].axvline(CFG.min_chicago, ls="--", c="k", lw=1,
                   label=f"cut = {CFG.min_chicago}")
axes[0, 1].set_title("CHiCAGO score (clipped at 40)")
axes[0, 1].set_xlabel("score"); axes[0, 1].set_ylabel("count (log)")
axes[0, 1].legend()

# 3. distance
d = loops.loc[loops["same_chrom"] == 1, "distance"].dropna()
d = d[d > 0]
axes[1, 0].hist(np.log10(d), bins=80, color=PALETTE[2])
axes[1, 0].set_title(f"cis contact distance  (n={len(d):,})")
axes[1, 0].set_xlabel("log10 distance (bp)")
for kb, lbl in [(1e4, "10 kb"), (1e5, "100 kb"), (1e6, "1 Mb")]:
    axes[1, 0].axvline(np.log10(kb), ls=":", c="grey", lw=1)
    axes[1, 0].text(np.log10(kb), axes[1, 0].get_ylim()[1]*0.92, lbl,
                    rotation=90, fontsize=7, ha="right", color="grey")

# 4. promoter degree
deg = loops.groupby(["sample_id", "gene"]).size()
axes[1, 1].hist(deg.clip(upper=100), bins=60, color=PALETTE[3], log=True)
axes[1, 1].axvline(CFG.max_neighbours, ls="--", c="k", lw=1,
                   label=f"MAX_NEIGHBOURS = {CFG.max_neighbours}")
axes[1, 1].set_title("Partners per (gene, tissue)")
axes[1, 1].set_xlabel("degree (clipped at 100)"); axes[1, 1].set_ylabel("count (log)")
axes[1, 1].legend()

fig.suptitle("PCHi-C interaction QC", fontsize=13, y=1.01)
fig.tight_layout(); savefig(fig, "01_interaction_qc"); plt.show()

print(f"degree: median={deg.median():.0f}  mean={deg.mean():.1f}  "
      f"p95={deg.quantile(.95):.0f}  max={deg.max()}")
print(f"fraction of (gene,tissue) pairs truncated by MAX_NEIGHBOURS="
      f"{CFG.max_neighbours}: {(deg > CFG.max_neighbours).mean():.1%}")
""")

# =============================================================================
MD(r"""
## 6 · Gene models, canonical TSS, and an assembly sanity check

The loop files give us gene *symbols* and fragment coordinates, but not
transcription start sites. We need TSS to define the promoter window for the
sequence tower, so we read UCSC `refGene`.

**Canonical TSS rule.** RefSeq lists many transcripts per symbol. We take, per
gene symbol, the transcript with the longest span on the most common chromosome
for that symbol, then its 5' end (`txStart` on `+`, `txEnd` on `-`). This is a
pragmatic choice; if you have a preferred canonical set (MANE Select, APPRIS)
swap it in here and nothing else changes.

**Assembly verification — the important part.** We do not trust the assertion
that these coordinates are hg19. Instead we test it: for a sample of rows, does
the annotated bait gene's hg19 TSS actually fall inside the bait fragment? If
the coordinates were hg38 the hit rate would collapse, because hg19→hg38 shifts
most of chromosome 1 by hundreds of kilobases. A high hit rate confirms hg19; a
low one aborts the notebook rather than letting you extract nonsense sequence.
""")

CODE(r"""
REFGENE_COLS = ["bin", "name", "chrom", "strand", "txStart", "txEnd",
                "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
                "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames"]

rg = pd.read_csv(REFGENE_TXT, sep="\t", header=None, names=REFGENE_COLS,
                 dtype={"chrom": str, "name2": str}, low_memory=False)
print(f"refGene rows: {len(rg):,}")

# keep primary assembly only (drop _alt / _random / chrUn contigs)
rg = rg[rg["chrom"].str.fullmatch(r"chr(\d{1,2}|X|Y)")]
rg["span"] = rg["txEnd"] - rg["txStart"]

# canonical transcript per symbol: longest span, on that symbol's modal chromosome
modal_chrom = rg.groupby("name2")["chrom"].agg(lambda s: s.value_counts().index[0])
rg = rg[rg.apply(lambda r: modal_chrom.get(r["name2"]) == r["chrom"], axis=1)]
canon = (rg.sort_values("span", ascending=False)
           .drop_duplicates("name2")
           .set_index("name2"))
canon["tss"] = np.where(canon["strand"] == "+", canon["txStart"], canon["txEnd"])

GENE_INFO = canon[["chrom", "strand", "txStart", "txEnd", "tss"]].copy()
print(f"canonical models for {len(GENE_INFO):,} gene symbols")
print(GENE_INFO.head().to_string())

# ---------------------------------------------------------- assembly check
sample = loops.drop_duplicates("gene").sample(
    min(4000, loops["gene"].nunique()), random_state=CFG.seed)
sample = sample.join(GENE_INFO, on="gene", how="inner")
inside = (
    (sample["chrom"] == sample["bait_chr"]) &
    (sample["tss"] >= sample["bait_start"] - 5000) &
    (sample["tss"] <= sample["bait_end"] + 5000)
)
hit = inside.mean()
print(f"\nassembly check: {inside.sum():,}/{len(sample):,} sampled genes have their "
      f"hg19 TSS inside (+/-5 kb of) their own bait fragment  ->  {hit:.1%}")

if hit < 0.45:
    raise RuntimeError(
        f"Only {hit:.1%} of bait fragments contain their annotated gene's hg19 TSS.\n"
        "The loop coordinates are probably NOT hg19. Re-run download_data.sh with a\n"
        "matching assembly, or liftOver the fragments, before going further —\n"
        "otherwise every promoter sequence extracted below is from the wrong locus."
    )
print("  -> coordinates are consistent with hg19. Proceeding.")

matched = loops["gene"].isin(GENE_INFO.index)
print(f"\ngene symbols resolvable to a TSS: {matched.mean():.1%} of rows "
      f"({loops.loc[matched,'gene'].nunique():,} unique genes)")
print("unresolved symbols are mostly lncRNA/clone IDs (RP11-*, CTD-*, AC0*)")
print(loops.loc[~matched, "gene"].value_counts().head(8).to_string())

loops = loops[matched].reset_index(drop=True)
print(f"\nkept {len(loops):,} rows with a resolvable TSS")
""")

# =============================================================================
MD(r"""
## 7 · Promoter sequence from hg19

For every surviving gene we cut a `PROMOTER_WINDOW`-bp window centred on the
canonical TSS and one-hot encode it as a `4 x L` matrix (A, C, G, T).

Details that matter:

- **Strand.** For `-` strand genes we reverse-complement, so "upstream" means
  the same thing for every gene and the CNN sees a consistent orientation.
- **`N` and soft-masked bases.** `py2bit` returns lowercase for repeat-masked
  regions; we upper-case them (repeats are real sequence and promoters are
  GC-rich, so masking them would bias the encoder). Genuine `N` becomes an
  all-zero column, which the CNN reads as "no information" rather than a base.
- **Edge clipping.** Windows that run off a chromosome end are zero-padded and
  the gene is flagged, not dropped.
- **Cache.** The one-hot tensor for ~20k genes at 2 kb is ~640 MB in float32,
  so we store it as `uint8` (~160 MB) in a compressed `.npz` and re-use it on
  subsequent runs. Delete `cache/promoter_onehot_*.npz` to force a rebuild.

The GC-content plot afterwards is the sanity check: real promoter windows show
a pronounced GC peak relative to the genome background of ~41%. If your
histogram is centred at 41% you are extracting random genomic sequence, which
means the TSS mapping is wrong.
""")

CODE(r"""
BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")

def one_hot(seq: str) -> np.ndarray:
    arr = np.zeros((4, len(seq)), dtype=np.uint8)
    for i, b in enumerate(seq.upper()):
        j = BASE_IDX.get(b)
        if j is not None:
            arr[j, i] = 1
    return arr


genes_needed = sorted(loops["gene"].unique())
if CFG.fast_mode and len(genes_needed) > CFG.fast_n_genes:
    # keep the genes seen in the most tissues -> denser gene x tissue matrix
    freq = loops.groupby("gene")["sample_id"].nunique().sort_values(ascending=False)
    genes_needed = sorted(freq.head(CFG.fast_n_genes).index)
    loops = loops[loops["gene"].isin(genes_needed)].reset_index(drop=True)
    print(f"FAST_MODE: kept {len(genes_needed):,} most widely-observed genes "
          f"-> {len(loops):,} rows")

L = CFG.seq_len
cache_path = CACHE_DIR / f"promoter_onehot_{CFG.genome_build}_{L}_{len(genes_needed)}.npz"

if cache_path.exists():
    z = np.load(cache_path, allow_pickle=True)
    SEQ_GENES = list(z["genes"]); SEQ_ONEHOT = z["onehot"]; seq_flags = z["flags"]
    print(f"loaded cached promoter sequences: {cache_path.name}")
else:
    if not GENOME_2BIT.exists():
        raise FileNotFoundError(f"{GENOME_2BIT} missing — run: bash download_data.sh")
    tb = py2bit.open(str(GENOME_2BIT))
    chrom_sizes = tb.chroms()

    SEQ_GENES, mats, flags = [], [], []
    n_clip = n_bad = 0
    for g in tqdm(genes_needed, desc="extracting promoters"):
        info = GENE_INFO.loc[g]
        c, tss, strand = info["chrom"], int(info["tss"]), info["strand"]
        if c not in chrom_sizes:
            n_bad += 1; continue
        half = L // 2
        s, e = tss - half, tss + (L - half)
        pad_l = max(0, -s); pad_r = max(0, e - chrom_sizes[c])
        if pad_l or pad_r: n_clip += 1
        try:
            seq = tb.sequence(c, max(0, s), min(chrom_sizes[c], e))
        except Exception:
            n_bad += 1; continue
        seq = "N" * pad_l + seq + "N" * pad_r
        if strand == "-":
            seq = seq.translate(COMP)[::-1]
        oh = one_hot(seq)
        if oh.shape[1] != L:                       # defensive: exact width
            oh = np.pad(oh, ((0, 0), (0, max(0, L - oh.shape[1]))))[:, :L]
        SEQ_GENES.append(g); mats.append(oh); flags.append(int(pad_l or pad_r))
    tb.close()

    SEQ_ONEHOT = np.stack(mats).astype(np.uint8)
    seq_flags = np.array(flags, dtype=np.int32)   # pad-base COUNT, not a boolean -- can exceed 255 at large windows
    np.savez_compressed(cache_path, genes=np.array(SEQ_GENES),
                        onehot=SEQ_ONEHOT, flags=seq_flags)
    print(f"\nextracted {len(SEQ_GENES):,} promoters   "
          f"(clipped at chromosome end: {n_clip}, failed: {n_bad})")
    print(f"cached -> {cache_path.name}")

GENE2SEQ = {g: i for i, g in enumerate(SEQ_GENES)}
print(f"one-hot tensor: {SEQ_ONEHOT.shape}  dtype={SEQ_ONEHOT.dtype}  "
      f"{SEQ_ONEHOT.nbytes/1e6:.0f} MB")

# ---------------------------------------------------------------- QC
gc = (SEQ_ONEHOT[:, 1, :] + SEQ_ONEHOT[:, 2, :]).sum(1) / L
n_frac = 1 - SEQ_ONEHOT.sum(1).mean(1)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].hist(gc, bins=60, color=PALETTE[0])
ax[0].axvline(0.41, ls="--", c="r", lw=1.2, label="genome background 41%")
ax[0].axvline(gc.mean(), ls="-", c="k", lw=1.2, label=f"promoter mean {gc.mean():.1%}")
ax[0].set_title("GC content of extracted promoter windows")
ax[0].set_xlabel("GC fraction"); ax[0].legend()

pos_gc = (SEQ_ONEHOT[:, 1, :] + SEQ_ONEHOT[:, 2, :]).mean(0)
ax[1].plot(np.arange(L) - L // 2, pos_gc, color=PALETTE[2], lw=1)
ax[1].axvline(0, ls="--", c="k", lw=1, label="TSS")
ax[1].set_title("Mean GC by position relative to TSS")
ax[1].set_xlabel("bp from TSS"); ax[1].set_ylabel("GC fraction"); ax[1].legend()
fig.tight_layout(); savefig(fig, "02_promoter_sequence_qc"); plt.show()

print(f"mean GC = {gc.mean():.1%}   (expect ~0.50-0.60 for promoters; "
      f"~0.41 would mean the TSS mapping is wrong)")
print(f"mean N fraction = {n_frac.mean():.4f}")
if gc.mean() < 0.44:
    print("  WARNING: GC is close to genome background. Check the TSS mapping "
          "before trusting the sequence tower.")
""")

# =============================================================================
MD(r"""
## 8 · Histone-mark features from bigWig

Each node in a promoter graph is a restriction fragment, and we summarise every
available ChIP-seq mark over that fragment's span with `pyBigWig.stats(...,
"mean")`.

**Missingness is a first-class feature.** For every (tissue, mark) we emit two
numbers per node:

- the **signal** (0.0 when the mark is unavailable), and
- a **mask bit** — 1 if the track exists for this tissue, 0 if not.

That doubles the histone feature width but it is the honest encoding. Zero-
filling alone would tell the model "this fragment has no H3K27ac", which is a
different and false claim from "we never measured H3K27ac here". With the mask,
the network can learn to route around absent tracks.

**Replicate handling.** Files like `0056.H3K9me3 (1).bw` are averaged with
`0056.H3K9me3.bw` rather than treated as separate marks.

**Transform.** ChIP fold-change is heavy-tailed, so we apply `log1p` and then
clip at the 99.5th percentile per mark before standardising. Without this a
handful of hyper-enriched fragments dominate the scale and the GAT effectively
sees a binary feature.

Signal is extracted once per unique (tissue, fragment) and cached, since the
same fragment recurs in many gene graphs.

### Assembly: we lift the query, not the track

ENCODE has retired most of its hg19 output, so `fetch_histone_encode.py` falls
back to GRCh38 and saves those files as `{sid}.{mark}.GRCh38.bw`. Converting a
whole genome-wide bigWig to hg19 means rebuilding signal at ~3 billion
positions — tens of GB of scratch and minutes per track — when we only ever ask
about a few hundred thousand restriction fragments.

So `CoordLifter` converts each **query interval** hg19 → GRCh38 with
`pyliftover` instead. This is exact for interval lookups and essentially free.
Intervals that fail to lift (a real minority, in assembly-discordant regions)
return no signal for that fragment and are counted, so the failure rate is
visible rather than assumed to be zero.

The same lifter is reused for GRCh38 `.hic` files in the next cell.
""")

CODE(r"""
class CoordLifter:
    # hg19 <-> GRCh38 interval conversion, cached. Used for any track or matrix
    # whose assembly differs from our fragment coordinates.
    def __init__(self, ref_dir):
        self.ref_dir = Path(ref_dir)
        self._lo = {}
        self._cache = {}
        self.n_ok = self.n_fail = 0

    def _get(self, src, dst):
        key = (src, dst)
        if key in self._lo:
            return self._lo[key]
        obj = None
        if HAS_PYLIFTOVER:
            name = {("hg19", "GRCh38"): "hg19ToHg38.over.chain.gz",
                    ("GRCh38", "hg19"): "hg38ToHg19.over.chain.gz"}.get(key)
            chain = self.ref_dir / name if name else None
            if chain is not None and chain.exists():
                try:
                    obj = pyliftover.LiftOver(str(chain))
                except Exception as e:
                    print(f"  ! could not load {chain.name}: {e}")
            elif chain is not None:
                print(f"  ! chain file missing: {chain}  (run: bash download_data.sh)")
        self._lo[key] = obj
        return obj

    def point(self, chrom, pos, src="hg19", dst="GRCh38"):
        if src == dst:
            return chrom, int(pos)
        lo = self._get(src, dst)
        if lo is None:
            return None
        key = (chrom, int(pos) // 1000, src, dst)
        if key in self._cache:
            return self._cache[key]
        try:
            res = lo.convert_coordinate(chrom, int(pos))
        except Exception:
            res = None
        out = (res[0][0], int(res[0][1])) if res else None
        self._cache[key] = out
        return out

    def interval(self, chrom, start, end, src="hg19", dst="GRCh38"):
        # Lift both ends independently and keep the result only if they land on
        # the same contig in a sane order -- a split interval means the region is
        # rearranged between assemblies and any signal we took would be wrong.
        if src == dst:
            return chrom, int(start), int(end)
        a = self.point(chrom, start, src, dst)
        b = self.point(chrom, max(int(end) - 1, int(start)), src, dst)
        if a is None or b is None or a[0] != b[0]:
            self.n_fail += 1
            return None
        lo_, hi_ = sorted((a[1], b[1]))
        if hi_ <= lo_:
            hi_ = lo_ + 1
        self.n_ok += 1
        return a[0], lo_, hi_ + 1

    def report(self):
        tot = self.n_ok + self.n_fail
        if tot == 0:
            return "no cross-assembly lookups were needed"
        return (f"{self.n_ok:,}/{tot:,} intervals lifted "
                f"({100*self.n_fail/tot:.2f}% failed and returned no signal)")


LIFTER = CoordLifter(REF_DIR)


class HistoneExtractor:
    # Lazily opens bigWigs and caches per-(sample, fragment) mean signal.
    # hist_map: sid -> mark -> [(path, assembly)]
    def __init__(self, hist_map, marks, stat="mean", lifter=None,
                 native="hg19"):
        self.hist_map, self.marks, self.stat = hist_map, list(marks), stat
        self.lifter, self.native = lifter, native
        self._open, self._cache = {}, {}
        self.available = {
            sid: {mk: len(hist_map.get(sid, {}).get(mk, [])) > 0 for mk in self.marks}
            for sid in hist_map
        }

    def has(self, sid, mark):
        return self.available.get(sid, {}).get(mark, False)

    def mask_vector(self, sid):
        return np.array([1.0 if self.has(sid, mk) else 0.0 for mk in self.marks],
                        dtype=np.float32)

    def _bw(self, path):
        p = str(path)
        if p not in self._open:
            try:
                self._open[p] = pyBigWig.open(p)
            except Exception as e:
                print(f"  ! cannot open {Path(p).name}: {e}")
                self._open[p] = None
        return self._open[p]

    def signal(self, sid, chrom, start, end):
        # -> (values[n_marks], mask[n_marks]); coordinates are given in self.native
        key = (sid, chrom, start, end)
        if key in self._cache:
            return self._cache[key], self.mask_vector(sid)
        out = np.zeros(len(self.marks), dtype=np.float32)
        for i, mk in enumerate(self.marks):
            entries = self.hist_map.get(sid, {}).get(mk, [])
            if not entries:
                continue
            vals = []
            for path, asm in entries:                       # average replicates
                bw = self._bw(path)
                if bw is None:
                    continue
                if asm == self.native:
                    q = (chrom, int(start), int(end))
                else:
                    q = self.lifter.interval(chrom, start, end,
                                             src=self.native, dst=asm) \
                        if self.lifter else None
                    if q is None:
                        continue                            # unliftable region
                c, s, e = q
                try:
                    chroms = bw.chroms()
                    # GRCh38 files from ENCODE use 'chr1'; some use '1'
                    if c not in chroms:
                        alt = c[3:] if c.startswith("chr") else "chr" + c
                        if alt not in chroms:
                            continue
                        c = alt
                    clen = chroms[c]
                    s, e = max(0, int(s)), min(int(e), clen)
                    if e <= s:
                        continue
                    v = bw.stats(c, s, e, type=self.stat)[0]
                    if v is not None and np.isfinite(v):
                        vals.append(float(v))
                except Exception:
                    continue
            if vals:
                out[i] = float(np.mean(vals))
        self._cache[key] = out
        return out, self.mask_vector(sid)

    def close(self):
        for bw in self._open.values():
            if bw is not None:
                try: bw.close()
                except Exception: pass


HIST = HistoneExtractor(hist_map, CFG.histone_marks, CFG.histone_stat,
                        lifter=LIFTER, native=CFG.genome_build)
N_MARKS = len(CFG.histone_marks)

avail_df = pd.DataFrame(HIST.available).T.reindex(sel["sample_id"]).fillna(False)
avail_df.index = sel.set_index("sample_id")["tissue"]
print("histone availability for the tissues in this run")
print(avail_df.astype(int).to_string())
frac = avail_df.values.mean()
print(f"\n{frac:.1%} of (tissue x mark) tracks are available; "
      f"the rest are masked (signal=0, mask=0)")

# ---- quick demo on a few fragments so the numbers are visible ---------------
demo_sid = next((s for s in sel["sample_id"] if any(HIST.available.get(s, {}).values())),
                None)
if demo_sid:
    demo = loops[loops["sample_id"] == demo_sid].head(5)
    print(f"\nexample signal, sample {demo_sid} "
          f"({sel.set_index('sample_id').loc[demo_sid,'tissue']}):")
    rows = []
    for _, r in demo.iterrows():
        v, m = HIST.signal(demo_sid, r["bait_chr"], r["bait_start"], r["bait_end"])
        rows.append([f"{r['gene']}", *np.round(v, 3)])
    print(pd.DataFrame(rows, columns=["gene", *CFG.histone_marks]).to_string(index=False))
else:
    print("\nNo histone tracks for any selected tissue — all histone features "
          "will be masked. The model still runs on topology + sequence.")

if n_grch38:
    print(f"\ncross-assembly lookups so far: {LIFTER.report()}")

# ---- fit the transform on a sample so cell 10 can apply it -----------------
samp = loops.sample(min(5000, len(loops)), random_state=CFG.seed)
raw = np.array([HIST.signal(r["sample_id"], r["bait_chr"],
                            r["bait_start"], r["bait_end"])[0]
                for _, r in samp.iterrows()], dtype=np.float32)
raw_log = np.log1p(np.clip(raw, 0, None))
HIST_CLIP = np.percentile(raw_log, 99.5, axis=0)
HIST_CLIP[HIST_CLIP <= 0] = 1.0
HIST_MEAN = raw_log.clip(0, HIST_CLIP).mean(0)
HIST_STD  = raw_log.clip(0, HIST_CLIP).std(0) + 1e-6

def transform_hist(v):
    return ((np.log1p(np.clip(v, 0, None)).clip(0, HIST_CLIP) - HIST_MEAN)
            / HIST_STD).astype(np.float32)

print("\nper-mark transform fitted on 5,000 sampled fragments")
print(pd.DataFrame({"mark": list(CFG.histone_marks), "clip_p99.5": HIST_CLIP.round(3),
                    "mean": HIST_MEAN.round(3), "std": HIST_STD.round(3)}
                   ).to_string(index=False))

if raw.sum() > 0:
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    present = [i for i in range(N_MARKS) if raw[:, i].sum() > 0]
    ax[0].boxplot([raw_log[:, i] for i in present],
                  labels=[CFG.histone_marks[i] for i in present], showfliers=False)
    ax[0].set_title("log1p signal over bait fragments"); ax[0].tick_params(axis="x", rotation=45)
    if len(present) > 1:
        cc = np.corrcoef(raw_log[:, present].T)
        sns.heatmap(cc, ax=ax[1], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    xticklabels=[CFG.histone_marks[i] for i in present],
                    yticklabels=[CFG.histone_marks[i] for i in present],
                    annot=True, fmt=".2f", annot_kws={"size": 7})
        ax[1].set_title("mark-mark correlation")
    fig.tight_layout(); savefig(fig, "03_histone_features"); plt.show()
    print("\nSanity: H3K4me3/H3K27ac (active) should correlate positively with each")
    print("other and negatively with H3K27me3/H3K9me3 (repressive).")
""")

# =============================================================================
MD(r"""
## 9 · Hi-C contact counts (the `.hic` folder)

The slide called for tissue Hi-C *in addition to* PCHi-C. `fetch_hic.py` pulls
real Juicer archives from ENCODE and the 4D Nucleome portal where they exist;
tissues with none keep a stub. This cell handles all three cases in one path:

1. Read `hic/HIC_manifest.csv` for each file's **source and assembly**.
2. Detect stubs by their header and skip them.
3. Open the rest with `hicstraw` and query observed / KR-normalised counts for
   each bait–other-end bin pair at `HIC_RESOLUTION`.

**Why the assembly matters here more than anywhere else.** Nearly all public
Hi-C is GRCh38. A `.hic` file is a binary, multi-resolution, *normalised* contact
matrix — there is no liftOver for one. Converting it would mean going back to
the raw reads and re-running Juicer, which is a pipeline in its own right and
not something this notebook should pretend to do. So we do the same thing as for
GRCh38 bigWigs: **lift the query coordinates** hg19 → GRCh38 with the `LIFTER`
from the previous cell, then ask the matrix. Exact, and free.

**The fallback is explicit, not silent.** When a tissue has no usable Hi-C the
`hic_count` edge feature is 0 *and* the companion `has_hic` bit is 0, exactly
like the histone masking. The model never confuses "no contact" with "not
measured". Drop a real `.hic` into `hic/` later and the same code path picks it
up with no other change.
""")

CODE(r"""
class HiCReader:
    # Reads .hic archives, lifting query coordinates when the file's assembly
    # differs from our fragment coordinates. Stubs and failures degrade to
    # (0.0, has_hic=0) so the feature is masked rather than faked.
    def __init__(self, hic_dir, resolution, lifter=None, native="hg19",
                 enabled=True):
        self.dir, self.res = Path(hic_dir), resolution
        self.lifter, self.native = lifter, native
        self.enabled = enabled and HAS_HICSTRAW
        self._files, self._cache = {}, {}
        self.status, self.assembly = {}, {}

        # assembly comes from the manifest fetch_hic.py wrote
        man = self.dir / "HIC_manifest.csv"
        self._man = {}
        if man.exists():
            try:
                m = pd.read_csv(man)
                for _, r in m.iterrows():
                    self._man[f"{str(r['sample_id']).zfill(4)}_{r['tissue']}"] = {
                        "assembly": str(r.get("assembly") or "").strip() or "unknown",
                        "source": r.get("source", ""), "status": r.get("status", ""),
                    }
            except Exception as e:
                print(f"  ! could not read HIC_manifest.csv: {e}")

    def _open(self, stem):
        if stem in self._files:
            return self._files[stem]
        path = self.dir / f"{stem}.hic"
        obj, asm = None, self._man.get(stem, {}).get("assembly", "unknown")
        if not path.exists():
            self.status[stem] = "fallback (file absent)"
        elif not self.enabled:
            self.status[stem] = "fallback (hicstraw unavailable)"
        else:
            try:
                head = path.read_bytes()[:24]
                if head.startswith(b"DUMMY_HIC_PLACEHOLDER"):
                    raise ValueError("stub placeholder, no real data")
                if head[:3] != b"HIC":
                    raise ValueError("not a Juicer .hic archive")
                obj = hicstraw.HiCFile(str(path))
                if asm in ("", "unknown"):
                    # trust the file over the manifest when it declares itself
                    asm = getattr(obj, "getGenomeID", lambda: "unknown")() or "unknown"
                    asm = {"GRCh38": "GRCh38", "hg38": "GRCh38",
                           "hg19": "hg19", "GRCh37": "hg19"}.get(asm, asm)
                src = self._man.get(stem, {}).get("source", "")
                self.status[stem] = (f"real .hic [{asm}]"
                                     + (f" from {src}" if src else ""))
            except Exception as e:
                self.status[stem] = f"fallback ({type(e).__name__}: {e})"
                obj = None
        self.assembly[stem] = asm
        self._files[stem] = obj
        return obj

    def count(self, stem, chrom, pos_a, pos_b):
        # -> (value, has_hic). Coordinates are given in self.native.
        f = self._open(stem)
        if f is None:
            return 0.0, 0.0
        asm = self.assembly.get(stem, "unknown")
        if asm not in ("hg19", "GRCh38"):
            asm = self.native            # undeclared: assume it matches ours

        if asm != self.native:
            a = self.lifter.point(chrom, pos_a, self.native, asm) if self.lifter else None
            b = self.lifter.point(chrom, pos_b, self.native, asm) if self.lifter else None
            if a is None or b is None or a[0] != b[0]:
                return 0.0, 0.0          # unliftable -> mask, do not guess
            chrom, pos_a, pos_b = a[0], a[1], b[1]

        key = (stem, chrom, int(pos_a) // self.res, int(pos_b) // self.res)
        if key in self._cache:
            return self._cache[key], 1.0
        try:
            c = chrom[3:] if chrom.startswith("chr") else chrom
            mzd = f.getMatrixZoomData(c, c, "observed", "KR", "BP", self.res)
            lo, hi = sorted((int(pos_a), int(pos_b)))
            recs = mzd.getRecords(lo, lo + self.res, hi, hi + self.res)
            val = float(recs[0].counts) if recs else 0.0
        except Exception:
            val = 0.0
        self._cache[key] = val
        return val, 1.0


HIC = HiCReader(HIC_DIR, CFG.hic_resolution, lifter=LIFTER,
                native=CFG.genome_build, enabled=True)

stems = {}
for _, r in sel.iterrows():
    stems[r["sample_id"]] = f"{r['sample_id']}_{r['tissue']}"

print("probing .hic files\n")
for sid, stem in stems.items():
    HIC._open(stem)
    print(f"  {stem:<48} {HIC.status.get(stem)}")

n_real = sum(1 for v in HIC.status.values() if v.startswith("real"))
n_lift = sum(1 for st, a in HIC.assembly.items()
             if a != CFG.genome_build and HIC.status.get(st, "").startswith("real"))
print(f"\n{n_real}/{len(stems)} tissues have usable .hic data")
if n_lift:
    print(f"  {n_lift} of them are not {CFG.genome_build}; their queries are "
          f"coordinate-lifted per lookup.")
    if not HAS_PYLIFTOVER:
        print("  ! pyliftover is NOT installed, so those lookups will all return")
        print("    has_hic=0. Fix with: bash setup_env.sh --verify-only")
if n_real == 0:
    print("  -> hic_count will be 0 with has_hic=0 on every edge, i.e. the")
    print("     feature is masked. Run:  python fetch_hic.py")
    print("     or drop Juicer .hic files named {sampleid}_{Tissue}.hic into hic/.")
""")

# =============================================================================
MD(r"""
## 10 · Build the promoter-centred subgraphs

This is the heart of the input pipeline: one `torch_geometric.data.Data` object
per **(gene, tissue)** pair, structured exactly as the slide's panel 1.

**Nodes.** Index 0 is always the bait fragment — the promoter anchor. Indices
1..k are the distal other-end fragments, capped at `MAX_NEIGHBOURS` and chosen
by descending CHiCAGO score.

Node feature vector (`F = 4 + 2*N_MARKS` dimensions):

| block | dims | content |
|---|---|---|
| `is_bait` | 1 | 1 for the anchor, 0 for partners |
| `log_len` | 1 | log10 fragment length |
| `signed_log_dist` | 1 | signed log10 distance to the TSS (0 for the bait) |
| `same_chrom` | 1 | 1 for cis partners, 0 for trans |
| histone signal | 7 | standardised log1p mark signal over the fragment |
| histone mask | 7 | 1 if that mark exists for this tissue |

**Edges.** Bidirectional bait↔partner edges, plus — following the GrapHi-C
convention from the earlier review — optional **backbone edges** joining
partners that are adjacent along the chromosome. Those carry no interaction
evidence but they give the attention layer a notion of linear genomic order.

Edge feature vector (6 dims): CHiCAGO score (log1p), `N_reads` (log1p), signed
log distance, `same_chrom`, `hic_count` (log1p), `has_hic`. Backbone edges get
score/reads of 0 and are distinguishable by their zeroed evidence channels.

**Link-prediction supervision.** For head H1 we hold out `LINK_HOLDOUT_FRAC` of
the real edges: they are removed from `edge_index` (so the encoder genuinely
cannot see them) and stored in `pos_edge_label_index`. An equal number of
**distance-matched negatives** is drawn — fragment pairs at a similar genomic
separation that are *not* connected. Distance matching is essential: with random
negatives, the head learns the distance-decay curve and reports AUC ≈ 0.95 while
having learned no biology at all.
""")

CODE(r"""
def build_graphs(loops_df, max_nb, add_backbone, holdout_frac, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    graphs, meta = [], []
    n_feat_hist = 2 * N_MARKS

    grouped = loops_df.groupby(["sample_id", "gene"], sort=False)
    for (sid, gene), g in tqdm(grouped, total=grouped.ngroups, desc="building graphs"):
        if gene not in GENE2SEQ:
            continue
        g = g.sort_values("score", ascending=False).head(max_nb)
        if len(g) < 2:                      # need at least two partners to be a graph
            continue

        tissue = g["tissue"].iloc[0]
        stem   = stems.get(sid, f"{sid}_{tissue}")
        tss    = int(GENE_INFO.loc[gene, "tss"])
        b_chr  = g["bait_chr"].iloc[0]
        b_s, b_e = int(g["bait_start"].iloc[0]), int(g["bait_end"].iloc[0])

        hmask = HIST.mask_vector(sid)

        # ---- node 0 : the bait / promoter anchor
        hv, _ = HIST.signal(sid, b_chr, b_s, b_e)
        nodes = [np.concatenate([
            [1.0, np.log10(max(b_e - b_s, 1)), 0.0, 1.0],
            transform_hist(hv), hmask]).astype(np.float32)]
        node_pos = [(b_s + b_e) // 2]

        # ---- nodes 1..k : the partner fragments
        e_src, e_dst, e_attr = [], [], []
        for _, r in g.iterrows():
            o_chr, o_s, o_e = r["other_chr"], int(r["other_start"]), int(r["other_end"])
            mid = (o_s + o_e) // 2
            same = float(r["same_chrom"])
            d = (mid - tss) if same else 0.0
            sgn_log_d = np.sign(d) * np.log10(abs(d) + 1)

            hv, _ = HIST.signal(sid, o_chr, o_s, o_e)
            nodes.append(np.concatenate([
                [0.0, np.log10(max(o_e - o_s, 1)), sgn_log_d, same],
                transform_hist(hv), hmask]).astype(np.float32))
            node_pos.append(mid)

            hic_v, has_hic = HIC.count(stem, o_chr, (b_s + b_e) // 2, mid)
            attr = np.array([
                np.log1p(float(r["score"])), np.log1p(float(r["N_reads"])),
                sgn_log_d, same, np.log1p(hic_v), has_hic], dtype=np.float32)

            j = len(nodes) - 1
            e_src += [0, j]; e_dst += [j, 0]; e_attr += [attr, attr]

        n_nodes = len(nodes)
        n_real_edges = len(e_src)

        # ---- optional backbone edges between genomically adjacent partners
        if add_backbone and n_nodes > 2:
            order = sorted(range(1, n_nodes), key=lambda i: node_pos[i])
            for a, b in zip(order[:-1], order[1:]):
                gap = abs(node_pos[a] - node_pos[b])
                attr = np.array([0.0, 0.0, np.log10(gap + 1), 1.0, 0.0, 0.0],
                                dtype=np.float32)
                e_src += [a, b]; e_dst += [b, a]; e_attr += [attr, attr]

        edge_index = np.array([e_src, e_dst], dtype=np.int64)
        edge_attr  = np.stack(e_attr).astype(np.float32)

        # ---- hold out real edges for the link-prediction head -----------------
        real_pairs = [(0, j) for j in range(1, n_nodes)]
        n_hold = max(1, int(round(holdout_frac * len(real_pairs))))
        n_hold = min(n_hold, len(real_pairs) - 1)      # always leave >=1 message edge
        hold_j = rng.choice(np.arange(1, n_nodes), size=n_hold, replace=False)
        hold_set = set(int(j) for j in hold_j)

        keep = []
        for k in range(edge_index.shape[1]):
            a, b = int(edge_index[0, k]), int(edge_index[1, k])
            is_real_bait_edge = k < n_real_edges
            if is_real_bait_edge and (a in hold_set or b in hold_set) and 0 in (a, b):
                continue                                # hide it from message passing
            keep.append(k)
        keep = np.array(keep, dtype=np.int64)

        pos_pairs = np.array([[0, j] for j in sorted(hold_set)], dtype=np.int64).T

        # ---- distance-matched negatives -------------------------------------
        # sample non-neighbours whose |log distance| is close to a held-out positive
        cand = [j for j in range(1, n_nodes) if j not in hold_set]
        neg_pairs = []
        if cand:
            pos_d = np.array([abs(node_pos[j] - node_pos[0]) for j in sorted(hold_set)])
            cand_d = np.array([abs(node_pos[j] - node_pos[0]) for j in cand])
            for pd_ in pos_d:
                # closest-in-distance non-edge, jittered so it is not deterministic
                order = np.argsort(np.abs(np.log10(cand_d + 1) - np.log10(pd_ + 1)))
                pick = cand[int(order[rng.integers(0, min(3, len(order)))])]
                neg_pairs.append([0, pick])
        if not neg_pairs:
            neg_pairs = [[0, int(rng.integers(1, n_nodes))]]
        neg_pairs = np.array(neg_pairs, dtype=np.int64).T

        d = Data(
            x=torch.from_numpy(np.stack(nodes)),
            edge_index=torch.from_numpy(edge_index[:, keep]),
            edge_attr=torch.from_numpy(edge_attr[keep]),
            pos_edge_label_index=torch.from_numpy(pos_pairs),
            neg_edge_label_index=torch.from_numpy(neg_pairs),
            seq_idx=torch.tensor([GENE2SEQ[gene]], dtype=torch.long),
        )
        d.num_nodes = n_nodes
        graphs.append(d)
        meta.append({"gene": gene, "sample_id": sid, "tissue": tissue,
                     "chrom": b_chr, "n_nodes": n_nodes,
                     "n_edges": int(keep.size), "n_pos": pos_pairs.shape[1],
                     "mean_score": float(g["score"].mean())})
    return graphs, pd.DataFrame(meta)


GRAPHS, META = build_graphs(loops, CFG.max_neighbours, CFG.add_backbone_edges,
                            CFG.link_holdout_frac, CFG.seed)

N_NODE_FEAT = GRAPHS[0].x.shape[1]
N_EDGE_FEAT = GRAPHS[0].edge_attr.shape[1]
print(f"\nbuilt {len(GRAPHS):,} promoter subgraphs")
print(f"  unique genes   : {META['gene'].nunique():,}")
print(f"  unique tissues : {META['tissue'].nunique()}")
print(f"  node features F   = {N_NODE_FEAT}  "
      f"(4 structural + {N_MARKS} marks + {N_MARKS} mask)")
print(f"  edge features F_e = {N_EDGE_FEAT}")
print()
print(META[["n_nodes", "n_edges", "n_pos", "mean_score"]].describe().to_string())
# ---- constant-feature audit -------------------------------------------------
# A feature with zero variance across the whole dataset contributes nothing but
# parameters. The usual culprit here is `same_chrom`: these PCHi-C files are
# cis-only, so every edge has same_chrom == 1 and the column is dead weight.
_nf = torch.cat([g.x for g in GRAPHS[:4000]]).numpy()
_ef = torch.cat([g.edge_attr for g in GRAPHS[:4000]]).numpy()
NODE_FEAT_NAMES = (["is_bait", "log_len", "sgn_log_d", "same_chrom"]
                   + list(CFG.histone_marks)
                   + [f"mask:{m}" for m in CFG.histone_marks])
EDGE_FEAT_NAMES = ["log_score", "log_reads", "sgn_log_d", "same_chrom",
                   "log_hic", "has_hic"]

const_node = [n for n, v in zip(NODE_FEAT_NAMES, _nf.std(0)) if v < 1e-8]
const_edge = [n for n, v in zip(EDGE_FEAT_NAMES, _ef.std(0)) if v < 1e-8]
print("\nfeature variance audit (first 4,000 graphs)")
print(pd.DataFrame({"feature": NODE_FEAT_NAMES, "mean": _nf.mean(0).round(3),
                    "std": _nf.std(0).round(3)}).to_string(index=False))
print()
print(pd.DataFrame({"edge_feature": EDGE_FEAT_NAMES, "mean": _ef.mean(0).round(3),
                    "std": _ef.std(0).round(3)}).to_string(index=False))
if const_node or const_edge:
    print(f"\n  CONSTANT (zero variance) -> carries no information:")
    if const_node: print(f"    node: {', '.join(const_node)}")
    if const_edge: print(f"    edge: {', '.join(const_edge)}")
    print("  These cost parameters and teach the model nothing. 'same_chrom' is")
    print("  constant whenever the loop files are cis-only, which is the case")
    print("  for this dataset. Leaving them in is harmless but wasteful; drop")
    print("  them if you want a tighter feature matrix.")
else:
    print("\n  no zero-variance features")

print("\nexample graph:")
d0 = GRAPHS[0]
print(f"  {META.iloc[0]['gene']} in {META.iloc[0]['tissue']}")
print(f"  x={tuple(d0.x.shape)}  edge_index={tuple(d0.edge_index.shape)}  "
      f"edge_attr={tuple(d0.edge_attr.shape)}")
print(f"  held-out positives={d0.pos_edge_label_index.shape[1]}  "
      f"negatives={d0.neg_edge_label_index.shape[1]}")
""")

# =============================================================================
MD(r"""
## 11 · Look at an actual graph

Numbers in a table are easy to get wrong without noticing. This cell draws one
promoter subgraph so you can see that it has the shape the model expects: a hub
(the bait, index 0) with spokes to partner fragments, laid out by genomic
position, edge width proportional to CHiCAGO score, plus the backbone edges
between adjacent partners.

It also prints the raw feature matrix of a small graph so every column can be
checked by eye against the table in cell 10.
""")

CODE(r"""
pick = META.sort_values("n_nodes", ascending=False).iloc[len(META)//20]
idx = int(pick.name)
d = GRAPHS[idx]

G = nx.Graph()
G.add_nodes_from(range(d.num_nodes))
ei = d.edge_index.numpy(); ea = d.edge_attr.numpy()
for k in range(ei.shape[1]):
    a, b = int(ei[0, k]), int(ei[1, k])
    if a < b:
        G.add_edge(a, b, w=float(ea[k, 0]))

pos = nx.spring_layout(G, seed=CFG.seed, k=0.9)
pos[0] = np.array([0.0, 0.0])

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
widths = [0.5 + 1.6 * G[a][b]["w"] for a, b in G.edges()]
colors = ["#B9C4D4" if G[a][b]["w"] == 0 else PALETTE[0] for a, b in G.edges()]
nx.draw_networkx_edges(G, pos, ax=ax[0], width=widths, edge_color=colors, alpha=.75)
nx.draw_networkx_nodes(G, pos, ax=ax[0], nodelist=[0], node_size=520,
                       node_color=PALETTE[2], edgecolors="k", linewidths=1.6)
nx.draw_networkx_nodes(G, pos, ax=ax[0], nodelist=list(range(1, d.num_nodes)),
                       node_size=150, node_color=PALETTE[3], edgecolors="k",
                       linewidths=.7)
ax[0].set_title(f"{pick['gene']} in {pick['tissue']}\n"
                f"{d.num_nodes} nodes, {G.number_of_edges()} undirected edges\n"
                f"grey = backbone, teal = PCHi-C (width ~ CHiCAGO)", fontsize=10)
ax[0].axis("off")

im = ax[1].imshow(d.x.numpy(), aspect="auto", cmap="RdBu_r",
                  vmin=-3, vmax=3, interpolation="nearest")
ax[1].set_title("node feature matrix  x  (rows = nodes, row 0 = bait)")
ax[1].set_xlabel("feature index"); ax[1].set_ylabel("node")
labels = (["is_bait", "log_len", "sgn_log_d", "same_chr"]
          + [f"{m}" for m in CFG.histone_marks]
          + [f"mask:{m}" for m in CFG.histone_marks])
ax[1].set_xticks(range(len(labels)))
ax[1].set_xticklabels(labels, rotation=90, fontsize=6)
plt.colorbar(im, ax=ax[1], fraction=.03)
fig.tight_layout(); savefig(fig, "04_example_graph"); plt.show()

small = min(GRAPHS, key=lambda g: g.num_nodes)
print("smallest graph, full feature matrix:")
print(pd.DataFrame(small.x.numpy(), columns=labels).round(2).to_string())
print("\nits edge features:")
print(pd.DataFrame(small.edge_attr.numpy(),
                   columns=["log_score", "log_reads", "sgn_log_d",
                            "same_chrom", "log_hic", "has_hic"]).round(2).to_string())
""")

# =============================================================================
MD(r"""
## 12 · Labels for all four heads

**H2 — expressed in tissue (binary).** The RSEM TPM matrix has 194 sample
columns named `{id}_{GSE}_{Tissue}_{rep}`. We average replicates within a
sample id to get a gene × tissue TPM matrix, then binarise at `TPM > 1`.
Averaging on the linear scale (not log) is deliberate: TPM is already a ratio
and the arithmetic mean of replicates is the standard summary.

**H3 — tissue-restricted vs housekeeping (binary).** We compute Yanai's tau
across all tissues that have RNA-seq:

$$\tau = \frac{\sum_{i=1}^{n}\left(1 - \frac{x_i}{\max_i x_i}\right)}{n-1}$$

on log2(TPM+1). tau = 0 means perfectly uniform (housekeeping), tau = 1 means
expressed in exactly one tissue. We label `tau > 0.85` as restricted. Note this
is a **per-gene** label: it is constant across the tissues of the same gene,
which is another reason the folds must be split by gene.

**H4 — tissue class (multiclass).** The tissue index of the graph. This is the
head most likely to saturate — the histone and contact profile largely gives the
tissue away — so watch for it hitting ceiling accuracy while contributing no
useful gradient.

**H1 — link prediction (binary).** Already constructed in cell 10 as held-out
positives plus distance-matched negatives, so there is nothing to build here;
we just report the balance.

Rows whose gene has no RNA-seq measurement are kept but get a `-1` label for
H2/H3, and the loss masks them. Dropping them instead would discard perfectly
good link-prediction and tissue-classification signal.
""")

CODE(r"""
# ------------------------------------------------ gene x tissue TPM matrix
usecols = ["gene_name"] + rna_cols
tpm = pd.read_csv(TPM_FILE, sep="\t", usecols=usecols, low_memory=False)
tpm = tpm[tpm["gene_name"].isin(set(META["gene"]))]
tpm = tpm.groupby("gene_name").max()          # duplicate symbols -> max

by_tissue = {}
for sid, cols in rna_by_tissue.items():
    cols = [c for c in cols if c in tpm.columns]
    if cols:
        by_tissue[sid] = tpm[cols].mean(axis=1)
TPM = pd.DataFrame(by_tissue)
print(f"TPM matrix: {TPM.shape[0]:,} genes x {TPM.shape[1]} tissues")
print(TPM.iloc[:5, :6].round(2).to_string())

# ------------------------------------------------ H2 expressed
EXPRESSED = (TPM > CFG.tpm_threshold).astype(int)
print(f"\nH2  expressed (TPM > {CFG.tpm_threshold}): "
      f"{EXPRESSED.values.mean():.1%} of gene x tissue cells positive")

# ------------------------------------------------ H3 tau
logtpm = np.log2(TPM + 1)
mx = logtpm.max(axis=1)
valid = mx > 0
tau = pd.Series(np.nan, index=TPM.index)
n_t = logtpm.shape[1]
tau[valid] = ((1 - logtpm[valid].div(mx[valid], axis=0)).sum(axis=1) / (n_t - 1))
RESTRICTED = (tau > CFG.tau_threshold).astype(int)
print(f"H3  tau computed for {int(valid.sum()):,} genes; "
      f"{RESTRICTED[valid].mean():.1%} are tissue-restricted (tau > {CFG.tau_threshold})")

# ------------------------------------------------ H4 tissue class
TISSUE_LIST = sorted(META["tissue"].unique())
TISSUE2IDX = {t: i for i, t in enumerate(TISSUE_LIST)}
print(f"H4  {len(TISSUE_LIST)} tissue classes")

# ------------------------------------------------ attach to graphs
n_expr_known = n_tau_known = 0
for i, d in enumerate(GRAPHS):
    m = META.iloc[i]
    g, sid, t = m["gene"], m["sample_id"], m["tissue"]

    y_e = -1
    if g in EXPRESSED.index and sid in EXPRESSED.columns:
        y_e = int(EXPRESSED.at[g, sid]); n_expr_known += 1
    y_t = -1
    if g in RESTRICTED.index and not np.isnan(tau.get(g, np.nan)):
        y_t = int(RESTRICTED.at[g]); n_tau_known += 1

    d.y_expr   = torch.tensor([y_e], dtype=torch.float32)
    d.y_tau    = torch.tensor([y_t], dtype=torch.float32)
    d.y_tissue = torch.tensor([TISSUE2IDX[t]], dtype=torch.long)

META["y_expr"]   = [int(d.y_expr.item())   for d in GRAPHS]
META["y_tau"]    = [int(d.y_tau.item())    for d in GRAPHS]
META["y_tissue"] = [int(d.y_tissue.item()) for d in GRAPHS]

print(f"\nlabel coverage over {len(GRAPHS):,} graphs")
print(f"  H2 expression : {n_expr_known:,} labelled ({n_expr_known/len(GRAPHS):.1%}), "
      f"positive rate {META.loc[META.y_expr>=0,'y_expr'].mean():.1%}")
print(f"  H3 tau        : {n_tau_known:,} labelled ({n_tau_known/len(GRAPHS):.1%}), "
      f"positive rate {META.loc[META.y_tau>=0,'y_tau'].mean():.1%}")
print(f"  H4 tissue     : {len(GRAPHS):,} labelled, {len(TISSUE_LIST)} classes")
n_pos = int(META['n_pos'].sum())
print(f"  H1 links      : {n_pos:,} positives + {n_pos:,} distance-matched negatives")

# ------------------------------------------------ plots
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
axes[0].bar(["not expr", "expressed"],
            [(META.y_expr == 0).sum(), (META.y_expr == 1).sum()], color=PALETTE[:2])
axes[0].set_title("H2 expression balance")

axes[1].hist(tau.dropna(), bins=60, color=PALETTE[2])
axes[1].axvline(CFG.tau_threshold, ls="--", c="k", label=f"tau={CFG.tau_threshold}")
axes[1].set_title("H3 tissue-specificity index"); axes[1].set_xlabel("tau"); axes[1].legend()

tc = META["tissue"].value_counts()
axes[2].barh(tc.index, tc.values, color=PALETTE[3])
axes[2].set_title("H4 graphs per tissue"); axes[2].tick_params(axis="y", labelsize=7)

axes[3].bar(["positives", "negatives"], [n_pos, n_pos], color=PALETTE[4:6])
axes[3].set_title("H1 link supervision\n(distance-matched)")

fig.tight_layout(); savefig(fig, "05_label_distributions"); plt.show()

expr_by_tissue = (META[META.y_expr >= 0].groupby("tissue")["y_expr"].mean()
                  .sort_values(ascending=False))
print("\npositive rate for H2 by tissue (watch for a tissue that is trivially separable)")
print(expr_by_tissue.round(3).to_string())
""")

# =============================================================================
MD(r"""
## 13 · Cross-validation splits

Five folds, split by **gene** using `GroupKFold`.

Why this matters more than any hyperparameter: the same gene appears in up to 46
tissue graphs, and its promoter sequence is *identical* in all of them (the
sequence tower is tissue-invariant by construction). If gene G is in the
training set for tissue A and the test set for tissue B, the model has already
seen G's promoter and — for the tau head, whose label is constant per gene —
already seen its exact answer. Random row-wise splitting would produce
beautiful, meaningless AUCs.

Grouping by gene closes that leak. The cost is that folds are not perfectly
balanced in size, which `GroupKFold` handles by greedily balancing group sizes.

The cell prints, for every fold: sizes, the positive rate of each head, and —
the key check — that the train/test gene sets are disjoint. It also reports
tissue coverage per fold, since a fold missing a tissue entirely would break the
multiclass head.

An optional `chrom` grouping is available in the config for a stricter,
chromosome-held-out evaluation, which additionally removes any linkage between
nearby genes sharing regulatory neighbourhoods.
""")

CODE(r"""
if CFG.group_split_by == "gene":
    groups = META["gene"].values
elif CFG.group_split_by == "chrom":
    groups = META["chrom"].values
else:
    groups = np.arange(len(META))

gkf = GroupKFold(n_splits=CFG.n_folds)
FOLDS = list(gkf.split(np.arange(len(GRAPHS)), groups=groups))

rows = []
for k, (tr, te) in enumerate(FOLDS):
    mtr, mte = META.iloc[tr], META.iloc[te]
    leak = len(set(mtr["gene"]) & set(mte["gene"]))
    rows.append({
        "fold": k, "n_train": len(tr), "n_test": len(te),
        "genes_train": mtr["gene"].nunique(), "genes_test": mte["gene"].nunique(),
        "gene_leak": leak,
        "tissues_test": mte["tissue"].nunique(),
        "expr_pos_train": round(mtr.loc[mtr.y_expr >= 0, "y_expr"].mean(), 3),
        "expr_pos_test":  round(mte.loc[mte.y_expr >= 0, "y_expr"].mean(), 3),
        "tau_pos_test":   round(mte.loc[mte.y_tau  >= 0, "y_tau"].mean(), 3),
    })
split_df = pd.DataFrame(rows)
print(f"{CFG.n_folds}-fold GroupKFold, grouped by '{CFG.group_split_by}'\n")
print(split_df.to_string(index=False))

assert split_df["gene_leak"].sum() == 0, "gene leakage between train and test"
print("\n  OK: no gene appears in both train and test of any fold")

missing = split_df[split_df["tissues_test"] < len(TISSUE_LIST)]
if len(missing):
    print(f"  note: {len(missing)} fold(s) do not see every tissue in test; "
          "macro tissue metrics are computed over present classes only")
else:
    print("  OK: every fold's test set covers all tissue classes")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].bar(split_df["fold"] - .2, split_df["n_train"], .4, label="train", color=PALETTE[0])
ax[0].bar(split_df["fold"] + .2, split_df["n_test"], .4, label="test", color=PALETTE[1])
ax[0].set_title("graphs per fold"); ax[0].set_xlabel("fold"); ax[0].legend()
ax[1].bar(split_df["fold"] - .2, split_df["expr_pos_train"], .4, label="train", color=PALETTE[0])
ax[1].bar(split_df["fold"] + .2, split_df["expr_pos_test"], .4, label="test", color=PALETTE[1])
ax[1].set_title("H2 positive rate per fold"); ax[1].set_xlabel("fold")
ax[1].axhline(META.loc[META.y_expr >= 0, "y_expr"].mean(), ls="--", c="k", lw=1,
              label="overall"); ax[1].legend()
fig.tight_layout(); savefig(fig, "06_cv_splits"); plt.show()

SEQ_TENSOR = torch.from_numpy(SEQ_ONEHOT).float()
print(f"\nsequence tensor for the model: {tuple(SEQ_TENSOR.shape)} "
      f"({SEQ_TENSOR.nbytes/1e6:.0f} MB float32)")
""")
