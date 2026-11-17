# %% [markdown]
# # SERVER RUN v2 — trimodal PangenomicGAT, 1 Mb window, leak-free two-stage training
#
# Forked from `pangenomic_gat_pipeline_trimodal_1mb_SERVER_latest.py`. The
# architecture (three towers, GatedFusion, four heads, both H2 framings), data,
# labels, splits, metrics and every downstream Part are the same. What changed,
# and why:
#
# **1. Two-stage training, leak-free (replaces the external warm start).**
# The previous run warm-started every fold, arm and validation scheme from ONE
# checkpoint (`results_trimodal_1mb_stratified_fast/model_fold0.pt`) trained on
# another run's fold-0 genes, then froze its encoders. Genes the frozen
# encoders were trained on ended up in this run's test folds. Now, for every
# split (each gene-CV fold, chromosome-CV fold, LOTO tissue):
#
# - **Stage A** trains the full `all3` model end to end on that split's own
#   training graphs only (gene-grouped inner validation for early stopping).
# - **Stage B** freezes those encoders, evaluates them ONCE (eval mode, no
#   grad) over every graph, caches `z_seq` / `z_str` / `z_epi` and the node
#   states at the held-out link pairs, and trains every ablation arm's
#   GatedFusion + heads on the cached tensors. A Stage-B epoch takes seconds.
#
# `CFG.encoder_source = "checkpoint"` still allows an external warm start, but
# only from a checkpoint that records its `train_genes`; test genes it has
# seen are then dropped from scoring and the count is printed.
# `"none"` trains every arm end to end (the slow original behaviour).
#
# **2. Speed.** The sequence bank is stored as uint8 base indices
# (`n_genes x L`, ~20 GB), resident on the GPU when it fits, one-hot encoded per
# batch on the device. No more 330 GB float32 host copy and no per-batch
# 256 MB host->device transfer. bf16 autocast plus TF32, no per-batch
# `.item()` syncs, predictions collected only when scored, 8 DataLoader workers.
#
# **3. Correctness fixes.**
# - Only `seq.*` / `struc.*` / `epi.*` are ever transferred from a checkpoint;
#   the heads and `link_bilinear` are always fresh, as the comments always claimed.
# - A failed bigWig read is no longer a silent 0.0 flagged "measured": it is
#   counted, logged, and imputed like a missing mark.
# - Inf CHiCAGO scores are capped at the max finite score while loading,
#   instead of being patched in the graph cache afterwards.
# - Caches are keyed on a content hash (genes, tissues, histone files,
#   transform stats, schema version), the key is stored inside the file, and
#   writes are atomic (tmp + rename).
# - Each fold reseeds from (seed, run key), so a resumed run is identical to an
#   uninterrupted one. Checkpoints store a config fingerprint and refuse to
#   resume across a config change. Finished folds store their test
#   predictions, so they are not re-scored on restart.
# - Multiclass AUROC no longer turns NaN when a tissue is absent from a test set.
#
# **4. Hygiene.** Missing packages are reported, never pip-installed
# automatically (set `PANGENOMIC_AUTO_INSTALL=1` to opt in). The one-off
# debugging and patch cells are gone. Smoke tests are opt-in
# (`PANGENOMIC_SMOKE=1`).
#
# **Restarting IS resuming.** Every Stage-A and Stage-B run checkpoints after
# each epoch to `RESULTS_DIR/checkpoints/`. Re-run the script top to bottom
# after any interruption. Results go to `results/results_trimodal_1mb_v2_full/`,
# so this never touches the files of the v1 run.

# %% [markdown]
# ## 0 · Preflight — environment and packages
#
# Checks the interpreter first, then every package this script imports. A
# missing package is reported and the run stops. It is never installed behind
# your back into whatever interpreter happens to be running. Set
# `PANGENOMIC_AUTO_INSTALL=1` to opt in to installing missing packages (still
# only after the environment check has passed).


# %%
import importlib, os, subprocess, sys

EXPECTED_ENV = "pchic"
_active_env = os.environ.get("CONDA_DEFAULT_ENV", "")
if f"/envs/{EXPECTED_ENV}/" not in sys.executable and _active_env != EXPECTED_ENV:
    raise RuntimeError(
        f"\n\nThis notebook must run inside the '{EXPECTED_ENV}' conda env.\n"
        f"  Currently: {sys.executable}\n\n"
        f"  Fix:  conda activate {EXPECTED_ENV}\n"
        f"        then Kernel > Change Kernel > Python ({EXPECTED_ENV})\n")

# import name -> pip distribution name
REQUIRED_PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "matplotlib": "matplotlib", "seaborn": "seaborn", "networkx": "networkx",
    "tqdm": "tqdm", "torch": "torch", "torch_geometric": "torch-geometric",
    "pyBigWig": "pyBigWig", "py2bit": "py2bit", "sklearn": "scikit-learn",
    "statsmodels": "statsmodels", "requests": "requests",
}
OPTIONAL_PACKAGES = {"pyliftover": "pyliftover", "umap": "umap-learn",
                     "torch_scatter": "torch-scatter"}


def _import_status(mod_name):
    # Only ImportError means "not installed". Anything else (a CUDA ABI
    # mismatch, a broken extension module) is a real error and must surface.
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        return False


_missing_pkgs = [(m, p) for m, p in REQUIRED_PACKAGES.items() if not _import_status(m)]
if _missing_pkgs:
    names = " ".join(p for _, p in _missing_pkgs)
    if os.environ.get("PANGENOMIC_AUTO_INSTALL") == "1":
        print(f"installing missing package(s): {names}")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            *[p for _, p in _missing_pkgs]], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"pip install failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        importlib.invalidate_caches()
        still = [m for m, _ in _missing_pkgs if not _import_status(m)]
        if still:
            raise RuntimeError(f"still not importable after install: {still}")
    else:
        raise RuntimeError(
            f"missing required package(s): {names}\n"
            f"  install them into '{EXPECTED_ENV}' (pinned versions: environment.yml), or\n"
            f"  re-run with PANGENOMIC_AUTO_INSTALL=1")
print(f"all {len(REQUIRED_PACKAGES)} required packages present")

print("\noptional packages (used if present):")
for mod_name, pip_name in OPTIONAL_PACKAGES.items():
    print(f"  {'present' if _import_status(mod_name) else 'absent '}  {mod_name}"
          + ("" if _import_status(mod_name) else f"  (pip install {pip_name} to enable)"))


# %% [markdown]
# # Pangenomic AI — promoter-centred graph attention network (PCHi-C only)
#
# This notebook is a variant of `pangenomic_gat_pipeline.ipynb` that builds every
# graph **from PCHi-C loop calls only** — the `.hic` (Juicer / straw) matrices are
# never opened and contribute no edge feature. Everything else (sequence tower,
# histone features, labels, cross-validation, ablation) is unchanged. It implements,
# end to end, the architecture from the pipeline slide:
#
# ```
# tissue-specific PCHi-C subgraphs               promoter DNA sequence
#                 |                                        |
#         STRUCTURE TOWER (GATv2, L layers x K heads)   SEQUENCE TOWER (CNN)
#                 |                                        |
#              z_str  (per gene x tissue)              z_seq  (per gene)
#                 \                                        /
#                  \------  contrastive alignment  -------/
#                                   |
#                         FiLM fusion  ->  z(g,t)
#                                   |
#     +--------------+--------------+--------------+--------------+
#     | H1 link pred | H2 expression| H3 tissue cls| H4 embedding  |
#     +--------------+--------------+--------------+--------------+
# ```
#
# **Data this notebook consumes** (all paths configured in the next cell):
#
# | folder | contents | role |
# |---|---|---|
# | `interection_data/` | 46 `*_merged_loop.txt` PCHi-C loop files | graph edges |
# | `histone/` | `{sample}.{mark}.bw` bigWigs | node features |
# | `rna_seq/` | RSEM TPM / FPKM / counts matrices | expression labels |
# | `reference/` | `hg19.2bit`, `refGene_hg19.txt` | promoter sequence + TSS |
#
# **Four supervised heads**, all evaluated with 5-fold cross-validation:
#
# 1. **H1 link prediction** — is this promoter–fragment contact real? (binary)
# 2. **H2 expression** — two heads on the same embedding: the continuous signal
#    `log2(TPM+1)` (H2a, regression) and expressed / not expressed, TPM > 1 (H2b)
# 3. **H3 tissue-specificity** — is this gene tissue-restricted? (binary, tau index)
# 4. **H4 tissue classification** — which tissue is this graph from? (multiclass)
#
# Run it with `conda activate pchic`. The preflight cell refuses any other
# interpreter. Full scale is the only mode (all genes, all 46 tissues).


# %% [markdown]
# ## 1 · Configuration
#
# Everything tunable lives here so that no magic numbers are buried further down.
# A few choices worth flagging:
#
# - **`GENOME_BUILD = "hg19"`** — the fragment coordinates in your loop files are
#   GRCh37. Cell 6 *verifies* this rather than trusting it: it checks that
#   annotated genes actually fall inside their own bait fragments, and aborts if
#   the hg38 hypothesis fits better.
# - **`promoter_window = 1_000_000`** — 1 Mb centred on the canonical TSS, read
#   by the Hyena tower with a stride-64 stem (~15.6k positions).
# - **`max_neighbours = 10000`** — cap on promoter degree, by CHiCAGO score. In
#   practice no promoter reaches it (median degree 5, max ~3.5k).
# - **`TPM_THRESHOLD = 1.0`** — the standard "expressed" cut for RSEM TPM.
# - **`TAU_THRESHOLD = 0.85`** — Yanai's tau above this is conventionally called
#   tissue-restricted.
# - **`GROUP_SPLIT_BY = "gene"`** — folds are split by *gene*, not by row. If the
#   same gene appeared in train and test (in different tissues) the model could
#   memorise gene identity and the metrics would be inflated. This is the single
#   most important line in the notebook for making the numbers mean anything.


# %%
from pathlib import Path
import os, sys, json, math, random, warnings, time
from dataclasses import dataclass, asdict, field

# --------------------------------------------------------------- paths
def _find_project_root(start: Path) -> Path:
    """Notebooks live in notebooks/, one level below the repo root, so
    Path.cwd() (Jupyter sets it to the notebook's own directory) is no
    longer the project root by itself. Walk upward for the marker that
    identifies the root instead of assuming a fixed number of parent hops."""
    for p in (start, *start.parents):
        if (p / "setup_env.sh").exists():
            return p
    return start

PROJECT_ROOT   = Path(os.environ.get("PANGENOMIC_ROOT", _find_project_root(Path.cwd()))).resolve()
INTERACT_DIR   = PROJECT_ROOT / "interection_data"
HISTONE_DIR    = PROJECT_ROOT / "histone"
RNASEQ_DIR     = PROJECT_ROOT / "rna_seq"
REF_DIR        = PROJECT_ROOT / "reference"
CACHE_DIR      = PROJECT_ROOT / "cache";   CACHE_DIR.mkdir(exist_ok=True)
# RESULTS_DIR / FIG_DIR: created below, once CFG.fast_mode is known (trimodal notebook picks the folder name from it)

GENOME_2BIT    = REF_DIR / "hg19.2bit"
REFGENE_TXT    = REF_DIR / "refGene_hg19.txt"
TPM_FILE       = RNASEQ_DIR / "rnaseq.rsem_genes_tpm.txt"

# --------------------------------------------------------------- parallelism
# Embarrassingly-parallel per-item work (loop-file parsing, promoter sequence
# extraction, per-(gene,tissue) graph construction) is farmed out to a process
# pool sized off the real core count. Training itself stays single-process
# (one model on one device) but uses DataLoader workers for batch collation
# and, on CUDA, automatic mixed precision -- see cells 15/16 (model + training).
N_CPU = os.cpu_count() or 1
# This box has 128 cores / ~2 TB RAM and exactly ONE GPU -- graph
# construction (CPU-bound, this pool) can use nearly all of both; model
# training cannot be parallelised across GPUs there aren't, so its speed is
# bounded by the single L40 regardless of how much CPU/RAM sits idle.
# Leave 2 cores for the OS/kernel driver rather than the previous 4/cap-116.
N_WORKERS = max(1, N_CPU - 2)


@dataclass
class Config:
    # ---- reproducibility
    seed: int = 42

    # ---- scale
    # Controlled by env var so ONE notebook file drives both the fast dry-run
    # and the full-scale run from the command line, without hand-editing
    # this cell between the two (e.g. PANGENOMIC_FAST_MODE=1 for the dry run).
    # This SERVER notebook always runs full scale -- PANGENOMIC_FAST_MODE is
    # deliberately NOT consulted here (unlike the notebook this forks), so an
    # environment variable left over from an earlier fast-mode smoke test on
    # this same machine can never silently downgrade an unattended full run.
    # Use the FINAL notebook (not this one) for a fast-mode dry run.
    fast_mode: bool = False
    fast_n_tissues: int = 8           # tissues kept when fast_mode
    fast_n_genes: int = 2500          # genes kept when fast_mode
    explicit_tissue_ids: tuple = ()   # if set, use exactly these sample_ids (in order) instead of fast_mode subsampling

    # ---- genome / promoter
    genome_build: str = "hg19"
    # Long context is the point of the Hyena tower: a dilated CNN needs
    # log2(L) layers to reach 16 kb, a Hyena operator reaches it in one.
    promoter_window: int = 1_000_000  # bp, centred on the canonical TSS -- 1 Mb variant
    seq_len_fast: int = 1_000_000     # same 1 Mb window in fast_mode too -- fast_mode only shrinks tissues/genes here

    # ---- graph construction
    max_neighbours: int = 10000         # cap on promoter degree (by CHiCAGO score)
    min_chicago: float = 5.0          # standard CHiCAGO significance cut
    add_backbone_edges: bool = True   # GrapHi-C style linear-adjacency edges

    # ---- histone
    histone_marks: tuple = ("H3K4me1", "H3K4me3", "H3K27ac",
                            "H3K27me3", "H3K36me3", "H3K9me3", "CTCF")
    histone_stat: str = "mean"        # bigWig summary over each fragment

    # ---- labels
    tpm_threshold: float = 1.0        # expressed if TPM > this
    tau_threshold: float = 0.85       # tissue-restricted if tau > this
    # Housekeeping = tau below this run's own MEDIAN tau (computed in the labels
    # cell). A fixed 0.2 cut is unreachable for this panel (tau floor ~0.226).
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
    stem_stride: int = 64             # forced up from 4: keeps L_eff ~15.6k at a 1 Mb window
    rc_equivariant: bool = True       # average forward and reverse-complement

    # ---- ablation
    # Every one of PangenomicGAT's 7 tower combinations, not a subset --
    # "structure + epigenome" (str_epi) and "sequence + epigenome" (seq_epi)
    # are both in here alongside the three single-tower arms and all3.
    variants: tuple = ("all3", "seq_str", "seq_epi", "str_epi", "seq", "str", "epi")
    run_hybrid: bool = True           # DHL-style classical heads on the embeddings

    # ---- encoders: where the frozen towers of Stage B come from
    # "per_split"  (default) Stage A trains all3 end to end on each split's own
    #              training graphs; Stage B trains every arm's fusion + heads on
    #              those frozen encoders. No test gene or held-out tissue ever
    #              reaches an encoder.
    # "checkpoint" load seq/struc/epi from `pretrained_checkpoint`. The file must
    #              record `train_genes`; test genes it was trained on are dropped
    #              from scoring (count printed). Legacy files without that
    #              record are refused.
    # "none"       no Stage A: every arm trains all towers end to end (slow).
    encoder_source: str = "per_split"
    pretrained_checkpoint: str = ""
    encoder_epochs: int = 20          # Stage A max epochs (all3, end to end)
    encoder_patience: int = 4         # Stage A early stopping
    # Stage B (fusion + heads on cached frozen features). A FRESH GatedFusion on
    # mature encoders needs many epochs to learn to down-weight the
    # tissue-invariant z_seq for H4; early stopping on the aggregate loss used to
    # cut it off at epoch 0, hence the floor. Stage B epochs cost seconds.
    head_lr_scale: float = 0.3        # Stage B lr = lr * this
    head_batch_size: int = 256
    min_epochs_heads: int = 20

    # ---- loss weights  (H1 link, H2 expression, H3 tau, H4 tissue, alignment)
    w_link: float = 1.0
    w_expr: float = 1.0
    # H2's SECOND head: expressed/not-expressed (TPM > tpm_threshold), a
    # binary classifier on the same fused embedding z as head_expr. Weighted
    # lower than w_expr because the continuous signal is the primary target
    # this notebook cares about -- the binary head is reported in full
    # alongside it, not as an afterthought, but it does not get to dominate
    # the shared trunk's gradient.
    w_expr_bin: float = 0.5
    w_tau: float = 0.5
    w_tissue: float = 0.5
    w_align: float = 0.3
    temperature: float = 0.1          # InfoNCE temperature
    # Tolerance bands for the continuous H2 head's "match accuracy" (see
    # regression_metrics): a prediction within this many log2(TPM+1) units
    # of the truth counts as a match. 0.5/1.0/2.0 log2 units are
    # ~1.4x/2x/4x fold-change -- reported together, not cherry-picked.
    expr_match_tolerances: tuple = (0.5, 1.0, 2.0)

    # ---- training
    n_folds: int = 5
    group_split_by: str = "gene"      # 'gene' | 'chrom' | 'none'
    epochs: int = 40                  # Stage B (and "none" mode) max epochs
    epochs_fast: int = 12
    # 16, not larger: raising it to 128 didn't speed up training at all
    # (epoch time was identical either way -- the bottleneck is real compute,
    # not batch-launch overhead) and DID cause a CUDA OOM once a batch
    # happened to cluster several high-degree promoter graphs together
    # (max_neighbours=10,000 means individual graph size is variable, so
    # batch memory isn't a fixed quantity the way it is for dense tensors).
    # 16 has run clean across many multi-hour fast- and full-scale attempts.
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 8                 # early stopping on validation loss
    # DataLoader workers only collate small PyG graphs; the sequence gather
    # happens on the GPU. More workers just multiply fork/COW cost.
    num_workers: int = min(8, N_CPU)
    # bf16 autocast (Stage A / end-to-end training and every eval pass). fp16
    # overflowed in Tower A (train loss NaN in fast runs #1-4); bf16 has fp32's
    # exponent range, needs no GradScaler, and fftconv upcasts to fp32 anyway.
    # Non-finite steps are skipped and counted; see run_epoch.
    use_amp: bool = True
    amp_dtype: str = "bf16"
    allow_tf32: bool = True
    # cuDNN determinism costs real conv throughput; per-run seeding still makes
    # splits, init and shuffling reproducible.
    deterministic: bool = False
    # Keep the uint8 sequence bank on the GPU when it fits next to training.
    seq_bank_on_gpu: str = "auto"     # "auto" | "yes" | "no"

    # ---- validation-scheme scale
    # Leave-one-tissue-out costs one Stage A per held-out tissue. 0 = all tissues.
    loto_n_tissues: int = 5

    @property
    def seq_len(self) -> int:
        return self.seq_len_fast if self.fast_mode else self.promoter_window

    @property
    def n_epochs(self) -> int:
        return self.epochs_fast if self.fast_mode else self.epochs


CFG = Config()
if CFG.encoder_source not in ("per_split", "checkpoint", "none"):
    raise ValueError(f"unknown encoder_source {CFG.encoder_source!r}")
if CFG.encoder_source == "checkpoint" and not Path(CFG.pretrained_checkpoint).is_file():
    raise FileNotFoundError(f"encoder_source='checkpoint' but no file at {CFG.pretrained_checkpoint!r}")

RESULTS_DIR = PROJECT_ROOT / "results" / ("results_trimodal_1mb_v2_fast" if CFG.fast_mode
                                          else "results_trimodal_1mb_v2_full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = RESULTS_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------- seeding
def set_all_seeds(seed: int):
    # PYTHONHASHSEED only matters at interpreter start-up, so it is not set here.
    random.seed(seed)
    try:
        import numpy as _np; _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _t
        _t.manual_seed(seed); _t.cuda.manual_seed_all(seed)
        _t.backends.cudnn.deterministic = CFG.deterministic
        _t.backends.cudnn.benchmark = not CFG.deterministic
        _t.backends.cuda.matmul.allow_tf32 = CFG.allow_tf32
        _t.backends.cudnn.allow_tf32 = CFG.allow_tf32
    except ImportError:
        pass


def run_seed(*parts) -> int:
    # Stable 32-bit seed from (base seed, run key, ...): sha256, not hash(),
    # which is salted per interpreter.
    import hashlib as _h
    return int.from_bytes(_h.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")

set_all_seeds(CFG.seed)
warnings.filterwarnings("ignore", category=FutureWarning)

print("PROJECT_ROOT :", PROJECT_ROOT)
print("CPU cores    :", N_CPU, f" (graph-building process pool: {N_WORKERS} workers)")
print("mode         :", "FAST (subsampled)" if CFG.fast_mode else "FULL")
print("promoter win :", CFG.seq_len, "bp")
print("epochs       :", CFG.n_epochs)
print("arms         :", list(CFG.variants), f"({len(CFG.variants)} total)")
print("encoders     :", CFG.encoder_source
      + (f" ({CFG.pretrained_checkpoint})" if CFG.encoder_source == "checkpoint" else ""))
print("results dir  :", RESULTS_DIR.relative_to(PROJECT_ROOT))
print()
for k, v in asdict(CFG).items():
    print(f"  {k:<20} {v}")


# %% [markdown]
# ## 2 · Environment guard and imports
#
# The interpreter was already checked by the preflight cell. This cell
#
# 1. **imports everything up front** and prints versions, so the run is
#    self-documenting;
# 2. **picks a device**, preferring CUDA, then Apple MPS, then CPU.
#
# Optional packages (`umap`, `torch_scatter`) are imported inside
# `try` blocks and set an availability flag; the pipeline degrades in a way that
# is announced rather than hidden. `hicstraw` is not imported at all in this
# notebook — no `.hic` file is ever opened.


# %%
import platform
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

print(f"python      : {platform.python_version()}")
print(f"executable  : {sys.executable}")
print(f"conda env   : {os.environ.get('CONDA_DEFAULT_ENV', '') or '(none detected)'}")

# ---------------------------------------------------------------- hard deps
import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib
# Headless-safe: use Agg only when NOT inside Jupyter (nohup/screen/sbatch runs
# with no $DISPLAY). Inside JupyterLab keep the inline backend so figures show
# in the notebook; savefig() still writes every figure to disk either way.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
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
    "scipy": __import__("scipy").__version__,
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

pyliftover = _try("pyliftover")
umap_mod   = _try("umap")
_          = _try("torch_scatter")
HAS_PYLIFTOVER = pyliftover is not None
HAS_UMAP       = umap_mod is not None

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
print(f"slurm cpus  : {os.environ.get('SLURM_CPUS_PER_TASK', 'n/a')}")

# ---------------------------------------------------------------- multiprocessing
# Worker pools below submit functions defined at notebook top level. That only
# pickles under the 'fork' start method (worker = copy of this interpreter).
# 'fork' is the Linux default; pin it explicitly anyway.
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass   # already set (e.g. this cell was re-run)
print(f"mp start method : {mp.get_start_method()}"
      + ("" if mp.get_start_method() == "fork" else
         "  ! expected 'fork', worker pools below may fail to pickle"))

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


# %% [markdown]
# ## 3 · Data inventory
#
# Before parsing anything we take stock of what is actually on disk. This cell is
# pure reconnaissance — it opens no large files — but it determines the shape of
# everything downstream:
#
# - **Which tissues exist**, parsed from the loop file names
#   (`0077_Liver_merged_loop.txt` → id `0077`, tissue `Liver`).
# - **Histone coverage matrix** — a tissue × mark grid of what is present. This
#   is the honest picture of your feature completeness. Any tissue/mark cell that
#   is empty here has its signal imputed later as the average of that mark across
#   the other tissues in this run, at the same fragment (section 8); the mask bit
#   still flags it as not directly measured for that tissue.
# - **RNA-seq sample columns** grouped back to tissue ids, so we know which
#   tissues have expression labels at all.
#
# Note the duplicate-mark files in `histone/` (`... (1).bw`). Those are treated as
# replicates of the same mark and averaged, not as separate features.


# %%
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

# plain nested dict, not a lambda-backed defaultdict: defaultdict(lambda: ...)
# cannot be pickled, and every process-pool worker below needs to receive
# this map as an argument to build its own independent HistoneExtractor.
hist_map = {sid: dict(mk_map) for sid, mk_map in hist_map.items()}

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
print("histone coverage (files per tissue x mark; 0 = imputed from other tissues)")
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
    print("     whatever remains missing is imputed as the cross-tissue average")
    print("     for that mark (section 8), not zero-filled.")

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

# ------------------------------------------------- reference
for f, why in [(GENOME_2BIT, "promoter sequence"), (REFGENE_TXT, "canonical TSS")]:
    print(f"\n{f.name:<22} {'present' if f.exists() else 'MISSING'}  ({why})")
    if not f.exists():
        print("   -> bash download_data.sh")


# %% [markdown]
# ## 4 · Parse the PCHi-C loop files
#
# Each `*_merged_loop.txt` has 12 tab-separated columns:
#
# | column | example | meaning |
# |---|---|---|
# | `bait_frag` | `chr1,927395,936954` | the promoter-capture bait fragment |
# | `other_frag` | `chr1,1000704,1005126` | the distal ("other end") fragment |
# | `N_reads` | `8` | supporting read pairs |
# | `score` | `5.16` | CHiCAGO score |
# | `bait_gene` | `RP11-54O7.17,HES4,ISG15` | genes annotated to the bait |
# | `oe_gene` | `RP11-465B22.3` | genes at the other end |
# | ...`_strand`, `_closest_gene` | | strand and nearest-gene fallbacks |
#
# Two things are worth being explicit about:
#
# **A bait fragment can carry several genes.** `HindIII`/`MboI` fragments are
# kilobases long and often span more than one TSS. We `explode` the comma-list so
# each (gene, bait, other-end) triple becomes one row. That means a single
# physical contact can support several gene-level graphs — correct, but it also
# means rows are *not* independent, which is exactly why the cross-validation
# splits by gene later on.
#
# **We filter on CHiCAGO score ≥ 5**, the threshold the CHiCAGO authors recommend
# for calling a significant interaction. Everything below is dropped rather than
# kept as a weak edge, because sub-threshold contacts are dominated by the
# distance-decay background and would teach the attention layer to attend to
# proximity rather than regulation.


# %%
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
if CFG.explicit_tissue_ids:
    order = {sid: i for i, sid in enumerate(CFG.explicit_tissue_ids)}
    sel = sel[sel["sample_id"].isin(order)].copy()
    sel["_order"] = sel["sample_id"].map(order)
    sel = sel.sort_values("_order").drop(columns="_order")
    print(f"EXPLICIT_TISSUE_IDS: using {len(sel)} of {len(tissue_df)} tissues: "
          f"{list(sel['tissue'])}\n")
elif CFG.fast_mode:
    # prefer tissues that have BOTH histone marks and RNA-seq, then top up
    has_hist = sel["sample_id"].isin(hist_map.keys())
    has_rna  = sel["sample_id"].isin(rna_by_tissue.keys())
    sel = pd.concat([sel[has_hist & has_rna], sel[~(has_hist & has_rna)]])
    sel = sel.head(CFG.fast_n_tissues)
    print(f"FAST_MODE: using {len(sel)} of {len(tissue_df)} tissues\n")

def _load_loops_task(args):
    path, sid, tissue, min_score = args
    return tissue, load_loops(path, sid, tissue, min_score)


tasks = [(r["path"], r["sample_id"], r["tissue"], CFG.min_chicago)
         for _, r in sel.iterrows()]
n_workers_here = max(1, min(N_WORKERS, len(tasks)))
print(f"parsing {len(tasks)} loop files across {n_workers_here} processes")

frames, raw_counts = [], {}
with ProcessPoolExecutor(max_workers=n_workers_here) as ex:
    futures = [ex.submit(_load_loops_task, t) for t in tasks]
    for fut in tqdm(as_completed(futures), total=len(futures),
                    desc="parsing loop files (parallel)"):
        tissue, (d, n0) = fut.result()
        frames.append(d)
        raw_counts[tissue] = (n0, len(d))

loops = pd.concat(frames, ignore_index=True)
del frames

# Some loop files report score = inf (CHiCAGO saturating). +inf is "maximally
# confident", not missing, so cap it at the largest finite score in the data
# HERE, before graph construction, rather than zeroing it later and patching
# the graph cache afterwards (what v1 did).
_inf = np.isinf(loops["score"].to_numpy())
SCORE_CAP = float(loops.loc[~_inf, "score"].max())
loops.loc[_inf, "score"] = SCORE_CAP
print(f"capped {int(_inf.sum()):,} infinite CHiCAGO scores at the max finite score {SCORE_CAP:.3f}")

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


# %% [markdown]
# ## 5 · Interaction QC plots
#
# Four diagnostics, each of which can invalidate the run if it looks wrong:
#
# 1. **Rows per tissue** — wildly uneven depth means the tissue-classification
#    head can cheat by learning graph size. We check the spread and, if it is
#    extreme, the `MAX_NEIGHBOURS` cap in cell 10 partially equalises it.
# 2. **CHiCAGO score distribution** — should be a long right tail starting at our
#    cut of 5. A spike exactly at the threshold means the upstream caller already
#    filtered, and our filter is a no-op.
# 3. **Contact distance** — the classic PCHi-C profile peaks in the tens-to-
#    hundreds of kb. A distribution peaked at the resolution limit suggests the
#    file contains raw proximity rather than called loops.
# 4. **Promoter degree** — how many partners each gene's bait has. This is the
#    distribution `MAX_NEIGHBOURS` truncates, so we look at it before choosing.


# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

# 1. rows per tissue
cnt = loops["tissue"].value_counts().sort_values()
axes[0, 0].barh(cnt.index, cnt.values, color=PALETTE[0])
axes[0, 0].set_title("Gene-exploded interaction rows per tissue")
axes[0, 0].set_xlabel("rows")
axes[0, 0].tick_params(axis="y", labelsize=7)

# 2. CHiCAGO score
axes[0, 1].hist(loops["score"].clip(upper=100), bins=80,
                color=PALETTE[1], log=True)
axes[0, 1].axvline(CFG.min_chicago, ls="--", c="k", lw=1,
                   label=f"cut = {CFG.min_chicago}")
axes[0, 1].set_title("CHiCAGO score (clipped at 100)")
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


# %% [markdown]
# ## 6 · Gene models, canonical TSS, and an assembly sanity check
#
# The loop files give us gene *symbols* and fragment coordinates, but not
# transcription start sites. We need TSS to define the promoter window for the
# sequence tower, so we read UCSC `refGene`.
#
# **Canonical TSS rule.** RefSeq lists many transcripts per symbol. We take, per
# gene symbol, the transcript with the longest span on the most common chromosome
# for that symbol, then its 5' end (`txStart` on `+`, `txEnd` on `-`). This is a
# pragmatic choice; if you have a preferred canonical set (MANE Select, APPRIS)
# swap it in here and nothing else changes.
#
# **Assembly verification — the important part.** We do not trust the assertion
# that these coordinates are hg19. Instead we test it: for a sample of rows, does
# the annotated bait gene's hg19 TSS actually fall inside the bait fragment? If
# the coordinates were hg38 the hit rate would collapse, because hg19→hg38 shifts
# most of chromosome 1 by hundreds of kilobases. A high hit rate confirms hg19; a
# low one aborts the notebook rather than letting you extract nonsense sequence.


# %%
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
rg = rg[rg["chrom"] == rg["name2"].map(modal_chrom)]
canon = (rg.sort_values("span", ascending=False)
           .drop_duplicates("name2")
           .set_index("name2"))
canon["tss"] = np.where(canon["strand"] == "+", canon["txStart"], canon["txEnd"])

GENE_INFO = canon[["chrom", "strand", "txStart", "txEnd", "tss"]].copy()

# ---------------------------------------------------------- protein-coding vs non-coding
# RefSeq's own accession prefix already tells us this, for free, from the same
# table used to build GENE_INFO above: NM_/XM_ = mRNA (protein-coding),
# NR_/XR_ = non-coding RNA. A symbol can have both kinds of transcript (e.g.
# a coding gene with an annotated non-coding isoform); it counts as
# protein-coding if ANY of its transcripts does.
GENE_CODING = (rg.assign(is_coding=rg["name"].str.startswith(("NM_", "XM_")))
                .groupby("name2")["is_coding"].any()
                .reindex(GENE_INFO.index))
print(f"\nprotein-coding vs non-coding, by RefSeq accession prefix "
      f"({GENE_CODING.sum():,} coding / {(~GENE_CODING).sum():,} non-coding "
      f"of {len(GENE_CODING):,} gene symbols with a canonical model)")
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


# %% [markdown]
# ## 7 · Promoter sequence from hg19
#
# For every surviving gene we cut a `PROMOTER_WINDOW`-bp window centred on the
# canonical TSS and one-hot encode it as a `4 x L` matrix (A, C, G, T).
#
# Details that matter:
#
# - **Strand.** For `-` strand genes we reverse-complement, so "upstream" means
#   the same thing for every gene and the CNN sees a consistent orientation.
# - **`N` and soft-masked bases.** `py2bit` returns lowercase for repeat-masked
#   regions; we upper-case them (repeats are real sequence and promoters are
#   GC-rich, so masking them would bias the encoder). Genuine `N` becomes an
#   all-zero column, which the CNN reads as "no information" rather than a base.
# - **Edge clipping.** Windows that run off a chromosome end are N-padded and
#   the gene is flagged, not dropped.
# - **Storage.** Bases are stored as uint8 codes (A,C,G,T,N = 0..4), one byte
#   per base: `n_genes x L`, ~20 GB at full scale, a quarter of a one-hot
#   array. The cache is an uncompressed `.npy` loaded with `mmap_mode`, so a
#   restart maps it instantly instead of decompressing for minutes. The model
#   one-hot encodes each batch on the GPU (see `SeqBank`).
#
# The GC-content plot afterwards is the sanity check: real promoter windows show
# a pronounced GC peak relative to the genome background of ~41%. If your
# histogram is centred at 41% you are extracting random genomic sequence, which
# means the TSS mapping is wrong.


# %%
COMP = str.maketrans("ACGTacgtN", "TGCAtgcaN")
# byte -> base code; anything that is not A/C/G/T (either case) is N (4)
_BASE_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _BASE_LUT[ord(_b)] = _i; _BASE_LUT[ord(_b.lower())] = _i
N_CODE = 4


def encode_bases(seq: str) -> np.ndarray:
    # vectorised: one table lookup per byte instead of a Python loop per base
    return _BASE_LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def atomic_save(save_fn, path: Path):
    # write to <path>.tmp, fsync, then rename: a killed job leaves either the
    # old file or the new one, never a truncated file that the next run loads
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        save_fn(fh)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def content_key(*parts) -> str:
    # short, stable hash of everything that determines a cache's contents
    h = hashlib.sha256()
    for part in parts:
        h.update(json.dumps(part, sort_keys=True, default=str).encode()); h.update(b"\0")
    return h.hexdigest()[:16]


def _seq_worker_init(genome_2bit_path):
    # one py2bit handle per worker process; nothing shared across processes
    global _W_TB, _W_CHROM_SIZES
    import py2bit as _py2bit
    _W_TB = _py2bit.open(str(genome_2bit_path))
    _W_CHROM_SIZES = _W_TB.chroms()


def _extract_one_promoter(args):
    gene, chrom, tss, strand, L = args
    if chrom not in _W_CHROM_SIZES:
        return gene, None, None
    half = L // 2
    s, e = tss - half, tss + (L - half)
    pad_l = max(0, -s); pad_r = max(0, e - _W_CHROM_SIZES[chrom])
    try:
        seq = _W_TB.sequence(chrom, max(0, s), min(_W_CHROM_SIZES[chrom], e))
    except RuntimeError:
        return gene, None, None
    seq = "N" * pad_l + seq + "N" * pad_r
    if strand == "-":
        seq = seq.translate(COMP)[::-1]
    codes = encode_bases(seq)
    if codes.shape[0] != L:                    # defensive: exact width
        codes = np.pad(codes, (0, max(0, L - codes.shape[0])), constant_values=N_CODE)[:L]
    return gene, codes, int(pad_l + pad_r)     # total padded bases


genes_needed = sorted(loops["gene"].unique())
if CFG.fast_mode and len(genes_needed) > CFG.fast_n_genes:
    # keep the genes seen in the most tissues -> denser gene x tissue matrix
    freq = loops.groupby("gene")["sample_id"].nunique().sort_values(ascending=False)
    genes_needed = sorted(freq.head(CFG.fast_n_genes).index)
    loops = loops[loops["gene"].isin(genes_needed)].reset_index(drop=True)
    print(f"FAST_MODE: kept {len(genes_needed):,} most widely-observed genes "
          f"-> {len(loops):,} rows")

L = CFG.seq_len
_seq_key = content_key("promoter_codes_v1", CFG.genome_build, L, genes_needed,
                       GENE_INFO.loc[genes_needed, ["chrom", "tss", "strand"]].values.tolist())
SEQ_CODES_PATH = CACHE_DIR / f"promoter_codes_{CFG.genome_build}_{L}_{_seq_key}.npy"
SEQ_META_PATH  = CACHE_DIR / f"promoter_codes_{CFG.genome_build}_{L}_{_seq_key}.meta.npz"
_legacy_npz = CACHE_DIR / f"promoter_onehot_{CFG.genome_build}_{L}_{len(genes_needed)}.npz"

if SEQ_CODES_PATH.exists() and SEQ_META_PATH.exists():
    with np.load(SEQ_META_PATH) as _m:
        SEQ_GENES = [str(g) for g in _m["genes"]]; seq_flags = _m["flags"]
    SEQ_CODES = np.load(SEQ_CODES_PATH, mmap_mode="r")
    print(f"mapped cached promoter codes: {SEQ_CODES_PATH.name}")
else:
    if _legacy_npz.exists():
        # One-off migration from the v1 one-hot cache (same windows, same
        # extraction), only if it holds exactly the genes needed here.
        print(f"converting legacy one-hot cache {_legacy_npz.name} (one-off)")
        with np.load(_legacy_npz, allow_pickle=False) as _z:
            _genes = [str(g) for g in _z["genes"]]
            _have = set(_genes)
            if _genes == [g for g in genes_needed if g in _have]:
                _oh = _z["onehot"]; seq_flags = _z["flags"].astype(np.int64)
                SEQ_GENES = _genes
                SEQ_CODES = np.empty((len(_genes), L), dtype=np.uint8)
                for i in tqdm(range(0, len(_genes), 256), desc="one-hot -> codes"):
                    blk = _oh[i:i + 256]
                    SEQ_CODES[i:i + 256] = np.where(blk.any(1), blk.argmax(1), N_CODE)
                del _oh
            else:
                print("  gene list differs from this run -- re-extracting instead")
                SEQ_CODES = None
    else:
        SEQ_CODES = None

    if SEQ_CODES is None:
        if not GENOME_2BIT.exists():
            raise FileNotFoundError(f"{GENOME_2BIT} missing — run: bash download_data.sh")
        tasks = [(g, GENE_INFO.at[g, "chrom"], int(GENE_INFO.at[g, "tss"]),
                  GENE_INFO.at[g, "strand"], L) for g in genes_needed]
        n_proc = max(1, min(N_WORKERS, len(tasks)))
        chunksize = max(1, len(tasks) // (n_proc * 4))
        print(f"extracting {len(tasks):,} promoters across {n_proc} processes "
              f"(chunksize={chunksize})")
        # filled in place: no list of per-gene arrays + np.stack (2x peak memory)
        SEQ_CODES = np.empty((len(tasks), L), dtype=np.uint8)
        SEQ_GENES, flags = [], []
        n_clip = n_bad = 0
        with ProcessPoolExecutor(max_workers=n_proc, initializer=_seq_worker_init,
                                 initargs=(GENOME_2BIT,)) as ex:
            for g, codes, flag in tqdm(ex.map(_extract_one_promoter, tasks, chunksize=chunksize),
                                       total=len(tasks), desc="extracting promoters (parallel)"):
                if codes is None:
                    n_bad += 1; continue
                n_clip += bool(flag)
                SEQ_CODES[len(SEQ_GENES)] = codes
                SEQ_GENES.append(g); flags.append(flag)
        SEQ_CODES = SEQ_CODES[:len(SEQ_GENES)]
        seq_flags = np.array(flags, dtype=np.int64)
        print(f"\nextracted {len(SEQ_GENES):,} promoters   "
              f"(clipped at chromosome end: {n_clip}, failed: {n_bad})")

    atomic_save(lambda fh: np.save(fh, SEQ_CODES), SEQ_CODES_PATH)
    atomic_save(lambda fh: np.savez(fh, genes=np.array(SEQ_GENES), flags=seq_flags), SEQ_META_PATH)
    print(f"cached -> {SEQ_CODES_PATH.name}")
    del SEQ_CODES
    SEQ_CODES = np.load(SEQ_CODES_PATH, mmap_mode="r")

GENE2SEQ = {g: i for i, g in enumerate(SEQ_GENES)}
print(f"promoter codes: {SEQ_CODES.shape}  dtype={SEQ_CODES.dtype}  "
      f"{SEQ_CODES.nbytes/1e9:.1f} GB (memory-mapped)")

# ---------------------------------------------------------------- QC
# chunked, so no (n_genes, L) temporaries in wider dtypes
gc = np.empty(len(SEQ_GENES)); n_frac = np.empty(len(SEQ_GENES))
pos_gc = np.zeros(L)
for i in range(0, len(SEQ_GENES), 512):
    blk = np.asarray(SEQ_CODES[i:i + 512])
    is_gc = (blk == 1) | (blk == 2)
    gc[i:i + 512] = is_gc.mean(1)
    n_frac[i:i + 512] = (blk == N_CODE).mean(1)
    pos_gc += is_gc.sum(0)
pos_gc /= len(SEQ_GENES)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].hist(gc, bins=60, color=PALETTE[0])
ax[0].axvline(0.41, ls="--", c="r", lw=1.2, label="genome background 41%")
ax[0].axvline(gc.mean(), ls="-", c="k", lw=1.2, label=f"promoter mean {gc.mean():.1%}")
ax[0].set_title("GC content of extracted promoter windows")
ax[0].set_xlabel("GC fraction"); ax[0].legend()

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


# %% [markdown]
# ## 8 · Histone-mark features from bigWig
#
# Each node in a promoter graph is a restriction fragment, and we summarise every
# available ChIP-seq mark over that fragment's span with `pyBigWig.stats(...,
# "mean")`.
#
# **Missing tracks are imputed from other tissues, not zero-filled.** For every
# (tissue, mark) we emit two numbers per node:
#
# - the **signal** — this tissue's own bigWig value when the track exists; when it
#   does not, the **mean of that mark over every other tissue in this run that does
#   have it, at the same fragment** (falling back to 0.0 only if no tissue anywhere
#   carries the mark), and
# - a **mask bit** — 1 if the value is a direct measurement for this tissue, 0 if it
#   is a cross-tissue average.
#
# That doubles the histone feature width. Plain zero-filling would tell the model
# "this fragment has no H3K27ac", which is a different and false claim from "we
# never measured H3K27ac here"; the cross-tissue average is a better point estimate
# than zero when a mark is genuinely missing for one tissue but present in others,
# and the mask bit still lets the network tell a real measurement from an imputed
# one apart and weight them differently.
#
# **Replicate handling.** Files like `0056.H3K9me3 (1).bw` are averaged with
# `0056.H3K9me3.bw` rather than treated as separate marks.
#
# **Transform.** ChIP fold-change is heavy-tailed, so we apply `log1p` and then
# clip at the 99.5th percentile per mark before standardising. Without this a
# handful of hyper-enriched fragments dominate the scale and the GAT effectively
# sees a binary feature.
#
# Signal is extracted once per unique (tissue, fragment) and cached, since the
# same fragment recurs in many gene graphs.
#
# ### Assembly: we lift the query, not the track
#
# ENCODE has retired most of its hg19 output, so `fetch_histone_encode.py` falls
# back to GRCh38 and saves those files as `{sid}.{mark}.GRCh38.bw`. Converting a
# whole genome-wide bigWig to hg19 means rebuilding signal at ~3 billion
# positions — tens of GB of scratch and minutes per track — when we only ever ask
# about a few hundred thousand restriction fragments.
#
# So `CoordLifter` converts each **query interval** hg19 → GRCh38 with
# `pyliftover` instead. This is exact for interval lookups and essentially free.
# Intervals that fail to lift (a real minority, in assembly-discordant regions)
# return no signal for that fragment and are counted, so the failure rate is
# visible rather than assumed to be zero.
#
# **A failed read is not a measurement.** If a track exists for this tissue but
# the read fails (unliftable interval, chromosome absent from the file,
# unreadable file, pyBigWig error), the value is imputed from the other tissues
# exactly like a missing track and its mask bit is 0. Failures are counted and
# the first few are printed. A region with no coverage in the bigWig is
# different: that is a real 0 and stays "measured".
#
# This notebook has no other consumer for `CoordLifter` — the `.hic` (Juicer)
# matrices that would otherwise reuse it for GRCh38 lookups are never opened here.


# %%
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


def _format_imputation_report(n_direct, n_imputed, n_unavailable, n_failed=0):
    tot = n_direct + n_imputed + n_unavailable
    if tot == 0:
        return "no signal lookups yet"
    return (f"{n_direct:,} direct  +  {n_imputed:,} imputed "
            f"(cross-tissue average)  +  {n_unavailable:,} unavailable "
            f"(no tissue anywhere has the mark)  =  {tot:,} (tissue, mark) "
            f"lookups  ({100*n_imputed/tot:.1f}% imputed; {n_failed:,} of the "
            f"imputed were failed reads of an existing track)")


class HistoneExtractor:
    # Lazily opens bigWigs and caches per-(sample, fragment) mean signal. When a
    # tissue lacks a given mark, its signal is imputed as the mean of that same
    # mark over every OTHER tissue in this run that does have it, at the exact
    # same fragment -- never a silent zero. The mask bit still records whether
    # the value shown for THIS tissue is a direct measurement (1) or an
    # imputed cross-tissue average (0), so the network can tell them apart.
    # hist_map: sid -> mark -> [(path, assembly)]
    def __init__(self, hist_map, marks, stat="mean", lifter=None,
                 native="hg19", tissue_ids=None):
        self.hist_map, self.marks, self.stat = hist_map, list(marks), stat
        self.lifter, self.native = lifter, native
        self._open, self._cache, self._avg_cache = {}, {}, {}
        # tissues averaged over when imputing -- defaults to every sid seen in
        # hist_map, but should normally be the tissues actually used in this run
        self.tissue_ids = list(tissue_ids) if tissue_ids is not None else list(hist_map.keys())
        self.available = {
            sid: {mk: len(hist_map.get(sid, {}).get(mk, [])) > 0 for mk in self.marks}
            for sid in hist_map
        }
        self.n_direct = self.n_imputed = self.n_unavailable = self.n_failed = 0
        self.failures = []        # first few failed reads, for the log

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
            except RuntimeError as e:
                print(f"  ! cannot open {Path(p).name}: {e}")
                self._open[p] = None
        return self._open[p]

    def _fail(self, why):
        self.n_failed += 1
        if len(self.failures) < 10:
            self.failures.append(why)

    def _raw_mark_signal(self, sid, mark, chrom, start, end):
        # -> float, or None if no replicate could be READ. A replicate that
        # reads fine but has no coverage over the fragment contributes 0.0
        # (bigWig convention: no data = no signal). No caching here.
        entries = self.hist_map.get(sid, {}).get(mark, [])
        if not entries:
            return None
        vals, errs = [], []
        for path, asm in entries:                       # average replicates
            bw = self._bw(path)
            if bw is None:
                errs.append(f"unopenable {Path(path).name}"); continue
            if asm == self.native:
                q = (chrom, int(start), int(end))
            else:
                q = self.lifter.interval(chrom, start, end,
                                         src=self.native, dst=asm) \
                    if self.lifter else None
                if q is None:
                    errs.append("unliftable"); continue
            c, s, e = q
            chroms = bw.chroms()
            # GRCh38 files from ENCODE use 'chr1'; some use '1'
            if c not in chroms:
                alt = c[3:] if c.startswith("chr") else "chr" + c
                if alt not in chroms:
                    errs.append(f"{c} not in {Path(path).name}"); continue
                c = alt
            s, e = max(0, int(s)), min(int(e), chroms[c])
            if e <= s:
                errs.append("empty interval after clipping"); continue
            try:
                v = bw.stats(c, s, e, type=self.stat)[0]
            except RuntimeError as ex:
                errs.append(f"stats error: {ex}"); continue
            vals.append(float(v) if v is not None and np.isfinite(v) else 0.0)
        if vals:
            return float(np.mean(vals))
        self._fail(f"{sid}/{mark} {chrom}:{start}-{end}: {'; '.join(errs)}")
        return None

    def _cross_tissue_average(self, sid, mark, chrom, start, end):
        # Mean of `mark` at this exact fragment over every OTHER tissue in this
        # run that has the track. Cached per (mark, fragment) rather than per
        # sid -- the result does not depend on who is asking, and the same
        # fragment recurs across many tissues' graphs for the same gene.
        key = (mark, chrom, int(start), int(end))
        if key in self._avg_cache:
            return self._avg_cache[key]
        vals = []
        for other in self.tissue_ids:
            if other == sid or not self.has(other, mark):
                continue
            v = self._raw_mark_signal(other, mark, chrom, start, end)
            if v is not None:
                vals.append(v)
        out = float(np.mean(vals)) if vals else None
        self._avg_cache[key] = out
        return out

    def signal(self, sid, chrom, start, end):
        # -> (values[n_marks], mask[n_marks]); coordinates are given in self.native.
        # values[i] is this tissue's own measurement when it could be read
        # (mask=1); otherwise the cross-tissue average of the same mark at this
        # fragment (mask=0); otherwise 0.0 if no tissue anywhere has it.
        key = (sid, chrom, start, end)
        if key in self._cache:
            return self._cache[key]
        out = np.zeros(len(self.marks), dtype=np.float32)
        mask = np.zeros(len(self.marks), dtype=np.float32)
        for i, mk in enumerate(self.marks):
            v = self._raw_mark_signal(sid, mk, chrom, start, end) if self.has(sid, mk) else None
            if v is not None:
                out[i] = v; mask[i] = 1.0
                self.n_direct += 1
                continue
            avg = self._cross_tissue_average(sid, mk, chrom, start, end)
            if avg is not None:
                out[i] = avg
                self.n_imputed += 1
            else:
                self.n_unavailable += 1
        self._cache[key] = (out, mask)
        return out, mask

    def imputation_report(self):
        return _format_imputation_report(self.n_direct, self.n_imputed,
                                         self.n_unavailable, self.n_failed)

    def close(self):
        for bw in self._open.values():
            if bw is not None:
                bw.close()
        self._open.clear()


HIST = HistoneExtractor(hist_map, CFG.histone_marks, CFG.histone_stat,
                        lifter=LIFTER, native=CFG.genome_build,
                        tissue_ids=sel["sample_id"].tolist())
N_MARKS = len(CFG.histone_marks)

avail_df = pd.DataFrame(HIST.available).T.reindex(sel["sample_id"]).fillna(False)
avail_df.index = sel.set_index("sample_id")["tissue"]
print("histone availability for the tissues in this run")
print(avail_df.astype(int).to_string())
frac = avail_df.values.mean()
print(f"\n{frac:.1%} of (tissue x mark) tracks are directly measured; "
      f"the rest have their signal imputed as the average of that mark "
      f"over the other tissues in this run (mask stays 0, so the model "
      f"can still tell a real measurement from an imputed one)")

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
    print("\nNo histone tracks for any selected tissue — there is nothing to "
          "average over, so all histone features fall back to 0. The model "
          "still runs on topology + sequence.")

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
# Floor the divisor: a near-constant mark carries no information, and dividing
# by a tiny std manufactures huge values from noise. Hard-clip the result too.
HIST_STD  = np.maximum(raw_log.clip(0, HIST_CLIP).std(0), 0.05).astype(np.float32)
HIST_Z_CLIP = 10.0


def transform_hist(v, clip=None, mean=None, std=None):
    clip = HIST_CLIP if clip is None else clip
    mean = HIST_MEAN if mean is None else mean
    std = HIST_STD if std is None else std
    z = (np.log1p(np.clip(v, 0, None)).clip(0, clip) - mean) / std
    return np.clip(z, -HIST_Z_CLIP, HIST_Z_CLIP).astype(np.float32)

print("\nper-mark transform fitted on 5,000 sampled fragments")
print(pd.DataFrame({"mark": list(CFG.histone_marks), "clip_p99.5": HIST_CLIP.round(3),
                    "mean": HIST_MEAN.round(3), "std": HIST_STD.round(3)}
                   ).to_string(index=False))
print(f"\nsignal provenance so far: {HIST.imputation_report()}")
for _why in HIST.failures:
    print(f"  failed read: {_why}")

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


# %% [markdown]
# ## 9 · Build the promoter-centred subgraphs
#
# This is the heart of the input pipeline: one `torch_geometric.data.Data` object
# per **(gene, tissue)** pair, structured exactly as the slide's panel 1.
#
# **Nodes.** Index 0 is always the bait fragment — the promoter anchor. Indices
# 1..k are the distal other-end fragments, capped at `MAX_NEIGHBOURS` and chosen
# by descending CHiCAGO score.
#
# Node feature vector (`F = 4 + 2*N_MARKS` dimensions):
#
# | block | dims | content |
# |---|---|---|
# | `is_bait` | 1 | 1 for the anchor, 0 for partners |
# | `log_len` | 1 | log10 fragment length |
# | `signed_log_dist` | 1 | signed log10 distance to the TSS (0 for the bait) |
# | `same_chrom` | 1 | 1 for cis partners, 0 for trans |
# | histone signal | 7 | standardised log1p mark signal over the fragment (imputed from other tissues' average when this tissue lacks the mark) |
# | histone mask | 7 | 1 if that mark is a direct measurement for this tissue, 0 if imputed |
#
# **Edges.** Bidirectional bait↔partner edges, plus — following the GrapHi-C
# convention from the earlier review — optional **backbone edges** joining
# partners that are adjacent along the chromosome. Those carry no interaction
# evidence but they give the attention layer a notion of linear genomic order.
#
# Edge feature vector (4 dims): CHiCAGO score (log1p), `N_reads` (log1p), signed
# log distance, `same_chrom`. Backbone edges get score/reads of 0 and are
# distinguishable by their zeroed evidence channels. **No Hi-C matrix feature is
# added here** — this notebook's graphs are built from PCHi-C loop calls only,
# so there is no `hic_count` / `has_hic` channel to include.
#
# **Link-prediction supervision.** For head H1 we hold out `LINK_HOLDOUT_FRAC` of
# the real edges: they are removed from `edge_index` (so the encoder genuinely
# cannot see them) and stored in `pos_edge_label_index`. An equal number of
# **distance-matched negatives** is drawn — fragment pairs at a similar genomic
# separation that are *not* connected. Distance matching is essential: with random
# negatives, the head learns the distance-decay curve and reports AUC ≈ 0.95 while
# having learned no biology at all.
#
# **Parallel construction.** Each (gene, tissue) pair is independent, so this is
# run as a `ProcessPoolExecutor` pool of `N_WORKERS` processes (cell 1). Every
# worker builds its own `HistoneExtractor` — its own `CoordLifter`, its own
# lazily-opened bigWig handles — so no file descriptor is ever shared across
# processes; only picklable data (row dicts, small lookup tables, plain-dict
# `hist_map`) crosses the process boundary. The one place this changes behaviour:
# the holdout/negative-sampling RNG is no longer a single sequential stream, so
# each group gets its own seed derived by hashing `(CFG.seed, sample_id, gene)`
# with `sha256` — deterministic given `CFG.seed`, independent of worker/schedule
# order, but not bit-identical to a single-process run with the same seed.


# %%
def _bg_worker_init(hist_map_plain, marks, stat, ref_dir, native, tissue_ids,
                    gene_info_records, gene2seq, hist_clip, hist_mean, hist_std,
                    max_nb, add_backbone, holdout_frac, rng_seed):
    # Runs once per worker process. Builds an entirely independent
    # HistoneExtractor -- its own CoordLifter, its own lazily-opened bigWig
    # handles -- so no open file descriptor or read position is ever shared
    # across processes; each worker owns its I/O end to end. CoordLifter and
    # HistoneExtractor themselves are not redefined here: under the 'fork'
    # start method pinned in cell 4, the worker already has them (and every
    # other class/function/import from cells run so far) inherited from the
    # parent's memory image at fork time.
    global _W_HIST, _W_GENE_INFO, _W_GENE2SEQ
    global _W_CLIP, _W_MEAN, _W_STD, _W_MAX_NB, _W_BACKBONE, _W_HOLDOUT, _W_SEED
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    lifter = CoordLifter(ref_dir)
    _W_HIST = HistoneExtractor(hist_map_plain, marks, stat, lifter=lifter,
                               native=native, tissue_ids=tissue_ids)
    # pool workers leave via os._exit, so atexit never runs there;
    # multiprocessing's own finaliser does
    from multiprocessing.util import Finalize
    Finalize(_W_HIST, _W_HIST.close, exitpriority=10)
    _W_GENE_INFO = gene_info_records          # gene -> (chrom, strand, tss)
    _W_GENE2SEQ  = gene2seq
    _W_CLIP, _W_MEAN, _W_STD = hist_clip, hist_mean, hist_std
    _W_MAX_NB, _W_BACKBONE, _W_HOLDOUT, _W_SEED = (
        max_nb, add_backbone, holdout_frac, rng_seed)


def _w_transform_hist(v):
    return transform_hist(v, _W_CLIP, _W_MEAN, _W_STD)


def _stable_group_seed(rng_seed, sid, gene):
    # Deterministic, process-independent seed for this (sample, gene)'s
    # holdout/negative sampling -- reproducible under a fixed CFG.seed no
    # matter which worker or in what order the pool schedules the group.
    # Python's built-in hash() is randomised per-process via PYTHONHASHSEED,
    # so it cannot be used to seed across processes; sha256 can.
    h = hashlib.sha256(f"{rng_seed}|{sid}|{gene}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _build_one_group(task):
    # One (sample_id, gene) promoter subgraph, built entirely from the
    # per-worker globals set up by _bg_worker_init plus this task's own rows
    # -- nothing here reaches back into the parent process's state. Mirrors
    # the body of the old single-process build_graphs() loop exactly, just
    # operating on a list of row-dicts instead of a DataFrame group.
    sid, gene, rows = task
    if gene not in _W_GENE2SEQ:
        return None
    n0 = (_W_HIST.n_direct, _W_HIST.n_imputed, _W_HIST.n_unavailable, _W_HIST.n_failed)

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)[:_W_MAX_NB]
    if len(rows) < 2:                       # need at least two partners to be a graph
        return None

    tissue = rows[0]["tissue"]
    tss    = int(_W_GENE_INFO[gene][2])
    b_chr  = rows[0]["bait_chr"]
    b_s, b_e = int(rows[0]["bait_start"]), int(rows[0]["bait_end"])

    # ---- node 0 : the bait / promoter anchor
    # mask is per node: a failed read of an existing track is imputed, mask 0
    hv, hmask = _W_HIST.signal(sid, b_chr, b_s, b_e)
    nodes = [np.concatenate([
        [1.0, np.log10(max(b_e - b_s, 1)), 0.0, 1.0],
        _w_transform_hist(hv), hmask]).astype(np.float32)]
    node_pos = [(b_s + b_e) // 2]

    # ---- nodes 1..k : the partner fragments
    e_src, e_dst, e_attr = [], [], []
    for r in rows:
        o_chr, o_s, o_e = r["other_chr"], int(r["other_start"]), int(r["other_end"])
        mid = (o_s + o_e) // 2
        same = float(r["same_chrom"])
        d = (mid - tss) if same else 0.0
        sgn_log_d = np.sign(d) * np.log10(abs(d) + 1)

        hv, hmask = _W_HIST.signal(sid, o_chr, o_s, o_e)
        nodes.append(np.concatenate([
            [0.0, np.log10(max(o_e - o_s, 1)), sgn_log_d, same],
            _w_transform_hist(hv), hmask]).astype(np.float32))
        node_pos.append(mid)

        attr = np.array([
            np.log1p(float(r["score"])), np.log1p(float(r["N_reads"])),
            sgn_log_d, same], dtype=np.float32)

        j = len(nodes) - 1
        e_src += [0, j]; e_dst += [j, 0]; e_attr += [attr, attr]

    n_nodes = len(nodes)
    n_real_edges = len(e_src)

    # ---- optional backbone edges between genomically adjacent partners
    if _W_BACKBONE and n_nodes > 2:
        order = sorted(range(1, n_nodes), key=lambda i: node_pos[i])
        for a, b in zip(order[:-1], order[1:]):
            gap = abs(node_pos[a] - node_pos[b])
            attr = np.array([0.0, 0.0, np.log10(gap + 1), 1.0], dtype=np.float32)
            e_src += [a, b]; e_dst += [b, a]; e_attr += [attr, attr]

    edge_index = np.array([e_src, e_dst], dtype=np.int64)
    edge_attr  = np.stack(e_attr).astype(np.float32)

    rng = np.random.default_rng(_stable_group_seed(_W_SEED, sid, gene))

    # ---- hold out real edges for the link-prediction head -----------------
    real_pairs = [(0, j) for j in range(1, n_nodes)]
    n_hold = max(1, int(round(_W_HOLDOUT * len(real_pairs))))
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
    cand = [j for j in range(1, n_nodes) if j not in hold_set]
    neg_pairs = []
    if cand:
        pos_d = np.array([abs(node_pos[j] - node_pos[0]) for j in sorted(hold_set)])
        cand_d = np.array([abs(node_pos[j] - node_pos[0]) for j in cand])
        for pd_ in pos_d:
            order = np.argsort(np.abs(np.log10(cand_d + 1) - np.log10(pd_ + 1)))
            pick = cand[int(order[rng.integers(0, min(3, len(order)))])]
            neg_pairs.append([0, pick])
    if not neg_pairs:
        neg_pairs = [[0, int(rng.integers(1, n_nodes))]]
    neg_pairs = np.array(neg_pairs, dtype=np.int64).T

    # Return plain numpy arrays, not a torch_geometric.data.Data / torch.Tensor.
    # PyTorch installs a custom multiprocessing reducer for CPU tensors that
    # shares them via file descriptors (one fd per tensor) rather than copying
    # bytes; at the scale of tens of thousands of graphs x 6 tensors each, that
    # exhausts the kernel's ancillary-data buffer for passing fds between
    # processes ("RuntimeError: received 0 items of ancdata" ->
    # BrokenProcessPool). Numpy arrays pickle as plain bytes, so the Data
    # object is assembled from them in the main process instead (below).
    arrs = {
        "x": np.stack(nodes),
        "edge_index": edge_index[:, keep],
        "edge_attr": edge_attr[keep],
        "pos_edge_label_index": pos_pairs,
        "neg_edge_label_index": neg_pairs,
        "seq_idx": np.array([_W_GENE2SEQ[gene]], dtype=np.int64),
        "num_nodes": n_nodes,
    }

    meta = {"gene": gene, "sample_id": sid, "tissue": tissue,
           "chrom": b_chr, "n_nodes": n_nodes,
           "n_edges": int(keep.size), "n_pos": pos_pairs.shape[1],
           "mean_score": float(np.mean([r["score"] for r in rows]))}

    hist_delta = tuple(now - before for now, before in zip(
        (_W_HIST.n_direct, _W_HIST.n_imputed, _W_HIST.n_unavailable, _W_HIST.n_failed), n0))
    return arrs, meta, hist_delta


def build_graphs_parallel(loops_df, gene_info, gene2seq, hist_map_plain,
                          marks, stat, ref_dir, native, tissue_ids,
                          hist_clip, hist_mean, hist_std,
                          max_nb, add_backbone, holdout_frac, rng_seed,
                          n_workers):
    used = gene_info.loc[gene_info.index.isin(set(loops_df["gene"].unique()))]
    gene_info_records = dict(zip(used.index, zip(used["chrom"], used["strand"],
                                                 used["tss"].astype(int))))

    # Grouped by gene first, so all tissues of a gene land in the same worker
    # chunk and share that worker's cross-tissue imputation cache. The graph
    # order is re-sorted to (sample_id, gene) at the end, as before.
    tasks = [(sid, gene, grp.to_dict("records"))
             for (gene, sid), grp in loops_df.groupby(["gene", "sample_id"], sort=True)]
    n_workers_here = max(1, min(n_workers, len(tasks)))
    chunksize = max(1, len(tasks) // (n_workers_here * 8))
    print(f"building {len(tasks):,} candidate (gene, tissue) graphs across "
          f"{n_workers_here} processes (chunksize={chunksize})")

    graphs, meta_rows = [], []
    totals = np.zeros(4, dtype=np.int64)
    with ProcessPoolExecutor(
            max_workers=n_workers_here, initializer=_bg_worker_init,
            initargs=(hist_map_plain, marks, stat, ref_dir, native, tissue_ids,
                     gene_info_records, gene2seq, hist_clip, hist_mean, hist_std,
                     max_nb, add_backbone, holdout_frac, rng_seed)) as ex:
        for result in tqdm(ex.map(_build_one_group, tasks, chunksize=chunksize),
                           total=len(tasks), desc="building graphs (parallel)"):
            if result is None:
                continue
            arrs, meta, delta = result
            d = Data(
                x=torch.from_numpy(arrs["x"]),
                edge_index=torch.from_numpy(arrs["edge_index"]),
                edge_attr=torch.from_numpy(arrs["edge_attr"]),
                pos_edge_label_index=torch.from_numpy(arrs["pos_edge_label_index"]),
                neg_edge_label_index=torch.from_numpy(arrs["neg_edge_label_index"]),
                seq_idx=torch.from_numpy(arrs["seq_idx"]),
            )
            d.num_nodes = arrs["num_nodes"]
            graphs.append(d); meta_rows.append(meta)
            totals += delta
    meta_df = pd.DataFrame(meta_rows)
    order = meta_df.sort_values(["sample_id", "gene"], kind="stable").index.to_numpy()
    graphs = [graphs[i] for i in order]
    meta_df = meta_df.loc[order].reset_index(drop=True)
    return graphs, meta_df, tuple(int(t) for t in totals)


# ---- disk cache: rebuilding ~350k graphs is the single most expensive step.
# The key hashes everything that determines a graph's contents: the gene and
# tissue lists, every histone file (path, size, mtime), the fitted transform,
# the score cap, the graph settings and a schema version. It is stored inside
# the file and checked on load, so a stale cache can never load silently.
GRAPH_SCHEMA_VERSION = "v2-failedread-imputed-zclip10"
_hist_files = sorted((str(pth), Path(pth).stat().st_size, Path(pth).stat().st_mtime_ns)
                     for sid in sel["sample_id"] for entries in hist_map.get(sid, {}).values()
                     for pth, _asm in entries)
GRAPH_CACHE_KEY = content_key(
    GRAPH_SCHEMA_VERSION, CFG.genome_build, CFG.seq_len, genes_needed,
    sel["sample_id"].tolist(), _hist_files, list(CFG.histone_marks), CFG.histone_stat,
    HIST_CLIP.tolist(), HIST_MEAN.tolist(), HIST_STD.tolist(), SCORE_CAP,
    CFG.max_neighbours, CFG.min_chicago, CFG.add_backbone_edges,
    CFG.link_holdout_frac, CFG.seed)
GRAPH_CACHE_PATH = CACHE_DIR / f"graphs_v2_{GRAPH_CACHE_KEY}.pt"

_cached = None
if GRAPH_CACHE_PATH.exists():
    print(f"loading cached graphs -> {GRAPH_CACHE_PATH.name}")
    _cached = torch.load(GRAPH_CACHE_PATH, weights_only=False)
    if _cached.get("key") != GRAPH_CACHE_KEY:
        print("  ! stored key does not match this run's inputs -- rebuilding")
        _cached = None
if _cached is not None:
    GRAPHS, META, _hist_deltas = _cached["graphs"], _cached["meta"], _cached["hist_deltas"]
    print(f"  {len(GRAPHS):,} graphs loaded from cache (rebuild skipped)")
else:
    GRAPHS, META, _hist_deltas = build_graphs_parallel(
        loops, GENE_INFO, GENE2SEQ, hist_map, CFG.histone_marks, CFG.histone_stat,
        REF_DIR, CFG.genome_build, sel["sample_id"].tolist(),
        HIST_CLIP, HIST_MEAN, HIST_STD,
        CFG.max_neighbours, CFG.add_backbone_edges, CFG.link_holdout_frac, CFG.seed,
        N_WORKERS)
    atomic_save(lambda fh: torch.save({"key": GRAPH_CACHE_KEY, "graphs": GRAPHS, "meta": META,
                                       "hist_deltas": _hist_deltas}, fh), GRAPH_CACHE_PATH)
    print(f"cached graphs -> {GRAPH_CACHE_PATH.name}  "
          f"({GRAPH_CACHE_PATH.stat().st_size/1e6:.0f} MB)")
del _cached
HIST.close()

# Non-finite features would mean a bug upstream (inf scores are capped at load
# time now), so fail loudly instead of silently patching tensors.
_bad = [i for i, g in enumerate(GRAPHS)
        if not (torch.isfinite(g.x).all() and torch.isfinite(g.edge_attr).all())]
assert not _bad, f"{len(_bad)} graphs have non-finite features, e.g. index {_bad[:5]}"

N_NODE_FEAT = GRAPHS[0].x.shape[1]
N_EDGE_FEAT = GRAPHS[0].edge_attr.shape[1]
print(f"\nbuilt {len(GRAPHS):,} promoter subgraphs")
print(f"  unique genes   : {META['gene'].nunique():,}")
print(f"  unique tissues : {META['tissue'].nunique()}")
print(f"  node features F   = {N_NODE_FEAT}  "
      f"(4 structural + {N_MARKS} marks + {N_MARKS} mask)")
print(f"  edge features F_e = {N_EDGE_FEAT}")
print(f"  histone signal provenance: {_format_imputation_report(*_hist_deltas)}")
if _hist_deltas[3]:
    print(f"  ! {_hist_deltas[3]:,} reads of existing tracks failed and were imputed "
          f"(mask 0) -- check the bigWig files / liftover chain if this is large")
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
EDGE_FEAT_NAMES = ["log_score", "log_reads", "sgn_log_d", "same_chrom"]

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


# %% [markdown]
# ## 10 · Look at an actual graph
#
# Numbers in a table are easy to get wrong without noticing. This cell draws one
# promoter subgraph so you can see that it has the shape the model expects: a hub
# (the bait, index 0) with spokes to partner fragments, laid out by genomic
# position, edge width proportional to CHiCAGO score, plus the backbone edges
# between adjacent partners.
#
# It also prints the raw feature matrix of a small graph so every column can be
# checked by eye against the table in cell 10.


# %%
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
                            "same_chrom"]).round(2).to_string())


# %% [markdown]
# ## 11 · Labels for all four heads
#
# **H2 — expressed in tissue (binary).** The RSEM TPM matrix has 194 sample
# columns named `{id}_{GSE}_{Tissue}_{rep}`. We average replicates within a
# sample id to get a gene × tissue TPM matrix, then binarise at `TPM > 1`.
# Averaging on the linear scale (not log) is deliberate: TPM is already a ratio
# and the arithmetic mean of replicates is the standard summary.
#
# **H3 — tissue-restricted vs housekeeping (binary).** We compute Yanai's tau
# across all tissues that have RNA-seq:
#
# $$\tau = \frac{\sum_{i=1}^{n}\left(1 - \frac{x_i}{\max_i x_i}\right)}{n-1}$$
#
# on log2(TPM+1). tau = 0 means perfectly uniform (housekeeping), tau = 1 means
# expressed in exactly one tissue. We label `tau > 0.85` as restricted. Note this
# is a **per-gene** label: it is constant across the tissues of the same gene,
# which is another reason the folds must be split by gene.
#
# **H4 — tissue class (multiclass).** The tissue index of the graph. This is the
# head most likely to saturate — the histone and contact profile largely gives the
# tissue away — so watch for it hitting ceiling accuracy while contributing no
# useful gradient.
#
# **H1 — link prediction (binary).** Already constructed in cell 10 as held-out
# positives plus distance-matched negatives, so there is nothing to build here;
# we just report the balance.
#
# Rows whose gene has no RNA-seq measurement are kept but get a `-1` label for
# H2/H3, and the loss masks them. Dropping them instead would discard perfectly
# good link-prediction and tissue-classification signal.


# %%
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

# ------------------------------------------------ H2 signal (continuous, replaces expressed/not binary)
# log2(TPM + 1) is the continuous expression SIGNAL this notebook predicts
# directly now, instead of thresholding it into expressed/not-expressed at
# CFG.tpm_threshold. log2(x+1) is the standard variance-stabilising transform
# for RNA-seq (raw TPM is heavy-tailed; a few genes would otherwise dominate
# a plain MSE). Safe as a -1-sentinel-maskable target: log2(TPM+1) >= 0
# always, so -1 can never collide with a real value.
SIGNAL = np.log2(TPM + 1)
_flat = SIGNAL.values[np.isfinite(SIGNAL.values)]
print(f"\nH2  signal = log2(TPM + 1): mean {_flat.mean():.3f}, std {_flat.std():.3f} "
      f"over {SIGNAL.size:,} gene x tissue cells")
EXPRESSED = (TPM > CFG.tpm_threshold).astype(int)   # H2's SECOND head's target
print(f"    {(TPM.values > CFG.tpm_threshold).mean():.1%} of cells are 'expressed' "
      f"at TPM > {CFG.tpm_threshold} -- this notebook trains a SECOND H2 head "
      f"(head_expr_bin) on exactly this binary target, alongside the continuous "
      f"signal head above. Both are real labels off the same TPM matrix, not one "
      f"derived from the other's prediction.")

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

# ------------------------------------------------ housekeeping vs non-housekeeping
# MEDIAN split of this run's own tau. A fixed 0.2 cut is unreachable for this
# panel (tau floor ~0.226) and produced an all-non-housekeeping label before.
HK_MEDIAN_CUT = float(tau[valid].median())
HOUSEKEEPING = pd.Series(-1, index=TPM.index, dtype=int)
HOUSEKEEPING[valid] = (tau[valid] < HK_MEDIAN_CUT).astype(int)
print(f"H*  housekeeping (tau < {HK_MEDIAN_CUT:.3f}, this run's median tau): "
      f"{(HOUSEKEEPING[valid] == 1).mean():.1%} of the {int(valid.sum()):,} genes "
      f"with a defined tau")
assert 0.1 < (HOUSEKEEPING[valid] == 1).mean() < 0.9, (
    "housekeeping split degenerated (should be ~50% by construction of a "
    "median split) -- something upstream of this cell is wrong")

# ------------------------------------------------ H4 tissue class
TISSUE_LIST = sorted(META["tissue"].unique())
TISSUE2IDX = {t: i for i, t in enumerate(TISSUE_LIST)}
print(f"H4  {len(TISSUE_LIST)} tissue classes")

# ------------------------------------------------ attach to graphs
# row_idx is this graph's own position in GRAPHS/META -- deliberately does
# NOT contain the substring "index", so PyG's DataLoader concatenates it on
# dim 0 with no cross-graph offset (same mechanism seq_idx already relies
# on), giving one plain row number per graph that survives batching. That is
# what lets Part IV join pooled eval-time predictions back to this row's
# gene, and from there to housekeeping/protein-coding status, after the
# fact -- without changing anything about how training or batching works.
def _lookup_2d(frame, rows, cols, missing):
    # frame.at[row, col] for every (row, col) pair at once; `missing` where the
    # gene or tissue is absent from the matrix
    ri = frame.index.get_indexer(rows); ci = frame.columns.get_indexer(cols)
    ok = (ri >= 0) & (ci >= 0)
    out = np.full(len(rows), missing, dtype=np.float64)
    out[ok] = frame.values[ri[ok], ci[ok]]
    return out


_g, _sid = META["gene"].to_numpy(), META["sample_id"].to_numpy()
y_expr = _lookup_2d(SIGNAL, _g, _sid, np.nan)
y_expr = np.where(np.isfinite(y_expr), y_expr, -1.0)
y_expr_bin = _lookup_2d(EXPRESSED, _g, _sid, -1)
y_tau = META["gene"].map(RESTRICTED.where(tau.notna())).fillna(-1).to_numpy()
y_hk = META["gene"].map(HOUSEKEEPING).fillna(-1).to_numpy()
_coding = META["gene"].map(GENE_CODING)
y_coding = np.where(_coding.isna(), -1, _coding.fillna(0).astype(bool).astype(int))
y_tissue = META["tissue"].map(TISSUE2IDX).to_numpy()

META["y_expr"]         = y_expr.astype(float)
META["y_expr_bin"]     = y_expr_bin.astype(int)
META["y_tau"]          = y_tau.astype(int)
META["y_tissue"]       = y_tissue.astype(int)
META["y_housekeeping"] = y_hk.astype(int)
META["y_coding"]       = y_coding.astype(int)
n_expr_known   = int((META.y_expr >= 0).sum())
n_tau_known    = int((META.y_tau >= 0).sum())
n_hk_known     = int((META.y_housekeeping >= 0).sum())
n_coding_known = int((META.y_coding >= 0).sum())

# row_idx is the graph's own position in GRAPHS/META. Its name deliberately
# lacks "index", so PyG concatenates it without a per-graph offset (same
# mechanism as seq_idx) and pooled predictions can be joined back to META.
_cols = {k: META[k].to_numpy() for k in
         ("y_expr", "y_expr_bin", "y_tau", "y_tissue", "y_housekeeping", "y_coding")}
for i, d in enumerate(GRAPHS):
    d.y_expr         = torch.tensor([_cols["y_expr"][i]], dtype=torch.float32)      # head_expr
    d.y_expr_bin     = torch.tensor([_cols["y_expr_bin"][i]], dtype=torch.float32)  # head_expr_bin
    d.y_tau          = torch.tensor([_cols["y_tau"][i]], dtype=torch.float32)
    d.y_tissue       = torch.tensor([_cols["y_tissue"][i]], dtype=torch.long)
    d.y_housekeeping = torch.tensor([_cols["y_housekeeping"][i]], dtype=torch.long)
    d.y_coding       = torch.tensor([_cols["y_coding"][i]], dtype=torch.long)
    d.row_idx        = torch.tensor([i], dtype=torch.long)

print(f"\nlabel coverage over {len(GRAPHS):,} graphs")
print(f"  H2 signal     : {n_expr_known:,} labelled ({n_expr_known/len(GRAPHS):.1%}), "
      f"mean {META.loc[META.y_expr>=0,'y_expr'].mean():.3f}, "
      f"std {META.loc[META.y_expr>=0,'y_expr'].std():.3f}")
print(f"  H2 expressed  : {int((META.y_expr_bin>=0).sum()):,} labelled "
      f"({(META.y_expr_bin>=0).mean():.1%}), "
      f"{META.loc[META.y_expr_bin>=0,'y_expr_bin'].mean():.1%} positive "
      f"(TPM > {CFG.tpm_threshold})")
print(f"  H3 tau        : {n_tau_known:,} labelled ({n_tau_known/len(GRAPHS):.1%}), "
      f"positive rate {META.loc[META.y_tau>=0,'y_tau'].mean():.1%}")
print(f"  H4 tissue     : {len(GRAPHS):,} labelled, {len(TISSUE_LIST)} classes")
print(f"  housekeeping  : {n_hk_known:,} labelled ({n_hk_known/len(GRAPHS):.1%}), "
      f"{META.loc[META.y_housekeeping>=0,'y_housekeeping'].mean():.1%} housekeeping")
print(f"  protein-coding: {n_coding_known:,} labelled ({n_coding_known/len(GRAPHS):.1%}), "
      f"{META.loc[META.y_coding>=0,'y_coding'].mean():.1%} protein-coding")
n_pos = int(META['n_pos'].sum())
print(f"  H1 links      : {n_pos:,} positives + {n_pos:,} distance-matched negatives")

# ------------------------------------------------ plots
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
axes[0].hist(META.loc[META.y_expr >= 0, "y_expr"], bins=60, color=PALETTE[0])
axes[0].axvline(np.log2(CFG.tpm_threshold + 1), ls="--", c="k",
                label=f"log2({CFG.tpm_threshold}+1) (expressed cut, head_expr_bin)")
axes[0].set_title("H2 signal distribution"); axes[0].set_xlabel("log2(TPM + 1)")
axes[0].legend(fontsize=7)

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
print("\nmean H2 signal by tissue (watch for a tissue trivially separable by scale alone)")
print(expr_by_tissue.round(3).to_string())


# %% [markdown]
# ## 12 · Cross-validation splits
#
# Five folds, split by **gene** using `GroupKFold`.
#
# Why this matters more than any hyperparameter: the same gene appears in up to 46
# tissue graphs, and its promoter sequence is *identical* in all of them (the
# sequence tower is tissue-invariant by construction). If gene G is in the
# training set for tissue A and the test set for tissue B, the model has already
# seen G's promoter and — for the tau head, whose label is constant per gene —
# already seen its exact answer. Random row-wise splitting would produce
# beautiful, meaningless AUCs.
#
# Grouping by gene closes that leak. The cost is that folds are not perfectly
# balanced in size, which `GroupKFold` handles by greedily balancing group sizes.
#
# The cell prints, for every fold: sizes, the positive rate of each head, and —
# the key check — that the train/test gene sets are disjoint. It also reports
# tissue coverage per fold, since a fold missing a tissue entirely would break the
# multiclass head.
#
# An optional `chrom` grouping is available in the config for a stricter,
# chromosome-held-out evaluation, which additionally removes any linkage between
# nearby genes sharing regulatory neighbourhoods.


# %%
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

_short_folds = split_df[split_df["tissues_test"] < len(TISSUE_LIST)]
if len(_short_folds):
    print(f"  note: {len(_short_folds)} fold(s) do not see every tissue in test; "
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


# %% [markdown]
# ## 13 · The model
#
# This is panels 2–4 of the slide, in code.
#
# ### Tower A — sequence encoder: a HyenaDNA-style long convolution
#
# Reimplemented from [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna)
# with no external dependency (the original pulls in einops, flash-attn and a
# config registry). The maths is theirs; the plumbing is ours.
#
# **The problem with the dilated CNN this replaces.** Four dilated conv layers
# reach roughly 500 bp. Promoter-proximal regulatory signal routinely sits several
# kilobases from the TSS, so the old tower was structurally incapable of seeing
# most of what it was being asked about. Stacking more layers reaches further but
# costs depth linearly in log(L) and blurs base-resolution detail through repeated
# pooling.
#
# **What a Hyena operator does instead.** One layer computes
#
# ```
# y = out_proj( x_0 * FFTConv(x_1 * v, h) )
# ```
#
# an `order`-step recurrence alternating element-wise gating with a long
# convolution whose filter `h` is **as long as the input**. So the receptive field
# of block 1 is the entire 16 kb window. Three properties follow:
#
# 1. **The filter is generated, not stored.** A small MLP maps a positional code
#    to filter values (`HyenaFilter`), so parameter count is independent of
#    sequence length — a 16 kb filter costs the same as a 1 kb one.
# 2. **`ExponentialModulation` gives channels different effective lengths.**
#    Without it, every channel is free to average over the whole window and the
#    filters collapse. With it, channels span a spectrum from a few hundred bp to
#    the full window, which is exactly the range between a core-promoter motif and
#    a distal element.
# 3. **Cost is O(L log L)**, via FFT, rather than O(L·k) for an explicit kernel or
#    O(L²) for attention.
#
# **Reverse-complement equivariance.** A promoter and its reverse complement are
# the same physical locus, so we embed both strands with shared weights and
# average. This halves the effective hypothesis space and removes an arbitrary
# strand convention from the features. The model cell asserts that
# `z(x) == z(rc(x))` — if that assertion ever fails, the features depend on which
# strand the annotation happened to pick.
#
# Modernised plumbing throughout: RMSNorm instead of BatchNorm, SwiGLU instead of
# a plain MLP, pre-norm residuals.
#
# ### Tower B — structure encoder (`StructureEncoder`)
# `L` × `GATv2Conv` with `K` heads and `edge_dim=N_EDGE_FEAT`, so the attention
# logits see the CHiCAGO score and distance, not just node states. GATv2 rather
# than the original GAT because GAT's attention is *static* — its ranking of
# neighbours is the same for every query node, which is exactly the wrong
# inductive bias when the whole question is "which partners matter for *this*
# promoter". Each layer is followed by residual + `LayerNorm` + dropout. Layers
# 1..L−1 concatenate heads; layer L averages them.
#
# Readout follows the slide: **gather the anchor node** (index 0 of each graph)
# and concatenate it with an attention-pooled summary of all nodes, then project
# to `z_str ∈ R^d`. **One vector per (gene, tissue).**
#
# ### Fusion — FiLM
# `z(g,t) = γ(z_str) ⊙ z_seq + β(z_str)`. The tissue-varying structure embedding
# *modulates* the tissue-invariant sequence embedding. The direction matters: this
# way tissue identity gates promoter identity, rather than being a weak additive
# nudge.
#
# ### Heads
# - **H1** inner-product edge decoder over the node states `H_L` (not the fused
#   graph vector — link prediction needs per-node representations).
# - **H2**, **H3** MLPs on `z(g,t)` → logit.
# - **H4** linear on `z(g,t)` → `T` logits.
# - **Alignment** InfoNCE between `z_seq` and `z_str` within the batch.
#
# Masked BCE is used for H2/H3 so that graphs with unknown labels (`-1`)
# contribute zero gradient instead of being dropped from the batch.


# %%
# =============================================================================
#  TOWER A - Hyena-style long-convolution sequence encoder
#
#  Reimplemented from the HyenaDNA operator (Nguyen et al. 2023,
#  github.com/HazyResearch/hyena-dna) with no external dependency: the original
#  needs einops, flash-attn and a config registry we do not want to drag in.
#  The maths is the same; the plumbing is ours.
# =============================================================================

def fftconv(u, k, D):
    # Circular-safe long convolution via FFT: O(L log L) instead of O(L * k).
    # u: (B, C, L)   k: (C, L)   D: (C,) skip-connection gain
    L = u.shape[-1]
    fft_size = 2 * L
    k_f = torch.fft.rfft(k.float(), n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.float(), n=fft_size)
    if u.dim() > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :L]
    return (y + u.float() * D.unsqueeze(-1)).to(u.dtype)


class Sin(nn.Module):
    # Periodic activation. A ReLU MLP cannot represent an oscillating filter
    # without enormous width; sine gives the implicit filter its frequency
    # content almost for free.
    def __init__(self, dim, w=10.0):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    # Complex-exponential position code, the input to the implicit filter MLP.
    def __init__(self, emb_dim, seq_len):
        super().__init__()
        assert emb_dim % 2 == 1 and emb_dim >= 3, "emb_dim must be odd and >= 3"
        t = torch.linspace(0, 1, seq_len)[None, :, None]              # (1, L, 1)
        bands = (emb_dim - 1) // 2
        t_res = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_res / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        self.register_buffer("z", torch.cat([t, z.real, z.imag], dim=-1))
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    # Multiplies the implicit filter by a learned per-channel exponential decay.
    # Without it every channel is free to attend uniformly over the whole
    # window and the filters collapse to a global average. With it, channels
    # span a spectrum of effective lengths - a few hundred bp to the full
    # window - which is what makes one layer able to model both a core-promoter
    # motif and a distal regulatory signal.
    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5,
                 target=1e-2, shift=0.0):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        self.register_buffer("deltas",
                             torch.linspace(min_decay, max_decay, d_model)[None, None])

    def forward(self, t, x):
        return x * (torch.exp(-t * self.deltas.abs()) + self.shift)


class HyenaFilter(nn.Module):
    # The filter is not stored, it is GENERATED by a small MLP from the position
    # code. That is what makes the parameter count independent of sequence
    # length: a 16 kb filter and a 1 kb filter cost the same weights.
    def __init__(self, d_model, emb_dim=33, order=64, seq_len=1024,
                 n_inner=2, w=10.0, modulate=True):
        super().__init__()
        self.d_model, self.modulate = d_model, modulate
        self.bias = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)
        act = Sin(dim=order, w=w)
        layers = [nn.Linear(emb_dim, order), act]
        for _ in range(n_inner):
            layers += [nn.Linear(order, order), Sin(dim=order, w=w)]
        layers += [nn.Linear(order, d_model, bias=False)]
        self.implicit_filter = nn.Sequential(*layers)
        self.modulation = ExponentialModulation(d_model)

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)
        return h                                   # (1, L, d_model)

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        k = k.squeeze(0).transpose(0, 1)           # (d_model, L)
        b = self.bias if bias is None else bias
        return fftconv(x, k, b)


class HyenaOperator(nn.Module):
    # y = out_proj( x_0 * FFTConv(x_1 * v, h) )
    #
    # An `order`-step recurrence of element-wise gating and long convolution.
    # The receptive field of ONE operator is the whole sequence, which is the
    # entire reason for using it: a dilated CNN needs log2(L) layers to see
    # 16 kb, and self-attention needs O(L^2) memory to do it.
    def __init__(self, d_model, l_max, order=2, filter_order=64,
                 short_filter_order=3, dropout=0.0, filter_emb_dim=33):
        super().__init__()
        assert order >= 2
        self.d_model, self.l_max, self.order = d_model, l_max, order
        total = d_model * (order + 1)
        self.in_proj = nn.Linear(d_model, total)
        self.out_proj = nn.Linear(d_model, d_model)
        self.short_filter = nn.Conv1d(total, total, short_filter_order,
                                      groups=total, padding=short_filter_order - 1)
        self.filter_fn = HyenaFilter(d_model * (order - 1), emb_dim=filter_emb_dim,
                                     order=filter_order, seq_len=l_max)
        self.drop = nn.Dropout(dropout)

    def forward(self, u):                          # u: (B, L, d_model)
        B, L, _ = u.shape
        lf = min(L, self.l_max)
        x = self.in_proj(u).transpose(1, 2)        # (B, total, L)
        x = self.short_filter(x)[..., :lf]         # local context, depthwise
        parts = x.split(self.d_model, dim=1)       # order+1 tensors of (B, d, L)
        *gates, v = parts

        k = self.filter_fn.filter(lf).squeeze(0).transpose(0, 1)   # (d*(order-1), L)
        k = k.view(self.order - 1, self.d_model, lf)
        bias = self.filter_fn.bias.view(self.order - 1, self.d_model)

        for o, g in enumerate(reversed(gates[1:])):
            v = self.drop(v * g)                   # element-wise gate
            v = fftconv(v, k[o], bias[o])          # global long convolution
        y = (v * gates[0]).transpose(1, 2)         # (B, L, d)
        return self.out_proj(y)


class RMSNorm(nn.Module):
    # Cheaper than LayerNorm and what current long-context stacks use.
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d, mult=2, dropout=0.0):
        super().__init__()
        h = int(d * mult)
        self.w1, self.w2, self.w3 = (nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class HyenaBlock(nn.Module):
    # Pre-norm residual, as in every modern sequence stack.
    def __init__(self, d, l_max, order, filter_order, dropout):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mixer = HyenaOperator(d, l_max, order, filter_order, dropout=dropout)
        self.mlp = SwiGLU(d, 2, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.mixer(self.n1(x)))
        return x + self.mlp(self.n2(x))


class RMSNorm_1d(nn.Module):
    # RMSNorm over the channel dim of a (B, C, L) tensor, for use inside the stem
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)


class SequenceEncoder(nn.Module):
    # HyenaDNA-style promoter encoder: one-hot (B, 4, L) -> z_seq (B, d).
    #
    # Three deliberate choices:
    #
    # 1. Single-nucleotide stem, no pooling before the mixer. HyenaDNA's
    #    headline claim is that you do not need k-mer tokenisation; a stride-1
    #    stem keeps base-resolution information available to the long
    #    convolution. We optionally stride the stem when l_max is large purely
    #    to bound FFT cost, and the stride is reported.
    #
    # 2. Global receptive field from layer 1. Each HyenaOperator convolves with
    #    a filter as long as the input, so the first block already sees the
    #    whole promoter window. The dilated CNN this replaces needed log2(L)
    #    layers and still only reached ~500 bp.
    #
    # 3. Reverse-complement equivariance. A promoter and its reverse complement
    #    are the same physical locus. We embed both strands with shared weights
    #    and average, which halves the effective hypothesis space and removes an
    #    arbitrary strand convention from the features. Costs 2x forward
    #    compute; usually worth more than 2x the parameters.
    def __init__(self, d_model, l_max, n_blocks=4, order=2, filter_order=64,
                 stem_kernel=15, stem_stride=1, dropout=0.1, d_out=128,
                 rc_equivariant=True):
        super().__init__()
        self.rc_equivariant = rc_equivariant
        self.stem_stride = stem_stride
        l_eff = l_max // stem_stride
        self.l_eff = l_eff
        self.stem = nn.Sequential(
            nn.Conv1d(4, d_model, stem_kernel, stride=stem_stride,
                      padding=stem_kernel // 2),
            RMSNorm_1d(d_model), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            HyenaBlock(d_model, l_eff, order, filter_order, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = RMSNorm(d_model)
        self.pool_gate = nn.Linear(d_model, 1)
        self.proj = nn.Sequential(nn.Linear(2 * d_model, d_out), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_out, d_out))

    def _embed(self, x):                            # x: (B, 4, L)
        h = self.stem(x).transpose(1, 2)            # (B, L', d)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        # mean pool ++ attention pool: the mean carries composition, the gate
        # lets a single strong motif dominate when it should
        a = torch.softmax(self.pool_gate(h), dim=1)
        return torch.cat([h.mean(1), (a * h).sum(1)], dim=-1)

    def forward(self, x):
        z = self._embed(x)
        if self.rc_equivariant:
            # reverse complement: flip length, and swap A<->T, C<->G
            xr = x.flip(-1)[:, [3, 2, 1, 0], :]
            z = 0.5 * (z + self._embed(xr))
        return self.proj(z)



class StructureEncoder(nn.Module):
    # promoter subgraph -> node states H_L (d_h each) and graph embedding z_str
    #
    # Width is held constant at d_h through the whole stack: layers 1..L-1 use
    # concat=True with d_h//heads channels per head (heads * d_h//heads == d_h),
    # and layer L averages its heads. That keeps every residual an identity and
    # removes the between-layer projections that a widening stack would need.
    def __init__(self, f_in, f_edge, d_hidden, d_out, layers, heads,
                 dropout, attn_dropout):
        super().__init__()
        if d_hidden % heads != 0:
            raise ValueError(f"d_hidden ({d_hidden}) must be divisible by "
                             f"gat_heads ({heads})")
        per_head = d_hidden // heads
        d_edge = max(8, d_hidden // 2)

        self.in_proj   = nn.Linear(f_in, d_hidden)
        self.edge_proj = nn.Sequential(nn.Linear(f_edge, d_edge), nn.GELU())

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            last = (i == layers - 1)
            self.convs.append(GATv2Conv(
                d_hidden,
                d_hidden if last else per_head,
                heads=heads,
                concat=not last,
                dropout=attn_dropout,
                edge_dim=d_edge,
                add_self_loops=True,
            ))
            self.norms.append(nn.LayerNorm(d_hidden))

        self.drop = nn.Dropout(dropout)
        self.pool_gate = nn.Linear(d_hidden, 1)
        self.out = nn.Sequential(nn.Linear(2 * d_hidden, d_out), nn.GELU(),
                                 nn.Linear(d_out, d_out))

    def forward(self, x, edge_index, edge_attr, batch, anchor_index):
        h = self.in_proj(x)
        e = self.edge_proj(edge_attr)
        attn_last = None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if i == len(self.convs) - 1:
                h_new, (ei, alpha) = conv(h, edge_index, e,
                                          return_attention_weights=True)
                attn_last = (ei.detach(), alpha.detach())
            else:
                h_new = conv(h, edge_index, e)
            h = self.drop(F.gelu(norm(h_new + h)))          # residual + norm

        # readout: anchor node  ++  attention-pooled summary.
        # pyg_softmax normalises the gate WITHIN each graph; a plain
        # torch.softmax would normalise across the whole batch, which silently
        # makes the pooled vector depend on batch composition.
        gate = pyg_softmax(self.pool_gate(h), batch)
        pooled = global_add_pool(gate * h, batch)
        anchor = h[anchor_index]
        return h, self.out(torch.cat([anchor, pooled], dim=-1)), attn_last


class EpiEncoder(nn.Module):
    # TOWER C - epigenomic state encoder: a small MLP over the promoter
    # fragment's own histone/CTCF summary -- signal + direct-measurement mask
    # per mark, exactly the block already computed for node 0 (the bait) in
    # the structure tower's node features (columns 4:4+2*n_marks). Pulling it
    # into its own tower and its own embedding is what makes "epigenomic
    # state" a modality the model can be trained to align and ablate on its
    # own terms, rather than a feature buried inside the graph tower.
    #
    # Note this does NOT remove histone/CTCF signal from the structure
    # tower's node features -- Tower B still sees it, on every node, not just
    # the anchor. So Tower B and Tower C see overlapping but not identical
    # information (Tower B: per-node, whole subgraph; Tower C: anchor only,
    # summarised). That overlap is a deliberate scope decision, not an
    # oversight -- see the notebook banner cell for the alternative (stripping
    # histone features from Tower B) and why this notebook does not do that.
    def __init__(self, f_epi, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f_epi, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        return self.norm(self.net(x))


class GatedFusion(nn.Module):
    # Generalises the two-tower FiLM fusion to 1, 2, or 3 active modalities,
    # since ablation arms now range from a single tower to all three. Two
    # mechanisms, combined:
    #   1. an attention-style gate gives each present modality a learned,
    #      per-example weight (softmax over the *present* modalities only)
    #      and pools them -- this is order-independent and works for any
    #      subset size.
    #   2. a concat -> MLP "mix" path (repeating the last embedding to pad
    #      out missing slots) gives the fusion a learned cross-term the pure
    #      gated sum cannot express on its own.
    # This replaces the original two-tower model's FiLMFusion
    # (z = gamma(z_str)*z_seq + beta(z_str)) exactly when only seq+str are
    # active is NOT bit-identical to that formula -- an intentional,
    # documented change, not an oversight. See the banner cell.
    def __init__(self, d, max_modalities=3):
        super().__init__()
        self.gate = nn.Linear(d, 1)
        self.mix = nn.Sequential(nn.Linear(d * max_modalities, d), nn.GELU())
        self.norm = nn.LayerNorm(d)
        self.max_modalities = max_modalities

    def forward(self, embeddings):                    # list of (B, d), len 1..max_modalities
        if len(embeddings) == 1:
            return self.norm(embeddings[0])
        stacked = torch.stack(embeddings, dim=1)        # (B, M, d)
        gate = torch.softmax(self.gate(stacked), dim=1)  # (B, M, 1)
        pooled = (gate * stacked).sum(1)                 # (B, d)
        pad = [embeddings[-1]] * (self.max_modalities - len(embeddings))
        mixed = self.mix(torch.cat(embeddings + pad, dim=-1))
        return self.norm(pooled + mixed)


class SeqBank:
    # Promoter sequences as uint8 base codes (n_genes, L), A,C,G,T,N = 0..4.
    # Lives on the GPU when it fits (~20 GB at full scale on a 141 GB H200),
    # otherwise in pinned host memory. onehot() gathers a batch of genes and
    # expands it to (B, 4, L) on the device: N becomes an all-zero column.
    def __init__(self, codes, device, place="auto", headroom_gb=40.0):
        self.device = torch.device(device)
        nbytes = codes.shape[0] * codes.shape[1]
        on_gpu = self.device.type == "cuda" and (
            place == "yes" or (place == "auto"
                               and torch.cuda.mem_get_info()[0] > nbytes + headroom_gb * 1e9))
        if on_gpu:
            self.codes = torch.empty(codes.shape, dtype=torch.uint8, device=self.device)
        else:
            self.codes = torch.empty(codes.shape, dtype=torch.uint8,
                                     pin_memory=(self.device.type == "cuda"))
        for i in range(0, codes.shape[0], 1024):      # chunked: never a second full copy
            self.codes[i:i + 1024] = torch.from_numpy(np.ascontiguousarray(codes[i:i + 1024]))
        self._lut = {}

    @property
    def shape(self):
        return tuple(self.codes.shape)

    def __len__(self):
        return self.codes.shape[0]

    def onehot(self, idx, device=None, dtype=torch.float32):
        device = torch.device(device) if device is not None else self.device
        if self.codes.is_cuda:
            c = self.codes[idx.to(self.codes.device)]
        else:
            c = self.codes[idx.cpu()].to(device, non_blocking=True)
        key = (str(device), dtype)
        if key not in self._lut:
            self._lut[key] = torch.eye(5, 4, device=device, dtype=dtype)   # row 4 (N) = 0
        return self._lut[key][c.long()].transpose(1, 2)                  # (B, 4, L)


class PangenomicGAT(nn.Module):
    # variant selects which of the three towers are live. Tokens separated
    # by "_", or the literal "all3" for all of them:
    #
    #   "all3"      z = fuse(z_seq, z_str, z_epi)   full model
    #   "seq_str"   z = fuse(z_seq, z_str)          original two-tower model
    #   "seq_epi"   z = fuse(z_seq, z_epi)          no graph tower
    #   "str_epi"   z = fuse(z_str, z_epi)          no sequence tower
    #   "seq"       z = z_seq                       sequence only
    #   "str"       z = z_str                       structure only
    #   "epi"       z = z_epi                       epigenomic state only
    #
    # As in the base model, the ablation is a constructor flag rather than
    # separate classes so heads/losses/splits/metric code are provably
    # identical across arms -- any difference in the numbers comes from the
    # towers. CFG.variants controls which of these 7 are actually trained;
    # see the banner cell for the default subset and why.
    VARIANTS = ("all3", "seq_str", "seq_epi", "str_epi", "seq", "str", "epi")

    @staticmethod
    def _parse(variant):
        if variant == "all3":
            return True, True, True
        tokens = set(variant.split("_"))
        return "seq" in tokens, "str" in tokens, "epi" in tokens

    def __init__(self, cfg, f_node, f_edge, n_tissues, variant="all3"):
        super().__init__()
        assert variant in self.VARIANTS, f"variant must be one of {self.VARIANTS}"
        self.variant = variant
        d = cfg.d_embed
        self.use_seq, self.use_str, self.use_epi = self._parse(variant)
        f_epi = 2 * len(cfg.histone_marks)   # signal + direct-measurement mask, per mark

        self.seq = SequenceEncoder(
            d_model=cfg.hyena_d_model, l_max=cfg.seq_len,
            n_blocks=cfg.hyena_blocks, order=cfg.hyena_order,
            filter_order=cfg.hyena_filter_order, stem_kernel=cfg.stem_kernel,
            stem_stride=cfg.stem_stride, dropout=cfg.dropout, d_out=d,
            rc_equivariant=cfg.rc_equivariant) if self.use_seq else None

        self.struc = StructureEncoder(f_node, f_edge, cfg.d_hidden, d,
                                      cfg.gat_layers, cfg.gat_heads,
                                      cfg.dropout, cfg.attn_dropout) \
            if self.use_str else None

        self.epi = EpiEncoder(f_epi, max(32, d // 2), d, cfg.dropout) \
            if self.use_epi else None

        n_active = sum([self.use_seq, self.use_str, self.use_epi])
        self.fuse = GatedFusion(d, max_modalities=3) if n_active > 1 else None

        self.head_expr   = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                         nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        # H2's SECOND head -- same input z, own weights, own target
        # (expressed 0/1 rather than the continuous signal). Structurally
        # identical to head_expr on purpose: any difference in what the two
        # heads learn comes from the loss/label, not from head capacity.
        self.head_expr_bin = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                           nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        self.head_tau    = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                         nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        self.head_tissue = nn.Linear(d, n_tissues)
        d_h = cfg.d_hidden
        self.link_bilinear = nn.Parameter(torch.eye(d_h) + 0.01 * torch.randn(d_h, d_h))
        self.temperature = cfg.temperature

    def encode(self, batch, seq_bank):
        # batch.ptr[:-1] is the global index of node 0 of every graph, i.e. the
        # promoter anchor. For a single un-batched graph it degenerates to [0].
        # seq_bank is a SeqBank; it one-hot encodes the batch's genes on `dev`.
        dev = batch.x.device
        z_seq = z_str = z_epi = h = attn = None

        if self.use_seq:
            z_seq = self.seq(seq_bank.onehot(batch.seq_idx, dev))
        if self.use_str:
            anchor = (batch.ptr[:-1] if hasattr(batch, "ptr")
                      else torch.zeros(1, dtype=torch.long, device=dev))
            h, z_str, attn = self.struc(batch.x, batch.edge_index,
                                        batch.edge_attr, batch.batch, anchor)
        if self.use_epi:
            z_epi = self.epi(batch.epi.to(dev))

        active = [z for z in (z_seq, z_str, z_epi) if z is not None]
        z = self.fuse(active) if self.fuse is not None else active[0]
        return z_seq, z_str, z_epi, z, h, attn

    def forward(self, batch, seq_bank):
        z_seq, z_str, z_epi, z, h, attn = self.encode(batch, seq_bank)
        return {
            "z_seq": z_seq, "z_str": z_str, "z_epi": z_epi, "z": z, "h": h, "attn": attn,
            "logit_expr":     self.head_expr(z).squeeze(-1),
            "logit_expr_bin": self.head_expr_bin(z).squeeze(-1),
            "logit_tau":      self.head_tau(z).squeeze(-1),
            "logit_tissue":   self.head_tissue(z),
        }

    def score_edges(self, h, pair_index):
        return self.score_pairs(h[pair_index[0]], h[pair_index[1]])

    def score_pairs(self, a, b):
        # bilinear link score for node-state pairs a, b: (P, d_h) each
        return ((a @ self.link_bilinear.to(a.dtype)) * b).sum(-1)

    def fuse_embeddings(self, z_seq, z_str, z_epi):
        active = [z for z in (z_seq, z_str, z_epi) if z is not None]
        return self.fuse(active) if self.fuse is not None else active[0]

    def towers(self):
        return {name: getattr(self, name) for name in ("seq", "struc", "epi")
                if getattr(self, name) is not None}


# Masked losses as weighted means: no boolean indexing and no "if m.sum() == 0"
# branch, so no host<->device sync per batch. target == -1 marks an unknown
# label (log2(TPM+1) >= 0, so -1 never collides with a real H2 value).
def masked_bce(logit, target):
    m = (target >= 0).to(logit.dtype)
    l = F.binary_cross_entropy_with_logits(logit, target.clamp(min=0), reduction="none")
    return (l * m).sum() / m.sum().clamp_min(1.0)


def masked_mse(pred, target):
    m = (target >= 0).to(pred.dtype)
    return (((pred - target) ** 2) * m).sum() / m.sum().clamp_min(1.0)


def info_nce(z_a, z_b, temperature):
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))


def multi_align_loss(embeddings, temperature):
    # embeddings: dict like {"seq": z_seq, "str": z_str, "epi": z_epi}, any
    # value may be None (that modality inactive for this arm). Averages
    # InfoNCE over every PRESENT pair -- 3 pairs for all3, 1 pair for a
    # two-tower arm, zero (returns 0) for a single-tower arm.
    present = {k: v for k, v in embeddings.items() if v is not None}
    keys = list(present.keys())
    if len(keys) < 2 or len(present[keys[0]]) <= 1:
        return None   # caller substitutes torch.tensor(0.0, device=...)
    losses = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            losses.append(info_nce(present[keys[i]], present[keys[j]], temperature))
    return sum(losses) / len(losses)


rows = []
probe = None
for v in CFG.variants:
    m = PangenomicGAT(CFG, N_NODE_FEAT, N_EDGE_FEAT, len(TISSUE_LIST), variant=v)
    tot = sum(p.numel() for p in m.parameters())
    nseq = sum(p.numel() for p in m.seq.parameters()) if m.seq else 0
    nstr = sum(p.numel() for p in m.struc.parameters()) if m.struc else 0
    nepi = sum(p.numel() for p in m.epi.parameters()) if m.epi else 0
    rows.append({"variant": v, "sequence_tower": nseq, "structure_tower": nstr,
                 "epigenomic_tower": nepi, "fusion+heads": tot - nseq - nstr - nepi,
                 "total": tot})
    if v == "all3" or (probe is None and m.use_seq):
        probe = m
print("parameter budget per ablation arm")
print(pd.DataFrame(rows).to_string(index=False))

L_eff = CFG.seq_len // CFG.stem_stride
print()
print("  TOWER A  HyenaDNA-style long convolution")
print(f"    promoter window   {CFG.seq_len:,} bp  "
      f"(stem stride {CFG.stem_stride} -> {L_eff:,} positions)")
print(f"    blocks            {CFG.hyena_blocks} x HyenaOperator(order={CFG.hyena_order})")
print(f"    d_model           {CFG.hyena_d_model}")
print(f"    implicit filter   MLP width {CFG.hyena_filter_order}, sine activations")
print(f"    receptive field   {L_eff:,} positions from block 1 "
      f"(filter length == input length)")
print(f"    rev-comp equiv.   {CFG.rc_equivariant}")
print("  TOWER B  GATv2")
print(f"    layers x heads    {CFG.gat_layers} x {CFG.gat_heads}, d_h={CFG.d_hidden}")
print(f"    node / edge feat  F={N_NODE_FEAT}  F_e={N_EDGE_FEAT}")
print("  TOWER C  epigenomic MLP")
print(f"    input             {2 * len(CFG.histone_marks)} dims "
      f"(signal + direct-measurement mask, per mark, promoter fragment only)")
print("  SHARED")
print(f"    embedding dim     d={CFG.d_embed}")
print(f"    tissue classes    {len(TISSUE_LIST)}")
print(f"    arms trained      {list(CFG.variants)}")

# ---- smoke tests (opt-in): a full 1 Mb forward+backward on CPU is slow, and
# pointless on every resume of a production run. PANGENOMIC_SMOKE=1 to run.
if os.environ.get("PANGENOMIC_SMOKE") == "1":
    # ---- shape + gradient smoke test before we spend hours training -------------
    if probe is not None and probe.use_seq:
        _x = torch.randn(2, 4, CFG.seq_len)
        _z = probe.seq(_x)
        print(f"\nsmoke test: SequenceEncoder {tuple(_x.shape)} -> {tuple(_z.shape)}")
        assert _z.shape == (2, CFG.d_embed), "sequence tower output shape is wrong"

        if CFG.rc_equivariant:
            probe.seq.eval()
            with torch.no_grad():
                z_f = probe.seq(_x)
                z_r = probe.seq(_x.flip(-1)[:, [3, 2, 1, 0], :])
            dev = (z_f - z_r).abs().max().item()
            print(f"  reverse-complement equivariance: max |z(x) - z(rc(x))| = {dev:.2e}")
            assert dev < 1e-4, "RC averaging is not making the encoder strand-symmetric"
            probe.seq.train()

        _z.sum().backward()
        g = [p.grad.abs().sum().item() for p in probe.seq.parameters() if p.grad is not None]
        print(f"  backward OK: {len(g)} tensors received gradient, "
              f"total |grad| = {sum(g):.3e}")
        assert sum(g) > 0, "no gradient reached the sequence tower"
        del _x, _z
    else:
        print("\n(no arm in CFG.variants uses the sequence tower -- skipping its smoke test)")

    # ---- epigenomic tower smoke test ---------------------------------------------
    _epi_probe = next((m for v in CFG.variants
                       for m in [PangenomicGAT(CFG, N_NODE_FEAT, N_EDGE_FEAT, len(TISSUE_LIST), variant=v)]
                       if m.use_epi), None)
    if _epi_probe is not None:
        _xe = torch.randn(5, 2 * len(CFG.histone_marks))
        _ze = _epi_probe.epi(_xe)
        print(f"smoke test: EpiEncoder {tuple(_xe.shape)} -> {tuple(_ze.shape)}")
        assert _ze.shape == (5, CFG.d_embed), "epigenomic tower output shape is wrong"
        del _epi_probe, _xe, _ze
    del probe
else:
    del probe
    print("\n(smoke tests skipped; set PANGENOMIC_SMOKE=1 to run them)")


# %% [markdown]
# ## 14 · Training and evaluation loops
#
# `run_epoch` trains or evaluates the full graph model (Stage A, and every arm
# when `encoder_source="none"`). `heads_and_losses` is shared with the Stage-B
# trainer in section 16, so both stages compute the loss identically.
#
# **Anchor indexing.** PyG's `DataLoader` concatenates graphs into one big
# disconnected graph and gives us `batch.ptr`, the node offset of each graph.
# `batch.ptr[:-1]` is therefore exactly the global index of every graph's node 0 —
# its promoter anchor. The same offsets shift the held-out edge pairs into the
# batched coordinate frame.
#
# **Loss.** Weighted sum of H1 (BCE over positive + distance-matched negative
# pairs), H2a (masked MSE), H2b / H3 (masked BCE), H4 (cross-entropy) and the
# InfoNCE alignment between present towers.
#
# **No per-batch host syncs.** Loss totals accumulate on the GPU and are read
# once per epoch; predictions are only gathered (`collect=True`) for the test
# pass. The one per-step read is the finite-check that lets a non-finite bf16
# step be skipped and counted instead of corrupting the weights.
#
# **Frozen towers stay in eval mode** during training, so their dropout never
# fires and their features are identical in training and validation.


# %%
LOSS_KEYS = ("loss", "link", "expr", "expr_bin", "tau", "tissue", "align")
PRED_KEYS = ("expr_p", "expr_y", "expr_bin_p", "expr_bin_y", "tau_p", "tau_y",
             "tis_p", "tis_y", "link_p", "link_y", "z", "z_seq", "z_str", "z_epi", "row_idx")


def amp_ctx(cfg, device):
    dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype,
                          enabled=(cfg.use_amp and torch.device(device).type == "cuda"))


def set_train_mode(model, train):
    # train() everything, then put fully frozen towers back in eval mode
    model.train(train)
    if train:
        for tower in model.towers().values():
            if not any(p.requires_grad for p in tower.parameters()):
                tower.eval()


def heads_and_losses(model, z, y, s_link, y_link, cfg, align=None):
    # z: fused embedding (B, d); y: dict of label tensors; s_link / y_link:
    # link scores and 0/1 targets for this batch's held-out pairs (or None).
    z = z.float()
    logits = {"expr": model.head_expr(z).squeeze(-1),
              "expr_bin": model.head_expr_bin(z).squeeze(-1),
              "tau": model.head_tau(z).squeeze(-1),
              "tissue": model.head_tissue(z)}
    zero = z.new_zeros(())
    l = {"link": (F.binary_cross_entropy_with_logits(s_link.float(), y_link)
                  if s_link is not None and s_link.numel() else zero),
         "expr": masked_mse(logits["expr"], y["expr"]),
         "expr_bin": masked_bce(logits["expr_bin"], y["expr_bin"]),
         "tau": masked_bce(logits["tau"], y["tau"]),
         "tissue": F.cross_entropy(logits["tissue"], y["tissue"]),
         "align": align.float() if align is not None else zero}
    l["loss"] = (cfg.w_link * l["link"] + cfg.w_expr * l["expr"]
                 + cfg.w_expr_bin * l["expr_bin"] + cfg.w_tau * l["tau"]
                 + cfg.w_tissue * l["tissue"] + cfg.w_align * l["align"])
    return logits, l


class PredCollector:
    # gathers eval-time outputs on the device; one transfer to host at the end
    def __init__(self):
        self.d = {k: [] for k in PRED_KEYS}

    def add(self, logits, y, s_link, y_link, rows, z, z_seq=None, z_str=None, z_epi=None):
        d = self.d
        d["expr_p"].append(logits["expr"]); d["expr_y"].append(y["expr"])
        d["expr_bin_p"].append(torch.sigmoid(logits["expr_bin"])); d["expr_bin_y"].append(y["expr_bin"])
        d["tau_p"].append(torch.sigmoid(logits["tau"])); d["tau_y"].append(y["tau"])
        d["tis_p"].append(torch.softmax(logits["tissue"], -1)); d["tis_y"].append(y["tissue"])
        if s_link is not None and s_link.numel():
            d["link_p"].append(torch.sigmoid(s_link.float())); d["link_y"].append(y_link)
        d["z"].append(z); d["row_idx"].append(rows)
        for k, v in (("z_seq", z_seq), ("z_str", z_str), ("z_epi", z_epi)):
            if v is not None:
                d[k].append(v)

    def result(self):
        out = {}
        for k, v in self.d.items():
            if not v:
                out[k] = np.array([]); continue
            t = torch.cat([x.detach() for x in v])
            out[k] = t.cpu().numpy() if k in ("tis_y", "row_idx") else t.float().cpu().numpy()
        return out


def batch_labels(batch):
    return {"expr": batch.y_expr, "expr_bin": batch.y_expr_bin,
            "tau": batch.y_tau, "tissue": batch.y_tissue}


def optimizer_step(model, optim, loss, cfg):
    # -> True if the step was applied, False if it was skipped (non-finite)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                  cfg.grad_clip)
    if not bool(torch.isfinite(loss) & torch.isfinite(gn)):
        optim.zero_grad(set_to_none=True)
        return False
    optim.step()
    return True


def run_epoch(model, loader, seq_bank, cfg, device, optim=None, collect=False):
    # -> (mean losses incl. "n_skipped", predictions dict or None)
    train = optim is not None
    set_train_mode(model, train)
    tot = torch.zeros(len(LOSS_KEYS), device=device)
    n_used = n_skipped = 0
    col = PredCollector() if collect else None
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with amp_ctx(cfg, device):
                z_seq, z_str, z_epi, z, h, attn = model.encode(batch, seq_bank)
                s_link = y_link = None
                pos = getattr(batch, "pos_edge_label_index", None)
                neg = getattr(batch, "neg_edge_label_index", None)
                # H1 scores node pairs, so it exists only for arms with the graph tower
                if h is not None and pos is not None and neg is not None:
                    s_pos, s_neg = model.score_edges(h, pos), model.score_edges(h, neg)
                    s_link = torch.cat([s_pos, s_neg])
                    y_link = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)]).float()
                align = multi_align_loss({"seq": z_seq, "str": z_str, "epi": z_epi},
                                         cfg.temperature)
                y = batch_labels(batch)
                logits, l = heads_and_losses(model, z, y, s_link, y_link, cfg, align)
            if train and not optimizer_step(model, optim, l["loss"], cfg):
                n_skipped += 1
                continue
            tot += torch.stack([l[k].detach().float() for k in LOSS_KEYS])
            n_used += 1
            if collect:
                col.add(logits, y, s_link, y_link, batch.row_idx, z, z_seq, z_str, z_epi)
    avg = dict(zip(LOSS_KEYS, (tot / max(n_used, 1)).tolist()))
    avg["n_skipped"] = n_skipped
    return avg, (col.result() if collect else None)


def make_loader(graphs, indices, bs, shuffle, persistent=False, cfg=None, device=None):
    cfg = cfg or CFG
    device = torch.device(device) if device is not None else DEVICE
    nw = cfg.num_workers
    return DataLoader([graphs[i] for i in indices], batch_size=bs, shuffle=shuffle,
                      num_workers=nw, pin_memory=(device.type == "cuda"),
                      persistent_workers=(persistent and nw > 0),
                      prefetch_factor=(4 if nw > 0 else None))


# ---- verify the batching assumption rather than trusting it -----------------
_probe = DataLoader([GRAPHS[0], GRAPHS[1]], batch_size=2, shuffle=False)
_b = next(iter(_probe))
_n0 = int(GRAPHS[0].num_nodes)
_expected = int(GRAPHS[1].pos_edge_label_index[0, 0]) + _n0
_got = int(_b.pos_edge_label_index[0, GRAPHS[0].pos_edge_label_index.shape[1]])
print("batching self-check")
print(f"  graph 0 has {_n0} nodes; graph 1's first held-out pair should be "
      f"offset by {_n0}")
print(f"  expected {_expected}, got {_got}")
assert _got == _expected, (
    "PyG did not offset pos_edge_label_index as expected. Rename the attribute "
    "so it contains 'index', or offset manually by batch.ptr.")
assert int(_b.ptr[1]) == _n0, "batch.ptr does not mark graph boundaries"
assert _b.seq_idx.shape[0] == 2, "seq_idx should be one entry per graph"
print("  OK: edge pairs offset correctly, ptr marks anchors, seq_idx per graph")

print("\ntraining utilities defined")
print("  run_epoch(...) -> (mean losses, predictions if collect=True)")
print("  anchors  = batch.ptr[:-1]  (node 0 of every graph)")
print("  H2a masked MSE, H2b/H3 masked BCE: y = -1 rows contribute no gradient")


# %% [markdown]
# ## 15 · The metric suite
#
# One function, every metric, computed identically for every head so the results
# table is comparable across tasks.
#
# **Binary heads (H1, H3)** — threshold-free first, then thresholded:
#
# **Regression head (H2)** — continuous log2(TPM+1) signal, not thresholded: Pearson r and Spearman rho (ranking-style agreement), R^2 and RMSE/MAE (magnitude-of-error), computed by `regression_metrics`.
#
# - *Ranking*: AUROC, AUPRC (= average precision). AUPRC is the one to trust when
#   classes are imbalanced; AUROC flatters a model on skewed data because the
#   huge negative pool makes the false-positive rate look small.
# - *At threshold 0.5 and at the F1-optimal threshold*: accuracy, balanced
#   accuracy, precision, recall (sensitivity), specificity, NPV, F1, F2, MCC,
#   Cohen's kappa, Jaccard.
# - *Probability quality*: Brier score and log loss. A model can have a great
#   AUROC and useless probabilities; these two catch that.
# - The **baseline AUPRC** (= positive prevalence) is reported alongside, because
#   an AUPRC of 0.4 is excellent at 5% prevalence and terrible at 60%.
#
# **Multiclass head (H4)** — accuracy, balanced accuracy, macro/micro/weighted
# precision/recall/F1, macro one-vs-rest AUROC and AUPRC, and Cohen's kappa.
#
# MCC is included deliberately: it is the metric that stays honest under class
# imbalance, where accuracy and even F1 can be gamed by predicting the majority.


# %%
def regression_metrics(y_true, y_pred, prefix="", tolerances=(0.5, 1.0, 2.0)):
    # Used for H2's continuous head (head_expr). H2's SECOND head
    # (head_expr_bin) and H1/H3 keep binary_metrics; H4 keeps
    # multiclass_metrics -- unaffected by this notebook's change.
    #
    # `tolerances`: match-within-tolerance rates, in addition to the usual
    # correlation/error metrics. Exact agreement is not the right bar for a
    # continuous target -- a prediction within `tol` log2(TPM+1) units of
    # the truth (~1.4x/2x/4x fold-change for 0.5/1.0/2.0) is a correct
    # prediction of the expression level, not a near-miss being excused.
    # Reported for every tolerance given, never just the flattering one, and
    # never in place of MAE/RMSE/r -- a reader who wants a stricter or
    # looser bar than the ones chosen here still has the raw error to judge
    # by.
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    m = y_true >= 0                                     # drop unknown labels (sentinel -1)
    y_true, y_pred = y_true[m], y_pred[m]
    tol_keys = [f"match_acc_{t}" for t in tolerances]
    out = {f"{prefix}n": len(y_true)}
    keys = ("mean", "std", "mse", "rmse", "mae", "r2",
            "pearson_r", "pearson_p", "spearman_r", "spearman_p", *tol_keys)
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]):
        return {**out, **{f"{prefix}{k}": np.nan for k in keys}}
    resid = y_true - y_pred
    abs_err = np.abs(resid)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    pear = sstats.pearsonr(y_true, y_pred)
    spear = sstats.spearmanr(y_true, y_pred)
    out.update({
        f"{prefix}mean": float(y_true.mean()), f"{prefix}std": float(y_true.std()),
        f"{prefix}mse": ss_res / len(y_true), f"{prefix}rmse": float(np.sqrt(ss_res / len(y_true))),
        f"{prefix}mae": float(np.mean(np.abs(resid))),
        f"{prefix}r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        f"{prefix}pearson_r": float(pear[0]), f"{prefix}pearson_p": float(pear[1]),
        f"{prefix}spearman_r": float(spear.correlation), f"{prefix}spearman_p": float(spear.pvalue),
    })
    for t, tk in zip(tolerances, tol_keys):
        out[f"{prefix}{tk}"] = float((abs_err <= t).mean())
    return out


def binary_metrics(y_true, y_prob, prefix=""):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    m = y_true >= 0                                     # drop unknown labels
    y_true, y_prob = y_true[m].astype(int), y_prob[m]
    out = {f"{prefix}n": len(y_true), f"{prefix}pos_rate": float(y_true.mean())
           if len(y_true) else np.nan}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {**out, **{f"{prefix}{k}": np.nan for k in
                          ("auroc", "auprc", "auprc_baseline", "accuracy",
                           "balanced_accuracy", "precision", "recall",
                           "specificity", "npv", "f1", "f2", "mcc", "kappa",
                           "jaccard", "brier", "logloss", "best_threshold",
                           "f1_at_best", "precision_at_best", "recall_at_best")}}

    out[f"{prefix}auroc"] = roc_auc_score(y_true, y_prob)
    out[f"{prefix}auprc"] = average_precision_score(y_true, y_prob)
    out[f"{prefix}auprc_baseline"] = float(y_true.mean())

    def at(th):
        yp = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true, yp, average="binary", zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        npv  = tn / (tn + fn) if (tn + fn) else np.nan
        f2 = (5 * pre * rec / (4 * pre + rec)) if (4 * pre + rec) else 0.0
        return dict(accuracy=accuracy_score(y_true, yp),
                    balanced_accuracy=balanced_accuracy_score(y_true, yp),
                    precision=pre, recall=rec, specificity=spec, npv=npv,
                    f1=f1, f2=f2, mcc=matthews_corrcoef(y_true, yp),
                    kappa=cohen_kappa_score(y_true, yp),
                    jaccard=jaccard_score(y_true, yp, zero_division=0),
                    tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

    for k, v in at(0.5).items():
        out[f"{prefix}{k}"] = v

    pr, rc, th = precision_recall_curve(y_true, y_prob)
    f1s = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    bi = int(np.nanargmax(f1s[:-1])) if len(th) else 0
    best_th = float(th[bi]) if len(th) else 0.5
    out[f"{prefix}best_threshold"]    = best_th
    out[f"{prefix}f1_at_best"]        = float(f1s[bi])
    out[f"{prefix}precision_at_best"] = float(pr[bi])
    out[f"{prefix}recall_at_best"]    = float(rc[bi])

    out[f"{prefix}brier"]   = brier_score_loss(y_true, y_prob)
    out[f"{prefix}logloss"] = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7),
                                       labels=[0, 1])
    return out


def multiclass_metrics(y_true, y_prob, n_classes, prefix="tissue_"):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return {}
    y_pred = y_prob.argmax(1)
    present = np.unique(y_true)
    out = {
        f"{prefix}n": len(y_true),
        f"{prefix}n_classes_present": len(present),
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}kappa": cohen_kappa_score(y_true, y_pred),
    }
    for avg in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        out[f"{prefix}precision_{avg}"] = p
        out[f"{prefix}recall_{avg}"] = r
        out[f"{prefix}f1_{avg}"] = f
    # One-vs-rest per present class, like AUPRC below. sklearn's multiclass
    # path needs rows summing to 1 over the passed columns, which fails (and
    # used to become a silent NaN) whenever a tissue is absent from y_true,
    # e.g. always under leave-one-tissue-out.
    aucs, aps = [], []
    for c in present:
        yc = (y_true == c).astype(int)
        if 0 < yc.sum() < len(yc):
            aucs.append(roc_auc_score(yc, y_prob[:, c]))
            aps.append(average_precision_score(yc, y_prob[:, c]))
    out[f"{prefix}auroc_macro_ovr"] = float(np.mean(aucs)) if aucs else np.nan
    out[f"{prefix}auprc_macro_ovr"] = float(np.mean(aps)) if aps else np.nan
    return out


def all_metrics(pred, n_classes):
    m = {}
    m.update(binary_metrics(pred["link_y"], pred["link_p"], "link_"))
    m.update(regression_metrics(pred["expr_y"], pred["expr_p"], "expr_",
                                tolerances=CFG.expr_match_tolerances))
    m.update(binary_metrics(pred["expr_bin_y"], pred["expr_bin_p"], "expr_bin_"))
    m.update(binary_metrics(pred["tau_y"],  pred["tau_p"],  "tau_"))
    m.update(multiclass_metrics(pred["tis_y"], pred["tis_p"], n_classes))
    return m


# ---- self-test on synthetic data so a silent metric bug shows up here -------
_rng = np.random.default_rng(0)
_y = _rng.integers(0, 2, 800)
_p = np.clip(_y * 0.6 + _rng.normal(0.2, 0.25, 800), 0, 1)
_t = binary_metrics(_y, _p, "")
print("metric self-test on synthetic data (should be well above chance):")
for k in ("n", "pos_rate", "auroc", "auprc", "auprc_baseline", "accuracy",
          "precision", "recall", "specificity", "f1", "mcc", "kappa",
          "brier", "logloss", "best_threshold", "f1_at_best"):
    print(f"  {k:<18} {_t[k]:.4f}")
assert _t["auroc"] > 0.7 and 0 <= _t["brier"] <= 1
print("\n  OK: metric suite behaves sensibly")


# %% [markdown]
# ## 16 · Two-stage, leak-free cross-validation
#
# For every split (here: each gene-grouped CV fold; Part III reuses the same
# machinery for chromosome-block CV and leave-one-tissue-out):
#
# **Stage A — encoders.** The full `all3` model trains end to end on the split's
# own training graphs only, with a 10% gene-grouped validation slice for early
# stopping. Nothing from the test fold (or the held-out tissue) reaches it.
#
# **Feature cache.** The best Stage-A encoders run ONCE, in eval mode, over every
# graph: `z_seq` per gene, `z_str` / `z_epi` per graph, and the GAT node states
# at every held-out link pair. Saved to `RESULTS_DIR/features/`, so restarts skip it.
#
# **Stage B — every ablation arm.** Each arm gets those encoders frozen and
# trains only its GatedFusion and heads (and the link bilinear form) on the cached
# tensors. The towers never change, so recomputing them every step would be pure
# waste; an epoch now takes seconds. Early stopping on the same inner
# validation slice, with `min_epochs_heads` as a floor (a fresh fusion on mature
# encoders needs time to learn to down-weight z_seq for H4). The alignment loss is
# omitted in Stage B: with frozen towers it is a constant.
#
# Every run checkpoints after each epoch (atomic write, config fingerprint,
# per-run seed from `(seed, run key)`), and a finished run stores its test
# predictions, so a restart neither retrains nor re-scores it.


# %%
# Tower C input: the promoter (anchor / node 0) histone + mask block from the
# node features (columns 4 : 4 + 2*n_marks), attached per graph. PyG batches a
# per-graph attribute whose name lacks "index" by concatenating on dim 0.
EPI_DIM = 2 * len(CFG.histone_marks)
for _g in GRAPHS:
    _g.epi = _g.x[0, 4:4 + EPI_DIM].clone().unsqueeze(0)
print(f"epigenomic modality attached: {EPI_DIM}-dim vector per graph "
      f"(promoter-fragment histone/CTCF signal + measurement mask)")

SEQ_BANK = SeqBank(SEQ_CODES, DEVICE, CFG.seq_bank_on_gpu)
print(f"sequence bank: {SEQ_BANK.shape} uint8 on "
      f"{'GPU' if SEQ_BANK.codes.is_cuda else 'pinned host memory'}")

ROW_SEQ_IDX = torch.as_tensor(META["gene"].map(GENE2SEQ).to_numpy(), dtype=torch.long)
ROW_GENES = META["gene"].to_numpy()


# %%
from torch.optim.lr_scheduler import CosineAnnealingLR

CKPT_DIR = RESULTS_DIR / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)
FEAT_DIR = RESULTS_DIR / "features";    FEAT_DIR.mkdir(exist_ok=True)
TOWER_PREFIXES = ("seq.", "struc.", "epi.")

# config fields that do not change what a run computes
_RUNTIME_ONLY = {"num_workers", "seq_bank_on_gpu", "run_hybrid", "variants", "loto_n_tissues"}


def cfg_fingerprint(cfg, **extra):
    fields = {k: v for k, v in asdict(cfg).items() if k not in _RUNTIME_ONLY}
    return content_key(fields, extra)


def load_towers(model, state_dict):
    # Copy ONLY encoder tensors (seq.*, struc.*, epi.*), by name and shape.
    # Heads, GatedFusion and link_bilinear always stay freshly initialised.
    dst = model.state_dict()
    picked = {k: v for k, v in state_dict.items()
              if k.startswith(TOWER_PREFIXES) and k in dst and dst[k].shape == v.shape}
    model.load_state_dict(picked, strict=False)
    expected = [k for k in dst if k.startswith(TOWER_PREFIXES)]
    missing = [k for k in expected if k not in picked]
    if missing:
        raise RuntimeError(f"encoder state lacks {len(missing)} tower tensors, e.g. {missing[:3]}")
    return len(picked)


def freeze_towers(model):
    n = 0
    for tower in model.towers().values():
        for p in tower.parameters():
            p.requires_grad_(False); n += p.numel()
    return n


def _rng_state():
    return {"torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def _restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = [t.cpu() for t in state["torch_cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            print(f"  ! checkpoint has {len(cuda_states)} CUDA RNG states, this machine "
                  f"{torch.cuda.device_count()} -- CUDA RNG not restored, run not bit-identical")
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _ckpt_path(run_key):
    return CKPT_DIR / f"ckpt_{run_key}.pt"


def save_ckpt(run_key, payload):
    atomic_save(lambda fh: torch.save(payload, fh), _ckpt_path(run_key))


def load_ckpt(run_key, fingerprint):
    p = _ckpt_path(run_key)
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"{p.name} was written under a different config/data split. Restore the "
            f"config it was trained with, or delete the file to retrain this run.")
    return ck


def fit_resumable(run_key, *, model, optim, sched, train_fn, val_fn, n_epochs, patience,
                  min_epochs, fingerprint, label, quiet=False, extra=None):
    # Shared epoch loop with early stopping and per-epoch atomic checkpoints.
    # The caller seeds (set_all_seeds(run_seed(...))) BEFORE building the
    # model, so a fresh start is reproducible; a resume restores the RNG state.
    ck = load_ckpt(run_key, fingerprint)
    best = {"loss": np.inf, "state": None, "epoch": -1}
    history, bad, start = [], 0, 0
    if ck is not None:
        if ck["done"]:
            if not quiet:
                print(f"  {label}: complete in {_ckpt_path(run_key).name} "
                      f"(best epoch {ck['best_epoch']}) -- not retraining")
            model.load_state_dict(ck["best_state"])
            return ck
        model.load_state_dict(ck["model_state"])
        optim.load_state_dict(ck["optim_state"]); sched.load_state_dict(ck["sched_state"])
        best = {"loss": ck["best_loss"], "state": ck["best_state"], "epoch": ck["best_epoch"]}
        bad, history, start = ck["bad"], ck["history"], ck["epoch"] + 1
        _restore_rng(ck["rng"])
        if not quiet:
            print(f"  {label}: resuming at epoch {start} "
                  f"(best so far: epoch {best['epoch']}, loss {best['loss']:.4f})")

    payload = None
    for ep in range(start, n_epochs):
        t0 = time.time()
        tr = train_fn(); va = val_fn()
        sched.step()
        history.append({"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()},
                        "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        flag = ""
        if np.isfinite(va["loss"]) and va["loss"] < best["loss"] - 1e-4:
            best = {"loss": va["loss"], "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad, flag = 0, "  <- best"
        else:
            bad += 1
        if not quiet:
            skipped = f"  skipped {tr['n_skipped']}" if tr.get("n_skipped") else ""
            parts = " ".join(f"{k} {va[k]:.3f}" for k in LOSS_KEYS[1:] if k in va)
            print(f"  {label} ep{ep:>3}  train {tr['loss']:.4f}  val {va['loss']:.4f}  "
                  f"[{parts}]  {time.time()-t0:.0f}s{skipped}{flag}")
        stop = bad >= patience and ep + 1 >= min_epochs
        done = stop or ep == n_epochs - 1
        if done and best["state"] is None:
            print(f"  ! {label}: validation loss was never finite -- keeping the last weights")
            best = {"loss": np.nan, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        payload = {"fingerprint": fingerprint, "epoch": ep, "done": done,
                   "model_state": model.state_dict(), "optim_state": optim.state_dict(),
                   "sched_state": sched.state_dict(), "best_loss": best["loss"],
                   "best_epoch": best["epoch"], "best_state": best["state"], "bad": bad,
                   "history": history, "rng": _rng_state(), "extra": extra or {}}
        save_ckpt(run_key, payload)
        if stop:
            if not quiet:
                print(f"  {label}: early stop at epoch {ep} (best was {best['epoch']})")
            break
    if payload is None:     # n_epochs == start: nothing left to run
        raise RuntimeError(f"{label}: no epochs to run (start={start}, n_epochs={n_epochs})")
    model.load_state_dict(best["state"])
    return payload


def store_test_preds(run_key, pred):
    # embeddings stored as fp16 to keep finished checkpoints small
    ck = torch.load(_ckpt_path(run_key), map_location="cpu", weights_only=False)
    ck["test_pred"] = {k: (v.astype(np.float16) if k.startswith("z") else v)
                       for k, v in pred.items()}
    save_ckpt(run_key, ck)


def stored_test_preds(ck):
    tp = ck.get("test_pred")
    if tp is None:
        return None
    return {k: (v.astype(np.float32) if k.startswith("z") else v) for k, v in tp.items()}


def inner_split(train_idx):
    # 90/10 train/validation, grouped by gene (same rule for both stages)
    sub_tr, sub_va = next(GroupKFold(n_splits=10).split(
        np.arange(len(train_idx)), groups=ROW_GENES[train_idx]))
    return train_idx[sub_tr], train_idx[sub_va]


def _hist_df(ck, **cols):
    return pd.DataFrame(ck["history"]).assign(**cols)


# ---------------------------------------------------------------- Stage A / end-to-end
def train_graph_model(run_key, variant, train_idx, cfg, device, *, n_epochs, patience,
                      min_epochs, lr, quiet=False, label=None):
    set_all_seeds(run_seed(cfg.seed, run_key))
    tr_idx, va_idx = inner_split(train_idx)
    model = PangenomicGAT(cfg, N_NODE_FEAT, N_EDGE_FEAT, len(TISSUE_LIST),
                          variant=variant).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=n_epochs)
    fp = cfg_fingerprint(cfg, stage="graph", variant=variant, n_epochs=n_epochs,
                         patience=patience, min_epochs=min_epochs, lr=lr,
                         train=content_key(np.sort(train_idx).tolist()))
    dl_tr = make_loader(GRAPHS, tr_idx, cfg.batch_size, True, persistent=True, cfg=cfg, device=device)
    dl_va = make_loader(GRAPHS, va_idx, cfg.batch_size, False, persistent=True, cfg=cfg, device=device)
    ck = fit_resumable(
        run_key, model=model, optim=optim, sched=sched,
        train_fn=lambda: run_epoch(model, dl_tr, SEQ_BANK, cfg, device, optim)[0],
        val_fn=lambda: run_epoch(model, dl_va, SEQ_BANK, cfg, device)[0],
        n_epochs=n_epochs, patience=patience, min_epochs=min_epochs, fingerprint=fp,
        label=label or run_key, quiet=quiet,
        extra={"train_genes": sorted(set(ROW_GENES[train_idx]))})
    return model, ck


# ---------------------------------------------------------------- frozen features
@torch.no_grad()
def extract_features(model, cfg, device, bs_seq=8, bs_graph=64):
    # One eval-mode pass of the frozen towers over every gene / graph.
    model.eval()
    feats, N = {}, len(GRAPHS)
    with amp_ctx(cfg, device):
        if model.seq is not None:
            zs = []
            for i in tqdm(range(0, len(SEQ_BANK), bs_seq), desc="  z_seq per gene", leave=False):
                idx = torch.arange(i, min(i + bs_seq, len(SEQ_BANK)), device=device)
                zs.append(model.seq(SEQ_BANK.onehot(idx, device)).half())
            feats["z_seq_gene"] = torch.cat(zs).cpu()
        if model.struc is not None or model.epi is not None:
            z_str = torch.zeros(N, cfg.d_embed, dtype=torch.float16)
            z_epi = torch.zeros(N, cfg.d_embed, dtype=torch.float16)
            pa, pb, py, prow = [], [], [], []
            for batch in tqdm(make_loader(GRAPHS, np.arange(N), bs_graph, False, cfg=cfg, device=device),
                              desc="  z_str / z_epi / link pairs", leave=False):
                batch = batch.to(device, non_blocking=True)
                rows = batch.row_idx
                rows_c = rows.cpu()
                if model.epi is not None:
                    z_epi[rows_c] = model.epi(batch.epi).half().cpu()
                if model.struc is not None:
                    h, zst, _ = model.struc(batch.x, batch.edge_index, batch.edge_attr,
                                            batch.batch, batch.ptr[:-1])
                    z_str[rows_c] = zst.half().cpu()
                    for pairs, lab in ((batch.pos_edge_label_index, 1.0),
                                       (batch.neg_edge_label_index, 0.0)):
                        pa.append(h[pairs[0]].half().cpu()); pb.append(h[pairs[1]].half().cpu())
                        py.append(torch.full((pairs.shape[1],), lab))
                        prow.append(rows[batch.batch[pairs[0]]].cpu())
            if model.epi is not None:
                feats["z_epi"] = z_epi
            if model.struc is not None:
                feats.update(z_str=z_str, pair_a=torch.cat(pa), pair_b=torch.cat(pb),
                             pair_y=torch.cat(py), pair_row=torch.cat(prow))
    return feats


# ---------------------------------------------------------------- encoders per split
ENCODERS = {}      # split_key -> {"state", "feats", "seen_genes", "key"}


def get_encoder(split_key, train_idx, quiet=False):
    # Stage A for this split (or the external checkpoint), plus its feature
    # cache. Always built from the base CFG, so seed-replicate runs share them.
    train_hash = content_key(np.sort(train_idx).tolist())
    if split_key in ENCODERS:
        enc = ENCODERS[split_key]
        if CFG.encoder_source == "per_split" and enc["train_hash"] != train_hash:
            raise RuntimeError(f"split_key {split_key!r} reused with a different training set")
        return enc
    if CFG.encoder_source == "checkpoint":
        if "external" not in ENCODERS:
            ck = torch.load(CFG.pretrained_checkpoint, map_location="cpu", weights_only=False)
            genes = ck.get("train_genes") or ck.get("extra", {}).get("train_genes")
            if not genes:
                raise RuntimeError(
                    f"{CFG.pretrained_checkpoint} does not record the genes it was trained on, "
                    f"so test-gene leakage cannot be ruled out. Use encoder_source='per_split'.")
            state = ck.get("best_state") or ck.get("state_dict") or ck
            p = Path(CFG.pretrained_checkpoint)
            ENCODERS["external"] = {"state": state, "seen_genes": set(genes), "train_hash": None,
                                    "key": content_key(str(p.resolve()), p.stat().st_mtime_ns)}
        enc = ENCODERS["external"]
    else:
        run_key = f"encA_{split_key}"
        model, ck = train_graph_model(
            run_key, "all3", train_idx, CFG, DEVICE, n_epochs=CFG.encoder_epochs,
            patience=CFG.encoder_patience, min_epochs=0, lr=CFG.lr, quiet=quiet,
            label=f"Stage A {split_key}")
        enc = {"state": ck["best_state"], "seen_genes": set(ck["extra"]["train_genes"]),
               "train_hash": train_hash, "key": content_key(ck["fingerprint"], ck["best_epoch"])}
        del model
    if "feats" not in enc:
        fpath = FEAT_DIR / f"feat_{enc['key']}.pt"
        if fpath.exists():
            enc["feats"] = torch.load(fpath, map_location="cpu", weights_only=True)
        else:
            m = PangenomicGAT(CFG, N_NODE_FEAT, N_EDGE_FEAT, len(TISSUE_LIST), "all3").to(DEVICE)
            load_towers(m, enc["state"])
            t0 = time.time()
            enc["feats"] = extract_features(m, CFG, DEVICE)
            atomic_save(lambda fh: torch.save(enc["feats"], fh), fpath)
            if not quiet:
                print(f"  features for {split_key}: {time.time()-t0:.0f}s -> {fpath.name}")
            del m
    ENCODERS[split_key] = enc
    return enc


# ---------------------------------------------------------------- Stage B
def heads_epoch(model, Fd, rows, cfg, optim=None, collect=False):
    # One pass of fusion + heads over cached features for `rows` (GPU tensor).
    train = optim is not None
    set_train_mode(model, train)
    dev = rows.device
    if train:
        rows = rows[torch.randperm(len(rows), device=dev)]
    in_batch = torch.zeros(Fd["n"], dtype=torch.bool, device=dev)
    tot = torch.zeros(len(LOSS_KEYS), device=dev); n_used = n_skipped = 0
    col = PredCollector() if collect else None
    with torch.set_grad_enabled(train):
        for i in range(0, len(rows), cfg.head_batch_size):
            r = rows[i:i + cfg.head_batch_size]
            z_seq = Fd["z_seq_gene"][Fd["seq_idx"][r]].float() if model.use_seq else None
            z_str = Fd["z_str"][r].float() if model.use_str else None
            z_epi = Fd["z_epi"][r].float() if model.use_epi else None
            z = model.fuse_embeddings(z_seq, z_str, z_epi)
            s_link = y_link = None
            if model.use_str:
                in_batch[r] = True
                m = in_batch[Fd["pair_row"]]
                in_batch[r] = False
                s_link = model.score_pairs(Fd["pair_a"][m].float(), Fd["pair_b"][m].float())
                y_link = Fd["pair_y"][m]
            y = {k: Fd[k][r] for k in ("expr", "expr_bin", "tau", "tissue")}
            logits, l = heads_and_losses(model, z, y, s_link, y_link, cfg)
            if train and not optimizer_step(model, optim, l["loss"], cfg):
                n_skipped += 1
                continue
            tot += torch.stack([l[k].detach().float() for k in LOSS_KEYS])
            n_used += 1
            if collect:
                col.add(logits, y, s_link, y_link, r, z, z_seq, z_str, z_epi)
    avg = dict(zip(LOSS_KEYS, (tot / max(n_used, 1)).tolist()))
    avg["n_skipped"] = n_skipped
    return avg, (col.result() if collect else None)


def _features_on_device(feats, device):
    Fd = {k: v.to(device, non_blocking=True) for k, v in feats.items()}
    Fd["n"] = len(GRAPHS)
    Fd["seq_idx"] = ROW_SEQ_IDX.to(device)
    for k, col, dt in (("expr", "y_expr", torch.float32), ("expr_bin", "y_expr_bin", torch.float32),
                       ("tau", "y_tau", torch.float32), ("tissue", "y_tissue", torch.long)):
        Fd[k] = torch.as_tensor(META[col].to_numpy(), dtype=dt, device=device)
    return Fd


def train_heads(run_key, variant, enc, train_idx, test_idx, cfg, device, quiet=False, label=None):
    set_all_seeds(run_seed(cfg.seed, run_key))
    tr_idx, va_idx = inner_split(train_idx)
    model = PangenomicGAT(cfg, N_NODE_FEAT, N_EDGE_FEAT, len(TISSUE_LIST),
                          variant=variant).to(device)
    load_towers(model, enc["state"])
    freeze_towers(model)
    lr = cfg.lr * cfg.head_lr_scale
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                              lr=lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=cfg.n_epochs)
    fp = cfg_fingerprint(cfg, stage="heads", variant=variant, encoder=enc["key"],
                         train=content_key(np.sort(train_idx).tolist()))
    Fd = _features_on_device(enc["feats"], device)
    rows_tr = torch.as_tensor(tr_idx, device=device)
    rows_va = torch.as_tensor(va_idx, device=device)
    ck = fit_resumable(
        run_key, model=model, optim=optim, sched=sched,
        train_fn=lambda: heads_epoch(model, Fd, rows_tr, cfg, optim)[0],
        val_fn=lambda: heads_epoch(model, Fd, rows_va, cfg)[0],
        n_epochs=cfg.n_epochs, patience=cfg.patience, min_epochs=cfg.min_epochs_heads,
        fingerprint=fp, label=label or run_key, quiet=quiet)
    pred = stored_test_preds(ck)
    if pred is None:
        _, pred = heads_epoch(model, Fd, torch.as_tensor(test_idx, device=device), cfg, collect=True)
        store_test_preds(run_key, pred)
    return model, pred, ck


# ---------------------------------------------------------------- one split, one arm
def train_one_fold(fold, train_idx, test_idx, graphs, seq_bank, cfg, device,
                   variant="all3", quiet=False, run_key=None, split_key=None):
    # -> (model, test predictions, history DataFrame, best epoch)
    # split_key names the data split (shared by every arm/seed on it) and so
    # selects the Stage-A encoders; run_key names this arm's own run.
    assert run_key and split_key, "train_one_fold needs run_key and split_key"
    if CFG.encoder_source == "none":
        model, ck = train_graph_model(run_key, variant, train_idx, cfg, device,
                                      n_epochs=cfg.n_epochs, patience=cfg.patience,
                                      min_epochs=0, lr=cfg.lr, quiet=quiet,
                                      label=f"f{fold} {variant}")
        pred = stored_test_preds(ck)
        if pred is None:
            dl_te = make_loader(graphs, test_idx, cfg.batch_size, False, cfg=cfg, device=device)
            _, pred = run_epoch(model, dl_te, seq_bank, cfg, device, collect=True)
            store_test_preds(run_key, pred)
        return model, pred, _hist_df(ck, variant=variant, fold=fold), ck["best_epoch"]

    enc = get_encoder(split_key, train_idx, quiet=quiet)
    if CFG.encoder_source == "checkpoint":
        # an external encoder may have been trained on some of this split's
        # test genes: score only the genes it has never seen
        keep = ~np.isin(ROW_GENES[test_idx], list(enc["seen_genes"]))
        if not quiet:
            print(f"  f{fold}: dropping {int((~keep).sum()):,} of {len(test_idx):,} test graphs "
                  f"whose gene the external encoder was trained on")
        test_idx = test_idx[keep]
    model, pred, ck = train_heads(run_key, variant, enc, train_idx, test_idx, cfg, device,
                                  quiet=quiet, label=f"f{fold} {variant}")
    return model, pred, _hist_df(ck, variant=variant, fold=fold), ck["best_epoch"]


VARIANT_LABEL = {
    "all3": "seq + str + epi", "seq_str": "seq + str (no epi)",
    "seq_epi": "seq + epi (no str)", "str_epi": "str + epi (no seq)",
    "seq": "sequence only", "str": "structure only", "epi": "epigenomic only",
}
print(f"device               : {DEVICE}")
print(f"mixed precision      : "
      f"{CFG.amp_dtype + ' autocast' if (DEVICE.type == 'cuda' and CFG.use_amp) else 'off (fp32)'}"
      f", TF32 {'on' if CFG.allow_tf32 else 'off'}")
print(f"DataLoader workers   : {CFG.num_workers}  (pin_memory={DEVICE.type == 'cuda'})")
print(f"encoders             : {CFG.encoder_source}")


# %%
RESULTS = {}                               # variant -> dict(preds, metrics, hist, models)
t_start = time.time()

if CFG.encoder_source != "none":
    print(f"{'#'*74}\n#  STAGE A: one all3 encoder per gene-CV fold\n{'#'*74}")
    for k, (tr, te) in enumerate(FOLDS):
        print(f"\n--- encoders for fold {k+1}/{CFG.n_folds}  train={len(tr):,} ---")
        get_encoder(f"gene_fold{k}", tr)
    print(f"\nStage A done in {(time.time()-t_start)/60:.1f} min")

for variant in CFG.variants:
    print(f"\n\n{'#'*74}\n#  ARM: {variant.upper():<10} ({VARIANT_LABEL[variant]})\n{'#'*74}")
    v_preds, v_metrics, v_hist, v_models = [], [], [], []
    for k, (tr, te) in enumerate(FOLDS):
        print(f"\n{variant} | FOLD {k+1}/{CFG.n_folds}   train={len(tr):,}  test={len(te):,}")
        model, pred, hist, best_ep = train_one_fold(
            k, tr, te, GRAPHS, SEQ_BANK, CFG, DEVICE, variant=variant,
            run_key=f"main_{variant}_fold{k}", split_key=f"gene_fold{k}")
        met = all_metrics(pred, len(TISSUE_LIST))
        met["fold"] = k; met["best_epoch"] = best_ep; met["variant"] = variant
        v_preds.append(pred); v_metrics.append(met)
        v_hist.append(hist); v_models.append(model)
        print(f"  {variant} fold {k} test:  "
              f"link AUPRC {met.get('link_auprc', np.nan):.3f} | "
              f"expr Pearson r {met.get('expr_pearson_r', np.nan):.3f} "
              f"(match@1.0 {met.get('expr_match_acc_1.0', np.nan):.1%}) | "
              f"expr_bin AUPRC {met.get('expr_bin_auprc', np.nan):.3f} | "
              f"tau AUPRC {met.get('tau_auprc', np.nan):.3f} | "
              f"tissue acc {met.get('tissue_accuracy', np.nan):.3f}")
    RESULTS[variant] = {"preds": v_preds, "metrics": v_metrics,
                        "hist": pd.concat(v_hist, ignore_index=True),
                        "models": v_models}

print(f"\n{'='*74}\nall {len(CFG.variants)} arms x {CFG.n_folds} folds done in "
      f"{(time.time()-t_start)/60:.1f} min")

# the detailed downstream cells report on the full model
PRIMARY = "all3" if "all3" in RESULTS else CFG.variants[0]
fold_preds   = RESULTS[PRIMARY]["preds"]
fold_metrics = RESULTS[PRIMARY]["metrics"]
fold_models  = RESULTS[PRIMARY]["models"]
TRAIN_HIST   = RESULTS[PRIMARY]["hist"]
print(f"cells below report the '{PRIMARY}' arm ({VARIANT_LABEL[PRIMARY]}); "
      f"the ablation cells compare all {len(RESULTS)}")


# %% [markdown]
# ## 17 · Results table
#
# Mean ± standard deviation across the five folds, for every metric and every
# head. The standard deviation is the number to look at hardest: a head with high
# mean AUPRC and a large spread across folds has probably latched onto something
# fold-specific, and the mean alone is misleading.
#
# Read the binary heads against their **AUPRC baseline** column. A head whose
# AUPRC barely exceeds its baseline has learned nothing, no matter how good its
# AUROC looks.


# %%
MET = pd.DataFrame(fold_metrics).drop(columns=["variant"]).set_index("fold")
MET.to_csv(RESULTS_DIR / "metrics_per_fold.csv")

def summarise(prefix, keys):
    rows = []
    for k in keys:
        col = f"{prefix}{k}"
        if col in MET.columns:
            v = MET[col].astype(float)
            rows.append({"metric": k, "mean": v.mean(), "std": v.std(),
                         "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows).set_index("metric")

BIN_KEYS = ["n", "pos_rate", "auroc", "auprc", "auprc_baseline",
            "accuracy", "balanced_accuracy", "precision", "recall",
            "specificity", "npv", "f1", "f2", "mcc", "kappa", "jaccard",
            "brier", "logloss", "best_threshold", "f1_at_best",
            "precision_at_best", "recall_at_best"]
MC_KEYS = ["n", "n_classes_present", "accuracy", "balanced_accuracy", "kappa",
           "precision_macro", "recall_macro", "f1_macro",
           "precision_micro", "recall_micro", "f1_micro",
           "precision_weighted", "recall_weighted", "f1_weighted",
           "auroc_macro_ovr", "auprc_macro_ovr"]

REG_KEYS = ["n", "mean", "std", "mse", "rmse", "mae", "r2",
            "pearson_r", "pearson_p", "spearman_r", "spearman_p",
            "match_acc_0.5", "match_acc_1.0", "match_acc_2.0"]

HEADS = [("H1  link prediction",              "link_",     BIN_KEYS),
         ("H2a expression signal (log2 TPM+1), continuous", "expr_",     REG_KEYS),
         ("H2b expressed / not-expressed (TPM>1), binary",  "expr_bin_", BIN_KEYS),
         ("H3  tissue-restricted (tau)",       "tau_",      BIN_KEYS),
         ("H4  tissue classification",         "tissue_",   MC_KEYS)]

SUMMARIES = {}
for title, pref, keys in HEADS:
    s = summarise(pref, keys)
    SUMMARIES[pref] = s
    print(f"\n{'='*74}\n{title}   (mean +/- sd over {CFG.n_folds} folds)\n{'='*74}")
    print(s.round(4).to_string())
    s.to_csv(RESULTS_DIR / f"metrics_summary_{pref.rstrip('_')}.csv")

print(f"\n\n{'='*74}\nHEADLINE\n{'='*74}")
head_rows = []
for title, pref, _ in HEADS:
    if pref == "tissue_":
        head_rows.append([title, f"acc {MET['tissue_accuracy'].mean():.3f} +/- {MET['tissue_accuracy'].std():.3f}",
                          f"F1 {MET['tissue_f1_macro'].mean():.3f}",
                          f"AUROC {MET['tissue_auroc_macro_ovr'].mean():.3f}",
                          f"AUPRC {MET['tissue_auprc_macro_ovr'].mean():.3f}"])
    elif pref == "expr_":
        head_rows.append([title, f"Pearson r {MET['expr_pearson_r'].mean():.3f} +/- {MET['expr_pearson_r'].std():.3f}",
                          f"R^2 {MET['expr_r2'].mean():.3f}",
                          f"MAE {MET['expr_mae'].mean():.3f}",
                          f"match@1.0 {MET['expr_match_acc_1.0'].mean():.1%}"])
    else:
        head_rows.append([title, f"acc {MET[pref+'accuracy'].mean():.3f} +/- {MET[pref+'accuracy'].std():.3f}",
                          f"F1 {MET[pref+'f1'].mean():.3f}",
                          f"AUROC {MET[pref+'auroc'].mean():.3f}",
                          f"AUPRC {MET[pref+'auprc'].mean():.3f}"])
print(pd.DataFrame(head_rows, columns=["head", "primary metric", "secondary", "tertiary", "quaternary"]
                   ).to_string(index=False))
print(f"\nper-fold metrics -> {(RESULTS_DIR/'metrics_per_fold.csv').name}")
print("\nH2 is reported TWICE, from two different heads on the same embedding:")
print("  H2a (expr_)     continuous head_expr      -- Pearson r / R^2 / MAE / RMSE")
print("                  + match-within-tolerance (match_acc_0.5/1.0/2.0 = fraction of")
print("                  predictions within +/-0.5/1.0/2.0 log2 units, i.e. within")
print("                  ~1.4x/2x/4x fold-change -- a correct continuous prediction is")
print("                  a close one, not only an exact one.")
print("  H2b (expr_bin_) binary head_expr_bin      -- AUPRC/AUROC/accuracy/F1, same")
print("                  metric family as H1/H3. A real second classifier head, not")
print("                  head_expr's output thresholded after the fact.")
print("Neither substitutes for the other -- 'is this gene on here?' and 'how much of")
print("it is there?' are different questions, reported separately throughout.")


# %% [markdown]
# ## 18 · ROC curves
#
# One panel per binary head. Each fold is drawn as a thin line, and the mean
# curve is computed by interpolating every fold onto a common false-positive-rate
# grid and averaging the true-positive rates, with a ±1 sd band.
#
# Interpolating then averaging is the right way to do this. Averaging the five
# AUC *numbers* and drawing one curve through the pooled predictions are both
# common and both hide fold-to-fold variance — which for a biological dataset
# split by gene is exactly the quantity you want to see.


# %%
def roc_panel(ax, key, title):
    grid = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    for k, pred in enumerate(fold_preds):
        y, p = pred[f"{key}_y"], pred[f"{key}_p"]
        m = y >= 0
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[m], p[m])
        auc = roc_auc_score(y[m], p[m]); aucs.append(auc)
        tprs.append(np.interp(grid, fpr, tpr)); tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, lw=.9, alpha=.45, color=PALETTE[k % len(PALETTE)],
                label=f"fold {k} (AUC {auc:.3f})")
    if tprs:
        mt = np.mean(tprs, 0); st = np.std(tprs, 0); mt[-1] = 1.0
        ax.plot(grid, mt, lw=2.4, color="k",
                label=f"mean (AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f})")
        ax.fill_between(grid, np.clip(mt-st, 0, 1), np.clip(mt+st, 0, 1),
                        color="k", alpha=.13, label="+/- 1 sd")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
roc_panel(axes[0], "link",     "H1  link prediction")
roc_panel(axes[1], "expr_bin", "H2b  expressed / not (head_expr_bin)")
roc_panel(axes[2], "tau",      "H3  tissue-restricted")
fig.suptitle("ROC, 5-fold cross-validation (gene-grouped splits) -- H2a (continuous\n"
            "head_expr) has no ROC; see the predicted-vs-observed plot below instead",
             y=1.05, fontsize=12)
fig.tight_layout(); savefig(fig, "07_roc_curves"); plt.show()


# %% [markdown]
# ## 19 · Precision–recall curves
#
# The same three heads, but on the axes that actually matter for imbalanced data.
#
# Each panel carries a dashed horizontal line at the **positive prevalence** —
# the AUPRC a random classifier achieves. The distance between the curve and that
# line is the real signal. For the expression head, where roughly half of
# gene×tissue pairs are positive, the baseline sits near 0.5 and a 0.6 AUPRC is
# weak; for a head with 5% positives, 0.4 would be strong.


# %%
def pr_panel(ax, key, title):
    grid = np.linspace(0, 1, 200)
    precs, aps, bases = [], [], []
    for k, pred in enumerate(fold_preds):
        y, p = pred[f"{key}_y"], pred[f"{key}_p"]
        m = y >= 0
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        pr, rc, _ = precision_recall_curve(y[m], p[m])
        ap = average_precision_score(y[m], p[m]); aps.append(ap)
        bases.append(float(np.mean(y[m])))
        ax.plot(rc, pr, lw=.9, alpha=.45, color=PALETTE[k % len(PALETTE)],
                label=f"fold {k} (AP {ap:.3f})")
        order = np.argsort(rc)
        precs.append(np.interp(grid, rc[order], pr[order]))
    if precs:
        mp = np.mean(precs, 0); sp = np.std(precs, 0)
        ax.plot(grid, mp, lw=2.4, color="k",
                label=f"mean (AP {np.mean(aps):.3f} +/- {np.std(aps):.3f})")
        ax.fill_between(grid, np.clip(mp-sp, 0, 1), np.clip(mp+sp, 0, 1),
                        color="k", alpha=.13, label="+/- 1 sd")
        b = float(np.mean(bases))
        ax.axhline(b, ls="--", c="r", lw=1.2, label=f"baseline = {b:.3f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pr_panel(axes[0], "link",     "H1  link prediction")
pr_panel(axes[1], "expr_bin", "H2b  expressed / not (head_expr_bin)")
pr_panel(axes[2], "tau",      "H3  tissue-restricted")
fig.suptitle("Precision-recall, 5-fold CV (dashed red = random baseline) -- H2a\n"
            "(continuous head_expr) has no PR curve; see the scatter plot below",
             y=1.05, fontsize=12)
fig.tight_layout(); savefig(fig, "08_pr_curves"); plt.show()

print("read each curve against its own red baseline, not against the other panels")


# %% [markdown]
# ## H2 predicted vs. observed signal, and residuals
#
# H2 is regression now, so its diagnostic is a predicted-vs-observed scatter
# and a residual plot, not a ROC/PR curve -- reused directly from
# `pangenomic_gat_pipeline_trimodal_1mb_h2_signal.ipynb`'s own H2-only
# version of this check.


# %%
def scatter_panel(ax, fold_preds, title):
    all_y, all_p = [], []
    for k, pred in enumerate(fold_preds):
        y, p = pred["expr_y"], pred["expr_p"]
        m = y >= 0
        if m.sum() == 0:
            continue
        ax.scatter(y[m], p[m], s=4, alpha=.25, color=PALETTE[k % len(PALETTE)],
                   linewidths=0, label=f"fold {k}")
        all_y.append(y[m]); all_p.append(p[m])
    if not all_y:
        ax.text(.5, .5, "no data", ha="center"); return None, None
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    diag = np.linspace(lo, hi, 50)
    for tol, alpha in zip(CFG.expr_match_tolerances, (0.18, 0.11, 0.06)):
        ax.fill_between(diag, diag - tol, diag + tol, color="k", alpha=alpha,
                        label=f"+/- {tol} log2 (~{2**tol:.1f}x)")
    ax.plot([lo, hi], [lo, hi], "--", c="k", lw=1.4, label="y = x (perfect)")
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    r, _ = sstats.pearsonr(y, p)
    rho, _ = sstats.spearmanr(y, p)
    match_str = "  ".join(f"match@{t}={float((np.abs(y-p)<=t).mean()):.1%}"
                          for t in CFG.expr_match_tolerances)
    ax.set_title(f"{title}\nR2={r2:.3f}  Pearson r={r:.3f}  Spearman rho={rho:.3f}\n{match_str}",
                fontsize=9.5)
    ax.set_xlabel("observed log2(TPM + 1)"); ax.set_ylabel("predicted")
    ax.legend(fontsize=6.5, loc="upper left")
    return y, p

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
y_pool, p_pool = scatter_panel(axes[0], fold_preds, "H2  expression signal, pooled over folds")
if y_pool is not None:
    resid = y_pool - p_pool
    axes[1].scatter(p_pool, resid, s=4, alpha=.25, color=PALETTE[2], linewidths=0)
    axes[1].axhline(0, ls="--", c="k", lw=1.2)
    axes[1].set_xlabel("predicted"); axes[1].set_ylabel("residual (observed - predicted)")
    axes[1].set_title(f"Residuals  (mean={resid.mean():.3f}, std={resid.std():.3f})")
fig.tight_layout(); savefig(fig, "07b_h2_predicted_vs_observed"); plt.show()

print("Points on the y=x line are correct; a slope < 1 (a fan tilted toward the mean)")
print("means the model is regressing toward the mean rather than using its inputs.")
print("\nThe shaded diagonal bands mark the tolerance windows behind match_acc_*: a")
print("point that lands in the darkest band, not exactly on the line, is still a")
print("correct prediction of the expression level -- being close IS being right for")
print("a continuous target; it is not a weaker, apologetic version of exact.")


# %% [markdown]
# ## 20 · Confusion matrices and threshold choice
#
# Left three panels: pooled confusion matrices for the binary heads at the
# **F1-optimal threshold** rather than at 0.5. With imbalanced data the 0.5 cut is
# arbitrary, and reporting only 0.5 numbers understates a model that ranks well
# but is poorly calibrated.
#
# Right: the tissue-classification confusion matrix, row-normalised. This is worth
# staring at — confusions here are informative rather than embarrassing. Related
# biosamples (the various CD4+ subsets, for instance) *should* be confusable; if
# they are perfectly separated, the model has probably found a batch signature
# rather than biology.
#
# Below, the F1-vs-threshold curve shows how sharply performance depends on where
# the cut is placed.


# %%
def pool(key):
    y = np.concatenate([p[f"{key}_y"] for p in fold_preds])
    s = np.concatenate([p[f"{key}_p"] for p in fold_preds])
    m = y >= 0
    return y[m].astype(int), s[m]

BINARY_HEADS = [("link", "H1 link"), ("expr_bin", "H2b expressed"), ("tau", "H3 restricted")]

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.8, 1])

for i, (key, name) in enumerate(BINARY_HEADS):
    ax = fig.add_subplot(gs[0, i])
    y, s = pool(key)
    if len(np.unique(y)) < 2:
        ax.text(.5, .5, "single class", ha="center"); ax.axis("off"); continue
    pr, rc, th = precision_recall_curve(y, s)
    f1 = np.divide(2*pr*rc, pr+rc, out=np.zeros_like(pr), where=(pr+rc) > 0)
    bt = float(th[int(np.nanargmax(f1[:-1]))]) if len(th) else .5
    cm = confusion_matrix(y, (s >= bt).astype(int), labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"{name}\nthreshold={bt:.3f}   F1={2*tp/(2*tp+fp+fn):.3f}", fontsize=10)

ax = fig.add_subplot(gs[1, :])
for i, (key, name) in enumerate(BINARY_HEADS):
    y, s = pool(key)
    if len(np.unique(y)) < 2:
        continue
    pr, rc, th = precision_recall_curve(y, s)
    f1 = np.divide(2*pr*rc, pr+rc, out=np.zeros_like(pr), where=(pr+rc) > 0)
    if len(th):
        ax.plot(th, f1[:-1], lw=1.6, color=PALETTE[i], label=name)
        ax.axvline(th[int(np.nanargmax(f1[:-1]))], ls=":", lw=1, color=PALETTE[i])
ax.axvline(.5, ls="--", c="k", lw=1, label="default 0.5")
ax.set_xlabel("threshold"); ax.set_ylabel("F1"); ax.set_title("F1 vs threshold, every binary head")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[2, :])
yt = np.concatenate([p["tis_y"] for p in fold_preds]).astype(int)
yp = np.concatenate([p["tis_p"] for p in fold_preds]).argmax(1)
present = sorted(set(yt) | set(yp))
cm = confusion_matrix(yt, yp, labels=present).astype(float)
cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
sns.heatmap(cm, cmap="magma", vmin=0, vmax=1, ax=ax,
            xticklabels=[TISSUE_LIST[i] for i in present],
            yticklabels=[TISSUE_LIST[i] for i in present],
            cbar_kws={"label": "row-normalised fraction"})
ax.set_title(f"H4 tissue classification, row-normalised  "
             f"(accuracy {accuracy_score(yt, yp):.3f})")
ax.tick_params(axis="x", rotation=90, labelsize=7)
ax.tick_params(axis="y", rotation=0, labelsize=7)

fig.tight_layout(); savefig(fig, "09_confusion_matrices"); plt.show()

print("\nH2a (continuous head_expr) has no confusion matrix -- see the scatter plot's")
print("match-within-tolerance rates instead. The H2b panel above is head_expr_bin's")
print("own probabilities, a second independent classifier, not head_expr thresholded.")
print("\nper-class report for H4 (pooled over folds):")
print(classification_report(yt, yp, labels=present,
                            target_names=[TISSUE_LIST[i] for i in present],
                            zero_division=0))


# %% [markdown]
# ## 21 · Calibration
#
# Ranking and calibration are different properties. A model can order examples
# perfectly (AUROC 1.0) while its probabilities are systematically wrong, which
# matters the moment anyone thresholds the output or uses it as a prior.
#
# The reliability diagram bins predictions by confidence and plots observed
# frequency against predicted probability; the diagonal is perfect calibration.
# Below the diagonal means overconfident. The Brier score and the **expected
# calibration error** quantify the gap.
#
# Multi-task training tends to produce mild overconfidence, because the shared
# trunk is optimised for four objectives at once and no single head's probability
# scale is directly regularised. If a head is badly miscalibrated, Platt scaling
# or temperature scaling fitted on the validation split is the standard fix — and
# it must be fitted on validation, never on test.


# %%
def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return e

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for i, (key, name) in enumerate(BINARY_HEADS):
    ax = axes[i]
    y, p = pool(key)
    if len(np.unique(y)) < 2:
        ax.text(.5, .5, "single class", ha="center"); ax.axis("off"); continue
    for k, pred in enumerate(fold_preds):
        yy, pp = pred[f"{key}_y"], pred[f"{key}_p"]
        m = yy >= 0
        if m.sum() < 50 or len(np.unique(yy[m])) < 2:
            continue
        try:
            ft, mp = calibration_curve(yy[m], pp[m], n_bins=10, strategy="quantile")
            ax.plot(mp, ft, "o-", ms=3, lw=.9, alpha=.5,
                    color=PALETTE[k % len(PALETTE)])
        except Exception:
            pass
    ft, mp = calibration_curve(y, p, n_bins=12, strategy="quantile")
    ax.plot(mp, ft, "o-", color="k", lw=2.2, ms=5, label="pooled")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="perfect")
    ax.set_title(f"{name}\nBrier={brier_score_loss(y, p):.4f}  ECE={ece(y, p):.4f}",
                 fontsize=10)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.legend(fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax = axes[3]
for i, (key, name) in enumerate(BINARY_HEADS):
    y, p = pool(key)
    if len(np.unique(y)) < 2:
        continue
    ax.hist(p, bins=40, histtype="step", lw=1.8, density=True,
            color=PALETTE[i], label=name)
ax.set_title("predicted probability distributions")
ax.set_xlabel("p"); ax.set_ylabel("density"); ax.legend(fontsize=8)

fig.tight_layout(); savefig(fig, "10_calibration"); plt.show()

print("A U-shaped probability histogram means confident predictions; a spike at")
print("the prevalence means the head has collapsed to predicting the base rate.")
print("(Calibration is a binary-probability concept -- H2a's continuous head_expr")
print("is not shown here; its equivalent honesty check is the residual plot above.)")


# %% [markdown]
# ## 22 · Training curves
#
# Total loss and each of the five components, train vs validation, averaged over
# folds with a min–max band.
#
# What to look for:
#
# - **Train and validation diverging** → overfitting; raise dropout or weight
#   decay, or lower `MAX_NEIGHBOURS`.
# - **A head whose loss goes flat almost immediately** → it has saturated and is
#   contributing no gradient. The tissue head is the usual suspect. Its loss
#   weight can then be reduced without hurting anything.
#
# These curves are Stage B (fusion + heads on frozen encoders), so the
# alignment term is 0 by construction; Stage-A curves are in each
# `ckpt_encA_*.pt` checkpoint's history.


# %%
comps = ["loss", "link", "expr", "expr_bin", "tau", "tissue", "align"]
fig, axes = plt.subplots(3, 3, figsize=(15, 10.5))
for ax, c in zip(axes.ravel(), comps):
    for phase, col, style in [("train", PALETTE[0], "-"), ("val", PALETTE[2], "-")]:
        piv = TRAIN_HIST.pivot_table(index="epoch", columns="fold", values=f"{phase}_{c}")
        m = piv.mean(1)
        ax.plot(m.index, m.values, style, color=col, lw=2, label=phase)
        ax.fill_between(piv.index, piv.min(1), piv.max(1), color=col, alpha=.15)
    ax.set_title(f"{c} loss"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
for ax in axes.ravel()[len(comps):]:
    ax.axis("off")
fig.suptitle("Training dynamics (mean over folds, band = min-max) -- 'expr' is head_expr's\n"
            "MSE, 'expr_bin' is head_expr_bin's BCE, tracked and optimised separately",
             y=1.01, fontsize=13)
fig.tight_layout(); savefig(fig, "11_training_curves"); plt.show()

print("best epoch per fold:", MET["best_epoch"].tolist())
print(f"mean seconds/epoch: {TRAIN_HIST['sec'].mean():.1f}")
TRAIN_HIST.to_csv(RESULTS_DIR / "training_history.csv", index=False)


# %% [markdown]
# ## 23 · The two embedding spaces
#
# The point of the whole architecture is the pair of embedding spaces, so we look
# at them directly. Using the final fold's model we embed the held-out graphs and
# project each space to 2-D (UMAP when available, PCA otherwise).
#
# Expectations, and what a violation would mean:
#
# - **`z_seq`** should cluster by *gene family / promoter type* and be blind to
#   tissue — it never sees tissue information, so visible tissue structure would
#   indicate a bug (most likely a leak through the shared trunk).
# - **`z_str`** should cluster by *tissue*, since contact profile and histone
#   state are tissue-specific.
# - **`z` (fused)** should show both: tissue structure inherited from `z_str`,
#   modulated by gene identity.
#
# The right-hand panel quantifies it with a silhouette score against tissue
# labels for each space — the numeric version of the same claim.


# %%
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

last_pred = fold_preds[-1]
te_idx = FOLDS[-1][1]
meta_te = META.iloc[te_idx].reset_index(drop=True)

def project(X, seed=0):
    X = np.asarray(X, dtype=np.float32)
    if len(X) > 6000:
        sub = np.random.default_rng(seed).choice(len(X), 6000, replace=False)
    else:
        sub = np.arange(len(X))
    Xs = X[sub]
    if HAS_UMAP:
        try:
            red = umap_mod.UMAP(n_neighbors=25, min_dist=.15,
                                random_state=seed).fit_transform(Xs)
            return red, sub, "UMAP"
        except Exception as e:
            print(f"  UMAP failed ({e}); using PCA")
    return PCA(n_components=2, random_state=seed).fit_transform(Xs), sub, "PCA"

# only include a space if the PRIMARY arm actually produced it (z_epi is
# None/empty for any arm without the epigenomic tower active)
spaces = [(name, last_pred[key]) for name, key in [
    ("z_seq  (sequence space)", "z_seq"),
    ("z_str  (structure space)", "z_str"),
    ("z_epi  (epigenomic space)", "z_epi"),
    ("z      (fused gene x tissue)", "z"),
] if last_pred.get(key) is not None and len(np.asarray(last_pred[key])) > 0]

n_panels = len(spaces) + 1
fig, axes = plt.subplots(1, n_panels, figsize=(4.9 * n_panels, 4.6))
axes = np.atleast_1d(axes)
sils = {}
for ax, (name, X) in zip(axes[:len(spaces)], spaces):
    emb, sub, how = project(X)
    lab = meta_te["tissue"].values[sub]
    codes = pd.Categorical(lab).codes
    ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=5, alpha=.75,
               linewidths=0)
    try:
        sils[name] = silhouette_score(X[sub], codes) if len(set(codes)) > 1 else np.nan
    except Exception:
        sils[name] = np.nan
    ax.set_title(f"{name}\n{how}, coloured by tissue\n"
                 f"silhouette={sils[name]:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

ax = axes[len(spaces)]
names = list(sils.keys()); vals = [sils[n] for n in names]
ax.barh([n.split("(")[0].strip() for n in names], vals, color=PALETTE[:len(names)])
ax.axvline(0, c="k", lw=1)
ax.set_title("Tissue silhouette by space\n(higher = more tissue structure)", fontsize=9)
ax.set_xlabel("silhouette score")

fig.tight_layout(); savefig(fig, "12_embedding_spaces"); plt.show()

print("\nsilhouette against tissue labels:")
for n, v in sils.items():
    print(f"  {n:<32} {v:.4f}")
print("\nz_seq should stay tissue-blind (it never receives tissue information);")
print("if it scores as high as the tissue-aware towers, tissue information is")
print("leaking into it. z_epi and z_str are both tissue-aware by construction")
print("(chromatin state and contact topology are both tissue-specific), so a")
print("similar silhouette between them is expected, not a red flag on its own.")


# %% [markdown]
# ## 24 · What the attention actually attends to
#
# The GAT's final-layer attention weights are returned by `GATv2Conv(...,
# return_attention_weights=True)`. We take a batch of test graphs and ask: for the
# edges pointing into the promoter anchor, what predicts a high attention weight?
#
# Three views:
#
# 1. **Attention vs CHiCAGO score** — if the model has learned nothing beyond the
#    input evidence, this is a tight monotone relationship. Scatter with structure
#    *around* the trend is the interesting case: contacts the model up-weights
#    beyond what their score alone would justify.
# 2. **Attention vs genomic distance** — a strong distance dependence would mean
#    the attention has rediscovered the distance-decay prior rather than
#    regulatory logic.
# 3. **Attention weight distribution** — concentrated (a few dominant partners)
#    or diffuse (near-uniform, i.e. the attention layer is doing nothing useful).
#
# This is the panel that tells you whether the architecture earned its complexity
# over a plain GCN.


# %%
model = fold_models[-1].eval()
dl = make_loader(GRAPHS, FOLDS[-1][1][:2048], CFG.batch_size, False)

_att = []
with torch.no_grad():
    for batch in dl:
        batch = batch.to(DEVICE)
        h, z_str, attn = model.struc(batch.x, batch.edge_index, batch.edge_attr,
                                     batch.batch, batch.ptr[:-1])
        if attn is None:
            break
        ei, alpha = attn
        a = alpha.float().mean(-1).cpu().numpy()  # average over heads
        dst = ei[1].cpu().numpy()
        ea = batch.edge_attr.cpu().numpy()
        # GATv2 appends self-loops after the original edges, so only the first
        # n_orig attention entries line up with edge_attr
        n = min(len(a), ea.shape[0])
        k = np.nonzero(np.isin(dst[:n], batch.ptr[:-1].cpu().numpy()))[0]   # edges into a promoter
        _att.append(pd.DataFrame({"alpha": a[k], "log_score": ea[k, 0], "log_reads": ea[k, 1],
                                  "sgn_log_d": ea[k, 2], "is_backbone": (ea[k, 0] == 0).astype(float)}))
        if sum(map(len, _att)) > 60000:
            break

A = pd.concat(_att, ignore_index=True) if _att else pd.DataFrame()
print(f"collected {len(A):,} attention weights on edges into promoter anchors")

if len(A) > 100:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    real = A[A.is_backbone == 0]

    axes[0].scatter(real["log_score"], real["alpha"], s=3, alpha=.15,
                    color=PALETTE[0], linewidths=0)
    if len(real) > 20:
        b = pd.cut(real["log_score"], 12)
        med = real.groupby(b, observed=True)["alpha"].median()
        cen = [iv.mid for iv in med.index]
        axes[0].plot(cen, med.values, "o-", color="k", lw=2, ms=4, label="median")
        r = sstats.spearmanr(real["log_score"], real["alpha"])
        axes[0].set_title(f"attention vs CHiCAGO score\nSpearman rho={r.correlation:.3f}")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("log1p(CHiCAGO score)"); axes[0].set_ylabel("attention weight")

    cis = real[real["sgn_log_d"].abs() > 0]
    axes[1].scatter(cis["sgn_log_d"].abs(), cis["alpha"], s=3, alpha=.15,
                    color=PALETTE[2], linewidths=0)
    if len(cis) > 20:
        r2 = sstats.spearmanr(cis["sgn_log_d"].abs(), cis["alpha"])
        axes[1].set_title(f"attention vs genomic distance\nSpearman rho={r2.correlation:.3f}")
    axes[1].set_xlabel("|log10 distance to TSS|"); axes[1].set_ylabel("attention weight")

    axes[2].hist(real["alpha"], bins=60, color=PALETTE[3], density=True,
                 label="PCHi-C edges")
    if (A.is_backbone == 1).sum() > 10:
        axes[2].hist(A[A.is_backbone == 1]["alpha"], bins=60, alpha=.6,
                     color=PALETTE[1], density=True, label="backbone edges")
    axes[2].axvline(real["alpha"].mean(), ls="--", c="k", lw=1,
                    label=f"mean {real['alpha'].mean():.3f}")
    axes[2].set_title("attention weight distribution")
    axes[2].set_xlabel("attention weight"); axes[2].legend(fontsize=8)

    fig.suptitle("Final-layer attention on edges into the promoter anchor",
                 y=1.03, fontsize=12)
    fig.tight_layout(); savefig(fig, "13_attention_analysis"); plt.show()

    gini = lambda x: (np.abs(np.subtract.outer(x, x)).mean() / (2 * np.mean(x))
                      if np.mean(x) > 0 else np.nan)
    g = gini(real["alpha"].sample(min(3000, len(real)), random_state=0).values)
    print(f"\nGini of attention weights: {g:.3f}")
    print("  ~0   = uniform attention (the GAT is behaving like a mean aggregator)")
    print("  >0.3 = genuinely selective, a few partners dominate")
else:
    print("Not enough attention records collected to plot.")


# %% [markdown]
# ## 25 · Export
#
# Everything a reader needs to check or reuse the run:
#
# - `results/metrics_per_fold.csv` and `metrics_summary_*.csv`
# - `results/predictions_fold*.npz` — raw scores and labels for every head, so any
#   metric or curve can be recomputed without retraining
# - `results/embeddings.parquet` — the gene × tissue embedding `z` plus `z_seq`
#   and `z_str` for the last fold's test graphs, keyed by gene and tissue. This is
#   head H4 from the slide: the embedding as the deliverable.
# - `results/model_fold*.pt` — weights plus the config used to build them
# - `results/config.json`, `results/run_manifest.json` — the exact settings, data
#   inventory and package versions of this run
# - `results/figures/*.png`


# %%
import platform as _pf

for k, pred in enumerate(fold_preds):
    np.savez_compressed(
        RESULTS_DIR / f"predictions_fold{k}.npz",
        **{n: pred[n] for n in ("link_p", "link_y", "expr_p", "expr_y", "expr_bin_p",
                                "expr_bin_y", "tau_p", "tau_y", "tis_p", "tis_y", "row_idx")})
print(f"predictions      -> results/predictions_fold0..{len(fold_preds)-1}.npz")

n_emb = min(len(meta_te), len(last_pred["z"]))
parts = [pd.DataFrame({
    "gene":      meta_te["gene"].values[:n_emb],
    "tissue":    meta_te["tissue"].values[:n_emb],
    "sample_id": meta_te["sample_id"].values[:n_emb],
})]
for name, arr in [("z", last_pred["z"]), ("z_seq", last_pred["z_seq"]),
                  ("z_str", last_pred["z_str"]), ("z_epi", last_pred["z_epi"])]:
    a = np.asarray(arr)
    if a.size == 0:
        continue
    a = a[:n_emb]
    parts.append(pd.DataFrame(a, columns=[f"{name}_{j}" for j in range(a.shape[1])]))
emb = pd.concat(parts, axis=1)
emb.to_parquet(RESULTS_DIR / "embeddings.parquet", index=False)
print(f"embeddings       -> results/embeddings.parquet  {emb.shape}")

for k, m in enumerate(fold_models):
    # train_genes makes this file usable as a leak-checked external encoder
    # (encoder_source="checkpoint") in a later run
    _st = {"state_dict": m.state_dict(), "config": asdict(CFG),
           "n_node_feat": N_NODE_FEAT, "n_edge_feat": N_EDGE_FEAT,
           "tissues": TISSUE_LIST, "train_genes": sorted(set(ROW_GENES[FOLDS[k][0]]))}
    atomic_save(lambda fh: torch.save(_st, fh), RESULTS_DIR / f"model_fold{k}.pt")
print(f"models           -> results/model_fold0..{len(fold_models)-1}.pt")

(RESULTS_DIR / "config.json").write_text(json.dumps(asdict(CFG), indent=2, default=str))

manifest = {
    "run_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python": _pf.python_version(), "platform": _pf.platform(),
    "device": str(DEVICE),
    "packages": {**HARD, **{k: str(v) for k, v in OPTIONAL.items()}},
    "data": {
        "tissues_used": int(META["tissue"].nunique()),
        "tissues_available": int(len(tissue_df)),
        "genes": int(META["gene"].nunique()),
        "graphs": int(len(GRAPHS)),
        "node_features": int(N_NODE_FEAT),
        "edge_features": int(N_EDGE_FEAT),
        "histone_tracks_present": int(len(hist_files)),
        "histone_coverage_frac": float(avail_df.values.mean()),
    },
    "cv": {"folds": CFG.n_folds, "grouped_by": CFG.group_split_by},
    "headline": {
        "link_auprc":  float(MET["link_auprc"].mean()),
        "expr_pearson_r":  float(MET["expr_pearson_r"].mean()),
        "tau_auprc":   float(MET["tau_auprc"].mean()),
        "tissue_acc":  float(MET["tissue_accuracy"].mean()),
    },
}
(RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
print("manifest         -> results/run_manifest.json")
print(json.dumps(manifest["headline"], indent=2))

print(f"\nfigures ({len(list(FIG_DIR.glob('*.png')))}):")
for p in sorted(FIG_DIR.glob("*.png")):
    print(f"  {p.name}")
TRAIN_HIST.to_csv(RESULTS_DIR / "training_history.csv", index=False)
print("\nexport complete")


# %% [markdown]
# ## 26 · What this run does and does not establish
#
# **What it does.** Every number above comes from graphs whose genes were never
# seen in training, with promoter sequence, histone state, PCHi-C contact
# topology and four supervised objectives learned jointly. Graph edges carry no
# `.hic` (Juicer) contact information — this run isolates what PCHi-C loop calls
# alone, plus sequence and histone state, can do. The metrics, curves and
# embeddings are reproducible from `results/` without retraining.
#
# **Where to be sceptical.**
#
# 1. **Histone coverage is the binding constraint.** Whatever fraction cell 8
#    reported as available is the fraction of the node-feature block that carries
#    real information; the rest is masked. If coverage is low, the structure tower
#    is running mostly on topology, and any claim about chromatin state driving
#    the prediction is unsupported. Run `fetch_histone_encode.py` and check
#    `histone/DOWNLOAD_manifest.csv` — rows marked `proxy` are a *different*
#    biosample and should be treated as noise, not signal.
#
# 2. **The tissue head probably saturates.** Contact profile plus histone state
#    nearly determines the tissue, so H4 is close to a free win and its high
#    accuracy is not evidence that the embedding is good. Treat it as a sanity
#    check, not a result.
#
# 3. **Tau is a per-gene label.** H3 is constant across a gene's tissues, so the
#    model can score well by recognising the gene rather than reasoning about
#    tissue. Gene-grouped folds prevent train/test leakage but do not make the
#    task harder than it is.
#
# 4. **Distance-matched negatives are a floor, not a guarantee.** They remove the
#    easiest shortcut for H1, but promoter–enhancer contacts also correlate with
#    local fragment density and mappability. An ablation against a distance-only
#    logistic baseline is the honest comparison, and it is not in this notebook.
#
# 5. **Ablation arms share their split's encoders.** Every arm on a split uses
#    the same frozen Stage-A towers (trained as all3), and only fusion + heads
#    are arm-specific. The ablation therefore measures what each tower's
#    representation contributes, not how well each tower trains on its own.


# %% [markdown]
# # Part II — Ablation: what is each tower actually contributing?
#
# Everything above reports the primary (`all3`) arm. This section reports the
# other 6 of `PangenomicGAT`'s 7 tower combinations, all trained through
# **identical** folds, heads, losses and metric code by the very same loop
# above (`CFG.variants` controls the full set; nothing here retrains anything):
#
# | arm | `z(g,t)` | what it can see |
# |---|---|---|
# | **all3** | fuse(z_seq, z_str, z_epi) | sequence + contact topology + promoter chromatin state |
# | **str_epi** | fuse(z_str, z_epi) | contact topology + chromatin state. No DNA. |
# | **seq_str** | fuse(z_seq, z_str) | sequence + contact topology. No dedicated chromatin-state tower (Tower B still sees histone/CTCF per-node). |
# | **seq_epi** | fuse(z_seq, z_epi) | sequence + the promoter's own chromatin state. No graph. |
# | **structure only** | z_str | contact topology + node-level histone/CTCF. No DNA, no dedicated epigenomic embedding. |
# | **sequence only** | z_seq | promoter DNA alone. No graph, no tissue information at all — one vector per gene, shared across every tissue. |
# | **epigenomic only** | z_epi | the promoter fragment's own histone/CTCF summary alone. No graph, no DNA. |
#
# Because the arm is a constructor flag on one class, any difference in the
# numbers comes from the towers rather than incidental differences in training
# code. Two things to predict before looking at the table:
#
# 1. **The sequence-only arm cannot do link prediction (H1).** H1 scores *node
#    pairs* from the graph tower's raw states; without Tower B there is
#    nothing to score. Reported as `n/a`, not zero.
# 2. **The sequence-only arm has no tissue information at all.** Its ceiling on
#    H2 and H4 is each gene's *marginal* behaviour across tissues — if it
#    still scores respectably, that is a genuine (if unflattering) result
#    about how tissue-independent the task actually is, not a bug.
#
# The interesting comparison is **all3 vs. the better single/pairwise arm**. If
# full fusion does not beat the best of the rest by more than the fold-to-fold
# spread, the extra tower(s) are costing parameters and buying nothing.


# %%
ABL = pd.DataFrame([m for v in RESULTS for m in RESULTS[v]["metrics"]])
ABL.to_csv(RESULTS_DIR / "ablation_per_fold.csv", index=False)

HEAD_SPECS = [
    ("H1   link prediction",           "link_",     "auprc",      "auroc",   "f1"),
    ("H2a  expression signal (cont.)", "expr_",     "pearson_r",  "r2",      "match_acc_1.0"),
    ("H2b  expressed / not (bin.)",    "expr_bin_", "auprc",      "auroc",   "f1"),
    ("H3   tissue-restricted",         "tau_",      "auprc",      "auroc",   "f1"),
    ("H4   tissue classification",     "tissue_",   "auprc_macro_ovr",
                                                     "auroc_macro_ovr", "f1_macro"),
]

def arm_table(prefix, keys):
    rows = []
    for v in CFG.variants:
        sub = ABL[ABL.variant == v]
        r = {"arm": VARIANT_LABEL[v]}
        for k in keys:
            col = f"{prefix}{k}"
            if col in sub.columns and sub[col].notna().any():
                r[k] = f"{sub[col].mean():.3f} +/- {sub[col].std():.3f}"
            else:
                r[k] = "n/a"
        return_rows = r
        rows.append(r)
    return pd.DataFrame(rows)

BIN_REPORT = ["auprc", "auprc_baseline", "auroc", "accuracy", "balanced_accuracy",
              "precision", "recall", "specificity", "f1", "mcc", "kappa",
              "brier", "logloss"]
REG_REPORT = ["n", "mean", "std", "mse", "rmse", "mae", "r2", "pearson_r", "spearman_r"]
MC_REPORT = ["auprc_macro_ovr", "auroc_macro_ovr", "accuracy", "balanced_accuracy",
             "precision_macro", "recall_macro", "f1_macro", "kappa"]

for title, pref, *_ in HEAD_SPECS:
    keys = MC_REPORT if pref == "tissue_" else (REG_REPORT if pref == "expr_" else BIN_REPORT)
    print(f"\n{'='*82}\n{title}   mean +/- sd over {CFG.n_folds} gene-grouped folds\n{'='*82}")
    print(arm_table(pref, keys).to_string(index=False))

# ---------------------------------------------------------------- headline
print(f"\n\n{'='*82}\nHEADLINE: primary metric per head per arm\n{'='*82}")
head_rows = []
for title, pref, main, sec, f1k in HEAD_SPECS:
    row = {"head": title}
    for v in CFG.variants:
        sub = ABL[ABL.variant == v]
        col = f"{pref}{main}"
        row[VARIANT_LABEL[v]] = (f"{sub[col].mean():.3f}"
                                 if col in sub.columns and sub[col].notna().any()
                                 else "n/a")
    if pref == "tissue_":
        row["random baseline"] = f"{1/len(TISSUE_LIST):.3f}"
    elif pref == "expr_":
        row["random baseline"] = "R^2=0 (predicting the mean)"   # regression has no prevalence baseline
    else:
        b = ABL[f"{pref}auprc_baseline"]
        row["random baseline"] = f"{b.mean():.3f}" if b.notna().any() else "-"
    head_rows.append(row)
HEADLINE = pd.DataFrame(head_rows)
print(HEADLINE.to_string(index=False))
HEADLINE.to_csv(RESULTS_DIR / "ablation_headline.csv", index=False)

# ---------------------------------------------------------------- is fusion real?
print(f"\n\n{'='*82}\nDoes fusion beat the better single tower?\n{'='*82}")
if "all3" in RESULTS and len(CFG.variants) > 1:
    verdict = []
    for title, pref, main, *_ in HEAD_SPECS:
        col = f"{pref}{main}"
        if col not in ABL.columns:
            continue
        both = ABL.loc[ABL.variant == "all3", col].dropna()
        singles = {v: ABL.loc[ABL.variant == v, col].dropna()
                   for v in CFG.variants if v != "all3"}
        singles = {k: s for k, s in singles.items() if len(s)}
        if not len(both) or not singles:
            continue
        best_v = max(singles, key=lambda k: singles[k].mean())
        best = singles[best_v]
        delta = both.mean() - best.mean()
        # paired across folds, so a paired t-test is the right (still weak) test
        try:
            t = sstats.ttest_rel(both.values, best.values)
            pval = float(t.pvalue)
        except Exception:
            pval = np.nan
        pooled_sd = np.sqrt((both.std()**2 + best.std()**2) / 2)
        verdict.append({
            "head": title.split()[0],
            "all3": round(both.mean(), 4),
            "best single": f"{VARIANT_LABEL[best_v]} {best.mean():.4f}",
            "delta": round(delta, 4),
            "delta / pooled sd": round(delta / pooled_sd, 2) if pooled_sd > 0 else np.nan,
            "paired p": round(pval, 4) if np.isfinite(pval) else "n/a",
            "beats spread?": "yes" if pooled_sd > 0 and delta > pooled_sd else "no",
        })
    V = pd.DataFrame(verdict)
    print(V.to_string(index=False))
    V.to_csv(RESULTS_DIR / "ablation_fusion_verdict.csv", index=False)
    print("\n  'delta / pooled sd' < 1 means the gain from fusion is smaller than")
    print("  the fold-to-fold noise. With only 5 folds the paired p-value is")
    print("  underpowered -- treat it as a direction, not a decision.")
else:
    print("  (need the 'all3' arm plus at least one single-tower arm)")


# %% [markdown]
# ## Ablation plots
#
# Four views of the same comparison:
#
# 1. **Grouped bars per head** — the primary metric for each arm with fold error
#    bars, and a dashed line at the random baseline. Bars that sit near that line
#    mean the head learned nothing regardless of arm.
# 2. **Per-fold paired lines** — every fold drawn as one line across the three
#    arms. Consistently sloping lines mean a real effect; crossing lines mean the
#    difference is noise.
# 3. **ROC and PR overlays** — all three arms on shared axes for H2, the head
#    where the towers are most directly comparable.
# 4. **Training curves** — how fast each arm converged, which tells you whether a
#    weaker arm is genuinely weaker or just under-trained.


# %%
ARMS = [v for v in CFG.variants]
ARM_COLOR = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(CFG.variants)}

# ---------------------------------------------------------------- 1. bars
fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
for ax, (title, pref, main, _, _) in zip(axes, HEAD_SPECS):
    col = f"{pref}{main}"
    xs, mus, sds, cols = [], [], [], []
    for v in ARMS:
        sub = ABL.loc[ABL.variant == v, col].dropna() if col in ABL.columns else []
        if len(sub):
            xs.append(VARIANT_LABEL[v]); mus.append(sub.mean()); sds.append(sub.std())
            cols.append(ARM_COLOR[v])
    if not xs:
        ax.text(.5, .5, "n/a", ha="center"); ax.axis("off"); continue
    ax.bar(xs, mus, yerr=sds, capsize=4, color=cols, edgecolor="k", linewidth=.6)
    if pref == "tissue_":
        base = 1 / len(TISSUE_LIST)
        ax.axhline(base, ls="--", c="r", lw=1.2, label=f"random {base:.3f}")
    elif pref == "expr_":
        ax.axhline(0.0, ls="--", c="r", lw=1.2, label="r=0 (no relationship)")
    elif pref == "expr_bin_":
        base = ABL[f"{pref}auprc_baseline"].mean()
        ax.axhline(base, ls="--", c="r", lw=1.2, label=f"random {base:.3f}")
    else:
        base = ABL[f"{pref}auprc_baseline"].mean()
        ax.axhline(base, ls="--", c="r", lw=1.2, label=f"random {base:.3f}")
    for i, (m, s) in enumerate(zip(mus, sds)):
        ax.text(i, m + s + .015, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_title(f"{title}\n({main})", fontsize=9)
    ax.set_ylim(0, min(1.05, max(mus) + max(sds) + .12))
    ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=20, labelsize=8)
fig.suptitle(f"Ablation: {list(CFG.variants)}", y=1.04, fontsize=13)
fig.tight_layout(); savefig(fig, "14_ablation_bars"); plt.show()

# ---------------------------------------------------------------- 2. paired folds
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (title, pref, main, _, _) in zip(axes, HEAD_SPECS):
    col = f"{pref}{main}"
    if col not in ABL.columns:
        ax.axis("off"); continue
    for k in range(CFG.n_folds):
        ys, xs = [], []
        for i, v in enumerate(ARMS):
            r = ABL[(ABL.variant == v) & (ABL.fold == k)][col]
            if len(r) and pd.notna(r.iloc[0]):
                xs.append(i); ys.append(float(r.iloc[0]))
        if ys:
            ax.plot(xs, ys, "o-", ms=4, lw=1.2, alpha=.8,
                    color=PALETTE[k % len(PALETTE)], label=f"fold {k}")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([VARIANT_LABEL[v] for v in ARMS], rotation=20, fontsize=8)
    ax.set_title(f"{title.split()[0]}  {main}", fontsize=9)
    ax.legend(fontsize=6)
fig.suptitle("Per-fold paired comparison (crossing lines = the gap is noise)",
             y=1.04, fontsize=12)
fig.tight_layout(); savefig(fig, "15_ablation_paired_folds"); plt.show()

# ---------------------------------------------------------------- 3. ROC / PR overlay
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, key, name in [(axes[0], "tau", "H3 restricted"), (axes[1], "link", "H1 link"),
                      (axes[2], "expr_bin", "H2b expressed")]:
    for v in ARMS:
        ys, ps = [], []
        for pred in RESULTS[v]["preds"]:
            y, p_ = pred[f"{key}_y"], pred[f"{key}_p"]
            if len(y) == 0:
                continue
            m = y >= 0
            ys.append(y[m]); ps.append(p_[m])
        if not ys:
            continue
        y = np.concatenate(ys); p_ = np.concatenate(ps)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p_)
        ax.plot(fpr, tpr, lw=2, color=ARM_COLOR[v],
                label=f"{VARIANT_LABEL[v]} (AUC {roc_auc_score(y, p_):.3f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax.set_title(f"ROC  {name}"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("ROC by arm, H1/H3/H2b (pooled over folds) -- H2a is regression, no ROC;\n"
            "see the per-arm Pearson r / match-accuracy in ablation_headline.csv instead",
             y=1.05, fontsize=11)
fig.tight_layout(); savefig(fig, "16_ablation_roc"); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, key, name in [(axes[0], "tau", "H3 restricted"), (axes[1], "link", "H1 link"),
                      (axes[2], "expr_bin", "H2b expressed")]:
    base = None
    for v in ARMS:
        ys, ps = [], []
        for pred in RESULTS[v]["preds"]:
            y, p_ = pred[f"{key}_y"], pred[f"{key}_p"]
            if len(y) == 0:
                continue
            m = y >= 0
            ys.append(y[m]); ps.append(p_[m])
        if not ys:
            continue
        y = np.concatenate(ys); p_ = np.concatenate(ps)
        if len(np.unique(y)) < 2:
            continue
        base = float(y.mean())
        pr, rc, _ = precision_recall_curve(y, p_)
        ax.plot(rc, pr, lw=2, color=ARM_COLOR[v],
                label=f"{VARIANT_LABEL[v]} (AP {average_precision_score(y, p_):.3f})")
    if base is not None:
        ax.axhline(base, ls="--", c="r", lw=1.2, label=f"baseline {base:.3f}")
    ax.set_title(f"PR  {name}"); ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.legend(fontsize=8, loc="lower left")
fig.suptitle("Precision-recall by arm, H1/H3/H2b (read against the red baseline) --\n"
            "H2a is regression, no PR curve", y=1.05, fontsize=11)
fig.tight_layout(); savefig(fig, "17_ablation_pr"); plt.show()

# ---------------------------------------------------------------- 4. convergence
ALLHIST = pd.concat([RESULTS[v]["hist"] for v in ARMS], ignore_index=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, comp in zip(axes, ["loss", "expr", "expr_bin"]):
    for v in ARMS:
        h = ALLHIST[ALLHIST.variant == v]
        piv = h.pivot_table(index="epoch", columns="fold", values=f"val_{comp}")
        ax.plot(piv.index, piv.mean(1), lw=2, color=ARM_COLOR[v],
                label=VARIANT_LABEL[v])
        ax.fill_between(piv.index, piv.min(1), piv.max(1),
                        color=ARM_COLOR[v], alpha=.13)
    ax.set_title(f"validation {comp} loss"); ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
fig.suptitle("Convergence by arm (a weak arm that is still falling is under-trained,"
             " not weaker)", y=1.04, fontsize=11)
fig.tight_layout(); savefig(fig, "18_ablation_convergence"); plt.show()

ALLHIST.to_csv(RESULTS_DIR / "ablation_training_history.csv", index=False)
print("\nablation tables -> results/ablation_*.csv")


# %% [markdown]
# ## Deep Hybrid Learning: classical heads on the learned embeddings
#
# Following [SFGLab/DHL](https://github.com/SFGLab/DHL), which pairs a deep
# sequence model (DNABERT there, our Hyena tower here) with classical learners —
# SVM, random forest, k-NN — rather than relying on the network's own softmax.
#
# **Why this is not redundant with the neural head.** The deep towers are trained
# to minimise a *multi-task* objective: their embedding is shaped by four heads
# and a contrastive term at once. A classifier fitted afterwards on the frozen
# embedding optimises one task with a different inductive bias — margins for SVM,
# axis-aligned partitions for trees. On a few thousand examples with a 128-d
# embedding, a gradient-boosted tree ensemble frequently beats a linear head that
# had to share its trunk with three other objectives.
#
# **Protocol, and the part that is easy to get wrong.** For each fold we freeze
# the model *trained on that fold's training set*, embed both splits, fit the
# classical models on train embeddings only, and score the test split. The
# embeddings for the test fold therefore come from a model that never saw them.
# Fitting the scaler or the classifier on all embeddings at once would leak, and
# would show up as a suspiciously large hybrid gain.
#
# We report:
#
# - the **deep head** (the network's own logit),
# - **logistic regression**, **linear SVM**, **RBF SVM**, **random forest**,
#   **k-NN**, **histogram gradient boosting** on the frozen embedding,
# - a **soft-voting ensemble** of deep + the best classical models.
#
# Set `CFG.run_hybrid = False` to skip. Kernel SVR and k-NN scale
# super-linearly, so every classical model is fitted on a seeded subsample of at
# most `HYBRID_MAX_TRAIN` labelled training rows (the deep head used them all;
# the table is therefore conservative for the classical models).


# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

HYBRID_TASK = "expr"          # H2: now a continuous-regression task, not classification
HYBRID_LABEL = {"expr": "H2 expression signal (log2 TPM+1)", "tau": "H3 tissue-restricted"}


HYBRID_MAX_TRAIN = 50_000


def embed_split(model, graphs, idx, seq_bank, device, bs, split_key=None):
    # From the split's cached frozen features when they exist (no tower
    # recomputation); otherwise a full forward pass.
    if split_key in ENCODERS:
        Fd = _features_on_device(ENCODERS[split_key]["feats"], device)
        _, pr = heads_epoch(model, Fd, torch.as_tensor(idx, device=device), CFG, collect=True)
        return pr["z"], pr["expr_y"], pr["tau_y"], pr["tis_y"]
    model.eval()
    Z, Y_expr, Y_tau, Y_tis = [], [], [], []
    with torch.no_grad(), amp_ctx(CFG, device):
        for batch in make_loader(graphs, idx, bs, False):
            batch = batch.to(device)
            _, _, _, z, _, _ = model.encode(batch, seq_bank)
            Z.append(z.float().cpu().numpy())
            Y_expr.append(batch.y_expr.cpu().numpy())
            Y_tau.append(batch.y_tau.cpu().numpy())
            Y_tis.append(batch.y_tissue.cpu().numpy())
    return (np.concatenate(Z), np.concatenate(Y_expr),
            np.concatenate(Y_tau), np.concatenate(Y_tis))


def classical_zoo(seed):
    # Regressors, not classifiers -- H2 is continuous now. Chosen to mirror
    # the previous classifier zoo's diversity of inductive bias (linear,
    # margin-based, tree-ensemble, instance-based, gradient-boosted).
    return {
        "linear (ridge)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "linear SVR":     make_pipeline(StandardScaler(), LinearSVR(C=0.5, max_iter=5000)),
        "RBF SVR":        make_pipeline(StandardScaler(), SVR(C=2.0, gamma="scale")),
        "random forest":  RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                n_jobs=-1, random_state=seed),
        "kNN (k=25)":     make_pipeline(StandardScaler(),
                                        KNeighborsRegressor(n_neighbors=25, weights="distance")),
        "hist gradboost": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                        random_state=seed),
    }


HYBRID = []
if CFG.run_hybrid and PRIMARY in RESULTS:
    print(f"DHL-style hybrid on the '{PRIMARY}' embedding, task = "
          f"{HYBRID_LABEL[HYBRID_TASK]}\n")
    for k, (tr, te) in enumerate(FOLDS):
        model = RESULTS[PRIMARY]["models"][k]
        Ztr, ytr_e, ytr_t, _ = embed_split(model, GRAPHS, tr, SEQ_BANK,
                                           DEVICE, CFG.batch_size, split_key=f"gene_fold{k}")
        Zte, yte_e, yte_t, _ = embed_split(model, GRAPHS, te, SEQ_BANK,
                                           DEVICE, CFG.batch_size, split_key=f"gene_fold{k}")
        ytr = ytr_e if HYBRID_TASK == "expr" else ytr_t
        yte = yte_e if HYBRID_TASK == "expr" else yte_t
        mtr, mte = ytr >= 0, yte >= 0
        Ztr, ytr = Ztr[mtr], ytr[mtr]
        Zte, yte = Zte[mte], yte[mte]
        if len(ytr) > HYBRID_MAX_TRAIN:
            _sub = np.random.default_rng(run_seed(CFG.seed, "hybrid", k)).choice(
                len(ytr), HYBRID_MAX_TRAIN, replace=False)
            Ztr, ytr = Ztr[_sub], ytr[_sub]
        if len(ytr) < 20 or len(yte) < 20:
            print(f"  fold {k}: too few labelled rows after masking, skipped")
            continue
        print(f"  fold {k}: train {Ztr.shape}  test {Zte.shape}  "
              f"mean signal {ytr.mean():.3f}/{yte.mean():.3f}")

        # deep head, restricted to the same labelled rows for a fair comparison
        deep_p = RESULTS[PRIMARY]["preds"][k][f"{HYBRID_TASK}_p"]
        deep_y = RESULTS[PRIMARY]["preds"][k][f"{HYBRID_TASK}_y"]
        dm = deep_y >= 0
        m = regression_metrics(deep_y[dm], deep_p[dm], "")
        m.update({"fold": k, "model": "deep head (neural)"})
        HYBRID.append(m)

        preds_by_model = {}
        for name, reg in classical_zoo(CFG.seed).items():
            t0 = time.time()
            try:
                reg.fit(Ztr, ytr)
                p_ = reg.predict(Zte)
            except (ValueError, MemoryError) as e:
                print(f"    {name:<16} failed: {type(e).__name__}: {e}")
                continue
            preds_by_model[name] = p_
            mm_ = regression_metrics(yte, p_, "")
            mm_.update({"fold": k, "model": name, "fit_sec": round(time.time()-t0, 1)})
            HYBRID.append(mm_)
            print(f"    {name:<16} Pearson r {mm_['pearson_r']:.3f}  R^2 {mm_['r2']:.3f}"
                  f"  ({time.time()-t0:.1f}s)")

        # simple averaging ensemble: deep head + the classical models, equal weight
        if preds_by_model and dm.sum() == len(yte):
            ens = np.mean([deep_p[dm]] + list(preds_by_model.values()), axis=0)
            mm_ = regression_metrics(yte, ens, "")
            mm_.update({"fold": k, "model": "ensemble (deep + classical)"})
            HYBRID.append(mm_)
            print(f"    {'ensemble':<16} Pearson r {mm_['pearson_r']:.3f}  "
                  f"R^2 {mm_['r2']:.3f}")
        elif preds_by_model:
            ens = np.mean(list(preds_by_model.values()), axis=0)
            mm_ = regression_metrics(yte, ens, "")
            mm_.update({"fold": k, "model": "ensemble (classical only)"})
            HYBRID.append(mm_)

    HYB = pd.DataFrame(HYBRID)
    HYB.to_csv(RESULTS_DIR / "hybrid_per_fold.csv", index=False)
else:
    HYB = pd.DataFrame()
    print("CFG.run_hybrid is False (or no primary arm) - skipping the hybrid section")


# %% [markdown]
# ## Hybrid results
#
# The table is ordered by mean Pearson r (H2 is regression now). Read it with two questions in mind:
#
# - **Does any classical head beat the deep head?** If yes, the embedding is
#   better than the classifier that was sitting on top of it, and the multi-task
#   trunk was holding the neural head back. That is DHL's central claim.
# - **Does the ensemble beat everything?** Usually by a little. If it beats the
#   best single model by less than the fold spread, it is not worth the
#   complexity of shipping six models.
#
# A caveat the DHL paper does not dwell on: classical heads fitted on a frozen
# embedding cannot fix a bad embedding. If every row of the table is close
# together, the ceiling is the representation, not the classifier — and the fix is
# upstream, in the towers or the features, not here.


# %%
if len(HYB):
    order_cols = ["pearson_r", "r2", "mae", "rmse", "spearman_r"]
    summ = (HYB.groupby("model")[order_cols]
              .agg(["mean", "std"]).round(4).sort_values(("pearson_r", "mean"),
                                                         ascending=False))
    print(f"{'='*88}\nDHL hybrid on the {PRIMARY} embedding - "
          f"{HYBRID_LABEL[HYBRID_TASK]}\n{'='*88}")
    print(summ.to_string())
    summ.to_csv(RESULTS_DIR / "hybrid_summary.csv")

    best = summ.index[0]
    deep = summ.loc["deep head (neural)", ("pearson_r", "mean")] \
        if "deep head (neural)" in summ.index else np.nan
    print(f"\n  best model     : {best}  "
          f"Pearson r {summ.loc[best, ('pearson_r','mean')]:.4f}")
    if np.isfinite(deep):
        d = summ.loc[best, ("pearson_r", "mean")] - deep
        sd = summ.loc[best, ("pearson_r", "std")]
        print(f"  deep head      : Pearson r {deep:.4f}")
        print(f"  delta          : {d:+.4f}  "
              f"({'within' if abs(d) < sd else 'beyond'} the fold spread of {sd:.4f})")
        if abs(d) < sd:
            print("  -> the hybrid gain is inside fold noise. The embedding, not the")
            print("     regressor, is the ceiling here.")

    # ---------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    mods = list(summ.index)
    mu = [summ.loc[m, ("pearson_r", "mean")] for m in mods]
    sd = [summ.loc[m, ("pearson_r", "std")] for m in mods]
    cols = ["#D14A73" if m.startswith("deep") else
            "#00A38C" if m.startswith("ensemble") else "#5B85D6" for m in mods]
    axes[0].barh(mods, mu, xerr=sd, color=cols, edgecolor="k", linewidth=.6)
    axes[0].axvline(0.0, ls="--", c="r", lw=1.2, label="r=0 (no relationship)")
    axes[0].set_xlabel("Pearson r"); axes[0].set_title("Hybrid heads, mean +/- sd")
    axes[0].legend(fontsize=7); axes[0].invert_yaxis()

    for m in mods:
        sub = HYB[HYB.model == m].sort_values("fold")
        axes[1].plot(sub["fold"], sub["pearson_r"], "o-", ms=4, lw=1.2, label=m, alpha=.85)
    axes[1].set_xlabel("fold"); axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Per-fold stability"); axes[1].legend(fontsize=6)

    piv = HYB.pivot_table(index="model", values=["pearson_r", "r2", "mae", "rmse"],
                          aggfunc="mean")
    sns.heatmap(piv.loc[mods], annot=True, fmt=".3f", cmap="YlGnBu",
                ax=axes[2], cbar_kws={"label": "score"})
    axes[2].set_title("Metric grid"); axes[2].set_ylabel("")
    fig.suptitle(f"Deep Hybrid Learning: classical regressors on the frozen "
                 f"{PRIMARY} embedding", y=1.04, fontsize=12)
    fig.tight_layout(); savefig(fig, "19_hybrid_heads"); plt.show()
else:
    print("no hybrid results to report")


# %% [markdown]
# ## What changed, and what to check first
#
# **The sequence tower is now a Hyena stack, not a dilated CNN.**
#
# | | before | now |
# |---|---|---|
# | operator | dilated Conv1d x 4 | HyenaOperator (implicit long conv, FFT) |
# | promoter window | 2 kb | 1 Mb (stride-64 stem, ~15.6k positions) |
# | receptive field | ~500 bp after 4 layers | full window from block 1 |
# | filter parameters | grow with kernel size | independent of length (generated by an MLP) |
# | cost in length | O(L k) | O(L log L) |
# | strand | arbitrary convention | reverse-complement equivariant |
# | norm / MLP | BatchNorm + GELU | RMSNorm + SwiGLU |
#
# The single most important line is the receptive field. The old stack physically
# could not see a signal 3 kb from the TSS; every Hyena block can see the whole
# window. That is the change that makes "long receptive field" true rather than
# aspirational.
#
# **Three things to verify on your first real run:**
#
# 1. **Does the ablation show the sequence tower doing anything?** If `GAT only`
#    and `both` are within a fold standard deviation, the Hyena tower is
#    expensive decoration. Check `ablation_fusion_verdict.csv`.
#
# 2. **Is the extra window used?** Re-run with a short `promoter_window` and
#    compare. If 1 Mb does not beat it, the signal is proximal and the long
#    context is not earning its FFT cost. This run does not test that.
#
# 3. **Did the RC-equivariance assertion pass?** It runs in the model cell. If it
#    ever fails, the two strands are being embedded differently and the features
#    depend on an arbitrary annotation choice.
#
# **Cost.** FFT convolution is O(L log L) but the constant is not small, and
# reverse-complement equivariance doubles the forward pass. At 16 kb with 4 blocks
# expect the sequence tower to dominate step time. `stem_stride` (default 4) is
# the dial: stride 1 is true single-nucleotide resolution and roughly 4x the cost.
#
# **Things I did not do, and would want before publishing:**
#
# - No pretrained HyenaDNA checkpoint. The real HyenaDNA is *pretrained* on the
#   human genome and then fine-tuned; ours is trained from scratch on a few
#   thousand promoters. That is the largest single gap between this and the paper,
#   and loading `hyenadna-small-32k-seqlen` from HuggingFace would likely matter
#   more than any architectural detail here.
# - No diffusion component. HiCDiffusionLooping generates contact maps from
#   sequence; that is a different task from ours (we consume contacts rather than
#   predicting them), so borrowing its decoder would have meant changing what the
#   model is for.
# - The DHL comparison uses our own embedding, not DNABERT's. A fair
#   reproduction of DHL would swap the encoder too.


# %% [markdown]
# # Part III — Alternative validation strategies
#
# Everything above validates by **gene**: `GroupKFold` on gene symbol, so no
# gene's promoter sequence or per-gene labels ever cross the train/test
# boundary. That answers "does this generalise to unseen genes". It does not
# answer two other, equally real, generalisation questions:
#
# 1. **Unseen chromosomes.** A gene-grouped fold can still put two genes that
#    are neighbours on the same chromosome — and therefore share regional
#    chromatin state, replication timing, TAD membership — on opposite sides
#    of the split. **Chromosome-block CV** removes that: every fold's test
#    genes come from chromosomes entirely absent from that fold's training
#    data.
# 2. **Unseen tissues.** Gene-grouped CV always trains on every tissue in
#    `TISSUE_LIST`; it never asks whether the model has learned something
#    tissue-transferable or has just memorised per-tissue quirks.
#    **Leave-one-tissue-out (LOTO)** holds out one tissue's graphs entirely
#    and trains on the rest (Stage A included, so the held-out tissue never
#    reaches an encoder). `CFG.loto_n_tissues` (default 5) picks the tissues
#    with the most complete histone coverage, then the most graphs; 0 = all 46.
#
# Both schemes train the **`all3`** arm through the same `train_one_fold`
# (each split gets its own Stage-A encoders), reusing `GRAPHS` / `META` and
# `all_metrics` — no graphs are rebuilt, no code is duplicated. Results are written to
# `RESULTS_DIR / "validation_strategies"`, separate from the gene-grouped-CV
# results saved earlier, so nothing already on disk is overwritten.
#
# **The same H3/H4 caveats from the base validations notebook apply:** H3
# (tau) is a per-gene label, so under a tissue-holdout split (LOTO) the same
# gene can appear on both sides via its non-held-out tissues — read LOTO's H3
# numbers as an upper bound, not a clean tissue-generalisation test. H4
# (tissue classification) is not meaningful under LOTO: the held-out tissue's
# class was never a training target.


# %%
VSTRAT_DIR = RESULTS_DIR / "validation_strategies"
VSTRAT_DIR.mkdir(exist_ok=True)
(VSTRAT_DIR / "chrom_cv").mkdir(exist_ok=True)
(VSTRAT_DIR / "leave_one_tissue_out").mkdir(exist_ok=True)
(VSTRAT_DIR / "blood_holdout").mkdir(exist_ok=True)

def _save_scheme_fold(subdir, tag, pred, met, hist, model, train_idx):
    out = VSTRAT_DIR / subdir
    np.savez_compressed(out / f"predictions_{tag}.npz",
                        **{n: pred[n] for n in ("link_p", "link_y", "expr_p", "expr_y", "expr_bin_p",
                                                "expr_bin_y", "tau_p", "tau_y", "tis_p", "tis_y",
                                                "row_idx")})
    hist.to_csv(out / f"history_{tag}.csv", index=False)
    _st = {"state_dict": model.state_dict(), "config": asdict(CFG),
           "n_node_feat": N_NODE_FEAT, "n_edge_feat": N_EDGE_FEAT,
           "tissues": TISSUE_LIST, "train_genes": sorted(set(ROW_GENES[train_idx]))}
    atomic_save(lambda fh: torch.save(_st, fh), out / f"model_{tag}.pt")
    return met

print(f"validation-strategy outputs -> {VSTRAT_DIR.relative_to(PROJECT_ROOT)}")


# %% [markdown]
# ## III.1 · Chromosome-block cross-validation
#
# Chromosomes are greedily binned into `CFG.n_folds` groups so total row
# count per fold stays balanced (the same greedy strategy `GroupKFold` uses
# internally, applied explicitly here so every fold really is a *block of
# whole chromosomes*, not a `GroupKFold` mix that happens to correlate with
# chromosome). Every fold's test genes come from chromosomes that contributed
# **zero** training rows.


# %%
def make_chrom_blocks(meta, n_blocks):
    counts = meta["chrom"].value_counts()          # descending by row count
    bins = [[] for _ in range(n_blocks)]
    bin_sizes = [0] * n_blocks
    for chrom, n in counts.items():                # largest chromosome first
        j = int(np.argmin(bin_sizes))               # always fill the lightest bin
        bins[j].append(chrom); bin_sizes[j] += n
    return bins

CHROM_BLOCKS = make_chrom_blocks(META, CFG.n_folds)
for k, block in enumerate(CHROM_BLOCKS):
    print(f"  block {k}: {len(block)} chromosome(s) -> {sorted(block)}")

CHROM_FOLDS = []
for block in CHROM_BLOCKS:
    test_mask = META["chrom"].isin(block).values
    te = np.where(test_mask)[0]; tr = np.where(~test_mask)[0]
    CHROM_FOLDS.append((tr, te))

chrom_rows = []
for k, (tr, te) in enumerate(CHROM_FOLDS):
    mtr, mte = META.iloc[tr], META.iloc[te]
    chrom_leak = len(set(mtr["chrom"]) & set(mte["chrom"]))
    chrom_rows.append({"fold": k, "n_train": len(tr), "n_test": len(te),
                       "chroms_test": mte["chrom"].nunique(), "chrom_leak": chrom_leak})
chrom_split_df = pd.DataFrame(chrom_rows)
print(chrom_split_df.to_string(index=False))
assert chrom_split_df["chrom_leak"].sum() == 0, "a chromosome appears in both train and test"
print("\n  OK: no chromosome appears in both train and test of any fold")


# %%
CHROM_CV_METRICS = []
CHROM_CV_PREDS = []     # kept in memory (row_idx included) for Part V's stratified cross-tabulation
t0 = time.time()
for k, (tr, te) in enumerate(CHROM_FOLDS):
    print(f"\n{'='*74}\nchrom-block CV | FOLD {k+1}/{len(CHROM_FOLDS)}   "
          f"train={len(tr):,}  test={len(te):,}\n{'='*74}")
    model, pred, hist, best_ep = train_one_fold(
        k, tr, te, GRAPHS, SEQ_BANK, CFG, DEVICE, variant="all3",
        run_key=f"chromcv_all3_fold{k}", split_key=f"chrom_fold{k}")
    met = all_metrics(pred, len(TISSUE_LIST))
    met["fold"] = k; met["best_epoch"] = best_ep
    met["n_chroms_test"] = META.iloc[te]["chrom"].nunique()
    _save_scheme_fold("chrom_cv", f"fold{k}", pred, met, hist, model, tr)
    CHROM_CV_METRICS.append(met)
    CHROM_CV_PREDS.append(pred)
    print(f"  fold {k}:  link AUPRC {met.get('link_auprc', np.nan):.3f} | "
          f"expr Pearson r {met.get('expr_pearson_r', np.nan):.3f} | "
          f"tau AUPRC {met.get('tau_auprc', np.nan):.3f} | "
          f"tissue acc {met.get('tissue_accuracy', np.nan):.3f}")

CHROM_CV_DF = pd.DataFrame(CHROM_CV_METRICS)
CHROM_CV_DF.to_csv(VSTRAT_DIR / "chrom_cv" / "metrics_per_fold.csv", index=False)
print(f"\nchrom-block CV done in {(time.time()-t0)/60:.1f} min -> "
      f"validation_strategies/chrom_cv/metrics_per_fold.csv")


# %% [markdown]
# ## III.2 · Leave-one-tissue-out cross-validation
#
# For every tissue in `LOTO_TISSUES`, train on every *other* tissue's graphs
# and test on that tissue alone. Each held-out tissue costs one Stage A, so
# `CFG.loto_n_tissues` (default 5) bounds the compute; 0 runs all of them.


# %%
# most complete histone coverage first, then most graphs
_sid_of = META.groupby("tissue")["sample_id"].first()
_cov = {t: sum(HIST.available.get(_sid_of[t], {}).values()) for t in TISSUE_LIST}
_ranked = sorted(TISSUE_LIST, key=lambda t: (-_cov[t], -(META["tissue"] == t).sum(), t))
LOTO_TISSUES = _ranked if CFG.loto_n_tissues == 0 else _ranked[:CFG.loto_n_tissues]
print(f"leave-one-tissue-out: {len(LOTO_TISSUES)} tissue(s) -> "
      f"{len(LOTO_TISSUES)} training run(s)")

LOTO_METRICS = []
LOTO_PREDS = []     # kept in memory (row_idx included) for Part V's stratified cross-tabulation
t0 = time.time()
for i, held_out in enumerate(LOTO_TISSUES):
    te = np.where(META["tissue"].values == held_out)[0]
    tr = np.where(META["tissue"].values != held_out)[0]
    if len(te) == 0 or len(tr) == 0:
        print(f"  [{i+1}/{len(LOTO_TISSUES)}] {held_out}: skipped (no graphs on one side)")
        continue
    print(f"\n{'='*74}\nLOTO | {held_out}  ({i+1}/{len(LOTO_TISSUES)})   "
          f"train={len(tr):,}  test={len(te):,}\n{'='*74}")
    tag = "".join(c if c.isalnum() else "_" for c in held_out)
    model, pred, hist, best_ep = train_one_fold(
        i, tr, te, GRAPHS, SEQ_BANK, CFG, DEVICE, variant="all3", quiet=True,
        run_key=f"loto_all3_{tag}", split_key=f"loto_{tag}")
    met = all_metrics(pred, len(TISSUE_LIST))
    met["held_out_tissue"] = held_out; met["best_epoch"] = best_ep
    _save_scheme_fold("leave_one_tissue_out", tag, pred, met, hist, model, tr)
    LOTO_METRICS.append(met)
    LOTO_PREDS.append(pred)
    print(f"  {held_out:<32} link AUPRC {met.get('link_auprc', np.nan):.3f} | "
          f"expr Pearson r {met.get('expr_pearson_r', np.nan):.3f} | "
          f"tau AUPRC {met.get('tau_auprc', np.nan):.3f}   "
          f"(H4 not meaningful for the held-out class)")

LOTO_DF = pd.DataFrame(LOTO_METRICS)
LOTO_DF.to_csv(VSTRAT_DIR / "leave_one_tissue_out" / "metrics_per_tissue.csv", index=False)
print(f"\nLOTO done in {(time.time()-t0)/60:.1f} min -> "
      f"validation_strategies/leave_one_tissue_out/metrics_per_tissue.csv")


# %% [markdown]
# ## III.3 · Comparing every validation strategy
#
# Same metric code, same trimodal model, three different ideas about what
# "generalises" means. The gap between gene-grouped CV and the others is the
# honest answer to "how much of the headline number was generalisation
# across genes specifically, versus generalisation the model does not
# actually have."


# %%
gene_cv_metrics = pd.DataFrame(RESULTS["all3"]["metrics"]) if "RESULTS" in globals() else None

def _mean_std(df, col):
    if df is None or col not in df or df[col].dropna().empty:
        return np.nan, np.nan
    return float(df[col].mean()), float(df[col].std())

_cmp_rows = []
for scheme, df in [
    ("gene-grouped CV (main, all3)", gene_cv_metrics),
    ("chromosome-block CV",           CHROM_CV_DF if len(CHROM_CV_DF) else None),
    ("leave-one-tissue-out",          LOTO_DF if len(LOTO_DF) else None),
]:
    row = {"scheme": scheme}
    for metric in ("link_auprc", "expr_pearson_r", "tau_auprc", "tissue_accuracy"):
        m, s = _mean_std(df, metric)
        row[f"{metric}_mean"] = m; row[f"{metric}_std"] = s
    _cmp_rows.append(row)

COMPARE_DF = pd.DataFrame(_cmp_rows)
COMPARE_DF.to_csv(VSTRAT_DIR / "comparison_summary.csv", index=False)
print(COMPARE_DF.to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, metric, title in zip(
        axes,
        ("link_auprc", "expr_pearson_r", "tau_auprc", "tissue_accuracy"),
        ("H1 link AUPRC", "H2 expression Pearson r", "H3 tau AUPRC", "H4 tissue accuracy")):
    means = COMPARE_DF[f"{metric}_mean"].values
    stds  = COMPARE_DF[f"{metric}_std"].values
    x = np.arange(len(COMPARE_DF))
    ax.bar(x, means, yerr=np.nan_to_num(stds), capsize=4, color=PALETTE[:len(x)])
    ax.set_xticks(x); ax.set_xticklabels(COMPARE_DF["scheme"], rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
fig.suptitle("Same trimodal (all3) model and metrics, three validation strategies "
            "(H4 under LOTO is not a trained-class result -- ignore that bar)")
fig.tight_layout(); savefig(fig, "20_validation_strategy_comparison"); plt.show()

print("\nsaved -> validation_strategies/comparison_summary.csv and "
      "figures/20_validation_strategy_comparison.png")


# %% [markdown]
# # Part IV — Stratified evaluation: housekeeping x protein-coding, every validation scheme
#
# Every number reported so far pools every gene together. This section splits
# that pool three ways, all attached per-graph back in the labels cell (11) and
# carried through training and batching via `row_idx`:
#
# - **Housekeeping vs non-housekeeping** — this dataset's own Yanai's tau,
#   split at this run's median tau (`HK_MEDIAN_CUT`, labels cell).
# - **Protein-coding vs non-coding** — the RefSeq accession prefix on each
#   gene's canonical transcript (`NM_`/`XM_` vs `NR_`/`XR_`), free from data
#   already loaded in cell 6.
# - **The full 2x2 combination of the two** — housekeeping+coding,
#   housekeeping+non-coding, non-housekeeping+coding, non-housekeeping+non-coding
#   — since a gene's coding status and its expression breadth are independent
#   axes and averaging over one can hide an effect that only shows up in a
#   specific corner of the other (e.g. non-coding + tissue-restricted genes,
#   such as many lncRNAs, may behave nothing like non-coding + housekeeping
#   ones).
#
# **Every ablation arm.** `row_idx` (cell 11) is a plain per-graph row number
# that survives PyG's batching unmodified (see cell 15). Every arm's stored
# test predictions in `RESULTS[v]["preds"]` therefore already carry, per
# prediction row, exactly which gene it came from, so the same stratification
# runs once per arm at no extra training cost.
#
# **Every validation scheme, not just the main gene-grouped CV.** Part III's
# chromosome-block CV and leave-one-tissue-out loops now keep their per-fold
# predictions in memory (`CHROM_CV_PREDS`, `LOTO_PREDS`) the same way the main
# loop already did, and row_idx is now also persisted into every saved
# `predictions_*.npz` (main and validation-strategy) so this table is
# reproducible from disk without retraining. Chromosome-CV and LOTO only ever
# train the `all3` arm (Part III's design, unchanged) — the combination table
# below is arm x stratum for gene-grouped CV, and scheme x stratum for `all3`
# specifically across all three schemes.
#
# **Why H1 is not included.** Link prediction scores one edge at a time, not
# one gene; there is no single housekeeping/coding status per prediction row.
#
# **Pooling.** Every row below pools across every fold of its scheme and
# therefore across every tissue those folds' test sets touch.


# %%
def stratum_frame(preds_list, meta):
    """Concatenate one (scheme, arm)'s per-fold prediction dicts into a single
    frame, joined back to this row's gene-level metadata via row_idx. One row
    per evaluated (gene, tissue) graph. tis_p (per-graph softmax over
    tissues) is kept as a separate parallel 2-D array, row-aligned with the
    frame, rather than embedded in it -- a DataFrame cell holding a vector is
    a fragile way to carry a fixed-width array through a mask operation.

    Carries BOTH H2 heads: expr_p/expr_y (continuous, H2a) and
    expr_bin_p/expr_bin_y (binary, H2b) -- previously only H2a made it into
    this frame, so every stratified table downstream reported H2 once, not
    twice. Falls back to all -1 for expr_bin_* if a given preds_list predates
    the dual-head model (keeps this function usable against older cached
    prediction dicts without crashing)."""
    row_idx_all, expr_p, expr_y, expr_bin_p, expr_bin_y = [], [], [], [], []
    tau_p, tau_y, tis_y, tis_p = [], [], [], []
    for pred in preds_list:
        n = len(pred["row_idx"])
        if n == 0:
            continue
        row_idx_all.append(pred["row_idx"].astype(int))
        expr_p.append(pred["expr_p"]); expr_y.append(pred["expr_y"])
        if "expr_bin_p" in pred and len(pred["expr_bin_p"]) == n:
            expr_bin_p.append(pred["expr_bin_p"]); expr_bin_y.append(pred["expr_bin_y"])
        else:
            expr_bin_p.append(np.full(n, np.nan)); expr_bin_y.append(np.full(n, -1.0))
        tau_p.append(pred["tau_p"]);   tau_y.append(pred["tau_y"])
        tis_y.append(pred["tis_y"]);   tis_p.append(pred["tis_p"])
    df = pd.DataFrame({
        "row_idx": np.concatenate(row_idx_all),
        "expr_p": np.concatenate(expr_p), "expr_y": np.concatenate(expr_y),
        "expr_bin_p": np.concatenate(expr_bin_p), "expr_bin_y": np.concatenate(expr_bin_y),
        "tau_p": np.concatenate(tau_p),   "tau_y": np.concatenate(tau_y),
        "tis_y": np.concatenate(tis_y),
    })
    tis_p_all = np.concatenate(tis_p, axis=0)
    gene_cols = meta[["gene", "tissue", "y_housekeeping", "y_coding"]].reset_index(drop=True)
    df = df.join(gene_cols, on="row_idx")
    return df, tis_p_all


CATEGORY_LEVELS = {
    "y_housekeeping": ("housekeeping",   {1: "housekeeping", 0: "non-housekeeping"}),
    "y_coding":       ("protein_coding", {1: "protein-coding", 0: "non-coding"}),
}
COMBO_LEVELS = {
    (1, 1): "housekeeping + protein-coding", (1, 0): "housekeeping + non-coding",
    (0, 1): "non-housekeeping + protein-coding", (0, 0): "non-housekeeping + non-coding",
}


def stratified_report(preds_list, meta, n_classes, scheme, arm):
    df, tis_p_all = stratum_frame(preds_list, meta)

    def score(mask, group, level):
        if mask.sum() < 20:
            return None
        sub = df[mask]
        m_expr = regression_metrics(sub["expr_y"].values, sub["expr_p"].values, "expr_")
        m_bin  = binary_metrics(sub["expr_bin_y"].values, sub["expr_bin_p"].values, "expr_bin_")
        m_tau  = binary_metrics(sub["tau_y"].values,  sub["tau_p"].values,  "tau_")
        m_tis  = multiclass_metrics(sub["tis_y"].values, tis_p_all[mask.values], n_classes)
        return {
            "scheme": scheme, "arm": arm, "group": group, "level": level,
            "n_graphs": int(mask.sum()),
            # H2a -- continuous head_expr
            "H2a_pearson_r": m_expr.get("expr_pearson_r", np.nan),
            "H2a_r2": m_expr.get("expr_r2", np.nan),   # R^2=0 is the natural "no-skill" reference
            "H2a_mae": m_expr.get("expr_mae", np.nan),
            "H2a_match_acc_1.0": m_expr.get("expr_match_acc_1.0", np.nan),
            # H2b -- binary head_expr_bin, same stratum, same rows
            "H2b_auprc": m_bin.get("expr_bin_auprc", np.nan),
            "H2b_auprc_baseline": m_bin.get("expr_bin_auprc_baseline", np.nan),
            "H2b_auroc": m_bin.get("expr_bin_auroc", np.nan),
            "H2b_f1": m_bin.get("expr_bin_f1", np.nan),
            "H3_auprc": m_tau.get("tau_auprc", np.nan),
            "H3_auprc_baseline": m_tau.get("tau_auprc_baseline", np.nan),
            "H4_accuracy": m_tis.get("tissue_accuracy", np.nan),
        }

    rows = []
    # explicit unstratified baseline -- every group/level below is judged
    # against this row, sitting in the SAME table rather than a separate one
    r = score(pd.Series(True, index=df.index), "all_genes", "all genes")
    if r is not None:
        rows.append(r)
    # single-factor marginals
    for col, (group_name, level_names) in CATEGORY_LEVELS.items():
        for level, level_name in level_names.items():
            r = score(df[col] == level, group_name, level_name)
            if r is not None:
                rows.append(r)
    # full 2x2 combination
    for (hk, cod), name in COMBO_LEVELS.items():
        mask = (df["y_housekeeping"] == hk) & (df["y_coding"] == cod)
        r = score(mask, "housekeeping_x_coding", name)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


STRAT_ALL = pd.concat(
    [stratified_report(RESULTS[v]["preds"], META, len(TISSUE_LIST), "gene-grouped CV", v)
     for v in RESULTS]
    + [stratified_report(CHROM_CV_PREDS, META, len(TISSUE_LIST), "chromosome-block CV", "all3")]
    + [stratified_report(LOTO_PREDS, META, len(TISSUE_LIST), "leave-one-tissue-out", "all3")],
    ignore_index=True,
)
STRAT_ALL.to_csv(RESULTS_DIR / "stratified_metrics_all_arms_all_schemes.csv", index=False)
print(f"{'='*110}\nStratified evaluation -- every arm (gene-grouped CV) x every validation "
      f"scheme (all3)\nBoth H2 heads reported per stratum, plus an explicit 'all genes' "
      f"unstratified row\n{'='*110}")
print(STRAT_ALL.round(4).to_string(index=False))
print(f"\nfull table -> results/stratified_metrics_all_arms_all_schemes.csv")
print("\nLOTO's H4 column is not meaningful (structurally ~0: the held-out tissue's")
print("class index was never a training target) -- read H2a/H2b/H3 there, ignore H4.")

# ---------------------------------------------------------------- headline: primary arm, gene-CV, all_genes + the 2x2 combination
HEADLINE = STRAT_ALL[(STRAT_ALL.scheme == "gene-grouped CV") & (STRAT_ALL.arm == PRIMARY)
                     & (STRAT_ALL.group.isin(["all_genes", "housekeeping_x_coding"]))].set_index("level")
print(f"\n{'='*110}\nHEADLINE: {PRIMARY} arm, gene-grouped CV -- all genes, then the full "
      f"housekeeping x coding combination\n{'='*110}")
print(HEADLINE[["n_graphs", "H2a_pearson_r", "H2a_r2", "H2a_mae", "H2b_auprc",
               "H2b_auprc_baseline", "H3_auprc", "H3_auprc_baseline", "H4_accuracy"]]
      .round(4).to_string())

fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
for ax, metric, base_col, title in [
        (axes[0], "H2a_pearson_r", None, "H2a expression signal (Pearson r)"),
        (axes[1], "H2b_auprc", "H2b_auprc_baseline", "H2b expressed/not (AUPRC)"),
        (axes[2], "H3_auprc", "H3_auprc_baseline", "H3 tissue-restricted (tau)"),
        (axes[3], "H4_accuracy", None, "H4 tissue classification")]:
    sub = HEADLINE.reset_index()
    ax.bar(sub["level"], sub[metric], color=PALETTE[:len(sub)])
    if base_col:
        for i, b in enumerate(sub[base_col]):
            ax.plot([i-.4, i+.4], [b, b], "r--", lw=1.4)
    ax.set_title(f"{title}\n({PRIMARY}, gene-grouped CV)", fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=20)
fig.suptitle("All genes vs. the full housekeeping x protein-coding combination -- both H2 heads",
            y=1.03)
fig.tight_layout(); savefig(fig, "21_stratified_combination"); plt.show()

# ---------------------------------------------------------------- how H2 (both heads) moves across validation schemes, per stratum
print(f"\n{'='*110}\nH2a Pearson r by stratum, across validation schemes ({PRIMARY} arm)\n{'='*110}")
piv_a = (STRAT_ALL[(STRAT_ALL.arm == PRIMARY) & (STRAT_ALL.group == "housekeeping_x_coding")]
        .pivot_table(index="level", columns="scheme", values="H2a_pearson_r"))
print(piv_a.round(4).to_string())

print(f"\n{'='*110}\nH2b AUPRC by stratum, across validation schemes ({PRIMARY} arm)\n{'='*110}")
piv_b = (STRAT_ALL[(STRAT_ALL.arm == PRIMARY) & (STRAT_ALL.group == "housekeeping_x_coding")]
        .pivot_table(index="level", columns="scheme", values="H2b_auprc"))
print(piv_b.round(4).to_string())

print("\nread each cell against the marginal rows in the full CSV for the group's own")
print("baseline -- a combination cell with few n_graphs (see the CSV) is noisier than")
print("its marginal. Compare H2a and H2b's OWN patterns across strata -- they do not")
print("always tell the same story (see this project's own report for why that matters).")


# %% [markdown]
# # Manuscript-completeness addendum: points 2-8 and the venue-dependent checks
#
# This notebook forks `pangenomic_gat_pipeline_trimodal_1mb_stratified.ipynb`
# **unchanged through Parts I-IV** (architecture, heads, 7-arm ablation,
# resumable checkpointing, stratified evaluation -- see that notebook's own
# banner). Everything below is new: **Part V onward** works through the
# publication-readiness punch list built up over this project's review
# discussion, item by item, with real numbers produced by actually running
# each check -- not a description of what *should* be done.
#
# **Scope note for this fork:** Parts I-II above give H2 two independent heads (continuous `head_expr` + binary `head_expr_bin`) and report both throughout. Part V onward (GC-MERGE baseline, multi-seed replication, H1 robustness, housekeeping validation, multiple-comparison correction, the archive bundle, the synthetic generalisation check) were written against the single continuous H2 head and continue to use it only -- extending each of those to also carry `head_expr_bin` is a mechanical follow-on, not done here, so as not to re-run every one of these checks for a result already fully covered in Parts I-II. Part IV's housekeeping/protein-coding stratification is likewise continuous-only here; a binary-head version of that same stratification was produced post-hoc, outside this notebook, from an earlier run's saved predictions (see `results/stratified_performance_matrix/`).
#
# | Part | Addresses |
# |---|---|
# | V | Point 2 -- a GC-MERGE-style baseline, re-implemented and re-run head-to-head on our own graphs and folds, not a literature number |
# | VI | Point 3 -- does the H4 fusion-harm result survive an epoch-count-matched control? (checked on this run's training histories) |
# | VII | Point 4 -- multi-seed replication of the central `all3` vs `str_epi` comparison |
# | VIII | Point 5 -- an H1 distance-only baseline, and a topologically-matched (not just distance-matched) negative-sampling check |
# | IX | Point 6 -- the housekeeping tau-median split, checked against an independent, externally curated housekeeping gene list |
# | X | Point 7 -- Benjamini-Hochberg correction across every p-value this project has computed |
# | XI | Point 8 -- an archive bundle assembled and checksummed, ready for a Zenodo upload and a DOI |
# | XII | CS-venue check -- does the fusion-harm mechanism generalise beyond genomics? A controlled synthetic multimodal experiment |
# | XIII | Bio-venue check -- do the structure tower's attention weights land preferentially on annotated regulatory elements (ENCODE cCREs)? |
#
# **Scope.** This addendum runs at the same scale as Parts I-IV: full scale
# (all genes, all 46 tissues) in this script.


# %% [markdown]
# # Part V -- A real head-to-head baseline: a GC-MERGE-style GCN
#
# `results/SOTA_COMPARISON.md` and `results/RELATED_WORK.md` already
# identified GC-MERGE (Bigness et al. 2022) as the one prior method that
# shares our H2 task -- and, now that H2 is continuous regression in this notebook, shares our metric too (Pearson r), closing the metric-family gap `results/SOTA_COMPARISON.md` flagged for every other comparable method --
# but the comparison there is against *their* published number on *their*
# data. This closes that gap: a GC-MERGE-style GCN (histone marks as node
# features, a graph convolution over the same contact graph, a single
# expression head), trained and evaluated on **our own graphs, our own
# gene-grouped folds, our own metric code** -- so this is a genuine
# head-to-head, not two numbers from two different experiments placed side
# by side.
#
# **What is, and is not, faithfully reproduced.** GC-MERGE's actual
# implementation uses Hi-C-derived contact strength as edge weight into a
# GCN; ours substitutes the log CHiCAGO score (our closest analogue, since
# this pipeline never opens a `.hic` matrix at all -- see Part I). The node
# features (histone signal + mask, per mark) and the expressed/not-expressed
# task and split are identical to our own model's. This is a faithful
# re-implementation of the *architecture family* GC-MERGE represents (a GCN
# over a chromatin contact graph with histone features), not a byte-for-byte
# port of their released code.


# %%
from torch_geometric.nn import GCNConv

class GCMergeBaseline(nn.Module):
    # A GCN over the promoter subgraph with histone-mark node features and a
    # single expression-signal (regression) head -- the architecture family
    # GC-MERGE (Bigness et al. 2022) represents, adapted to continuous
    # regression now that H2 itself is continuous. This is what finally makes
    # this baseline genuinely comparable to the regression-task literature
    # (EPInformer, Prism, Omics Graph Learning) on Pearson r -- the metric-
    # family mismatch results/SOTA_COMPARISON.md flagged for those methods
    # does not apply here, since both sides are now the same kind of number.
    def __init__(self, f_in, d_hidden=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = f_in
        for _ in range(n_layers):
            self.convs.append(GCNConv(d, d_hidden))
            self.norms.append(nn.LayerNorm(d_hidden))
            d = d_hidden
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, x, edge_index, edge_weight, batch):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = self.drop(F.relu(norm(conv(h, edge_index, edge_weight))))
        pooled = global_mean_pool(h, batch)
        return self.head(pooled).squeeze(-1)


def _gcmerge_edge_weight(batch):
    # log1p CHiCAGO score (edge_attr column 0, see cell 18's edge feature
    # spec) -- the closest analogue to a Hi-C contact-strength edge weight
    # this loop-calls-only pipeline has. Clamped positive: GCNConv's
    # symmetric normalisation assumes non-negative weights.
    return batch.edge_attr[:, 0].clamp(min=0)


def gcmerge_epoch(model, loader, device, optim=None):
    train = optim is not None
    model.train(train)
    tot = torch.zeros((), device=device); n_used = n_skipped = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with amp_ctx(CFG, device):
                logit = model(batch.x, batch.edge_index, _gcmerge_edge_weight(batch), batch.batch)
            loss = masked_mse(logit.float(), batch.y_expr)   # H2 is continuous regression
            if train and not optimizer_step(model, optim, loss, CFG):
                n_skipped += 1
                continue
            tot += loss.detach(); n_used += 1
    return {"loss": (tot / max(n_used, 1)).item(), "n_skipped": n_skipped}


@torch.no_grad()
def gcmerge_predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with amp_ctx(CFG, device):
            logit = model(batch.x, batch.edge_index, _gcmerge_edge_weight(batch), batch.batch)
        ys.append(batch.y_expr); ps.append(logit.float())   # raw regression output
    y, p = torch.cat(ys).cpu().numpy(), torch.cat(ps).cpu().numpy()
    return {"y": y[y >= 0], "p": p[y >= 0]}


def train_gcmerge_fold(fold, train_idx, test_idx, graphs, device,
                       n_epochs=25, patience=6, batch_size=16, run_key=None):
    # Same resumable loop as every other run (fit_resumable): per-run seed,
    # atomic checkpoints, RNG state, cosine schedule, gradient clipping.
    set_all_seeds(run_seed(CFG.seed, run_key))
    tr_idx, va_idx = inner_split(train_idx)
    model = GCMergeBaseline(f_in=N_NODE_FEAT).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(optim, T_max=n_epochs)
    fp = cfg_fingerprint(CFG, stage="gcmerge", n_epochs=n_epochs, patience=patience,
                         batch_size=batch_size, train=content_key(np.sort(train_idx).tolist()))
    dl_tr = make_loader(graphs, tr_idx, batch_size, True, persistent=True)
    dl_va = make_loader(graphs, va_idx, batch_size, False, persistent=True)
    ck = fit_resumable(
        run_key, model=model, optim=optim, sched=sched,
        train_fn=lambda: gcmerge_epoch(model, dl_tr, device, optim),
        val_fn=lambda: gcmerge_epoch(model, dl_va, device),
        n_epochs=n_epochs, patience=patience, min_epochs=0, fingerprint=fp,
        label=f"gcmerge f{fold}", quiet=True)
    pred = stored_test_preds(ck)
    if pred is None:
        pred = gcmerge_predict(model, make_loader(graphs, test_idx, batch_size, False), device)
        store_test_preds(run_key, pred)
    m = regression_metrics(pred["y"], pred["p"], "")
    m["epochs_run"] = ck["epoch"] + 1
    return m


print("Part V -- GC-MERGE-style baseline, same graphs/folds/metric as our own model\n")
gcmerge_rows = []
for k, (tr, te) in enumerate(FOLDS):
    m = train_gcmerge_fold(k, tr, te, GRAPHS, DEVICE, run_key=f"gcmerge_fold{k}")
    m["fold"] = k
    gcmerge_rows.append(m)
    print(f"  fold {k}: Pearson r {m['pearson_r']:.4f}  (R^2 {m['r2']:.4f}, "
          f"{m['epochs_run']} epochs)")

GCMERGE_DF = pd.DataFrame(gcmerge_rows)
GCMERGE_DF.to_csv(RESULTS_DIR / "gcmerge_baseline_per_fold.csv", index=False)

ours_h2 = ABL.loc[ABL.variant == "all3", "expr_pearson_r"].values
gcmerge_h2 = GCMERGE_DF["pearson_r"].values
t_gc = sstats.ttest_rel(ours_h2, gcmerge_h2)
P_VALUES = {}   # collected here and in every later part, corrected together in Part X
P_VALUES["ours_vs_gcmerge_H2"] = float(t_gc.pvalue)

print(f"\nours (all3)                H2 Pearson r: {ours_h2.mean():.4f} +/- {ours_h2.std():.4f}")
print(f"GC-MERGE-style baseline    H2 Pearson r: {gcmerge_h2.mean():.4f} +/- {gcmerge_h2.std():.4f}")
print(f"delta: {ours_h2.mean() - gcmerge_h2.mean():+.4f}   paired p = {t_gc.pvalue:.4f}")
print()
print("This is now the SAME metric family as EPInformer/Prism/Omics Graph Learning's")
print("Pearson r on continuous expression -- unlike results/SOTA_COMPARISON.md's")
print("H2-as-classification version, this comparison is not just to GC-MERGE but is")
print("finally on a directly comparable footing with the regression-task literature.")

fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.bar(["GC-MERGE-style\n(this data)", "Ours (all3)"],
       [gcmerge_h2.mean(), ours_h2.mean()],
       yerr=[gcmerge_h2.std(), ours_h2.std()],
       color=[PALETTE[1], PALETTE[0]], capsize=6)
ax.set_ylabel("H2 Pearson r"); ax.set_ylim(0, 1.02); ax.legend(["+/- 1 s.d."])
ax.set_title("Head-to-head on our own data/splits\n(not a literature comparison)")
fig.tight_layout(); savefig(fig, "22_gcmerge_head_to_head"); plt.show()


# %% [markdown]
# # Part VI -- Is the H4 fusion-harm result a training-dynamics artefact?
#
# v1 (the two-tower manuscript) found the fused arm trained for dramatically
# fewer effective epochs than the structure-only arm (0-4 vs 33-35) before
# early stopping fired, which is a real alternative explanation for "fusion
# hurts H4" distinct from an architectural one. Both the trimodal manuscript
# (`paper/v2_trimodal/main_v2.tex`, Section VIII) and the thesis addendum
# flagged this as unresolved for the trimodal result specifically and left it
# open.
#
# **Checked empirically here.** v1's analytic argument (a min-epoch floor above
# the fast-mode epoch budget, so every arm ran every epoch) does not carry over:
# in this run Stage B trains for up to `CFG.epochs` epochs with a floor of
# `CFG.min_epochs_heads`, so early stopping can fire after the floor. The
# cell below compares the epochs each arm actually trained, and the epoch with
# the best validation loss, paired per fold.


# %%
print("Part VI -- epoch-count confound, checked on this run's histories\n")
print(f"Stage B: up to {CFG.n_epochs} epochs, early stopping allowed from epoch "
      f"{CFG.min_epochs_heads} (patience {CFG.patience})\n")

_runs = pd.concat([RESULTS[v]["hist"] for v in RESULTS], ignore_index=True)
EPOCHS_RUN = (_runs.groupby(["variant", "fold"])["epoch"].max() + 1).rename("epochs_run").reset_index()
epoch_counts = ABL.groupby("variant")["best_epoch"].agg(["mean", "std", "min", "max", "count"])
print("best_epoch per arm:")
print(epoch_counts.round(2).to_string())
print("\nepochs actually trained per arm:")
print(EPOCHS_RUN.groupby("variant")["epochs_run"].agg(["mean", "min", "max"]).round(1).to_string())

_er = EPOCHS_RUN.pivot(index="fold", columns="variant", values="epochs_run")
all3_ep, strepi_ep = _er["all3"].values, _er["str_epi"].values
t_ep = sstats.ttest_rel(all3_ep, strepi_ep) if np.any(all3_ep != strepi_ep) else None
P_VALUES["epoch_count_all3_vs_strepi"] = float(t_ep.pvalue) if t_ep is not None else 1.0
print(f"\nall3 epochs trained:    {all3_ep.mean():.1f} +/- {all3_ep.std():.1f}")
print(f"str_epi epochs trained: {strepi_ep.mean():.1f} +/- {strepi_ep.std():.1f}")
print("paired p = " + (f"{t_ep.pvalue:.4f}" if t_ep is not None else "n/a (identical in every fold)"))
print("A large, significant gap here would reopen the training-dynamics explanation "
      "for any H4 fusion-harm result; a small one rules it out for this run.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=EPOCHS_RUN, x="variant", y="epochs_run",
            order=[v for v in CFG.variants if v in RESULTS], palette=PALETTE, ax=ax)
ax.axhline(CFG.min_epochs_heads, ls="--", c="k", lw=1, label=f"min epochs ({CFG.min_epochs_heads})")
ax.set_title("Stage-B epochs trained, by arm")
ax.tick_params(axis="x", rotation=30); ax.legend()
fig.tight_layout(); savefig(fig, "23_epoch_count_by_arm"); plt.show()


# %% [markdown]
# # Part VII -- Multi-seed replication
#
# Every significance test so far paired 5 folds under **one** seed. This
# reruns the central comparison from Table V (`all3` vs `str_epi` on H4) at
# two additional seeds, giving 15 paired observations (3 seeds x 5 folds)
# instead of 5, using the exact same `train_one_fold` (and its resumable
# checkpointing -- if this cell is interrupted, rerunning it resumes rather
# than restarting any completed seed/fold).
#
# In two-stage mode the seed changes Stage B (fusion/head init, shuffling)
# while each fold keeps its Stage-A encoders, so this measures head-training
# variance, not encoder-training variance.
#
# Scope: restricted to the two arms the paper's H4 claim is actually about,
# not all 7 -- rerunning the full ablation at 3 seeds would be 3x the
# already-substantial compute for arms that are not load-bearing for this
# specific claim.


# %%
import copy

print("Part VII -- multi-seed replication of the all3 vs str_epi H4 comparison\n")
EXTRA_SEEDS = [43, 44]
seed_rows = list(ABL.loc[ABL.variant.isin(["all3", "str_epi"]),
                        [c for c in ABL.columns if c in
                         ("variant", "fold", "tissue_accuracy")]].assign(seed=CFG.seed).to_dict("records"))

for seed in EXTRA_SEEDS:
    CFG_SEED = copy.deepcopy(CFG)
    CFG_SEED.seed = seed
    for variant in ("all3", "str_epi"):
        for k, (tr, te) in enumerate(FOLDS):
            model, pred, hist, best_ep = train_one_fold(
                k, tr, te, GRAPHS, SEQ_BANK, CFG_SEED, DEVICE, variant=variant, quiet=True,
                run_key=f"seed{seed}_{variant}_fold{k}", split_key=f"gene_fold{k}")
            met = all_metrics(pred, len(TISSUE_LIST))
            seed_rows.append({"variant": variant, "fold": k,
                              "tissue_accuracy": met["tissue_accuracy"], "seed": seed})
    print(f"  seed {seed}: done")
set_all_seeds(CFG.seed)   # every run seeds itself; this only resets the global stream

SEED_DF = pd.DataFrame(seed_rows)
SEED_DF.to_csv(RESULTS_DIR / "multiseed_replication.csv", index=False)

a = SEED_DF.loc[SEED_DF.variant == "all3"].sort_values(["seed", "fold"])["tissue_accuracy"].values
b = SEED_DF.loc[SEED_DF.variant == "str_epi"].sort_values(["seed", "fold"])["tissue_accuracy"].values
t_multi = sstats.ttest_rel(a, b)
P_VALUES["multiseed_all3_vs_strepi_H4"] = float(t_multi.pvalue)

print(f"\nacross {len(set(SEED_DF.seed))} seeds x {CFG.n_folds} folds ({len(a)} paired observations):")
print(f"  all3    H4 accuracy: {a.mean():.4f} +/- {a.std():.4f}")
print(f"  str_epi H4 accuracy: {b.mean():.4f} +/- {b.std():.4f}")
print(f"  delta: {a.mean()-b.mean():+.4f}   paired p = {t_multi.pvalue:.4f}  "
      f"(compare to the single-seed, 5-fold p=0.035 reported in the manuscript)")

fig, ax = plt.subplots(figsize=(6, 4.2))
sns.boxplot(data=SEED_DF, x="variant", y="tissue_accuracy",
           order=["all3", "str_epi"], palette=PALETTE[:2], ax=ax)
sns.stripplot(data=SEED_DF, x="variant", y="tissue_accuracy",
             order=["all3", "str_epi"], color="black", alpha=0.5, ax=ax)
ax.set_title(f"H4 accuracy, all3 vs str_epi, {len(set(SEED_DF.seed))} seeds x {CFG.n_folds} folds")
fig.tight_layout(); savefig(fig, "24_multiseed_h4"); plt.show()


# %% [markdown]
# # Part VIII -- H1 robustness: a distance-only baseline, and a topological-asymmetry check
#
# v1's discussion flagged H1's near-ceiling AUPRC as "a floor, not a solved
# task," and named the honest next comparison that was never run: a
# distance-only baseline. It also named a structural concern: held-out
# positive pairs come from nodes whose anchor edges were deleted from
# message passing, while distance-matched negatives come from nodes whose
# edges were retained -- so the two classes could differ topologically by
# construction, before the model learns anything. Both are addressed here,
# directly, not just re-stated as open questions.


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

print("Part VIII.1 -- H1 distance-only baseline\n")

_dist = []
for k, (tr, te) in enumerate(FOLDS):
    for batch in make_loader(GRAPHS, te, 256, False):
        x = batch.x  # column 2 = signed_log_dist (node feature spec, section 9)
        for ei, y in ((batch.pos_edge_label_index, 1), (batch.neg_edge_label_index, 0)):
            d = (x[ei[0], 2] - x[ei[1], 2]).abs().numpy()
            _dist.append(pd.DataFrame({"fold": k, "dist_feat": d, "y": y}))

DIST_DF = pd.concat(_dist, ignore_index=True)
print(f"collected {len(DIST_DF):,} labelled pairs across {CFG.n_folds} folds")

dist_aucs = []
for k in range(CFG.n_folds):
    tr_d = DIST_DF[DIST_DF.fold != k]
    te_d = DIST_DF[DIST_DF.fold == k]
    if te_d["y"].nunique() < 2 or tr_d["y"].nunique() < 2:
        continue
    clf = LogisticRegression().fit(tr_d[["dist_feat"]], tr_d["y"])
    p = clf.predict_proba(te_d[["dist_feat"]])[:, 1]
    m = binary_metrics(te_d["y"].values, p, "")
    m["fold"] = k
    dist_aucs.append(m)

DIST_RESULT = pd.DataFrame(dist_aucs)
DIST_RESULT.to_csv(RESULTS_DIR / "h1_distance_only_baseline.csv", index=False)
print(DIST_RESULT[["fold", "auprc", "auroc"]].to_string(index=False))
print(f"\ndistance-only AUPRC: {DIST_RESULT['auprc'].mean():.4f} +/- {DIST_RESULT['auprc'].std():.4f}")
print(f"full model (all3) H1 AUPRC: {ABL.loc[ABL.variant=='all3','link_auprc'].mean():.4f}")
print()
if DIST_RESULT["auprc"].mean() > 0.9:
    print("Distance ALONE already reaches near-ceiling AUPRC -- H1's headline number is")
    print("consistent with (not distinguishable from) a pure distance-decay shortcut,")
    print("exactly the concern v1 raised. This does not mean the model has learned")
    print("nothing beyond distance, only that this task construction cannot tell the")
    print("difference -- restated as a limitation, not resolved away.")
else:
    print("Distance alone does NOT reach the full model's AUPRC -- H1's headline number")
    print("reflects signal beyond pure distance decay, which is a materially stronger")
    print("claim than v1 was able to make.")


# %%
from scipy.stats import mannwhitneyu

print("Part VIII.2 -- topological-asymmetry check: do positive and negative pairs")
print("differ in node degree by construction, independent of the model?\n")

_deg = []
for k, (tr, te) in enumerate(FOLDS):
    for batch in make_loader(GRAPHS, te, 256, False):
        # degree in the RETAINED message-passing graph (edge_index), per node
        deg = torch.bincount(batch.edge_index[0], minlength=batch.num_nodes)
        for ei, label in ((batch.pos_edge_label_index, "positive (edges removed from message passing)"),
                          (batch.neg_edge_label_index, "negative (edges retained)")):
            d = (deg[ei[0]] + deg[ei[1]]).numpy()   # summed endpoint degree
            _deg.append(pd.DataFrame({"fold": k, "degree_sum": d, "label": label}))

DEG_DF = pd.concat(_deg, ignore_index=True)
pos_deg = DEG_DF.loc[DEG_DF.label.str.startswith("positive"), "degree_sum"]
neg_deg = DEG_DF.loc[DEG_DF.label.str.startswith("negative"), "degree_sum"]
u = mannwhitneyu(pos_deg, neg_deg)
P_VALUES["h1_topological_asymmetry"] = float(u.pvalue)

print(f"positive-pair endpoint degree (in the retained message-passing graph): "
      f"mean {pos_deg.mean():.2f}, median {pos_deg.median():.0f}")
print(f"negative-pair endpoint degree: mean {neg_deg.mean():.2f}, median {neg_deg.median():.0f}")
print(f"Mann-Whitney U p = {u.pvalue:.2e}")
if u.pvalue < 0.001 and abs(pos_deg.mean() - neg_deg.mean()) > 0.5:
    print("\nCONFIRMED: positive and negative pairs differ systematically in local graph")
    print("degree, independent of anything the model learns -- exactly the topological")
    print("asymmetry v1 raised as a plausible (not proven) leak mechanism for H1's")
    print("near-ceiling AUPRC. A bilinear scorer over node states can exploit this")
    print("degree difference directly. Fixing it requires drawing negatives from nodes")
    print("ALSO excluded from message passing (matching topology, not just distance) --")
    print("a data-pipeline change, not something this notebook retrains around.")
else:
    print("\nNo strong systematic degree difference detected between positive and")
    print("negative pairs in this run -- the topological-asymmetry concern is not")
    print("supported by this check, which somewhat rehabilitates H1's headline number.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=DEG_DF, x="label", y="degree_sum", palette=PALETTE[:2], ax=ax, showfliers=False)
ax.set_title("Endpoint-node degree: positive vs. negative H1 pairs")
ax.tick_params(axis="x", rotation=15)
fig.tight_layout(); savefig(fig, "25_h1_topological_asymmetry"); plt.show()


# %% [markdown]
# # Part IX -- External validation of the housekeeping tau-median split
#
# The housekeeping stratification (Part IV) uses a median split of this
# dataset's own tau distribution (0.636) as a corrected replacement for the
# codebase's unreachable 0.2 default -- but that correction was never
# cross-checked against an independent, externally curated housekeeping gene
# list, exactly as `results/pretrained_ablation_summary/README.md` and the
# manuscript's Discussion both flagged as still open. This closes that gap
# with a widely-used, independently curated compact reference set (canonical
# qPCR/Western-blot normalisation controls, chosen for consistent constitutive
# expression across tissues in the literature, predating and independent of
# this project's own tau computation).


# %%
print("Part IX -- housekeeping tau-median split vs. an external curated gene list\n")

# A compact, widely-cited set of canonical constitutively-expressed genes,
# independent of and pre-dating this project's own tau computation --
# standard qPCR/Western-blot normalisation controls (e.g. Eisenberg & Levanon
# 2013-style "always expressed" panels). Not exhaustive; chosen because their
# housekeeping status is essentially uncontested in the literature, which is
# exactly the property needed for an external sanity check.
EXTERNAL_HOUSEKEEPING = [
    "ACTB", "GAPDH", "TUBB", "B2M", "RPL13A", "RPLP0", "YWHAZ", "SDHA",
    "HPRT1", "TBP", "PPIA", "PGK1", "UBC", "POLR2A", "GUSB", "HMBS",
    "PUM1", "EEF1A1", "RPS18", "ALAS1",
]

present = [g for g in EXTERNAL_HOUSEKEEPING if g in tau.index and not np.isnan(tau.get(g, np.nan))]
missing = [g for g in EXTERNAL_HOUSEKEEPING if g not in present]
print(f"{len(present)}/{len(EXTERNAL_HOUSEKEEPING)} external housekeeping genes have a "
      f"computed tau in this gene panel (missing: {missing})")

if present:
    ext_tau = tau.loc[present].sort_values()
    print("\ntau for each external housekeeping gene (median split threshold = "
          f"{tau.median():.3f}):")
    print(ext_tau.round(3).to_string())

    n_below = int((ext_tau < tau.median()).sum())
    print(f"\n{n_below}/{len(present)} ({n_below/len(present):.0%}) of these externally-"
          f"curated housekeeping genes fall BELOW the tau-median split, i.e. are "
          f"correctly classified housekeeping by this project's own method.")
    print()
    print("A high match rate here is external validation that the tau-median split")
    print("captures real housekeeping biology, not an arbitrary statistical artefact")
    print("of this dataset. A low match rate would mean the correction from the")
    print("unreachable 0.2 default to a median split, while mathematically necessary,")
    print("still does not track independently-known housekeeping status well -- report")
    print("whichever this run actually shows, not the expected direction.")

    HK_EXTERNAL_VALIDATION = pd.DataFrame({
        "gene": ext_tau.index, "tau": ext_tau.values,
        "classified_housekeeping": (ext_tau.values < tau.median()),
    })
    HK_EXTERNAL_VALIDATION.to_csv(RESULTS_DIR / "housekeeping_external_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(tau.dropna(), bins=60, color=PALETTE[2], alpha=0.6, label="all genes in this panel")
    ax.axvline(tau.median(), ls="--", c="k", label=f"median split ({tau.median():.3f})")
    for g, t in ext_tau.items():
        ax.axvline(t, color=PALETTE[0], alpha=0.5, lw=1)
    ax.scatter(ext_tau.values, [ax.get_ylim()[1]*0.02]*len(ext_tau), color=PALETTE[0],
              zorder=5, label="external housekeeping genes", s=25)
    ax.set_xlabel("tau"); ax.set_ylabel("gene count")
    ax.set_title("Where externally-curated housekeeping genes fall in our tau distribution")
    ax.legend(fontsize=8)
    fig.tight_layout(); savefig(fig, "26_housekeeping_external_validation"); plt.show()
else:
    print("\nNone of the external housekeeping genes are present with a computed tau in")
    print("this gene panel -- this check cannot run at this scale (likely a fast-mode")
    print("gene-subsampling artefact; retry at full scale).")


# %% [markdown]
# # Part X -- Multiple-comparison correction across every p-value this project has computed
#
# Every significance test so far (the original 4-head ablation verdict, plus
# the new ones in Parts V, VI, VII, VIII) was reported and read on its own,
# uncorrected p-value. Reported together, they are 8+ tests drawn from the
# same project -- exactly the situation multiple-comparison correction exists
# for. This applies Benjamini-Hochberg (controls the false discovery rate,
# less conservative than Bonferroni, the standard choice when tests are not
# assumed independent) across the full set and reports which findings survive.


# %%
from statsmodels.stats.multitest import multipletests

print("Part X -- Benjamini-Hochberg correction across every p-value in this project\n")

# the original 7-arm ablation verdict (cell 60) -- reload from disk since it
# was computed in the base notebook, not recomputed here
_verdict = pd.read_csv(RESULTS_DIR / "ablation_fusion_verdict.csv")
_verdict["_p"] = pd.to_numeric(_verdict["paired p"], errors="coerce")   # handles 1e-05 etc.
for head, pv in _verdict.dropna(subset=["_p"])[["head", "_p"]].itertuples(index=False):
    P_VALUES[f"ablation_{head}_all3_vs_best_single"] = float(pv)

print("Every p-value collected across this project's significance tests:")
names = list(P_VALUES.keys())
pvals = [P_VALUES[n] for n in names]
for n, p in zip(names, pvals):
    print(f"  {n:<45} p = {p:.4f}")

reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
CORRECTED = pd.DataFrame({"test": names, "p_raw": pvals, "p_bh_adjusted": p_adj,
                          "significant_at_0.05_after_correction": reject})
CORRECTED = CORRECTED.sort_values("p_raw").reset_index(drop=True)
CORRECTED.to_csv(RESULTS_DIR / "multiple_comparison_correction.csv", index=False)

print(f"\n{CORRECTED.to_string(index=False)}")
print(f"\n{int(reject.sum())}/{len(reject)} tests remain significant at alpha=0.05 after "
      "Benjamini-Hochberg correction.")
n_sig_before = sum(1 for p in pvals if p < 0.05)
print(f"({n_sig_before}/{len(pvals)} were nominally significant before correction.)")
print()
print("This is the number to cite in a manuscript's statistics section, not any")
print("individual test's raw p-value taken in isolation.")


# %% [markdown]
# # Part XI -- Archive bundle for Zenodo / a DOI
#
# Every venue on the shortlist now expects code + trained checkpoints +
# configs to be archived with a persistent identifier, not just linked to a
# GitHub repo that can change. This assembles everything needed for that:
# config, this run's headline + ablation + stratification + SOTA-comparison
# CSVs, the checkpoint chain, and a checksummed manifest, into one folder
# ready to zip and upload to Zenodo (which mints the DOI on upload -- that
# last step is a human action on zenodo.org, not something this notebook can
# do on its own).


# %%
import hashlib as _hashlib
import shutil as _shutil

print("Part XI -- assembling the Zenodo-ready archive bundle\n")

ARCHIVE_DIR = RESULTS_DIR / "archive_bundle"
ARCHIVE_DIR.mkdir(exist_ok=True)
(ARCHIVE_DIR / "checkpoints").mkdir(exist_ok=True)
(ARCHIVE_DIR / "results").mkdir(exist_ok=True)

# config, as actually run
(ARCHIVE_DIR / "config.json").write_text(json.dumps(asdict(CFG), indent=2, default=str))

# every headline/ablation/stratification/new-analysis CSV this run produced
for csv_path in sorted(RESULTS_DIR.glob("*.csv")):
    _shutil.copy2(csv_path, ARCHIVE_DIR / "results" / csv_path.name)

# the best-fold checkpoint for the primary arm (all3), which is what a
# reader would actually want to load to reproduce headline numbers
for k in range(CFG.n_folds):
    src_ckpt = RESULTS_DIR / f"model_fold{k}.pt"
    if src_ckpt.exists():
        _shutil.copy2(src_ckpt, ARCHIVE_DIR / "checkpoints" / src_ckpt.name)

manifest = {"files": [], "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "config_summary": {"fast_mode": CFG.fast_mode, "seed": CFG.seed,
                              "variants": list(CFG.variants), "n_folds": CFG.n_folds}}
for p in sorted(ARCHIVE_DIR.rglob("*")):
    if p.is_file() and p.name != "MANIFEST.json":
        h = _hashlib.sha256(p.read_bytes()).hexdigest()
        manifest["files"].append({"path": str(p.relative_to(ARCHIVE_DIR)),
                                  "sha256": h, "bytes": p.stat().st_size})
(ARCHIVE_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

readme = f"""# Archive bundle -- {RESULTS_DIR.name}

Generated {manifest['generated']} by
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
    assert h == f["sha256"], f"MISMATCH: {{f['path']}}"
print("all files verified")
```
"""
(ARCHIVE_DIR / "README.md").write_text(readme)

n_files = len(manifest["files"])
total_mb = sum(f["bytes"] for f in manifest["files"]) / 1e6
print(f"archive bundle assembled at {ARCHIVE_DIR.relative_to(PROJECT_ROOT)}")
print(f"  {n_files} files, {total_mb:.1f} MB total, every one checksummed in MANIFEST.json")
print(f"  see {ARCHIVE_DIR.relative_to(PROJECT_ROOT)}/README.md for the Zenodo upload steps")


# %% [markdown]
# # Part XII -- CS-venue check: does the fusion-harm mechanism generalise beyond genomics?
#
# A CS-heavy reviewer's natural objection to the paper's central claim (naive
# fusion of a task-invariant modality dilutes a task-varying signal under
# gated fusion) is that it could be a genomics-specific artefact rather than
# a general property of `GatedFusion`. This tests it directly: a fully
# synthetic, non-biological 3-modality classification benchmark, built so
# that exactly one modality is **constant within a group** (mirroring
# `z_seq` being constant across every tissue graph of a gene) while the
# label depends on the other two. It uses the **exact same `GatedFusion`
# class** already defined for the real model (cell 27), not a re-implementation.
#
# **This is declared synthetic, not disguised as real data** -- its only job
# is to isolate the mechanism from the genomics domain, the way a controlled
# experiment is supposed to.


# %%
print("Part XII -- synthetic controlled multimodal fusion experiment\n")

def make_synthetic_multimodal(n_groups=400, group_size=6, d=32, seed=0):
    # n_groups x group_size samples. Modalities A, B vary per-sample and
    # jointly determine the label (XOR-like, so BOTH are needed -- mirrors
    # z_str / z_epi carrying real, complementary tissue-varying signal).
    # Modality C is IDENTICAL for every sample in a group regardless of
    # label -- mirrors z_seq being identical across every tissue graph of a
    # gene, i.e. constant per "gene", varying only across "genes".
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_groups * group_size, d)).astype(np.float32)
    B = rng.normal(size=(n_groups * group_size, d)).astype(np.float32)
    C = np.repeat(rng.normal(size=(n_groups, d)).astype(np.float32), group_size, axis=0)
    group_id = np.repeat(np.arange(n_groups), group_size)
    # label: XOR of the sign of two informative linear projections of A and B
    wa = rng.normal(size=d); wb = rng.normal(size=d)
    sa = (A @ wa) > 0
    sb = (B @ wb) > 0
    y = (sa ^ sb).astype(np.float32)   # needs BOTH A and B; C carries no label information
    return A, B, C, y.astype(np.float32), group_id


class SmallEncoder(nn.Module):
    def __init__(self, d_in, d_out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
    def forward(self, x): return self.net(x)


class SyntheticFusionModel(nn.Module):
    # Uses the SAME GatedFusion class the real trimodal model uses (cell 27)
    # -- this is a test of that class's behaviour, not a re-implementation.
    def __init__(self, d_in=32, d=64, use_c=True):
        super().__init__()
        self.enc_a, self.enc_b = SmallEncoder(d_in, d), SmallEncoder(d_in, d)
        self.enc_c = SmallEncoder(d_in, d) if use_c else None
        self.fuse = GatedFusion(d, max_modalities=3)
        self.head = nn.Linear(d, 1)
        self.use_c = use_c
    def forward(self, a, b, c):
        embeddings = [self.enc_a(a), self.enc_b(b)]
        if self.use_c:
            embeddings.append(self.enc_c(c))
        z = self.fuse(embeddings)
        return self.head(z).squeeze(-1)


def train_synthetic(use_c, seed, n_epochs=60, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    A, B, C, y, group_id = make_synthetic_multimodal(seed=seed)
    n = len(y)
    gkf_syn = GroupKFold(n_splits=5)
    accs = []
    for tr, te in gkf_syn.split(np.arange(n), groups=group_id):
        model = SyntheticFusionModel(d_in=A.shape[1], use_c=use_c)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        At, Bt, Ct, yt = (torch.tensor(A[tr]), torch.tensor(B[tr]),
                          torch.tensor(C[tr]), torch.tensor(y[tr]))
        Ae, Be, Ce, ye = (torch.tensor(A[te]), torch.tensor(B[te]),
                          torch.tensor(C[te]), torch.tensor(y[te]))
        for ep in range(n_epochs):
            model.train()
            optim.zero_grad()
            logit = model(At, Bt, Ct)
            loss = F.binary_cross_entropy_with_logits(logit, yt)
            loss.backward(); optim.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Ae, Be, Ce)).numpy()
        accs.append(float(((p > 0.5).astype(np.float32) == ye.numpy()).mean()))
    return np.array(accs)

SEEDS_SYN = [0, 1, 2, 3, 4]
acc_with_c = np.concatenate([train_synthetic(True, s) for s in SEEDS_SYN])
acc_without_c = np.concatenate([train_synthetic(False, s) for s in SEEDS_SYN])

t_syn = sstats.ttest_rel(acc_with_c, acc_without_c)
P_VALUES["synthetic_fusion_with_vs_without_invariant_modality"] = float(t_syn.pvalue)

print(f"fused A+B+C (C constant per group, uninformative): "
      f"{acc_with_c.mean():.4f} +/- {acc_with_c.std():.4f}  (n={len(acc_with_c)})")
print(f"fused A+B only (informative modalities only):       "
      f"{acc_without_c.mean():.4f} +/- {acc_without_c.std():.4f}  (n={len(acc_without_c)})")
print(f"delta: {acc_with_c.mean() - acc_without_c.mean():+.4f}   paired p = {t_syn.pvalue:.4f}")
print()
if acc_with_c.mean() < acc_without_c.mean() and t_syn.pvalue < 0.05:
    print("CONFIRMED on a fully synthetic, non-genomic benchmark: fusing in a")
    print("group-invariant (task-uninformative) modality through GatedFusion")
    print("significantly HURTS accuracy relative to fusing only the informative")
    print("modalities -- the mechanism this paper reports for H4 is not a")
    print("genomics-specific artefact. It is a general property of this fusion")
    print("class under this specific failure condition (one constant, uninformative")
    print("modality diluting informative ones).")
else:
    print("This synthetic benchmark did NOT reproduce the fusion-harm pattern at the")
    print("parameters tried here -- report this honestly rather than the wanted")
    print("direction; it means the genomics finding may depend on specifics (e.g.")
    print("the interaction structure between the informative modalities) this")
    print("simple synthetic setup does not capture, and the generality claim should")
    print("be narrowed accordingly.")

SYN_DF = pd.DataFrame({
    "condition": ["all3 (A+B+C)"] * len(acc_with_c) + ["A+B only"] * len(acc_without_c),
    "accuracy": np.concatenate([acc_with_c, acc_without_c]),
})
SYN_DF.to_csv(RESULTS_DIR / "synthetic_fusion_generalization.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4.2))
sns.boxplot(data=SYN_DF, x="condition", y="accuracy", palette=PALETTE[:2], ax=ax)
sns.stripplot(data=SYN_DF, x="condition", y="accuracy", color="black", alpha=0.4, ax=ax)
ax.set_title("Synthetic, non-genomic replication:\nGatedFusion with vs. without a group-invariant modality")
fig.tight_layout(); savefig(fig, "27_synthetic_fusion_generalization"); plt.show()


# %% [markdown]
# # Part XIII -- Bio-venue check: are real PCHi-C partners enriched for annotated regulatory elements?
#
# A biology-heavy reviewer's natural question is whether the contact graphs
# this model learns over are grounded in real regulatory biology at all,
# independent of what the model does with them. This checks it directly:
# do real PCHi-C distal partner fragments (the edges actually present in our
# graphs) overlap ENCODE candidate cis-regulatory elements (cCREs) at a
# higher rate than distance-matched random genomic locations? This is
# attempted as a genuine external-data fetch, not assumed to succeed --
# reported honestly either way.


# %%
import requests

print("Part XIII -- fetching ENCODE cCREs (hg19) for an external regulatory-annotation check\n")

CCRE_PATH = REF_DIR / "GRCh37-ccREs.bed"
CCRE_URLS = [
    "https://downloads.wenglab.org/GRCh37-ccREs.bed",
    "https://api.wenglab.org/screen_v13/fdownloads/GRCh37-ccREs.bed",
]

ccre_available = CCRE_PATH.exists()
if not ccre_available:
    for url in CCRE_URLS:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 10000:
                CCRE_PATH.write_bytes(r.content)
                ccre_available = True
                print(f"  fetched {url} -> {CCRE_PATH} ({len(r.content)/1e6:.1f} MB)")
                break
            else:
                print(f"  {url} -> HTTP {r.status_code}, skipping")
        except Exception as e:
            print(f"  {url} -> {type(e).__name__}: {e}")
else:
    print(f"  already present: {CCRE_PATH}")

if not ccre_available:
    print()
    print("ENCODE cCREs could not be fetched in this environment (no reachable URL")
    print("returned a usable file). This check is reported as NOT COMPLETED, not")
    print("faked -- if the target venue needs it, retry with network access to")
    print("downloads.wenglab.org, or substitute a locally available cCRE/ENCODE")
    print("candidate-enhancer BED file at reference/GRCh37-ccREs.bed and rerun this")
    print("cell alone (everything above it is independent of this one).")
else:
    ccre = pd.read_csv(CCRE_PATH, sep="\t", header=None, usecols=[0, 1, 2, 5],
                       names=["chrom", "start", "end", "kind"], low_memory=False)
    print(f"  loaded {len(ccre):,} cCREs across {ccre.chrom.nunique()} contigs")
    # per chromosome: starts sorted, plus the running max of ends, so overlap
    # queries are exact even if cCREs overlap each other (sorting starts and
    # ends independently, as v1 did, pairs the wrong ends with the starts)
    ccre_by_chrom = {}
    for c, g in ccre.groupby("chrom"):
        g = g.sort_values("start")
        ccre_by_chrom[c] = (g["start"].to_numpy(), np.maximum.accumulate(g["end"].to_numpy()))

    def overlaps_any(chrom, starts, ends, by_chrom):
        # vectorised: does [start, end) overlap any interval on `chrom`?
        if chrom not in by_chrom:
            return np.zeros(len(starts), dtype=bool)
        cs, cmax_end = by_chrom[chrom]
        n_before = np.searchsorted(cs, ends, side="left")      # intervals starting before `end`
        out = np.zeros(len(starts), dtype=bool)
        ok = n_before > 0
        out[ok] = cmax_end[n_before[ok] - 1] > starts[ok]
        return out

    real_partners = loops[["other_chr", "other_start", "other_end"]].dropna().drop_duplicates()
    rng = np.random.default_rng(CFG.seed)
    chrom_sizes = pd.read_csv(REF_DIR / "hg19.chrom.sizes", sep="\t", header=None,
                              names=["chrom", "size"]).set_index("chrom")["size"]
    real_hits, rand_hits = [], []
    for c, g in real_partners.groupby("other_chr", sort=True):
        st = g["other_start"].to_numpy(np.int64); en = g["other_end"].to_numpy(np.int64)
        real_hits.append(overlaps_any(c, st, en, ccre_by_chrom))
        if c in chrom_sizes.index:
            width = en - st
            rs = rng.integers(0, np.maximum(1, int(chrom_sizes[c]) - width))
            rand_hits.append(overlaps_any(c, rs, rs + width, ccre_by_chrom))
    real_hits = np.concatenate(real_hits) if real_hits else np.array([], dtype=bool)
    rand_hits = np.concatenate(rand_hits) if rand_hits else np.array([], dtype=bool)

    real_rate = float(np.mean(real_hits))
    rand_rate = float(np.mean(rand_hits)) if len(rand_hits) else float("nan")
    ct = np.array([[int(np.sum(real_hits)), len(real_hits) - int(np.sum(real_hits))],
                  [int(np.sum(rand_hits)), len(rand_hits) - int(np.sum(rand_hits))]])
    chi2 = sstats.chi2_contingency(ct)
    P_VALUES["ccre_enrichment_real_vs_random"] = float(chi2.pvalue)

    print(f"\nreal PCHi-C distal partners overlapping a cCRE:   {real_rate:.1%} "
          f"({int(np.sum(real_hits)):,}/{len(real_hits):,})")
    print(f"same-width random genomic locations overlapping:  {rand_rate:.1%} "
          f"({int(np.sum(rand_hits)):,}/{len(rand_hits):,})")
    print(f"chi-square p = {chi2.pvalue:.2e}")
    if real_rate > rand_rate and chi2.pvalue < 0.05:
        print()
        print("Real PCHi-C distal partners are SIGNIFICANTLY enriched for annotated")
        print("ENCODE cCREs relative to random genomic background -- the contact graphs")
        print("this model learns over are grounded in independently-annotated regulatory")
        print("biology, not arbitrary genomic intervals.")

    CCRE_RESULT = pd.DataFrame([{"real_overlap_rate": real_rate, "random_overlap_rate": rand_rate,
                                "n_real": len(real_hits), "n_random": len(rand_hits),
                                "chi2_p": float(chi2.pvalue)}])
    CCRE_RESULT.to_csv(RESULTS_DIR / "ccre_enrichment.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(["Real PCHi-C\npartners", "Random genomic\nlocations"],
          [real_rate, rand_rate], color=[PALETTE[0], PALETTE[1]])
    ax.set_ylabel("fraction overlapping an ENCODE cCRE")
    ax.set_title("Regulatory-annotation enrichment of real contact-graph edges")
    fig.tight_layout(); savefig(fig, "28_ccre_enrichment"); plt.show()


# %% [markdown]
# # Summary: what this notebook actually closed, and what it did not


# %%
print("="*78)
print("MANUSCRIPT-COMPLETENESS ADDENDUM -- FINAL SUMMARY")
print("="*78)
print(f"""
Point 2  (real baseline)          -- GC-MERGE-style GCN, head-to-head:
                                      see gcmerge_baseline_per_fold.csv
Point 3  (epoch-count confound)   -- epochs trained per arm, paired test:
                                      see 23_epoch_count_by_arm.png
Point 4  (multi-seed)             -- {len(set(SEED_DF.seed))} seeds x {CFG.n_folds} folds, all3 vs str_epi:
                                      see multiseed_replication.csv
Point 5  (H1 robustness)          -- distance-only baseline + topological
                                      asymmetry check: see h1_distance_only_baseline.csv,
                                      25_h1_topological_asymmetry.png
Point 6  (housekeeping external)  -- checked against {len(EXTERNAL_HOUSEKEEPING)} externally curated genes:
                                      see housekeeping_external_validation.csv
Point 7  (multiple comparisons)   -- Benjamini-Hochberg across {len(P_VALUES)} tests:
                                      see multiple_comparison_correction.csv
Point 8  (archive bundle)         -- assembled and checksummed at
                                      {ARCHIVE_DIR.relative_to(PROJECT_ROOT)}/
CS-venue check                    -- synthetic non-genomic replication of the
                                      fusion-harm mechanism: see
                                      synthetic_fusion_generalization.csv
Bio-venue check                   -- ENCODE cCRE enrichment of real PCHi-C
                                      partners vs. random background: see
                                      ccre_enrichment.csv ({'completed' if ccre_available else 'NOT completed -- no network access to fetch cCREs in this run'})
""")
print(f"Scale: {'FAST (subsampled)' if CFG.fast_mode else 'FULL'} -- "
      f"{META['tissue'].nunique()} tissues, {META['gene'].nunique():,} genes, "
      f"{len(GRAPHS):,} graphs; encoders: {CFG.encoder_source}.")
