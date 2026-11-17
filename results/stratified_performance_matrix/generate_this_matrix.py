"""
Post-hoc performance matrix: protein-coding x housekeeping, using the
ALREADY-TRAINED (pretrained/warm-started) all3-arm model checkpoints.
Inference only -- no retraining. Covers all three validation schemes that
have saved checkpoints: gene-grouped CV, chromosome-block CV,
leave-one-tissue-out.
"""
import sys, json, time, math
from pathlib import Path
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score, jaccard_score,
    brier_score_loss, log_loss, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/mnt/raid/ksengupta/pic_hic_ml")
CACHE_DIR = PROJECT_ROOT / "cache"
SRC_DIR = PROJECT_ROOT / "results" / "results_trimodal_1mb_stratified_fast"
VSTRAT_DIR = SRC_DIR / "validation_strategies"
OUT_DIR = PROJECT_ROOT / "results" / "stratified_performance_matrix"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# ============================================================== load metric functions
exec(open("/tmp/cell31_metrics.py").read().split("# ---- self-test")[0])

# ============================================================== load cached graphs + META
GRAPH_CACHE = CACHE_DIR / "graphs_hg19_1000000_2500_8_10000_5.0_True_0.15_42.pt"
print("loading graph cache:", GRAPH_CACHE.name)
d = torch.load(GRAPH_CACHE, map_location="cpu", weights_only=False)
GRAPHS, META = d["graphs"], d["meta"]
print(f"  {len(GRAPHS):,} graphs, META shape {META.shape}")

# ---- recompute housekeeping with a threshold that's actually reachable here.
# CFG.hk_tau_threshold=0.2 (the notebook default) gives ZERO housekeeping
# genes in this data: tau is computed over the FULL 43-tissue RNA-seq panel
# (independent of which/how many PCHi-C tissues are in play -- fast_mode only
# subsamples the graph side, not the RNA-seq side), and that panel's tau
# never drops below 0.226 for any of these genes. This is not a small-sample
# artifact that a bigger PCHi-C run fixes -- the RNA-seq panel is already at
# its full size -- so the fix is a threshold appropriate to this tau
# distribution, not more data. Using a median split instead: genuinely
# balanced, and "more/less broadly expressed than a typical gene in this
# panel" is still a meaningful reading of housekeeping-vs-not.
from collections import defaultdict
TPM_FILE = PROJECT_ROOT / "rna_seq" / "rnaseq.rsem_genes_tpm.txt"
tpm_header = pd.read_csv(TPM_FILE, sep="\t", nrows=0)
rna_cols = [c for c in tpm_header.columns if c not in ("gene_id", "gene_name")]
rna_by_tissue = defaultdict(list)
for c in rna_cols:
    rna_by_tissue[c.split("_")[0]].append(c)
_tpm = pd.read_csv(TPM_FILE, sep="\t", usecols=["gene_name"] + rna_cols, low_memory=False)
_tpm = _tpm[_tpm["gene_name"].isin(set(META["gene"]))].groupby("gene_name").max()
_by_tissue = {}
for sid, cols in rna_by_tissue.items():
    cols = [c for c in cols if c in _tpm.columns]
    if cols:
        _by_tissue[sid] = _tpm[cols].mean(axis=1)
_TPM = pd.DataFrame(_by_tissue)
_logtpm = np.log2(_TPM + 1)
_mx = _logtpm.max(axis=1)
_valid = _mx > 0
_n_t = _logtpm.shape[1]
TAU = pd.Series(np.nan, index=_TPM.index)
TAU[_valid] = ((1 - _logtpm[_valid].div(_mx[_valid], axis=0)).sum(axis=1) / (_n_t - 1))
HK_MEDIAN_CUT = float(TAU[_valid].median())
print(f"\nraw tau recomputed over the full {_n_t}-tissue RNA-seq panel "
      f"({int(_valid.sum())} genes with a defined value)")
print(f"  tau range [{TAU[_valid].min():.3f}, {TAU[_valid].max():.3f}], "
      f"median {HK_MEDIAN_CUT:.3f}  (0.2 threshold -> "
      f"{(TAU[_valid] < 0.2).sum()} genes: unusable)")
_tau_lookup = TAU.reindex(META["gene"]).values
META = META.copy()
META["tau_raw"] = _tau_lookup
META["y_housekeeping"] = np.where(np.isnan(_tau_lookup), -1,
                                  (_tau_lookup < HK_MEDIAN_CUT).astype(int))
print(f"  median-split housekeeping: {(META['y_housekeeping']==1).sum()} housekeeping / "
      f"{(META['y_housekeeping']==0).sum()} non-housekeeping / "
      f"{(META['y_housekeeping']==-1).sum()} undefined")

# re-attach epi (computed AFTER the cache save in the original notebook, cell 20)
CFG_HIST_MARKS = 7
EPI_DIM = 2 * CFG_HIST_MARKS
for g in GRAPHS:
    g.epi = g.x[0, 4:4 + EPI_DIM].clone().unsqueeze(0)

N_NODE_FEAT = GRAPHS[0].x.shape[1]
N_EDGE_FEAT = GRAPHS[0].edge_attr.shape[1]
TISSUE_LIST = sorted(META["tissue"].unique())
print(f"  N_NODE_FEAT={N_NODE_FEAT}  N_EDGE_FEAT={N_EDGE_FEAT}  tissues={len(TISSUE_LIST)}")

SEQ_CACHE = CACHE_DIR / "promoter_onehot_hg19_1000000_2500.npz"
print("loading sequence cache:", SEQ_CACHE.name)
z = np.load(SEQ_CACHE, allow_pickle=True)
SEQ_TENSOR = torch.from_numpy(z["onehot"]).float()
print("  SEQ_TENSOR", tuple(SEQ_TENSOR.shape))
SEQ_BANK = SEQ_TENSOR

# ============================================================== load model class defs
@dataclass
class Config:
    seed: int = 42
    fast_mode: bool = True
    promoter_window: int = 1_000_000
    seq_len_fast: int = 1_000_000
    histone_marks: tuple = ("H3K4me1", "H3K4me3", "H3K27ac",
                            "H3K27me3", "H3K36me3", "H3K9me3", "CTCF")
    d_hidden: int = 128
    d_embed: int = 128
    gat_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.2
    attn_dropout: float = 0.1
    hyena_d_model: int = 128
    hyena_blocks: int = 4
    hyena_order: int = 2
    hyena_filter_order: int = 64
    stem_kernel: int = 15
    stem_stride: int = 64
    rc_equivariant: bool = True
    temperature: float = 0.1
    batch_size: int = 16
    num_workers: int = 8
    use_amp: bool = False

    @property
    def seq_len(self):
        return self.seq_len_fast if self.fast_mode else self.promoter_window

CFG = Config()

_model_src = open("/tmp/cell27_model.py").read()
_model_src = _model_src.split("rows = []")[0]   # drop the smoke-test/param-budget tail; not needed here
exec(_model_src)
print("model classes loaded")

# ============================================================== reconstruct gene-grouped 5-fold CV split
groups = META["gene"].values
gkf = GroupKFold(n_splits=5)
FOLDS = list(gkf.split(np.arange(len(GRAPHS)), groups=groups))
print(f"reconstructed {len(FOLDS)} gene-grouped folds (deterministic given the same gene order + seed)")


def make_loader(graphs, indices, bs=32):
    return DataLoader([graphs[i] for i in indices], batch_size=bs, shuffle=False)


@torch.no_grad()
def infer(model, loader):
    model.eval()
    store = {"expr_p": [], "expr_y": [], "tau_p": [], "tau_y": [],
             "tis_p": [], "tis_y": [], "row_idx": []}
    for batch in loader:
        batch = batch.to(DEVICE)
        z_seq, z_str, z_epi, z, h, attn = model.encode(batch, SEQ_BANK)
        logit_expr = model.head_expr(z).squeeze(-1)
        logit_tau = model.head_tau(z).squeeze(-1)
        logit_tissue = model.head_tissue(z)
        store["expr_p"].append(torch.sigmoid(logit_expr).float().cpu())
        store["expr_y"].append(batch.y_expr.float().cpu())
        store["tau_p"].append(torch.sigmoid(logit_tau).float().cpu())
        store["tau_y"].append(batch.y_tau.float().cpu())
        store["tis_p"].append(torch.softmax(logit_tissue, -1).float().cpu())
        store["tis_y"].append(batch.y_tissue.cpu())
        store["row_idx"].append(batch.row_idx.cpu())
    return {k: torch.cat(v).numpy() for k, v in store.items()}


def load_model_for_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = PangenomicGAT(CFG, ckpt["n_node_feat"], ckpt["n_edge_feat"],
                          len(ckpt["tissues"]), variant="all3").to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model


def stratum_frame(preds_list, meta):
    row_idx_all, expr_p, expr_y, tau_p, tau_y, tis_y, tis_p = [], [], [], [], [], [], []
    for pred in preds_list:
        row_idx_all.append(pred["row_idx"].astype(int))
        expr_p.append(pred["expr_p"]); expr_y.append(pred["expr_y"])
        tau_p.append(pred["tau_p"]); tau_y.append(pred["tau_y"])
        tis_y.append(pred["tis_y"]); tis_p.append(pred["tis_p"])
    df = pd.DataFrame({
        "row_idx": np.concatenate(row_idx_all),
        "expr_p": np.concatenate(expr_p), "expr_y": np.concatenate(expr_y),
        "tau_p": np.concatenate(tau_p), "tau_y": np.concatenate(tau_y),
        "tis_y": np.concatenate(tis_y),
    })
    tis_p_all = np.concatenate(tis_p, axis=0)
    gene_cols = meta[["gene", "tissue", "y_housekeeping", "y_coding"]].reset_index(drop=True)
    df = df.join(gene_cols, on="row_idx")
    return df, tis_p_all


CATEGORY_LEVELS = {
    "y_housekeeping": ("housekeeping", {1: "housekeeping", 0: "non-housekeeping"}),
    "y_coding": ("protein_coding", {1: "protein-coding", 0: "non-coding"}),
}
COMBO_LEVELS = {
    (1, 1): "housekeeping + protein-coding", (1, 0): "housekeeping + non-coding",
    (0, 1): "non-housekeeping + protein-coding", (0, 0): "non-housekeeping + non-coding",
}


def stratified_report(preds_list, meta, n_classes, scheme):
    df, tis_p_all = stratum_frame(preds_list, meta)

    def score(mask, group, level):
        mask = mask.values if hasattr(mask, "values") else mask
        if mask.sum() < 10:
            return None
        sub = df[mask]
        m_expr = binary_metrics(sub["expr_y"].values, sub["expr_p"].values, "expr_")
        m_tau = binary_metrics(sub["tau_y"].values, sub["tau_p"].values, "tau_")
        m_tis = multiclass_metrics(sub["tis_y"].values, tis_p_all[mask], n_classes)
        return {
            "scheme": scheme, "arm": "all3", "group": group, "level": level,
            "n_graphs": int(mask.sum()),
            "H2_auprc": m_expr.get("expr_auprc", np.nan),
            "H2_auprc_baseline": m_expr.get("expr_auprc_baseline", np.nan),
            "H2_accuracy": m_expr.get("expr_accuracy", np.nan),
            "H2_f1": m_expr.get("expr_f1", np.nan),
            "H3_auprc": m_tau.get("tau_auprc", np.nan),
            "H3_auprc_baseline": m_tau.get("tau_auprc_baseline", np.nan),
            "H3_accuracy": m_tau.get("tau_accuracy", np.nan),
            "H4_accuracy": m_tis.get("tissue_accuracy", np.nan),
            "H4_f1_macro": m_tis.get("tissue_f1_macro", np.nan),
        }

    rows = []
    for col, (group_name, level_names) in CATEGORY_LEVELS.items():
        for level, level_name in level_names.items():
            r = score(df[col] == level, group_name, level_name)
            if r is not None:
                rows.append(r)
    for (hk, cod), name in COMBO_LEVELS.items():
        mask = (df["y_housekeeping"] == hk) & (df["y_coding"] == cod)
        r = score(mask, "housekeeping_x_coding", name)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


# ============================================================== 1. gene-grouped CV (5 saved fold models)
print("\n" + "=" * 80 + "\nGENE-GROUPED CV -- loading 5 saved fold models, running inference only\n" + "=" * 80)
gene_cv_preds = []
t0 = time.time()
for k, (tr, te) in enumerate(FOLDS):
    ckpt_path = SRC_DIR / f"model_fold{k}.pt"
    model = load_model_for_ckpt(ckpt_path)
    pred = infer(model, make_loader(GRAPHS, te))
    gene_cv_preds.append(pred)
    print(f"  fold {k}: {len(te):,} test graphs, loaded {ckpt_path.name}")
    del model
print(f"gene-CV inference done in {time.time()-t0:.1f}s")

# ============================================================== 2. chromosome-block CV (5 saved fold models)
chrom_preds = []
chrom_dir = VSTRAT_DIR / "chrom_cv"
if chrom_dir.exists():
    print("\n" + "=" * 80 + "\nCHROMOSOME-BLOCK CV\n" + "=" * 80)

    def make_chrom_blocks(meta, n_blocks, seed=42):
        counts = meta["chrom"].value_counts()
        bins = [[] for _ in range(n_blocks)]
        bin_sizes = [0] * n_blocks
        for chrom, n in counts.items():
            j = int(np.argmin(bin_sizes))
            bins[j].append(chrom); bin_sizes[j] += n
        return bins

    chrom_blocks = make_chrom_blocks(META, 5, seed=42)
    chrom_folds = []
    for block in chrom_blocks:
        test_mask = META["chrom"].isin(block).values
        te = np.where(test_mask)[0]
        chrom_folds.append(te)
    for k, te in enumerate(chrom_folds):
        ckpt_path = chrom_dir / f"model_fold{k}.pt"
        if not ckpt_path.exists():
            print(f"  fold {k}: no checkpoint, skipped"); continue
        model = load_model_for_ckpt(ckpt_path)
        pred = infer(model, make_loader(GRAPHS, te))
        chrom_preds.append(pred)
        print(f"  fold {k}: {len(te):,} test graphs, loaded {ckpt_path.name}")
        del model
else:
    print("no chrom_cv checkpoints found, skipping")

# ============================================================== 3. leave-one-tissue-out (per-tissue saved models)
loto_preds = []
loto_dir = VSTRAT_DIR / "leave_one_tissue_out"
if loto_dir.exists():
    print("\n" + "=" * 80 + "\nLEAVE-ONE-TISSUE-OUT\n" + "=" * 80)
    for held_out in TISSUE_LIST:
        tag = "".join(c if c.isalnum() else "_" for c in held_out)
        ckpt_path = loto_dir / f"model_{tag}.pt"
        if not ckpt_path.exists():
            continue
        te = np.where(META["tissue"].values == held_out)[0]
        if len(te) == 0:
            continue
        model = load_model_for_ckpt(ckpt_path)
        pred = infer(model, make_loader(GRAPHS, te))
        loto_preds.append(pred)
        print(f"  {held_out}: {len(te):,} test graphs, loaded {ckpt_path.name}")
        del model
else:
    print("no LOTO checkpoints found, skipping")

# ============================================================== build the full performance matrix
print("\n" + "=" * 80 + "\nBuilding the full protein-coding x housekeeping performance matrix\n" + "=" * 80)
parts = [stratified_report(gene_cv_preds, META, len(TISSUE_LIST), "gene-grouped CV")]
if chrom_preds:
    parts.append(stratified_report(chrom_preds, META, len(TISSUE_LIST), "chromosome-block CV"))
if loto_preds:
    parts.append(stratified_report(loto_preds, META, len(TISSUE_LIST), "leave-one-tissue-out"))
MATRIX = pd.concat(parts, ignore_index=True)
MATRIX.to_csv(OUT_DIR / "performance_matrix_full.csv", index=False)
print(MATRIX.round(4).to_string(index=False))

# ---- the headline 2x2 combination table, gene-grouped CV
COMBO = MATRIX[(MATRIX.scheme == "gene-grouped CV") & (MATRIX.group == "housekeeping_x_coding")]
COMBO.to_csv(OUT_DIR / "combination_2x2_gene_cv.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
for ax, metric, base_col, title in [
        (axes[0], "H2_auprc", "H2_auprc_baseline", "H2 expressed-in-tissue (AUPRC)"),
        (axes[1], "H3_auprc", "H3_auprc_baseline", "H3 tissue-restricted / tau (AUPRC)"),
        (axes[2], "H4_accuracy", None, "H4 tissue classification (accuracy)"),
        (axes[3], "n_graphs", None, "group size")]:
    sub = COMBO.reset_index()
    ax.bar(sub["level"], sub[metric], color=["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"])
    if base_col:
        for i, b in enumerate(sub[base_col]):
            ax.plot([i - .4, i + .4], [b, b], "r--", lw=1.4)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=7, rotation=25)
fig.suptitle("Full housekeeping x protein-coding performance matrix (all3, gene-grouped CV)", y=1.04)
fig.tight_layout()
fig.savefig(FIG_DIR / "performance_matrix_2x2.png", bbox_inches="tight")
print(f"\nfigure -> {FIG_DIR / 'performance_matrix_2x2.png'}")

# ---- heatmap view: rows = metric, cols = the 4 combined groups
heat_metrics = ["H2_auprc", "H3_auprc", "H4_accuracy"]
heat = COMBO.set_index("level")[heat_metrics].T
fig2, ax2 = plt.subplots(figsize=(7.5, 3.2))
im = ax2.imshow(heat.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
ax2.set_xticks(range(len(heat.columns))); ax2.set_xticklabels(heat.columns, rotation=20, ha="right", fontsize=8)
ax2.set_yticks(range(len(heat.index))); ax2.set_yticklabels(heat.index, fontsize=9)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        ax2.text(j, i, f"{heat.values[i,j]:.3f}", ha="center", va="center",
                 color="white" if heat.values[i, j] < 0.6 else "black", fontsize=9)
fig2.colorbar(im, ax=ax2, label="score")
ax2.set_title("Performance heatmap: protein-coding x housekeeping (all3, gene-grouped CV)", fontsize=10)
fig2.tight_layout()
fig2.savefig(FIG_DIR / "performance_matrix_heatmap.png", bbox_inches="tight")
print(f"figure -> {FIG_DIR / 'performance_matrix_heatmap.png'}")

print(f"\nfull matrix (all schemes) -> {OUT_DIR / 'performance_matrix_full.csv'}")
print(f"2x2 combination (gene-CV)  -> {OUT_DIR / 'combination_2x2_gene_cv.csv'}")
print("\nDONE")
