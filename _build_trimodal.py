"""Build pangenomic_gat_pipeline_trimodal.ipynb: the PCHi-C-only pipeline
with epigenomic (histone/CTCF) state pulled out as its own third input
modality -- its own small MLP encoder + embedding (Tower C) -- aligned to
the sequence and structure towers with a 3-way InfoNCE loss and fused with
a variant-count-agnostic gated fusion, instead of the two-tower FiLM.

Same 16,384 bp window, same graphs, same everything else as
pangenomic_gat_pipeline_pchic_only.ipynb. Config-driven fast_mode (as in
the base notebook) controls scale; results go to
results_trimodal_fast/ or results_trimodal_full/ automatically.
"""
from pathlib import Path
from _nb_tools import load_base, replace_cell, drop_cells, append_cells, build_ipynb, find_cell_idx
import _nb_edits as E

ROOT = Path(__file__).parent
SCRATCH = Path("/tmp/claude-1009/-mnt-raid-ksengupta-pic-hic-ml/a229cc82-8991-40c9-bca5-7311abab7598/scratchpad")

cell26_new_tail = (SCRATCH / "cell26_new_tail.py").read_text()
run_epoch_new = (SCRATCH / "run_epoch_new.py").read_text()

BANNER = '''# Variant note — a third input modality: epigenomic state

This notebook is `pangenomic_gat_pipeline_pchic_only.ipynb` (same 16,384 bp
promoter window, same PCHi-C graphs, same labels) with one architectural
change: **epigenomic state (histone marks + CTCF) becomes its own input
modality**, with its own encoder and its own embedding, instead of being
visible to the model only as node features buried inside the structure
tower.

**What "Tower C" sees.** The promoter (bait/anchor) fragment's own
histone/CTCF summary — the same 14-dim signal+mask block already computed
for node 0 during graph construction — is attached to every graph as
`g.epi` and encoded by a small MLP (`EpiEncoder`) into a 128-dim embedding
`z_epi`, in the same shared space as `z_seq` and `z_str`.

**This does not remove histone/CTCF features from the structure tower.**
Tower B still sees histone/CTCF signal on every node (not just the
anchor), same as the base notebook. Tower B and Tower C therefore see
overlapping — not disjoint — information: Tower B's view is per-node,
whole-subgraph; Tower C's is anchor-only, summarised. That is a deliberate
scope decision (it needs no change to graph construction or its cache, so
this notebook reuses the base notebook's cached graphs unchanged), not an
oversight. The natural harder ablation — strip histone/CTCF from Tower B's
node features entirely so Tower B is pure topology and Tower C is the only
source of chromatin state — would need a new node-feature schema and a new
graph cache; it is a one-line change in the graph-building cell
(`_w_transform_hist(hv)` in the histone block) if you want to run it, but
is out of scope here.

**Alignment and fusion, extended to three.** The alignment loss becomes the
average InfoNCE over every *present* pair among `(z_seq, z_str, z_epi)` —
3 pairs for the full model, 1 pair for a two-tower arm, none for a
single-tower arm. Fusion becomes a `GatedFusion` module (an attention-style
gate over whichever modalities are present, plus a concat-and-mix MLP term)
that works for 1, 2, or 3 active towers, replacing the original's two-input
`FiLMFusion` — for the seq+str-only arm this is *not* bit-identical to the
original FiLM formula, an intentional, documented change.

**Ablation arms trained by default:** `all3` (all three), `str_epi`
(structure + epigenomics, no DNA), `seq_str` (the original two-tower
model, for a direct before/after comparison), `epi` (epigenomic state
alone). The model supports all 7 combinations
(`PangenomicGAT.VARIANTS`) — edit `CFG.variants` to run more.

Results go to `results_trimodal_fast/` or `results_trimodal_full/`
(chosen automatically from `CFG.fast_mode`, same convention as the base
notebook). This notebook does not add the chromosome/tissue/blood
validation strategies from `pangenomic_gat_pipeline_pchic_only_validations.ipynb`
— that Part III section is a straightforward drop-in if you want it (it
only calls `train_one_fold`, which is unchanged here).'''

cells = load_base()
cells = [["markdown", BANNER]] + cells

# ---------------------------------------------------------------------------
# 1. config: default variants -> the 4-arm trimodal set; results dir depends
#    on fast_mode (evaluated once at the top, same pattern as CACHE_DIR etc).
# ---------------------------------------------------------------------------
i_config = find_cell_idx(cells, "class Config:")
replace_cell(cells, i_config,
            'variants: tuple = ("both", "gat", "seq")   # which arms to train',
            'variants: tuple = ("all3", "str_epi", "seq_str", "epi")   # which arms to train')

# RESULTS_DIR depends on fast_mode, which is a CFG field -- CFG does not
# exist yet at the point the base notebook creates RESULTS_DIR/FIG_DIR, so
# defer both until right after `CFG = Config()` instead of creating (and
# leaving behind) an unused placeholder directory.
replace_cell(cells, i_config,
            'RESULTS_DIR    = PROJECT_ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True)\n'
            'FIG_DIR        = RESULTS_DIR / "figures";  FIG_DIR.mkdir(exist_ok=True)',
            '# RESULTS_DIR / FIG_DIR: created below, once CFG.fast_mode is known '
            '(trimodal notebook picks the folder name from it)')
replace_cell(cells, i_config, "CFG = Config()",
            'CFG = Config()\n'
            'RESULTS_DIR = PROJECT_ROOT / ("results_trimodal_fast" if CFG.fast_mode else "results_trimodal_full")\n'
            'RESULTS_DIR.mkdir(exist_ok=True)\n'
            'FIG_DIR = RESULTS_DIR / "figures"; FIG_DIR.mkdir(exist_ok=True)')

# ---------------------------------------------------------------------------
# 2. graph cache (same mechanism as the validations / 1 Mb notebooks)
# ---------------------------------------------------------------------------
i_build = find_cell_idx(cells, "GRAPHS, META, _hist_deltas = build_graphs_parallel(")
replace_cell(cells, i_build, E.CELL18_OLD_BUILD_CALL, E.CELL18_NEW_BUILD_CALL)

i_cache36 = find_cell_idx(cells, "cache/graphs_repaired.pt")
replace_cell(cells, i_cache36, E.CELL36_OLD, E.CELL36_NEW)

# drop the two broken interactive-debug cells
i_diag1 = find_cell_idx(cells, "FIND THE 9 GRAPHS")
i_diag2 = find_cell_idx(cells, "non-finite content of the fold that failed")
drop_cells(cells, sorted({i_diag1, i_diag2}))

# ---------------------------------------------------------------------------
# 3. attach the epigenomic modality to every graph, right after the cache
#    is refreshed (post sanitise/repair) and before the model is defined.
# ---------------------------------------------------------------------------
i_cache36 = find_cell_idx(cells, "cache refreshed (post sanitise/repair)")
EPI_ATTACH = '''# Tower C input: attach the promoter (anchor / node 0) histone+mask block
# already computed during graph construction (node feature columns
# 4 : 4 + 2*n_marks) to every graph as its own attribute. PyG's DataLoader
# batches a per-graph attribute whose name does not contain "index" by
# concatenating it on dim 0 (the same mechanism seq_idx already relies on),
# so batch.epi comes out as (batch_size, 2*n_marks) with no further code
# needed anywhere else.
EPI_DIM = 2 * len(CFG.histone_marks)
for _g in GRAPHS:
    _g.epi = _g.x[0, 4:4 + EPI_DIM].clone().unsqueeze(0)
print(f"epigenomic modality attached: {EPI_DIM}-dim vector per graph "
      f"(promoter-fragment histone/CTCF signal + measurement mask)")
print(f"  example: gene={META.iloc[0]['gene']}  tissue={META.iloc[0]['tissue']}")
print(f"  epi = {GRAPHS[0].epi.numpy().round(3)}")'''
cells.insert(i_cache36 + 1, ["code", EPI_ATTACH])

# ---------------------------------------------------------------------------
# 4. the model: replace FiLMFusion..end-of-cell with GatedFusion + EpiEncoder
#    + the 3-modality PangenomicGAT + multi_align_loss.
# ---------------------------------------------------------------------------
i_model = find_cell_idx(cells, "class StructureEncoder")

# locate the exact FiLMFusion..EOF tail inside the model cell and swap it
_old_tail_marker = "class FiLMFusion(nn.Module):"
assert cells[i_model][1].count(_old_tail_marker) == 1
_head, _tail = cells[i_model][1].split(_old_tail_marker, 1)
cells[i_model][1] = _head + cell26_new_tail

# ---------------------------------------------------------------------------
# 5. run_epoch: full replacement (3-way align, z_epi bookkeeping)
# ---------------------------------------------------------------------------
i_run_epoch = find_cell_idx(cells, "def run_epoch(model, loader, seq_bank, cfg")
_old_run_epoch_marker = "def make_loader(graphs, indices, bs, shuffle):"
assert cells[i_run_epoch][1].count(_old_run_epoch_marker) == 1
_head2, _tail2 = cells[i_run_epoch][1].split(_old_run_epoch_marker, 1)
cells[i_run_epoch][1] = run_epoch_new + _old_run_epoch_marker + _tail2

# ---------------------------------------------------------------------------
# 6. main CV loop: VARIANT_LABEL for the 7 possible arms; PRIMARY = "all3"
# ---------------------------------------------------------------------------
i_main = find_cell_idx(cells, "def train_one_fold(fold, train_idx")
replace_cell(cells, i_main,
            'VARIANT_LABEL = {"both": "GAT + sequence", "gat": "GAT only", "seq": "sequence only"}',
            'VARIANT_LABEL = {\n'
            '    "all3": "seq + str + epi", "seq_str": "seq + str (no epi)",\n'
            '    "seq_epi": "seq + epi (no str)", "str_epi": "str + epi (no seq)",\n'
            '    "seq": "sequence only", "str": "structure only", "epi": "epigenomic only",\n'
            '}')
replace_cell(cells, i_main,
            'PRIMARY = "both" if "both" in RESULTS else CFG.variants[0]',
            'PRIMARY = "all3" if "all3" in RESULTS else CFG.variants[0]')

# ---------------------------------------------------------------------------
# 7. embedding-space plot: add a z_epi panel (dynamic panel count)
# ---------------------------------------------------------------------------
i_embed = find_cell_idx(cells, "last_pred = fold_preds[-1]")
cells[i_embed][1] = '''from sklearn.decomposition import PCA
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
    ax.set_title(f"{name}\\n{how}, coloured by tissue\\n"
                 f"silhouette={sils[name]:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

ax = axes[len(spaces)]
names = list(sils.keys()); vals = [sils[n] for n in names]
ax.barh([n.split("(")[0].strip() for n in names], vals, color=PALETTE[:len(names)])
ax.axvline(0, c="k", lw=1)
ax.set_title("Tissue silhouette by space\\n(higher = more tissue structure)", fontsize=9)
ax.set_xlabel("silhouette score")

fig.tight_layout(); savefig(fig, "12_embedding_spaces"); plt.show()

print("\\nsilhouette against tissue labels:")
for n, v in sils.items():
    print(f"  {n:<32} {v:.4f}")
print("\\nz_seq should stay tissue-blind (it never receives tissue information);")
print("if it scores as high as the tissue-aware towers, tissue information is")
print("leaking into it. z_epi and z_str are both tissue-aware by construction")
print("(chromatin state and contact topology are both tissue-specific), so a")
print("similar silhouette between them is expected, not a red flag on its own.")'''

# ---------------------------------------------------------------------------
# 8. export: add z_epi to the embeddings parquet, guarding for arms without it
# ---------------------------------------------------------------------------
i_export = find_cell_idx(cells, "import platform as _pf")
replace_cell(cells, i_export,
            'for name, arr in [("z", last_pred["z"]), ("z_seq", last_pred["z_seq"]),\n'
            '                  ("z_str", last_pred["z_str"])]:\n'
            '    a = np.asarray(arr)[:n_emb]\n'
            '    parts.append(pd.DataFrame(a, columns=[f"{name}_{j}" for j in range(a.shape[1])]))',
            'for name, arr in [("z", last_pred["z"]), ("z_seq", last_pred["z_seq"]),\n'
            '                  ("z_str", last_pred["z_str"]), ("z_epi", last_pred["z_epi"])]:\n'
            '    a = np.asarray(arr)\n'
            '    if a.size == 0:\n'
            '        continue\n'
            '    a = a[:n_emb]\n'
            '    parts.append(pd.DataFrame(a, columns=[f"{name}_{j}" for j in range(a.shape[1])]))')

# ---------------------------------------------------------------------------
# 9. ablation headline: "both" (the old primary-arm key) -> "all3"
# ---------------------------------------------------------------------------
i_abl = find_cell_idx(cells, 'ABL = pd.DataFrame([m for v in RESULTS')
replace_cell(cells, i_abl, '"both"', '"all3"', expect=4)

# ---------------------------------------------------------------------------
# 10. ablation plots: ARM_COLOR generic over CFG.variants (was hardcoded to
#     exactly {"both","gat","seq"})
# ---------------------------------------------------------------------------
i_ablplot = find_cell_idx(cells, 'ARM_COLOR = {"both": PALETTE[0]')
replace_cell(cells, i_ablplot,
            'ARM_COLOR = {"both": PALETTE[0], "gat": PALETTE[1], "seq": PALETTE[2]}',
            'ARM_COLOR = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(CFG.variants)}')
replace_cell(cells, i_ablplot,
            'fig.suptitle("Ablation: GAT only vs sequence only vs both", y=1.04, fontsize=13)',
            'fig.suptitle(f"Ablation: {list(CFG.variants)}", y=1.04, fontsize=13)')

# ---------------------------------------------------------------------------
# 11. DHL hybrid: encode() now returns a 6-tuple, not 5
# ---------------------------------------------------------------------------
i_dhl = find_cell_idx(cells, "from sklearn.linear_model import LogisticRegression")
replace_cell(cells, i_dhl,
            "_, _, z, _, _ = model.encode(batch, seq_bank)",
            "_, _, _, z, _, _ = model.encode(batch, seq_bank)")

build_ipynb(cells, ROOT / "pangenomic_gat_pipeline_trimodal.ipynb",
           title="Pangenomic AI - PCHi-C only, trimodal (seq + structure + epigenomic)")
