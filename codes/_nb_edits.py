"""Reusable source-text edits and new cells shared by the derivative
notebooks (graph caching + Part III validation strategies)."""

# ---------------------------------------------------------------------------
# Edit applied inside the base notebook's cell 18 (graph construction): wrap
# the expensive build_graphs_parallel(...) call in a disk-cache check keyed
# on every setting that changes what the graphs contain, so a second run (or
# a second notebook using the same window/gene/tissue selection) loads
# straight from disk instead of rebuilding ~350k graphs.
# ---------------------------------------------------------------------------
CELL18_OLD_BUILD_CALL = '''GRAPHS, META, _hist_deltas = build_graphs_parallel(
    loops, GENE_INFO, GENE2SEQ, hist_map, CFG.histone_marks, CFG.histone_stat,
    REF_DIR, CFG.genome_build, sel["sample_id"].tolist(),
    HIST_CLIP, HIST_MEAN, HIST_STD,
    CFG.max_neighbours, CFG.add_backbone_edges, CFG.link_holdout_frac, CFG.seed,
    N_WORKERS)

N_NODE_FEAT = GRAPHS[0].x.shape[1]'''

CELL18_NEW_BUILD_CALL = '''# ---- disk cache: rebuilding ~350k graphs is the single most expensive step
# in this notebook, so key a cache file on every setting that changes what a
# graph contains and skip straight to loading it on a second run.
_graph_cache_key = "_".join(str(v) for v in [
    CFG.genome_build, CFG.seq_len, len(genes_needed), len(sel), CFG.max_neighbours,
    CFG.min_chicago, CFG.add_backbone_edges, CFG.link_holdout_frac, CFG.seed])
GRAPH_CACHE_PATH = CACHE_DIR / f"graphs_{_graph_cache_key}.pt"

if GRAPH_CACHE_PATH.exists():
    print(f"loading cached graphs -> {GRAPH_CACHE_PATH.name}")
    _cached = torch.load(GRAPH_CACHE_PATH, weights_only=False)
    GRAPHS, META, _hist_deltas = _cached["graphs"], _cached["meta"], _cached["hist_deltas"]
    print(f"  {len(GRAPHS):,} graphs loaded from cache (rebuild skipped)")
else:
    GRAPHS, META, _hist_deltas = build_graphs_parallel(
        loops, GENE_INFO, GENE2SEQ, hist_map, CFG.histone_marks, CFG.histone_stat,
        REF_DIR, CFG.genome_build, sel["sample_id"].tolist(),
        HIST_CLIP, HIST_MEAN, HIST_STD,
        CFG.max_neighbours, CFG.add_backbone_edges, CFG.link_holdout_frac, CFG.seed,
        N_WORKERS)
    torch.save({"graphs": GRAPHS, "meta": META, "hist_deltas": _hist_deltas},
               GRAPH_CACHE_PATH)
    print(f"cached graphs -> {GRAPH_CACHE_PATH.name}  "
          f"({GRAPH_CACHE_PATH.stat().st_size/1e6:.0f} MB)")

N_NODE_FEAT = GRAPHS[0].x.shape[1]'''


# ---------------------------------------------------------------------------
# Cell 36 in the base notebook ("cache/graphs_repaired.pt", a one-off from a
# previous debugging session) is superseded by the general GRAPH_CACHE_PATH
# mechanism above -- replace it with a cache re-save so the *sanitised and
# repaired* tensors (cells 34/35, kept as-is) are what gets reloaded next time.
# ---------------------------------------------------------------------------
CELL36_OLD = '''import inspect
print("NaN-safe" if "n_nonfinite_dropped" in inspect.getsource(binary_metrics)
      else "*** ORIGINAL — re-run the NaN-safe metrics cell first ***")

torch.save({"graphs": GRAPHS, "meta": META}, CACHE_DIR / "graphs_repaired.pt")
print("saved ->", CACHE_DIR / "graphs_repaired.pt")'''

CELL36_NEW = '''# Re-save the cache now that the sanitise/repair patches above have run, so
# a reload from GRAPH_CACHE_PATH (cell 9) always yields clean tensors -- the
# patches are idempotent (re-applying them to already-clean graphs is a
# no-op) but there is no reason to redo even that work on every rerun.
torch.save({"graphs": GRAPHS, "meta": META, "hist_deltas": _hist_deltas},
           GRAPH_CACHE_PATH)
print(f"cache refreshed (post sanitise/repair) -> {GRAPH_CACHE_PATH.name}")'''


# ---------------------------------------------------------------------------
# Part III -- alternative validation strategies. Appended after the base
# notebook's own final cell. Reuses everything already defined earlier in
# the SAME run: GRAPHS, META, SEQ_BANK, CFG, DEVICE, TISSUE_LIST,
# train_one_fold, all_metrics, savefig, PALETTE, RESULTS, MET.
# ---------------------------------------------------------------------------
PART3_MD_INTRO = '''# Part III — Alternative validation strategies

Everything above validates by **gene**: `GroupKFold` on gene symbol, so no
gene's promoter sequence or per-gene labels ever cross the train/test
boundary. That answers "does this generalise to unseen genes". It does not
answer three other, equally real, generalisation questions:

1. **Unseen chromosomes.** A gene-grouped fold can still put two genes that
   are neighbours on the same chromosome — and therefore share regional
   chromatin state, replication timing, TAD membership — on opposite sides
   of the split. **Chromosome-block CV** removes that: every fold's test
   genes come from chromosomes entirely absent from that fold's training
   data.
2. **Unseen tissues.** Gene-grouped CV always trains on all 46 tissues; it
   never asks whether the model has learned something tissue-transferable
   or has just memorised per-tissue quirks. **Leave-one-tissue-out (LOTO)**
   holds out one tissue's graphs entirely and trains on the rest.
3. **A named, biologically motivated holdout.** Blood cell types are the
   best-represented lineage in this PCHi-C compendium (Javierre et al.'s
   original 17 blood cell types are the historical core of these loop
   calls). **Blood-as-test** trains on every non-blood tissue and asks how
   well that transfers to blood specifically — the single split a reviewer
   is most likely to ask for by name.

**A caveat that applies to (2) and (3), not (1):** H3 (tissue-restriction,
$\\tau$) is a **per-gene** label, constant across a gene's tissues. Under
gene-grouped CV that is harmless because genes never cross the split. Under
a *tissue*-holdout split, the same gene can appear in both train (via its
non-held-out tissues) and test (via the held-out tissue) — so the model can
in principle have already seen that gene's exact $\\tau$ answer through
another tissue's graph. H3 numbers under LOTO / blood-holdout should
therefore be read as an **upper bound**, not a clean tissue-generalisation
test, for that head specifically. H1 (link) and H2 (expression, a genuinely
per-gene-per-tissue label) do not have this problem.

**H4 (tissue class) is not meaningful under LOTO.** A model trained with
one tissue's graphs entirely removed has never seen that class as a
training target; scoring it is scoring an untrained output unit. H4 is
reported for completeness (and will be near zero for the held-out class)
but should not be interpreted as a tissue-classification result.

All three schemes reuse the *already-built* `GRAPHS` / `META` and the
already-trained-on `train_one_fold` / `all_metrics` from earlier in this
notebook — no graphs are rebuilt, and no code is duplicated. Results are
written to `RESULTS_DIR / "validation_strategies"`, separate from the
gene-grouped-CV results saved earlier, so nothing already on disk is
overwritten.'''

PART3_CODE_SETUP = '''VSTRAT_DIR = RESULTS_DIR / "validation_strategies"
VSTRAT_DIR.mkdir(exist_ok=True)
(VSTRAT_DIR / "chrom_cv").mkdir(exist_ok=True)
(VSTRAT_DIR / "leave_one_tissue_out").mkdir(exist_ok=True)
(VSTRAT_DIR / "blood_holdout").mkdir(exist_ok=True)

def _save_scheme_fold(subdir, tag, pred, met, hist, model):
    out = VSTRAT_DIR / subdir
    np.savez_compressed(out / f"predictions_{tag}.npz",
                        **{n: pred[n] for n in ("link_p", "link_y", "expr_p", "expr_y",
                                                "tau_p", "tau_y", "tis_p", "tis_y")})
    hist.to_csv(out / f"history_{tag}.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "config": asdict(CFG),
               "n_node_feat": N_NODE_FEAT, "n_edge_feat": N_EDGE_FEAT,
               "tissues": TISSUE_LIST}, out / f"model_{tag}.pt")
    return met

print(f"validation-strategy outputs -> {VSTRAT_DIR.relative_to(PROJECT_ROOT)}")'''

PART3_MD_CHROM = '''## III.1 · Chromosome-block cross-validation

Chromosomes are greedily binned into `CFG.n_folds` groups so total row
count per fold stays balanced (the same greedy strategy `GroupKFold` uses
internally, applied explicitly here so every fold really is a *block of
whole chromosomes*, not a `GroupKFold` mix that happens to correlate with
chromosome). Every fold's test genes come from chromosomes that contributed
**zero** training rows.'''

PART3_CODE_CHROM = '''def make_chrom_blocks(meta, n_blocks, seed=42):
    counts = meta["chrom"].value_counts()          # descending by row count
    bins = [[] for _ in range(n_blocks)]
    bin_sizes = [0] * n_blocks
    for chrom, n in counts.items():                # largest chromosome first
        j = int(np.argmin(bin_sizes))               # always fill the lightest bin
        bins[j].append(chrom); bin_sizes[j] += n
    return bins

CHROM_BLOCKS = make_chrom_blocks(META, CFG.n_folds, seed=CFG.seed)
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
print("\\n  OK: no chromosome appears in both train and test of any fold")'''

PART3_CODE_CHROM_TRAIN = '''CHROM_CV_METRICS = []
t0 = time.time()
for k, (tr, te) in enumerate(CHROM_FOLDS):
    print(f"\\n{'='*74}\\nchrom-block CV | FOLD {k+1}/{len(CHROM_FOLDS)}   "
          f"train={len(tr):,}  test={len(te):,}\\n{'='*74}")
    model, pred, hist, best_ep = train_one_fold(
        k, tr, te, GRAPHS, SEQ_BANK, CFG, DEVICE, variant="both")
    met = all_metrics(pred, len(TISSUE_LIST))
    met["fold"] = k; met["best_epoch"] = best_ep
    met["n_chroms_test"] = META.iloc[te]["chrom"].nunique()
    _save_scheme_fold("chrom_cv", f"fold{k}", pred, met, hist, model)
    CHROM_CV_METRICS.append(met)
    print(f"  fold {k}:  link AUPRC {met.get('link_auprc', np.nan):.3f} | "
          f"expr AUPRC {met.get('expr_auprc', np.nan):.3f} | "
          f"tau AUPRC {met.get('tau_auprc', np.nan):.3f} | "
          f"tissue acc {met.get('tissue_accuracy', np.nan):.3f}")

CHROM_CV_DF = pd.DataFrame(CHROM_CV_METRICS)
CHROM_CV_DF.to_csv(VSTRAT_DIR / "chrom_cv" / "metrics_per_fold.csv", index=False)
print(f"\\nchrom-block CV done in {(time.time()-t0)/60:.1f} min -> "
      f"validation_strategies/chrom_cv/metrics_per_fold.csv")'''

PART3_MD_LOTO = '''## III.2 · Leave-one-tissue-out cross-validation

For every tissue in `LOTO_TISSUES`, train on every *other* tissue's graphs
and test on that tissue alone. In `fast_mode` this is already cheap
(`TISSUE_LIST` only has `CFG.fast_n_tissues` entries); in a full run it is
one full training run **per tissue** — edit `LOTO_TISSUES` below to a
subset first if that is more compute than you want to spend.'''

PART3_CODE_LOTO = '''LOTO_TISSUES = list(TISSUE_LIST)   # <- trim this list to run a subset
print(f"leave-one-tissue-out: {len(LOTO_TISSUES)} tissue(s) -> "
      f"{len(LOTO_TISSUES)} training run(s)")

LOTO_METRICS = []
t0 = time.time()
for i, held_out in enumerate(LOTO_TISSUES):
    te = np.where(META["tissue"].values == held_out)[0]
    tr = np.where(META["tissue"].values != held_out)[0]
    if len(te) == 0 or len(tr) == 0:
        print(f"  [{i+1}/{len(LOTO_TISSUES)}] {held_out}: skipped (no graphs on one side)")
        continue
    print(f"\\n{'='*74}\\nLOTO | {held_out}  ({i+1}/{len(LOTO_TISSUES)})   "
          f"train={len(tr):,}  test={len(te):,}\\n{'='*74}")
    model, pred, hist, best_ep = train_one_fold(
        i, tr, te, GRAPHS, SEQ_BANK, CFG, DEVICE, variant="both", quiet=True)
    met = all_metrics(pred, len(TISSUE_LIST))
    met["held_out_tissue"] = held_out; met["best_epoch"] = best_ep
    tag = "".join(c if c.isalnum() else "_" for c in held_out)
    _save_scheme_fold("leave_one_tissue_out", tag, pred, met, hist, model)
    LOTO_METRICS.append(met)
    print(f"  {held_out:<32} link AUPRC {met.get('link_auprc', np.nan):.3f} | "
          f"expr AUPRC {met.get('expr_auprc', np.nan):.3f} | "
          f"tau AUPRC {met.get('tau_auprc', np.nan):.3f}   "
          f"(H4 not meaningful for the held-out class)")

LOTO_DF = pd.DataFrame(LOTO_METRICS)
LOTO_DF.to_csv(VSTRAT_DIR / "leave_one_tissue_out" / "metrics_per_tissue.csv", index=False)
print(f"\\nLOTO done in {(time.time()-t0)/60:.1f} min -> "
      f"validation_strategies/leave_one_tissue_out/metrics_per_tissue.csv")'''

PART3_MD_BLOOD = '''## III.3 · Named holdout — blood as the test lineage

A single split, not a K-fold: every tissue whose name contains "Blood" is
held out as test; every other tissue trains the model. Blood cell types are
the best-represented lineage in this PCHi-C compendium, so this is the
holdout most likely to be asked for by name and the one with the most
training tissues on the other side of the split.'''

PART3_CODE_BLOOD = '''BLOOD_TISSUES = [t for t in TISSUE_LIST if "blood" in t.lower()]
print(f"held-out (test) tissues: {BLOOD_TISSUES}")

if len(BLOOD_TISSUES) == 0:
    # Expected under fast_mode: TISSUE_LIST is a random subsample of the 46
    # tissues, and it can miss "Blood" entirely. Skip rather than hard-fail
    # the whole run over an optional, named holdout -- CFG.fast_mode = False
    # (or a TISSUE_LIST that happens to include a Blood tissue) always runs it.
    print("no tissue name contains 'Blood' in this run's TISSUE_LIST -- "
          "skipping the blood-as-test holdout for this run "
          "(expected when CFG.fast_mode subsamples tissues).")
    BLOOD_METRICS = None
else:
    te = np.where(META["tissue"].isin(BLOOD_TISSUES).values)[0]
    tr = np.where(~META["tissue"].isin(BLOOD_TISSUES).values)[0]
    print(f"train (non-blood): {len(tr):,} graphs, "
          f"{META.iloc[tr]['tissue'].nunique()} tissues, {META.iloc[tr]['gene'].nunique():,} genes")
    print(f"test  (blood)    : {len(te):,} graphs, "
          f"{META.iloc[te]['tissue'].nunique()} tissues, {META.iloc[te]['gene'].nunique():,} genes")
    gene_overlap = len(set(META.iloc[tr]["gene"]) & set(META.iloc[te]["gene"]))
    print(f"genes appearing on both sides (expected -- this is a tissue split, "
          f"not a gene split): {gene_overlap:,}")

    t0 = time.time()
    model, pred, hist, best_ep = train_one_fold(
        0, tr, te, GRAPHS, SEQ_BANK, CFG, DEVICE, variant="both")
    BLOOD_METRICS = all_metrics(pred, len(TISSUE_LIST))
    BLOOD_METRICS["best_epoch"] = best_ep
    _save_scheme_fold("blood_holdout", "test", pred, BLOOD_METRICS, hist, model)
    pd.DataFrame([BLOOD_METRICS]).to_csv(
        VSTRAT_DIR / "blood_holdout" / "metrics.csv", index=False)
    print(f"\\nblood holdout done in {(time.time()-t0)/60:.1f} min:")
    print(f"  link AUPRC {BLOOD_METRICS.get('link_auprc', np.nan):.3f} | "
          f"expr AUPRC {BLOOD_METRICS.get('expr_auprc', np.nan):.3f} | "
          f"tau AUPRC {BLOOD_METRICS.get('tau_auprc', np.nan):.3f} | "
          f"tissue acc {BLOOD_METRICS.get('tissue_accuracy', np.nan):.3f}"
          "  (tissue acc is over blood sub-classes only, all seen in training)")'''

PART3_MD_COMPARE = '''## III.4 · Comparing every validation strategy

Same metric code, same model, four different ideas about what
"generalises" means. The gap between gene-grouped CV and the others is the
honest answer to "how much of the headline number was generalisation
across genes specifically, versus generalisation the model does not
actually have."'''

PART3_CODE_COMPARE = '''gene_cv_metrics = pd.DataFrame(RESULTS["both"]["metrics"]) if "RESULTS" in globals() else None

def _mean_std(df, col):
    if df is None or col not in df or df[col].dropna().empty:
        return np.nan, np.nan
    return float(df[col].mean()), float(df[col].std())

rows = []
for scheme, df in [
    ("gene-grouped CV (main)", gene_cv_metrics),
    ("chromosome-block CV",     CHROM_CV_DF if len(CHROM_CV_DF) else None),
    ("leave-one-tissue-out",    LOTO_DF if len(LOTO_DF) else None),
]:
    row = {"scheme": scheme}
    for metric in ("link_auprc", "expr_auprc", "tau_auprc", "tissue_accuracy"):
        m, s = _mean_std(df, metric)
        row[f"{metric}_mean"] = m; row[f"{metric}_std"] = s
    rows.append(row)

blood_row = {"scheme": "blood-as-test (single split)"}
for metric in ("link_auprc", "expr_auprc", "tau_auprc", "tissue_accuracy"):
    blood_row[f"{metric}_mean"] = (BLOOD_METRICS.get(metric, np.nan)
                                    if "BLOOD_METRICS" in globals() and BLOOD_METRICS is not None
                                    else np.nan)
    blood_row[f"{metric}_std"] = 0.0
rows.append(blood_row)

COMPARE_DF = pd.DataFrame(rows)
COMPARE_DF.to_csv(VSTRAT_DIR / "comparison_summary.csv", index=False)
print(COMPARE_DF.to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, metric, title in zip(
        axes,
        ("link_auprc", "expr_auprc", "tau_auprc", "tissue_accuracy"),
        ("H1 link AUPRC", "H2 expression AUPRC", "H3 tau AUPRC", "H4 tissue accuracy")):
    means = COMPARE_DF[f"{metric}_mean"].values
    stds  = COMPARE_DF[f"{metric}_std"].values
    x = np.arange(len(COMPARE_DF))
    ax.bar(x, means, yerr=np.nan_to_num(stds), capsize=4, color=PALETTE[:len(x)])
    ax.set_xticks(x); ax.set_xticklabels(COMPARE_DF["scheme"], rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    if metric == "tissue_accuracy":
        ax.axhspan(0, 0, color="none")  # placeholder, LOTO bar for this metric is not meaningful
fig.suptitle("Same model and metrics, four validation strategies "
            "(H4 under LOTO is not a trained-class result -- ignore that bar)")
fig.tight_layout(); savefig(fig, "20_validation_strategy_comparison"); plt.show()

print("\\nsaved -> validation_strategies/comparison_summary.csv and "
      "figures/20_validation_strategy_comparison.png")'''
