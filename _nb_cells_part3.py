# Cells 28-33: three-arm ablation (GAT / sequence / both) + DHL-style hybrid heads
CELLS = []
def MD(s):   CELLS.append(("markdown", s.strip("\n")))
def CODE(s): CELLS.append(("code",     s.strip("\n")))

# =============================================================================
MD(r"""
# Part II — Ablation: what is each tower actually contributing?

Everything above reports the full model. That is not evidence that either tower
earns its place. This section trains three arms through **identical** folds,
heads, losses and metric code, differing only in which encoder is live:

| arm | `z(g,t)` | what it can see |
|---|---|---|
| **GAT only** | `z_str` | contact topology + histone state. No DNA. |
| **sequence only** | `z_seq` | 16 kb of promoter DNA. No graph, no tissue. |
| **GAT + sequence** | `FiLM(z_seq, z_str)` | both, fused |

Because the arm is a constructor flag on one class, any difference in the
numbers comes from the towers rather than from incidental differences in
training code.

### Two things to predict before you look

1. **The sequence-only arm cannot do link prediction.** H1 scores *node pairs*,
   and without the graph tower there are no node states to score. That row is
   reported as `n/a`, not as zero — a zero would read as "it tried and failed".

2. **The sequence-only arm has no tissue information at all.** `z_seq` is one
   vector per gene, identical across all tissues. So for H2 (expressed in
   tissue) and H4 (which tissue) its ceiling is the *marginal* — the best it can
   do is learn each gene's average behaviour across tissues. If it scores well
   on H2 anyway, that tells you H2 is largely a gene-level property rather than
   a tissue-level one, which is a genuinely useful negative result.

The interesting comparison is **both vs. the better single tower**. If fusion
does not beat max(GAT, sequence) by more than the fold-to-fold spread, the extra
tower is costing parameters and buying nothing.
""")

CODE(r"""
ABL = pd.DataFrame([m for v in RESULTS for m in RESULTS[v]["metrics"]])
ABL.to_csv(RESULTS_DIR / "ablation_per_fold.csv", index=False)

HEAD_SPECS = [
    ("H1  link prediction",       "link_",   "auprc",    "auroc",   "f1"),
    ("H2  expressed in tissue",   "expr_",   "auprc",    "auroc",   "f1"),
    ("H3  tissue-restricted",     "tau_",    "auprc",    "auroc",   "f1"),
    ("H4  tissue classification", "tissue_", "auprc_macro_ovr",
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
MC_REPORT = ["auprc_macro_ovr", "auroc_macro_ovr", "accuracy", "balanced_accuracy",
             "precision_macro", "recall_macro", "f1_macro", "kappa"]

for title, pref, *_ in HEAD_SPECS:
    keys = MC_REPORT if pref == "tissue_" else BIN_REPORT
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
    if pref != "tissue_":
        b = ABL[f"{pref}auprc_baseline"]
        row["random baseline"] = f"{b.mean():.3f}" if b.notna().any() else "-"
    else:
        row["random baseline"] = f"{1/len(TISSUE_LIST):.3f}"
    head_rows.append(row)
HEADLINE = pd.DataFrame(head_rows)
print(HEADLINE.to_string(index=False))
HEADLINE.to_csv(RESULTS_DIR / "ablation_headline.csv", index=False)

# ---------------------------------------------------------------- is fusion real?
print(f"\n\n{'='*82}\nDoes fusion beat the better single tower?\n{'='*82}")
if "both" in RESULTS and len(CFG.variants) > 1:
    verdict = []
    for title, pref, main, *_ in HEAD_SPECS:
        col = f"{pref}{main}"
        if col not in ABL.columns:
            continue
        both = ABL.loc[ABL.variant == "both", col].dropna()
        singles = {v: ABL.loc[ABL.variant == v, col].dropna()
                   for v in CFG.variants if v != "both"}
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
            "both": round(both.mean(), 4),
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
    print("  (need the 'both' arm plus at least one single-tower arm)")
""")

# =============================================================================
MD(r"""
## Ablation plots

Four views of the same comparison:

1. **Grouped bars per head** — the primary metric for each arm with fold error
   bars, and a dashed line at the random baseline. Bars that sit near that line
   mean the head learned nothing regardless of arm.
2. **Per-fold paired lines** — every fold drawn as one line across the three
   arms. Consistently sloping lines mean a real effect; crossing lines mean the
   difference is noise.
3. **ROC and PR overlays** — all three arms on shared axes for H2, the head
   where the towers are most directly comparable.
4. **Training curves** — how fast each arm converged, which tells you whether a
   weaker arm is genuinely weaker or just under-trained.
""")

CODE(r"""
ARMS = [v for v in CFG.variants]
ARM_COLOR = {"both": PALETTE[0], "gat": PALETTE[1], "seq": PALETTE[2]}

# ---------------------------------------------------------------- 1. bars
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
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
    base = (ABL[f"{pref}auprc_baseline"].mean() if pref != "tissue_"
            else 1 / len(TISSUE_LIST))
    ax.axhline(base, ls="--", c="r", lw=1.2, label=f"random {base:.3f}")
    for i, (m, s) in enumerate(zip(mus, sds)):
        ax.text(i, m + s + .015, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_title(f"{title}\n({main})", fontsize=9)
    ax.set_ylim(0, min(1.05, max(mus) + max(sds) + .12))
    ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=20, labelsize=8)
fig.suptitle("Ablation: GAT only vs sequence only vs both", y=1.04, fontsize=13)
fig.tight_layout(); savefig(fig, "14_ablation_bars"); plt.show()

# ---------------------------------------------------------------- 2. paired folds
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
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
for ax, key, name in [(axes[0], "expr", "H2 expressed"), (axes[1], "tau", "H3 restricted"),
                      (axes[2], "link", "H1 link")]:
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
fig.suptitle("ROC by arm (pooled over folds)", y=1.02, fontsize=12)
fig.tight_layout(); savefig(fig, "16_ablation_roc"); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, key, name in [(axes[0], "expr", "H2 expressed"), (axes[1], "tau", "H3 restricted"),
                      (axes[2], "link", "H1 link")]:
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
fig.suptitle("Precision-recall by arm (read against the red baseline)",
             y=1.02, fontsize=12)
fig.tight_layout(); savefig(fig, "17_ablation_pr"); plt.show()

# ---------------------------------------------------------------- 4. convergence
ALLHIST = pd.concat([RESULTS[v]["hist"] for v in ARMS], ignore_index=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, comp in zip(axes, ["loss", "expr", "tissue"]):
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
""")

# =============================================================================
MD(r"""
## Deep Hybrid Learning: classical heads on the learned embeddings

Following [SFGLab/DHL](https://github.com/SFGLab/DHL), which pairs a deep
sequence model (DNABERT there, our Hyena tower here) with classical learners —
SVM, random forest, k-NN — rather than relying on the network's own softmax.

**Why this is not redundant with the neural head.** The deep towers are trained
to minimise a *multi-task* objective: their embedding is shaped by four heads
and a contrastive term at once. A classifier fitted afterwards on the frozen
embedding optimises one task with a different inductive bias — margins for SVM,
axis-aligned partitions for trees. On a few thousand examples with a 128-d
embedding, a gradient-boosted tree ensemble frequently beats a linear head that
had to share its trunk with three other objectives.

**Protocol, and the part that is easy to get wrong.** For each fold we freeze
the model *trained on that fold's training set*, embed both splits, fit the
classical models on train embeddings only, and score the test split. The
embeddings for the test fold therefore come from a model that never saw them.
Fitting the scaler or the classifier on all embeddings at once would leak, and
would show up as a suspiciously large hybrid gain.

We report:

- the **deep head** (the network's own logit),
- **logistic regression**, **linear SVM**, **RBF SVM**, **random forest**,
  **k-NN**, **histogram gradient boosting** on the frozen embedding,
- a **soft-voting ensemble** of deep + the best classical models.

Set `CFG.run_hybrid = False` to skip; on the full dataset the RBF SVM is the
expensive one.
""")

CODE(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

HYBRID_TASK = "expr"          # H2: the head with the most labelled examples
HYBRID_LABEL = {"expr": "H2 expressed in tissue", "tau": "H3 tissue-restricted"}


def embed_split(model, graphs, idx, seq_bank, device, bs):
    model.eval()
    Z, Y_expr, Y_tau, Y_tis = [], [], [], []
    with torch.no_grad():
        for batch in make_loader(graphs, idx, bs, False):
            batch = batch.to(device)
            _, _, z, _, _ = model.encode(batch, seq_bank)
            Z.append(z.cpu().numpy())
            Y_expr.append(batch.y_expr.cpu().numpy())
            Y_tau.append(batch.y_tau.cpu().numpy())
            Y_tis.append(batch.y_tissue.cpu().numpy())
    return (np.concatenate(Z), np.concatenate(Y_expr),
            np.concatenate(Y_tau), np.concatenate(Y_tis))


def classical_zoo(seed):
    return {
        "logistic":      make_pipeline(StandardScaler(),
                                       LogisticRegression(max_iter=2000, C=1.0)),
        "linear SVM":    make_pipeline(StandardScaler(),
                                       CalibratedClassifierCV(
                                           LinearSVC(C=0.5, max_iter=5000), cv=3)),
        "RBF SVM":       make_pipeline(StandardScaler(),
                                       SVC(C=2.0, gamma="scale", probability=True,
                                           random_state=seed)),
        "random forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                n_jobs=-1, random_state=seed),
        "kNN (k=25)":    make_pipeline(StandardScaler(),
                                       KNeighborsClassifier(n_neighbors=25,
                                                            weights="distance")),
        "hist gradboost": HistGradientBoostingClassifier(max_iter=300,
                                                         learning_rate=0.08,
                                                         random_state=seed),
    }


HYBRID = []
if CFG.run_hybrid and PRIMARY in RESULTS:
    print(f"DHL-style hybrid on the '{PRIMARY}' embedding, task = "
          f"{HYBRID_LABEL[HYBRID_TASK]}\n")
    for k, (tr, te) in enumerate(FOLDS):
        model = RESULTS[PRIMARY]["models"][k]
        Ztr, ytr_e, ytr_t, _ = embed_split(model, GRAPHS, tr, SEQ_BANK,
                                           DEVICE, CFG.batch_size)
        Zte, yte_e, yte_t, _ = embed_split(model, GRAPHS, te, SEQ_BANK,
                                           DEVICE, CFG.batch_size)
        ytr = ytr_e if HYBRID_TASK == "expr" else ytr_t
        yte = yte_e if HYBRID_TASK == "expr" else yte_t
        mtr, mte = ytr >= 0, yte >= 0
        Ztr, ytr = Ztr[mtr], ytr[mtr].astype(int)
        Zte, yte = Zte[mte], yte[mte].astype(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            print(f"  fold {k}: single class after masking, skipped")
            continue
        print(f"  fold {k}: train {Ztr.shape}  test {Zte.shape}  "
              f"pos rate {ytr.mean():.3f}/{yte.mean():.3f}")

        # deep head, restricted to the same labelled rows for a fair comparison
        deep_p = RESULTS[PRIMARY]["preds"][k][f"{HYBRID_TASK}_p"]
        deep_y = RESULTS[PRIMARY]["preds"][k][f"{HYBRID_TASK}_y"]
        dm = deep_y >= 0
        m = binary_metrics(deep_y[dm], deep_p[dm], "")
        m.update({"fold": k, "model": "deep head (neural)"})
        HYBRID.append(m)

        probs = {}
        for name, clf in classical_zoo(CFG.seed).items():
            t0 = time.time()
            try:
                clf.fit(Ztr, ytr)
                p_ = clf.predict_proba(Zte)[:, 1]
            except Exception as e:
                print(f"    {name:<16} failed: {type(e).__name__}: {e}")
                continue
            probs[name] = p_
            mm_ = binary_metrics(yte, p_, "")
            mm_.update({"fold": k, "model": name, "fit_sec": round(time.time()-t0, 1)})
            HYBRID.append(mm_)
            print(f"    {name:<16} AUPRC {mm_['auprc']:.3f}  AUROC {mm_['auroc']:.3f}"
                  f"  ({time.time()-t0:.1f}s)")

        # soft-voting ensemble: deep head + the classical models, equal weight
        if probs and dm.sum() == len(yte):
            ens = np.mean([deep_p[dm]] + list(probs.values()), axis=0)
            mm_ = binary_metrics(yte, ens, "")
            mm_.update({"fold": k, "model": "ensemble (deep + classical)"})
            HYBRID.append(mm_)
            print(f"    {'ensemble':<16} AUPRC {mm_['auprc']:.3f}  "
                  f"AUROC {mm_['auroc']:.3f}")
        elif probs:
            ens = np.mean(list(probs.values()), axis=0)
            mm_ = binary_metrics(yte, ens, "")
            mm_.update({"fold": k, "model": "ensemble (classical only)"})
            HYBRID.append(mm_)

    HYB = pd.DataFrame(HYBRID)
    HYB.to_csv(RESULTS_DIR / "hybrid_per_fold.csv", index=False)
else:
    HYB = pd.DataFrame()
    print("CFG.run_hybrid is False (or no primary arm) - skipping the hybrid section")
""")

# =============================================================================
MD(r"""
## Hybrid results

The table is ordered by mean AUPRC. Read it with two questions in mind:

- **Does any classical head beat the deep head?** If yes, the embedding is
  better than the classifier that was sitting on top of it, and the multi-task
  trunk was holding the neural head back. That is DHL's central claim.
- **Does the ensemble beat everything?** Usually by a little. If it beats the
  best single model by less than the fold spread, it is not worth the
  complexity of shipping six models.

A caveat the DHL paper does not dwell on: classical heads fitted on a frozen
embedding cannot fix a bad embedding. If every row of the table is close
together, the ceiling is the representation, not the classifier — and the fix is
upstream, in the towers or the features, not here.
""")

CODE(r"""
if len(HYB):
    order_cols = ["auprc", "auroc", "f1", "precision", "recall",
                  "balanced_accuracy", "mcc", "brier"]
    summ = (HYB.groupby("model")[order_cols]
              .agg(["mean", "std"]).round(4).sort_values(("auprc", "mean"),
                                                         ascending=False))
    print(f"{'='*88}\nDHL hybrid on the {PRIMARY} embedding - "
          f"{HYBRID_LABEL[HYBRID_TASK]}\n{'='*88}")
    print(summ.to_string())
    summ.to_csv(RESULTS_DIR / "hybrid_summary.csv")

    best = summ.index[0]
    deep = summ.loc["deep head (neural)", ("auprc", "mean")] \
        if "deep head (neural)" in summ.index else np.nan
    print(f"\n  best model     : {best}  "
          f"AUPRC {summ.loc[best, ('auprc','mean')]:.4f}")
    if np.isfinite(deep):
        d = summ.loc[best, ("auprc", "mean")] - deep
        sd = summ.loc[best, ("auprc", "std")]
        print(f"  deep head      : AUPRC {deep:.4f}")
        print(f"  delta          : {d:+.4f}  "
              f"({'within' if abs(d) < sd else 'beyond'} the fold spread of {sd:.4f})")
        if abs(d) < sd:
            print("  -> the hybrid gain is inside fold noise. The embedding, not the")
            print("     classifier, is the ceiling here.")

    # ---------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    mods = list(summ.index)
    mu = [summ.loc[m, ("auprc", "mean")] for m in mods]
    sd = [summ.loc[m, ("auprc", "std")] for m in mods]
    cols = ["#D14A73" if m.startswith("deep") else
            "#00A38C" if m.startswith("ensemble") else "#5B85D6" for m in mods]
    axes[0].barh(mods, mu, xerr=sd, color=cols, edgecolor="k", linewidth=.6)
    base = HYB["auprc_baseline"].mean()
    axes[0].axvline(base, ls="--", c="r", lw=1.2, label=f"random {base:.3f}")
    axes[0].set_xlabel("AUPRC"); axes[0].set_title("Hybrid heads, mean +/- sd")
    axes[0].legend(fontsize=7); axes[0].invert_yaxis()

    for m in mods:
        sub = HYB[HYB.model == m].sort_values("fold")
        axes[1].plot(sub["fold"], sub["auprc"], "o-", ms=4, lw=1.2, label=m, alpha=.85)
    axes[1].set_xlabel("fold"); axes[1].set_ylabel("AUPRC")
    axes[1].set_title("Per-fold stability"); axes[1].legend(fontsize=6)

    piv = HYB.pivot_table(index="model", values=["auprc", "auroc", "f1", "mcc"],
                          aggfunc="mean")
    sns.heatmap(piv.loc[mods], annot=True, fmt=".3f", cmap="YlGnBu",
                ax=axes[2], cbar_kws={"label": "score"})
    axes[2].set_title("Metric grid"); axes[2].set_ylabel("")
    fig.suptitle(f"Deep Hybrid Learning: classical heads on the frozen "
                 f"{PRIMARY} embedding", y=1.04, fontsize=12)
    fig.tight_layout(); savefig(fig, "19_hybrid_heads"); plt.show()
else:
    print("no hybrid results to report")
""")

# =============================================================================
MD(r"""
## What changed, and what to check first

**The sequence tower is now a Hyena stack, not a dilated CNN.**

| | before | now |
|---|---|---|
| operator | dilated Conv1d x 4 | HyenaOperator (implicit long conv, FFT) |
| promoter window | 2 kb | 16 kb (4 kb in fast mode) |
| receptive field | ~500 bp after 4 layers | full window from block 1 |
| filter parameters | grow with kernel size | independent of length (generated by an MLP) |
| cost in length | O(L k) | O(L log L) |
| strand | arbitrary convention | reverse-complement equivariant |
| norm / MLP | BatchNorm + GELU | RMSNorm + SwiGLU |

The single most important line is the receptive field. The old stack physically
could not see a signal 3 kb from the TSS; every Hyena block can see the whole
16 kb. That is the change that makes "long receptive field" true rather than
aspirational.

**Three things to verify on your first real run:**

1. **Does the ablation show the sequence tower doing anything?** If `GAT only`
   and `both` are within a fold standard deviation, the Hyena tower is
   expensive decoration. Check `ablation_fusion_verdict.csv`.

2. **Is the extra window used?** Re-run with `promoter_window = 2000` and
   compare. If 16 kb does not beat 2 kb, the signal is proximal and the long
   context is not earning its FFT cost. This is the honest test of the whole
   change, and I have not run it for you.

3. **Did the RC-equivariance assertion pass?** It runs in the model cell. If it
   ever fails, the two strands are being embedded differently and the features
   depend on an arbitrary annotation choice.

**Cost.** FFT convolution is O(L log L) but the constant is not small, and
reverse-complement equivariance doubles the forward pass. At 16 kb with 4 blocks
expect the sequence tower to dominate step time. `stem_stride` (default 4) is
the dial: stride 1 is true single-nucleotide resolution and roughly 4x the cost.

**Things I did not do, and would want before publishing:**

- No pretrained HyenaDNA checkpoint. The real HyenaDNA is *pretrained* on the
  human genome and then fine-tuned; ours is trained from scratch on a few
  thousand promoters. That is the largest single gap between this and the paper,
  and loading `hyenadna-small-32k-seqlen` from HuggingFace would likely matter
  more than any architectural detail here.
- No diffusion component. HiCDiffusionLooping generates contact maps from
  sequence; that is a different task from ours (we consume contacts rather than
  predicting them), so borrowing its decoder would have meant changing what the
  model is for.
- The DHL comparison uses our own embedding, not DNABERT's. A fair
  reproduction of DHL would swap the encoder too.
""")
