"""
Follow-up analysis for pangenomic_exploratory_analysis.ipynb Cell 36.

Tests whether tissue clustering in the 'topology' representation is driven by
sequencing depth (degree, reads_sum, span) or by actual biological differences.

Method:
1. Regress out 'degree' (and optionally other depth proxies) from topology features
2. Re-run k-means clustering on the residual representation
3. Compare ARI/NMI/purity before and after regression

If ARI collapses after removing depth, the clustering was a batch effect.
If ARI stays high, the tissue structure is real biology, not just library size.

Usage:
  Run this after completing the main notebook (all variables in scope).
  Copy-paste into Cell 36, or run as a standalone script after loading the notebook state.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("=" * 80)
print("FOLLOW-UP: Depth Regression Test")
print("=" * 80)
print("\nTesting whether 'topology' tissue clustering is driven by sequencing depth.\n")

# -----------------------------------------------------------------
# 1. Extract topology representation
# -----------------------------------------------------------------
if "topology" not in REPS:
    print("ERROR: 'topology' representation not found. Run sections 1-6 first.")
else:
    topo_data = REPS["topology"]
    X_topo = topo_data["X"]  # Already standardized
    topo_cols = topo_data["cols"]
    valid_mask = topo_data["valid"]

    # Subset to valid rows
    X = X_topo[valid_mask]
    Lsub = LAB.iloc[np.where(valid_mask)[0]]
    y_tissue = LabelEncoder().fit_transform(Lsub["tissue_class"])
    n_tissues = len(np.unique(y_tissue))

    print(f"Working with {len(X):,} valid (gene, tissue) rows")
    print(f"Topology features: {len(topo_cols)}")
    print(f"  Columns: {', '.join(topo_cols[:10])}{'...' if len(topo_cols) > 10 else ''}")

    # -----------------------------------------------------------------
    # 2. Identify depth-proxy features
    # -----------------------------------------------------------------
    # Features that encode sequencing depth rather than biology
    depth_features = ["degree", "reads_sum", "reads_mean", "reads_max", "score_sum", "span"]
    depth_indices = [i for i, c in enumerate(topo_cols) if c in depth_features]

    if not depth_indices:
        print("\nWARNING: No depth features found in topology. Check column names.")
        print(f"Available columns: {topo_cols}")
    else:
        print(f"\nDepth-proxy features to regress out ({len(depth_indices)}):")
        for idx in depth_indices:
            print(f"  - {topo_cols[idx]}")

        # -----------------------------------------------------------------
        # 3. Baseline clustering (before regression)
        # -----------------------------------------------------------------
        print("\n" + "-" * 80)
        print("BASELINE: Clustering on original topology")
        print("-" * 80)

        # Subsample for speed
        sub_indices = subsample(len(X), cap=min(N_SUBSAMPLE, 6000))
        X_sub = X[sub_indices, :min(20, X.shape[1])]
        y_sub = y_tissue[sub_indices]

        km_baseline = KMeans(n_clusters=n_tissues, n_init=10, random_state=SEED).fit(X_sub)
        labels_baseline = km_baseline.labels_

        ari_baseline = adjusted_rand_score(y_sub, labels_baseline)
        nmi_baseline = normalized_mutual_info_score(y_sub, labels_baseline)
        sil_baseline = silhouette_score(X_sub, labels_baseline)

        print(f"  ARI (tissue):        {ari_baseline:.3f}")
        print(f"  NMI (tissue):        {nmi_baseline:.3f}")
        print(f"  Silhouette:          {sil_baseline:.3f}")

        # -----------------------------------------------------------------
        # 4. Regress out depth proxies
        # -----------------------------------------------------------------
        print("\n" + "-" * 80)
        print("REGRESSION: Removing depth-proxy features")
        print("-" * 80)

        # For each non-depth feature, regress out the depth features
        non_depth_indices = [i for i in range(len(topo_cols)) if i not in depth_indices]

        print(f"Regressing {len(depth_indices)} depth features out of {len(non_depth_indices)} other features...")

        X_depth = X[:, depth_indices]  # Predictors
        X_residual = np.zeros((len(X), len(non_depth_indices)), dtype=np.float32)

        for i, feat_idx in enumerate(non_depth_indices):
            y_feat = X[:, feat_idx]
            # Fit linear regression: y_feat ~ X_depth
            reg = LinearRegression().fit(X_depth, y_feat)
            residual = y_feat - reg.predict(X_depth)
            X_residual[:, i] = residual

        # Re-standardize residuals
        X_residual = StandardScaler().fit_transform(X_residual)

        print(f"Residual representation shape: {X_residual.shape}")

        # -----------------------------------------------------------------
        # 5. Re-cluster on residuals
        # -----------------------------------------------------------------
        print("\n" + "-" * 80)
        print("POST-REGRESSION: Clustering on residual (depth-free) topology")
        print("-" * 80)

        X_res_sub = X_residual[sub_indices, :min(20, X_residual.shape[1])]

        km_residual = KMeans(n_clusters=n_tissues, n_init=10, random_state=SEED).fit(X_res_sub)
        labels_residual = km_residual.labels_

        ari_residual = adjusted_rand_score(y_sub, labels_residual)
        nmi_residual = normalized_mutual_info_score(y_sub, labels_residual)
        sil_residual = silhouette_score(X_res_sub, labels_residual)

        print(f"  ARI (tissue):        {ari_residual:.3f}")
        print(f"  NMI (tissue):        {nmi_residual:.3f}")
        print(f"  Silhouette:          {sil_residual:.3f}")

        # -----------------------------------------------------------------
        # 6. Interpret the results
        # -----------------------------------------------------------------
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)

        ari_delta = ari_baseline - ari_residual
        nmi_delta = nmi_baseline - nmi_residual

        print(f"\nARI change:  {ari_baseline:.3f} → {ari_residual:.3f}  (Δ = {ari_delta:+.3f})")
        print(f"NMI change:  {nmi_baseline:.3f} → {nmi_residual:.3f}  (Δ = {nmi_delta:+.3f})")

        if abs(ari_delta) < 0.05:
            verdict = "✓ ROBUST: Tissue clustering survives depth regression."
            detail = (
                "The tissue structure is real biology, not a sequencing depth artifact.\n"
                "The clustering is driven by contact patterns, not library size."
            )
        elif ari_delta > 0.2:
            verdict = "⚠ DEPTH-DRIVEN: Tissue clustering collapses after removing depth."
            detail = (
                "The original clustering was mostly a batch effect (sequencing depth).\n"
                "Consider quantile-normalizing depth per tissue before building topology,\n"
                "or use only non-depth features (distance distributions, partner counts)."
            )
        else:
            verdict = "~ PARTIAL: Depth contributes, but biology remains."
            detail = (
                "Depth explains some of the clustering, but not all of it.\n"
                "The tissue structure has both biological and technical components."
            )

        print(f"\n{verdict}")
        print(f"\n{detail}")

        # -----------------------------------------------------------------
        # 7. Save results
        # -----------------------------------------------------------------
        depth_test = pd.DataFrame([
            {"representation": "topology (original)",
             "ARI": ari_baseline, "NMI": nmi_baseline, "silhouette": sil_baseline},
            {"representation": "topology (depth-regressed)",
             "ARI": ari_residual, "NMI": nmi_residual, "silhouette": sil_residual},
        ])

        out_path = EDA_DIR / "depth_regression_test.csv"
        depth_test.to_csv(out_path, index=False)
        print(f"\nResults saved to: {out_path.relative_to(PROJECT_ROOT)}")

        print("\n" + "=" * 80)
        print("Next steps if depth-driven:")
        print("  1. Quantile-normalize degree/reads_sum/span per tissue")
        print("  2. Rebuild TOPO with normalized depth features")
        print("  3. Re-run section 12 (clustering)")
        print("  4. Check if tissue ARI improves with true biological signal")
        print("=" * 80)
