# Manuscript abstract (draft)

**Title:** Naive multimodal fusion degrades tissue-specific prediction of
gene regulation

---

Gene expression is shaped jointly by promoter sequence, three-dimensional
chromatin contacts, and local chromatin state. Recent models increasingly
fuse two or three of these signals, and a small number of studies have
begun to show that naive fusion can introduce confounding or antagonistic
interactions rather than simply failing to help; however, no study has
exhaustively ablated every combination of exactly these three modalities
within one architecture, or tested whether complete fusion is significantly
worse — not just no better — than a specific subset for a specific
regulatory task. We present a trimodal graph attention network that
encodes a gene's 1-megabase promoter sequence via long-range convolution,
its promoter-capture Hi-C contact graph via graph attention, and its
histone/CTCF chromatin state via a dedicated encoder, fusing all three into
a tissue-specific gene representation. Evaluated across eight human tissues
under gene-, chromosome-, and tissue-held-out cross-validation, the model
predicts promoter-fragment contacts (AUPRC 1.00), tissue-specific
expression (AUPRC 0.88 versus 0.67 baseline), tissue-restriction (AUPRC
0.55 versus 0.21), and tissue identity (accuracy 0.80 versus 0.13 baseline).
Systematic ablation of all seven tower combinations shows that fusing every
modality never significantly outperforms the best single or paired
combination, and for tissue identity, adding sequence to structure and
chromatin state significantly degrades accuracy (P = 0.035): promoter
sequence, being tissue-invariant, dilutes tissue-discriminative signal under
naive fusion. Stratifying by gene class reveals the model's genuine
predictive skill is concentrated in tissue-restricted rather than
housekeeping genes (0.20 versus 0.03 gain over baseline) and reverses
direction between protein-coding and non-coding genes. Classical classifiers
trained on frozen embeddings matched the trained network, indicating that
representation learning, not the prediction head, remains the binding
constraint on model performance.

---

**Novelty claim, verified against prior art:** see `results/RELATED_WORK.md`
for the comparison against Chromoformer, GC-MERGE, GraphReg, EPInformer,
EpiExpr, Yan et al. 2024, Prism, and Omics Graph Learning that this opening
paragraph is based on.

---

## Provenance — every number traced to source

| Claim | Value | Source |
|---|---|---|
| Link prediction AUPRC | 1.00 (baseline 0.50) | `results/pretrained_ablation_summary/ablation/ablation_headline_all_heads_all_arms.csv`, row H1, `all3` |
| Tissue-specific expression AUPRC | 0.884 (baseline 0.672) | same file, row H2, `all3` |
| Tissue-restriction (τ) AUPRC | 0.545 (baseline 0.207) | same file, row H3, `all3` |
| Tissue identity accuracy | 0.802 (baseline 0.125) | same file, row H4, `all3` |
| All3 vs. best other arm, H4, P = 0.035 | Δ = −0.018, `str_epi` beats `all3` | `results/pretrained_ablation_summary/ablation/ablation_fusion_vs_best_single_arm.csv` |
| Housekeeping gain over baseline (+0.03) / non-housekeeping (+0.20) | 0.988−0.959 / 0.573−0.372 | `results/stratified_performance_matrix/performance_matrix_full.csv` |
| Protein-coding vs non-coding reversal (H2 0.911→0.289, H3 0.526→0.716) | | same file |
| Hybrid classical-vs-deep parity | deep 0.8845, best classical 0.8846 AUPRC | `results/pretrained_ablation_summary/hybrid_deep_vs_classical/hybrid_summary.csv` |

## Scope, stated plainly (read before reusing this abstract anywhere)

All numbers above come from the **fast-mode pilot**: 8 of 46 available
tissues, 1,911 of the full gene set, 5-fold gene-grouped cross-validation
(plus chromosome-block and leave-one-tissue-out). This is real, validated,
reproducible work — not a full genome-wide, all-46-tissue result, which was
attempted (`results_trimodal_1mb_stratified_full/`) but paused at an
estimated 3.5–7 months of GPU time at current settings. If this abstract is
used in an actual submission, either (a) caveat the tissue/gene count
explicitly in the main text, or (b) rerun at full scale first and refresh
the numbers above — do not silently upgrade the scope claims without
re-deriving the numbers.
