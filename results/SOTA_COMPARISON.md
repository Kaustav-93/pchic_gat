# SOTA comparison — read the caveat before the numbers

**The honest headline: only one number in this document is directly
comparable to ours. Everything else measures a different thing on a
different scale, and putting them in one column would be a metric-scale
category error, not a comparison.** Two task families exist across these
eight papers:

- **Binary expressed/not-expressed classification**, scored by AUROC/AUPRC —
  the same metric family as our H2. Only **GC-MERGE** reports this.
- **Continuous expression-magnitude regression** (predicted vs. measured
  CAGE-seq or log-TPM), scored by **Pearson correlation** — Chromoformer-reg,
  GraphReg, EPInformer, EpiExpr, Prism, OGL, and Yan et al. all report this.
  A Pearson r of 0.88 on that task and an AUPRC of 0.88 on ours are not the
  same kind of number and cannot be ranked against each other.

None of the eight papers report tissue-identity classification (our H4) or
tissue-restriction (our H3) as an endpoint at all — there is nothing to
compare those two heads against.

## What each paper reports, on its own task and its own terms

| Method | Task | Metric | Headline result | Protocol |
|---|---|---|---|---|
| Chromoformer-clf (Lee et al. 2022) | Binary high/low expression | ROC-AUC | Outperforms DeepChrome/AttentiveChrome/DeepDiff/HM-CRNN per cell type (reported as per-cell-type boxplots; **no single aggregate number is published**) | 11 Roadmap/ENCODE cell types, 4-fold CV within each |
| **GC-MERGE** (Bigness et al. 2022) | **Binary expressed/not-expressed** | **AUROC / AUPRC** | AUROC 0.893 (GM12878) / 0.910 (K562) / 0.880 (HUVEC); **AUPRC 0.865 / 0.884 / 0.848** | One model per cell line; genes held out within that line |
| Epi-GraphReg (Karbalayghareh et al. 2022) | Expression log-fold-change | Pearson R | R = 0.627, genes with chromatin interactions | Per cell type (e.g. K562), held-out genomic bins |
| EPInformer-PE-Activity(-HiC) (Lin et al. 2026) | CAGE-seq magnitude | Pearson R | Mean R = 0.83 (no HiC) → 0.84 (+HiC) across 6 cell lines; K562 0.841→0.875, GM12878 0.847→0.891 | 12-fold cross-chromosome CV, 6 cell lines |
| EpiExpr-3D (Bhattacharyya & Ay 2026) | Expression magnitude | Pearson R / MAE | Reported "comparable to sequence-transformer baselines"; exact aggregate R not available in accessible text | GM12878, K562, H1-hESC, HepG2 |
| Prism (Yang et al. 2026) | Expression magnitude | Pearson R | K562 R = 0.875 (vs. Seq2Exp-soft 0.872); GM12878 R = 0.902 (vs. 0.895) | 2 cell lines, 18,377 genes, held-out genes |
| Omics Graph Learning (Ho & Mills 2025) | Expression magnitude | Pearson R | Mean R = 0.83 across 20 tissue models; ~0.87 for cell-line-only models | 20 tissues/cell lines, within-tissue held-out genes |
| Yan et al. 2024 (NAR) | Expression magnitude, cross-cohort | Correlation | 0.77–0.86 depending on train→test cohort shift (GTEx→TCGA, TCGA→CPTAC, TCGA→CCLE) | Explicit cross-cohort/cross-platform generalization |
| **Ours — H1 link prediction** (all3, gene-grouped CV) | Promoter–fragment contact | AUPRC | **1.00** (baseline 0.50) | 8 tissues jointly, genes held out by gene-group |
| **Ours — H2 expressed-in-tissue** | Binary expressed/not | **AUPRC** | **0.884** (baseline 0.672) | same |
| **Ours — H3 tissue-restriction (τ)** | Binary restricted/not | AUPRC | 0.545 (baseline 0.207) | same |
| **Ours — H4 tissue identity** | 8-class classification | Accuracy | 0.802 (baseline 0.125) | same |

## The one genuinely comparable pair: our H2 vs. GC-MERGE

Both score **binary expressed/not-expressed classification** by **AUPRC** —
the same metric, the same task family.

| | AUPRC | Held-out unit | Tissue/cell-line scope |
|---|---|---|---|
| GC-MERGE | 0.848–0.884 (HUVEC / GM12878 / K562) | Genes, within one cell line's own model | One cell line per model |
| **Ours (all3)** | **0.884** | Genes, held out by gene-group, jointly across tissues | 8 tissues, one shared model |

Our number sits at the *top* of GC-MERGE's own range, from a harder
generalization protocol (a single shared model must generalize the
expressed/not decision rule to genes it has never seen, across multiple
tissues at once, rather than one model per cell line seeing more of that
line's own gene universe during training) — but from a much smaller gene/
tissue panel (8 tissues, ~1,900 genes, fast-mode) than GC-MERGE's full gene
sets. This is a fair, like-for-like data point. It is the only one in this
document.

## What this document does not license

Do not read this as "we are competitive with EPInformer" or "we underperform
Prism" or any similar claim — those numbers are Pearson correlations on a
regression task none of our four heads perform. Reusing this table in a
manuscript without preserving the task/metric distinction would misrepresent
both our results and theirs.

## What a real head-to-head would require (not done here)

Either (a) re-implement one of these methods' continuous-expression
regression head inside our own pipeline/data and report our own Pearson r,
or (b) re-implement our gene-grouped, tissue-joint, expressed/not-expressed
protocol against one of these methods' released code, so at least one shared
axis exists with both methods evaluated identically. Absent that, GC-MERGE
is the only legitimate SOTA reference point this project currently has.
