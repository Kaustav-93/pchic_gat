# Pretrained-model ablation & stratified results summary

Source run: `results/results_trimodal_1mb_stratified_fast/` — the trimodal
(sequence + structure + epigenome) 1 Mb model, warm-started from a pretrained
checkpoint (frozen encoders, fp32, fine-tuned fusion + heads), fast-mode scale:
**8 tissues, 1,911 genes, 3,462 (gene, tissue) graphs**, 5-fold gene-grouped CV.
Finished 2026-09-15 10:32.

This is the fast-mode (subsampled) run, used here because it is complete and
validated; the full-scale run (all 46 tissues, every gene) is separately
in progress/paused — see "Known gaps" below.

## Folders

| Folder | Contents |
|---|---|
| `ablation/` | H1–H4 accuracy for all 7 tower-combination arms (all3, seq_str, seq_epi, str_epi, seq, str, epi), plus the all3-vs-best-single-arm verdict and raw per-fold numbers |
| `validation_schemes/` | The same primary (all3) model scored under gene-grouped CV, chromosome-block CV, and leave-one-tissue-out |
| `stratified_housekeeping_coding/` | H2/H3/H4 broken out by housekeeping vs non-housekeeping, and by protein-coding vs non-coding, for every ablation arm |
| `hybrid_deep_vs_classical/` | Six classical models (logistic, SVMs, random forest, k-NN, gradient boosting) fitted on the frozen embedding, vs. the neural head itself |
| `figures/` | The corresponding plots from that run |

## Headline numbers (all3 arm, gene-grouped CV, pooled over 5 folds)

| head | metric | value | random baseline |
|---|---|---|---|
| H1 — link prediction | AUPRC | 1.000 | 0.500 |
| H2 — expressed in tissue | AUPRC | 0.884 | 0.672 |
| H3 — tissue-restricted (τ) | AUPRC | 0.545 | 0.207 |
| H4 — tissue classification | accuracy | 0.802 | 0.125 (1-in-8) |

## Key findings

**Ablation.** Fusing all three towers does not beat the best single/pairwise
arm by more than fold-to-fold noise for H1–H3. For H4 specifically, `str_epi`
(structure + epigenome, **no sequence**) is significantly *better* than the
full `all3` fusion (0.656 vs. 0.637, Δ = −0.018, paired p = 0.035) — adding
the sequence tower measurably *hurts* tissue classification, consistent with
`z_seq` being identical across every tissue of a gene and diluting the
tissue-varying signal `z_str`/`z_epi` carry on their own.
See `ablation/ablation_fusion_vs_best_single_arm.csv`.

**Validation schemes.** Gene-grouped CV and chromosome-block CV agree closely
(H2 AUPRC 0.884 vs. 0.874, H4 accuracy 0.802 vs. 0.803) — the model is not
just memorising gene-adjacency structure. Leave-one-tissue-out is where it
slips (H2 AUPRC 0.813, much higher fold spread); H4 = 0.000 under LOTO is
expected, not a bug (the held-out tissue's class index is never a training
target). See `validation_schemes/gene_cv_vs_chrom_cv_vs_loto.csv`.

**Housekeeping vs non-housekeeping** (this dataset's own Yanai's τ < 0.2 =
housekeeping; see the notebook's labels cell for the rationale). At that
threshold, **zero** genes qualify as housekeeping here — corrected finding:
this is *not* a small-sample artifact that the full 46-tissue run would fix.
τ is computed over the full 43-tissue RNA-seq panel regardless of how many
PCHi-C tissues are in play, and that panel's τ never drops below 0.226 for
these genes — 0.2 is below the floor this panel can produce at all. See
`results/stratified_performance_matrix/README.md` for the corrected
analysis, which uses a median τ split (0.636) instead and finds housekeeping
status is in fact the single strongest driver of H2 performance in this
whole project (AUPRC 0.988 housekeeping vs. 0.573 non-housekeeping).

**Protein-coding vs non-coding** (RefSeq accession prefix: NM_/XM_ vs
NR_/XR_). The non-coding subset is small (n=189 vs. 3,273 coding) and behaves
differently: H2 AUPRC drops sharply (0.29 vs. 0.91) while H3 (tau) AUPRC
*rises* (0.72 vs. 0.53) — i.e. this model is much better at telling whether a
non-coding gene is tissue-restricted than at predicting its raw
expressed/not-expressed status, the reverse of the pattern for protein-coding
genes. See `stratified_housekeeping_coding/housekeeping_and_coding_marginals_all_arms.csv`
for every arm.

**Hybrid (deep vs. classical).** Every classical model fitted on the frozen
embedding lands within ~0.03 AUPRC of the deep head itself (0.857–0.885) —
the embedding, not the head on top of it, is the ceiling.
See `hybrid_deep_vs_classical/hybrid_summary.csv`.

## Known gaps (why this isn't the full 2×2 combination across every scheme)

Two things were built but are **not** reflected in these CSVs yet:

1. **The full housekeeping × protein-coding 2×2 combination** (4 groups, not
   two separate marginal splits) — implemented, verified against synthetic
   data, but the real run that would have produced it crashed twice: once on
   a naming bug (fixed), once on an unrelated CUDA OOM caused by an
   over-aggressive `batch_size` change made in between attempts (since
   reverted to the known-safe value of 16).
2. **Stratifying chromosome-CV and leave-one-tissue-out by housekeeping/coding
   status**, not just gene-grouped CV — the code keeps those schemes'
   predictions in memory for exactly this, but the run never reached that
   cell successfully for the reason above.

Both are one clean fast-mode run away (~7h) from being real, at
`notebooks/pangenomic_gat_pipeline_trimodal_1mb_stratified.ipynb`. This
summary uses what is already on disk and validated rather than waiting on
that.

## Full-scale (46-tissue, all-gene) run

Separately attempted and paused: a single epoch over the full dataset
(363,617 graphs) takes ~92 minutes regardless of batch size (compute-bound,
not batching-inefficiency-bound), which puts the originally-scoped run
(7 arms × 5 folds + chromosome-CV + 46-tissue leave-one-tissue-out, each
needing 20–40 epochs) at an estimated 3.5–7 months of GPU time. Not
attempted further pending a scope decision.
