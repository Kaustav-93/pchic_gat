# SOTA comparison — full table and charts

This folder consolidates `../SOTA_COMPARISON.md` and `../RELATED_WORK.md` into
one table and three figures. **Read the caveat below before the table** — it
is not restated in full here; see those two files for the complete reasoning
and the paper-by-paper writeup.

## The caveat, in one paragraph

Eight prior methods split into two task families that are not
interconvertible: **binary expressed/not-expressed classification** scored by
AUROC/AUPRC (only GC-MERGE), and **continuous expression-magnitude
regression** scored by Pearson r (Chromoformer-reg, Epi-GraphReg, EPInformer,
EpiExpr, Prism, Omics Graph Learning, Yan et al.). Our four heads (H1–H4) are
all classification/accuracy metrics. **Only GC-MERGE vs. our H2 is a valid
apples-to-apples comparison.** Everything else in the table is reported for
completeness and landscape context, not for ranking against our numbers.

## Full table

See [`full_table.csv`](full_table.csv) / [`full_table.md`](full_table.md).
Every row carries a `comparable_to_ours` column stating explicitly whether it
is a valid comparison point — most rows say "No — regression metric,
different task" or "No — same metric family as H2 but no single number to
compare" (Chromoformer). Only three rows say "YES."

## Figures

| File | What it shows | Safe to compare across bars? |
|---|---|---|
| [`figures/classification_auprc_comparison.png`](figures/classification_auprc_comparison.png) | GC-MERGE's 3 cell-line AUPRCs vs. our H2 AUPRC — the one valid head-to-head | Yes — same metric, same task family |
| [`figures/regression_pearson_landscape.png`](figures/regression_pearson_landscape.png) | Pearson r for the five regression papers with a confirmed aggregate number (Chromoformer-reg and EpiExpr excluded — no aggregate published) | Only within this chart, among each other — never against the other two figures |
| [`figures/our_model_all_heads.png`](figures/our_model_all_heads.png) | Our own H1–H4 vs. each head's own random baseline | Only within this chart, against our own baselines |

The one-sentence read of the AUPRC comparison figure: our H2 (0.884) sits at
the top of GC-MERGE's own range (0.848–0.884), from a harder protocol — one
shared model generalizing to held-out genes across 8 tissues jointly, rather
than one model per cell line — but on a much smaller gene/tissue panel
(fast-mode: 8 tissues, ~1,900 genes) than GC-MERGE's full data. See
`../SOTA_COMPARISON.md` for the full discussion of what that protocol
difference does and doesn't license as a claim.

## What not to do with this folder

Do not lift the regression-landscape figure or table rows into a manuscript
sentence like "we are competitive with Prism/EPInformer" — those bars are
Pearson r on a task none of our heads perform. The only legitimate SOTA
sentence this project currently supports is the GC-MERGE/H2 comparison above.
