# Related work — comparison and correction to the novelty claim

**This corrects a real problem in `ABSTRACT.md`'s opening sentence** ("whether
combining these modalities improves computational prediction ... remains
untested"). That is false as a blanket claim — verified below against eight
real, correctly-characterized papers spanning 2022–2026. The abstract's
opening has been rewritten accordingly (see bottom of this file and the
updated `ABSTRACT.*` files).

## Comparison table

| Paper | Modalities fused | Fusion mechanism | Tests whether fusion helps/hurts? | Relation to our finding |
|---|---|---|---|---|
| **Chromoformer** (Lee et al. 2022, *Nat Commun*) | Histone marks + 3D promoter–pCRE contacts | Three-level hierarchical self-attention (core promoter / pCRE / 3D contact) | Partial — ablates individual histone marks (e.g. H3K36me3) and attention analysis; reports 3D contact *helps* expression prediction | Consistent with our structure+epigenome arms outperforming sequence-only, but never assembles a third (sequence) tower or tests a modality *hurting* |
| **GC-MERGE** (Bigness et al. 2022, *J Comput Biol*) | Histone marks + Hi-C | GCN over a contact graph with histone marks as node features | No systematic fusion-vs-single-modality ablation | Establishes a 2-modality integration baseline; does not test whether fusion beats either modality alone |
| **GraphReg** (Karbalayghareh et al. 2022, *Genome Res*) | Seq-GraphReg (sequence → predicted epigenome, multitask with CAGE) *or* Epi-GraphReg (measured epigenome) + Hi-C/HiChIP/Micro-C graph | GAT propagation over the contact graph; sequence and measured epigenome are **alternative input branches into the same graph architecture**, evaluated as separate models | Compares the two branches' accuracy against each other | Closest prior analogue to our *single-tower* ablation arms (seq-only vs. epi-only) — but the two are never trained as one jointly-fused three-modality model the way all3/seq_str/str_epi/etc. are here |
| **EPInformer** (Lin et al. 2026, *Nat Commun*; preprint 2024) | Sequence + epigenomic signals + chromatin contacts | Optional fusion layer merging sequence embeddings with distance/contact/epigenomic signals; cross-chromosome validated; 0.4M params | Reports integration improves over sequence-only; no report of full fusion *underperforming* a subset | The most directly comparable trimodal architecture in modality composition — motivates our ablation design but does not run it |
| **EpiExpr** (Bhattacharyya & Ay 2026, *bioRxiv*) | Epigenetic tracks alone (EpiExpr-1D) vs. epigenetics + 3D contacts (EpiExpr-3D), explicitly **sequence-free** | GAT / graph-transformer over contacts | Reports EpiExpr-3D *matches* sequence-transformer accuracy without using sequence at all | Compatible with, but weaker than, our finding: sequence is *unnecessary* (their result) vs. actively *harmful when fused* for a specific task (our result) |
| **Yan et al. 2024** (*Nucleic Acids Res*) | Histone marks (as spatial graph features) + regulatory network structure | Principal-node-aggregation GNN, benchmarked against 5 other model classes (incl. GCN) | Compares model *architectures*, not modality subsets | Different axis entirely (architecture choice, not modality ablation) |
| **Prism** (Yang et al. 2026, arXiv:2602.21550 / ICLR 2026) | Sequence + epigenomic signals, with DNase/Hi-C treated as background-chromatin-state confounders | Learns background-state clusters, deconfounds via backdoor adjustment | **Yes — the closest prior evidence that naive fusion hurts.** Shows naive concatenation induces spurious associations with background chromatin state, and that 2 kb sequence + deconfounding beats naive 200 kb-sequence fusion | Same *direction* of finding (naive fusion can hurt), different *mechanism* (confounding by background chromatin state, fixed architecturally) and different *scope* (general expression prediction, sequence *length* as the lever) vs. ours (tower-level tissue-invariance under gated-average pooling, tissue-*identity* classification specifically, shown by exhaustive combinatorial ablation with a paired significance test, not a proposed fix) |
| **Omics Graph Learning** (Ho & Mills 2025, *bioRxiv*) | Multiple epigenetic marks + regulatory graph structure | Heterogeneous gene/enhancer/promoter graph; in silico perturbation | **Yes.** Finds non-additive, tissue-specific, sometimes antagonistic interactions — but *among epigenetic marks* (feature-level), across 20 tissues, via perturbation | Antagonism at a different granularity (features within one modality) and by a different method (perturbation, not architectural ablation) than ours (whole-tower/modality-level, via ablation) |

## What survives as the actual, defensible novelty claim

Not: "no one has tested whether fusion helps." Two papers already show it can
hurt (Prism, OGL), and at least four already fuse 2–3 of these specific
modalities in one architecture (Chromoformer, GC-MERGE, EPInformer, and
GraphReg's shared graph backbone). What appears genuinely unaddressed by
this set:

1. **Exhaustive, symmetric ablation of all combinations of exactly these
   three modalities (sequence, 3D contact structure, chromatin state) within
   one architecture**, under identical heads, losses, and splits — not one
   component switched off (Chromoformer), not two alternative single-modality
   branches (GraphReg), not a flexible-but-not-combinatorially-enumerated
   fusion layer (EPInformer).
2. **A formal paired significance test showing complete fusion is
   significantly *worse* than a specific strict subset** for a specific task
   (tissue identity, P = 0.035) — not merely "no better than," and not
   requiring an architectural fix (contra Prism's deconfounding) to make the
   comparison meaningful.
3. **A mechanism pinned to tower-level tissue-invariance under gated-average
   pooling** — distinct from Prism's background-chromatin-confounding account
   and from OGL's within-modality mark-antagonism account.
4. **Gene-class stratification (housekeeping vs. non-housekeeping,
   protein-coding vs. non-coding) connecting the ablation finding to where
   the resulting model's real predictive skill concentrates** — not present
   in any of the eight papers above.
5. **A DHL-style hybrid check** (classical models on frozen embeddings match
   the trained network) — orthogonal to this fusion literature, not covered
   by it either.

Framed this way, the contribution is narrower and more defensible than the
original abstract claimed, and it is stronger for being narrower — it says
something specific that Prism and OGL's broader claims don't, rather than
re-claiming ground they already hold.

## Revised abstract opening (replaces the old first sentence)

> Gene expression is shaped jointly by promoter sequence, three-dimensional
> chromatin contacts, and local chromatin state. Recent models increasingly
> fuse two or three of these signals, and a small number of studies have
> begun to show that naive fusion can introduce confounding or antagonistic
> interactions rather than simply failing to help; however, no study has
> exhaustively ablated every combination of exactly these three modalities
> within one architecture, or tested whether complete fusion is
> significantly *worse* — not just no better — than a specific subset for a
> specific regulatory task.

This has been applied to `ABSTRACT.md`, `ABSTRACT.docx`, and `ABSTRACT.pdf`.
