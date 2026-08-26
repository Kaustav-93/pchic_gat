"""Verification harness: exercise the notebook's logic against the real data.

Runs the parts that do not need torch/pyBigWig/py2bit, using the actual cell
sources pulled out of the .ipynb so we are testing shipped code, not a copy.
"""
import json, re, sys, ast
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent  # repo root: this script lives in codes/
NB = json.loads((ROOT / "notebooks" / "pangenomic_gat_pipeline.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def grab(pattern, why):
    """Pull a function/class definition out of the notebook source by name."""
    for src in CODE:
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == pattern:
                return ast.get_source_segment(src, node)
    raise SystemExit(f"could not find {pattern} in the notebook ({why})")


print("=" * 70)
print("1. PCHi-C loop parsing on a real file")
print("=" * 70)
ns = {"np": np, "pd": pd, "Path": Path}
exec(grab("parse_frag", "loop parsing"), ns)
exec(grab("load_loops", "loop parsing"), ns)

check("parse_frag splits a fragment string",
      ns["parse_frag"]("chr1,927395,936954") == ("chr1", 927395, 936954))

f = sorted((ROOT / "interection_data").glob("*_merged_loop.txt"))[0]
sid, _, tname = f.name.replace("_merged_loop.txt", "").partition("_")
df, n0 = ns["load_loops"](f, sid, tname, 5.0)
print(f"  file: {f.name}  raw={n0:,}  parsed={len(df):,}")

check("returns rows", len(df) > 0, f"{len(df):,} rows")
check("expected columns present",
      set(["sample_id", "tissue", "gene", "bait_chr", "bait_start", "bait_end",
           "other_chr", "other_start", "other_end", "N_reads", "score",
           "distance", "same_chrom"]) <= set(df.columns))
check("CHiCAGO filter applied", df["score"].min() >= 5.0,
      f"min score {df['score'].min():.2f}")
check("gene column exploded to single symbols",
      not df["gene"].str.contains(",").any())
check("no empty gene symbols", (df["gene"].str.len() > 0).all())
check("coordinates are integers",
      df["bait_start"].dtype.kind in "iu" and df["other_end"].dtype.kind in "iu",
      f"{df['bait_start'].dtype}, {df['other_end'].dtype}")
check("trans rows have NaN distance",
      bool(df.loc[df.same_chrom == 0, "distance"].isna().all())
      if (df.same_chrom == 0).any() else True)
check("cis distances positive",
      bool((df.loc[df.same_chrom == 1, "distance"] >= 0).all()))
print(f"  genes={df['gene'].nunique():,}  cis={int(df.same_chrom.sum()):,}  "
      f"trans={int((1-df.same_chrom).sum()):,}  "
      f"median cis distance={df.loc[df.same_chrom==1,'distance'].median():,.0f} bp")

print()
print("=" * 70)
print("2. Histone filename parsing (incl. GRCh38 fallback + replicates)")
print("=" * 70)
mark_re = re.compile(
    r"^(?P<sid>\d+)\.(?P<mark>.+?)(?:\s*\(\d+\))?(?:\.(?P<asm>GRCh38|hg38|hg19))?\.bw$")
cases = [
    ("0056.H3K27ac.bw",           "0056", "H3K27ac", None),
    ("0056.H3K9me3 (1).bw",       "0056", "H3K9me3", None),
    ("0060.CTCF.bw",              "0060", "CTCF",    None),
    ("0077.H3K4me3.GRCh38.bw",    "0077", "H3K4me3", "GRCh38"),
    ("0009.H3K36me3.hg19.bw",     "0009", "H3K36me3", "hg19"),
]
for name, sid_, mk, asm in cases:
    m = mark_re.match(name)
    ok = m and m.group("sid") == sid_ and m.group("mark") == mk and m.group("asm") == asm
    check(f"parse {name!r}", bool(ok),
          f"-> {m.group('sid')}/{m.group('mark')}/{m.group('asm')}" if m else "no match")

real = sorted((ROOT / "histone").glob("*.bw"))
unmatched = [p.name for p in real if not mark_re.match(p.name)]
check(f"all {len(real)} real histone filenames parse", not unmatched,
      f"unmatched: {unmatched[:3]}" if unmatched else "")

print()
print("=" * 70)
print("3. Metric suite")
print("=" * 70)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score, jaccard_score,
    brier_score_loss, log_loss)
mns = dict(np=np, accuracy_score=accuracy_score,
           balanced_accuracy_score=balanced_accuracy_score,
           precision_recall_fscore_support=precision_recall_fscore_support,
           roc_auc_score=roc_auc_score, average_precision_score=average_precision_score,
           precision_recall_curve=precision_recall_curve,
           confusion_matrix=confusion_matrix, matthews_corrcoef=matthews_corrcoef,
           cohen_kappa_score=cohen_kappa_score, jaccard_score=jaccard_score,
           brier_score_loss=brier_score_loss, log_loss=log_loss)
exec(grab("binary_metrics", "metrics"), mns)
exec(grab("multiclass_metrics", "metrics"), mns)
bm, mm = mns["binary_metrics"], mns["multiclass_metrics"]

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 2000)
p = np.clip(y * 0.55 + rng.normal(0.22, 0.22, 2000), 0, 1)
m = bm(y, p, "")
check("informative model beats chance", m["auroc"] > 0.75, f"AUROC={m['auroc']:.3f}")
check("AUPRC above its baseline", m["auprc"] > m["auprc_baseline"],
      f"{m['auprc']:.3f} vs {m['auprc_baseline']:.3f}")
check("confusion counts sum to n", m["tn"]+m["fp"]+m["fn"]+m["tp"] == m["n"])
check("F1 at best threshold >= F1 at 0.5", m["f1_at_best"] >= m["f1"] - 1e-9,
      f"{m['f1_at_best']:.3f} >= {m['f1']:.3f}")
check("Brier in [0,1]", 0 <= m["brier"] <= 1, f"{m['brier']:.4f}")
check("MCC in [-1,1]", -1 <= m["mcc"] <= 1, f"{m['mcc']:.3f}")

rand = bm(y, rng.random(2000), "")
check("random predictor ~ chance AUROC", abs(rand["auroc"] - 0.5) < 0.06,
      f"{rand['auroc']:.3f}")

perfect = bm(np.array([0, 0, 1, 1]), np.array([0.01, 0.02, 0.98, 0.99]), "")
check("perfect separation -> AUROC 1", perfect["auroc"] == 1.0)

single = bm(np.zeros(50, int), rng.random(50), "")
check("single-class input returns NaN, not a crash", np.isnan(single["auroc"]))

masked = bm(np.array([-1, -1, 0, 1, 1, 0]), np.array([.9, .1, .2, .8, .7, .3]), "")
check("y = -1 rows are dropped", masked["n"] == 4, f"n={masked['n']}")

yt = rng.integers(0, 5, 600)
prob = rng.random((600, 5))
prob[np.arange(600), yt] += 1.5
prob /= prob.sum(1, keepdims=True)
mc = mm(yt, prob, 5, "tissue_")
check("multiclass accuracy above chance", mc["tissue_accuracy"] > 0.3,
      f"acc={mc['tissue_accuracy']:.3f}")
check("macro OvR AUROC computed", mc["tissue_auroc_macro_ovr"] > 0.6,
      f"{mc['tissue_auroc_macro_ovr']:.3f}")
check("missing class does not crash",
      "tissue_accuracy" in mm(np.where(yt == 4, 3, yt), prob, 5, "tissue_"))

print()
print("=" * 70)
print("4. Cross-assembly coordinate lifting")
print("=" * 70)
lns = {"np": np, "Path": Path, "HAS_PYLIFTOVER": False, "pyliftover": None}
exec(grab("CoordLifter", "assembly conversion"), lns)
L = lns["CoordLifter"](ROOT / "reference")
check("same-assembly interval is a passthrough",
      L.interval("chr1", 100, 200, "hg19", "hg19") == ("chr1", 100, 200))
check("cross-assembly without pyliftover returns None (feature masked)",
      L.interval("chr1", 100, 200, "hg19", "GRCh38") is None)
check("report() is honest when nothing was lifted", "failed" in L.report()
      or "no cross-assembly" in L.report(), L.report())

try:
    import pyliftover  # noqa: F401
    chain = ROOT / "reference" / "hg19ToHg38.over.chain.gz"
    if chain.exists():
        lns["HAS_PYLIFTOVER"] = True
        lns["pyliftover"] = pyliftover
        L2 = lns["CoordLifter"](ROOT / "reference")
        got = L2.interval("chr1", 1000000, 1000500, "hg19", "GRCh38")
        check("real hg19->GRCh38 lift returns a shifted interval",
              got is not None and got[0] == "chr1" and got[1] != 1000000, str(got))
    else:
        print("  SKIP  real lift (chain file absent; run download_data.sh)")
except ImportError:
    print("  SKIP  real lift (pyliftover not installed in this sandbox)")

print()
print("=" * 70)
print("5. Static checks over the whole notebook")
print("=" * 70)
allsrc = "\n".join(CODE)
check("no leftover shift_pairs helper", "shift_pairs" not in allsrc)
check("no self.str attribute shadowing", ".str = StructureEncoder" not in allsrc)
check("StructureEncoder width guard present",
      "must be divisible by" in allsrc)
check("per-graph pooling uses pyg_softmax", "pyg_softmax(self.pool_gate" in allsrc)
check("pyg_softmax is imported", "softmax as pyg_softmax" in allsrc)
check("folds are grouped, not random", "GroupKFold" in allsrc)
check("gene-leak assertion present", "gene leakage between train and test" in allsrc)
check("batching self-check present", "batching self-check" in allsrc)
check("histone mask channel built", "hmask" in allsrc and "mask_vector" in allsrc)
check("hic masked via has_hic", "has_hic" in allsrc)
check("distance-matched negatives", "distance-matched" in allsrc.lower())
check("assembly verified, not assumed", "assembly check" in allsrc)

names = set()
for src in CODE:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
dupes = [n for n in names if sum(
    1 for src in CODE for nd in ast.parse(src).body
    if isinstance(nd, (ast.FunctionDef, ast.ClassDef)) and nd.name == n) > 1]
check("no duplicated top-level definitions", not dupes, str(dupes))

print()
print("=" * 70)
print(f"{len(FAIL)} failure(s)" if FAIL else "ALL CHECKS PASSED")
if FAIL:
    for f_ in FAIL:
        print(f"  - {f_}")
print("=" * 70)
sys.exit(1 if FAIL else 0)
