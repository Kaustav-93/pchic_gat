#!/usr/bin/env python3
"""
train_test_on_histone_samples.py

Train AND test the Pangenomic GAT structure tower on the *only* tissues that
have histone bigWigs on disk.

Why this script exists
----------------------
The shipped checkpoints in results/model_fold*.pt were trained with the full
18-dim node feature vector:

    [is_bait, log_len, signed_log_dist, same_chrom] + [7 histone signal] + [7 histone mask]

i.e. 14 of the 18 input dimensions are histone-derived (run_manifest.json:
"node_features": 18, "histone_tracks_present": 305). Those 305 tracks are no
longer on disk -- histone/ now holds 69 bigWigs over 12 sample ids. So a
faithful re-test of the old checkpoints is impossible, and testing them with
zeroed histone dims (as test_models_on_pchic_saikat.py does) is meaningless.

This script does the self-consistent thing instead: restrict BOTH training and
testing to the histone-covered tissues, so every graph really carries histone
signal (mask=1 where measured), and report gene-grouped CV metrics.

Only the GAT (structure) tower is trained. It is the tower that consumes the
18-dim histone+topology node features and the 4-dim edge features, so it is the
tower the histone question is actually about. No promoter sequence is needed.

Usage:
    conda activate pangenomic-gat
    python train_test_on_histone_samples.py --device cpu --epochs 24
    python train_test_on_histone_samples.py --device cpu --max-genes 3000 --folds 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
HISTONE_MARKS = ("H3K4me1", "H3K4me3", "H3K27ac",
                 "H3K27me3", "H3K36me3", "H3K9me3", "CTCF")


@dataclass
class Config:
    seed: int = 42
    genome_build: str = "hg19"
    max_neighbours: int = 10000
    min_chicago: float = 5.0
    add_backbone_edges: bool = True
    link_holdout_frac: float = 0.15
    tpm_threshold: float = 1.0
    tau_threshold: float = 0.85

    d_hidden: int = 128
    d_embed: int = 128
    gat_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.2
    attn_dropout: float = 0.1

    w_link: float = 1.0
    w_expr: float = 1.0
    w_tau: float = 0.5
    w_tissue: float = 0.5

    n_folds: int = 5
    epochs: int = 24
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 6
    num_workers: int = 0


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
INTERACT_DIR = PROJECT_ROOT / "interection_data"
HISTONE_DIR = PROJECT_ROOT / "histone"
RNASEQ_DIR = PROJECT_ROOT / "rna_seq"
REF_DIR = PROJECT_ROOT / "reference"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "histone_subset"

GENOME_BUILD = "hg19"
REFGENE_TXT = REF_DIR / "refGene_hg19.txt"
TPM_FILE = RNASEQ_DIR / "rnaseq.rsem_genes_tpm.txt"

REFGENE_COLS = ["bin", "name", "chrom", "strand", "txStart", "txEnd",
                "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds",
                "score", "name2", "cdsStartStat", "cdsEndStat", "exonFrames"]
# --------------------------------------------------------------------------- #
#  Data inventory
# --------------------------------------------------------------------------- #
def list_loop_tissues() -> pd.DataFrame:
    """Return one row per {sample_id, tissue, path} PCHi-C loop file."""
    files = sorted(INTERACT_DIR.glob("*_merged_loop.txt"))
    if not files:
        raise FileNotFoundError(f"No *_merged_loop.txt in {INTERACT_DIR}")
    rows = []
    for p in files:
        stem = p.name.replace("_merged_loop.txt", "")
        sid, _, tname = stem.partition("_")
        rows.append({"sample_id": sid, "tissue": tname, "path": p})
    return pd.DataFrame(rows)


def build_hist_map() -> dict:
    """
    Parse histone/{sid}.{mark}.bw (and .GRCh38.bw) into
        {sample_id: {mark: [(path, assembly), ...]}}
    Replicates with a " (1)" suffix are averaged later, not treated as marks.
    """
    import re
    mark_re = re.compile(
        r"^(?P<sid>\d+)\.(?P<mark>.+?)(?:\s*\(\d+\))?(?:\.(?P<asm>GRCh38|hg38|hg19))?\.bw$")
    hist_map: dict = defaultdict(lambda: defaultdict(list))
    n_files = 0
    for p in sorted(HISTONE_DIR.glob("*.bw")):
        m = mark_re.match(p.name)
        if not m:
            print(f"  ! unparsed histone filename: {p.name}")
            continue
        asm = m.group("asm") or "hg19"
        asm = "GRCh38" if asm in ("GRCh38", "hg38") else "hg19"
        hist_map[m.group("sid")][m.group("mark")].append((str(p), asm))
        n_files += 1
    # plain nested dict (picklable, and comparable to the training notebook)
    out = {sid: dict(marks) for sid, marks in hist_map.items()}
    print(f"histone: {n_files} bigWigs over {len(out)} sample ids")
    return out


def rna_columns_by_tissue() -> dict:
    """{sample_id: [rna column names]} using the leading id of each column."""
    header = pd.read_csv(TPM_FILE, sep="\t", nrows=0)
    cols = [c for c in header.columns if c not in ("gene_id", "gene_name")]
    by_tissue: dict = defaultdict(list)
    for c in cols:
        by_tissue[c.split("_")[0]].append(c)
    return dict(by_tissue)


# --------------------------------------------------------------------------- #
#  Canonical TSS (UCSC refGene, one transcript per symbol)
# --------------------------------------------------------------------------- #
def load_gene_info() -> pd.DataFrame:
    rg = pd.read_csv(REFGENE_TXT, sep="\t", header=None, names=REFGENE_COLS,
                     dtype={"chrom": str, "name2": str}, low_memory=False)
    rg = rg[rg["chrom"].str.fullmatch(r"chr(\d{1,2}|X|Y)")]
    rg["span"] = rg["txEnd"] - rg["txStart"]
    modal = rg.groupby("name2")["chrom"].agg(lambda s: s.value_counts().index[0])
    rg = rg[rg.apply(lambda r: modal.get(r["name2"]) == r["chrom"], axis=1)]
    canon = (rg.sort_values("span", ascending=False)
               .drop_duplicates("name2")
               .set_index("name2"))
    canon["tss"] = np.where(canon["strand"] == "+", canon["txStart"], canon["txEnd"])
    info = canon[["chrom", "strand", "txStart", "txEnd", "tss"]].copy()
    print(f"refGene: canonical models for {len(info):,} gene symbols")
    return info


# --------------------------------------------------------------------------- #
#  Loop loading (CHiCAGO filter + bait-side gene annotation)
# --------------------------------------------------------------------------- #
def parse_frag(s: str):
    c, a, b = s.split(",")
    return c, int(a), int(b)


def load_loops(path: Path, sample_id: str, tissue: str, min_score: float):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["N_reads"] = pd.to_numeric(df["N_reads"], errors="coerce")
    n0 = len(df)
    df = df[df["score"] >= min_score].copy()

    for side in ("bait", "other"):
        col = "bait_frag" if side == "bait" else "other_frag"
        parts = df[col].str.split(",", expand=True)
        df[f"{side}_chr"] = parts[0]
        df[f"{side}_start"] = pd.to_numeric(parts[1], errors="coerce")
        df[f"{side}_end"] = pd.to_numeric(parts[2], errors="coerce")

    # gene annotated to the bait; fall back to the closest gene when absent
    genes = df["bait_gene"].fillna("").where(
        df["bait_gene"].fillna("").str.len() > 0,
        df["bait_closest_gene"].fillna(""))
    df["gene"] = genes.str.split(",")
    df = df.explode("gene")
    df["gene"] = df["gene"].str.strip()
    df = df[(df["gene"] != "") & df["gene"].notna()]

    df["sample_id"] = sample_id
    df["tissue"] = tissue
    df["distance"] = (
        (df["other_start"] + df["other_end"]) / 2
        - (df["bait_start"] + df["bait_end"]) / 2
    ).abs()
    df["same_chrom"] = (df["bait_chr"] == df["other_chr"]).astype(int)
    df.loc[df["same_chrom"] == 0, "distance"] = np.nan

    keep = ["sample_id", "tissue", "gene",
            "bait_chr", "bait_start", "bait_end",
            "other_chr", "other_start", "other_end",
            "N_reads", "score", "distance", "same_chrom"]
    return df[keep].reset_index(drop=True), n0
# --------------------------------------------------------------------------- #
#  Histone signal: bigWig extraction with cross-tissue imputation
# --------------------------------------------------------------------------- #
#  Mirrors the training notebook exactly: a missing (tissue, mark) is filled
#  with the mean of that mark over the *other* tissues in this run at the same
#  fragment (never a silent zero), and the mask bit records whether the value
#  shown is a direct measurement (1) or an imputed cross-tissue average (0).
# --------------------------------------------------------------------------- #
class CoordLifter:
    """hg19 <-> GRCh38 interval conversion, cached. Used for GRCh38-only tracks."""

    def __init__(self, ref_dir: Path):
        self.ref_dir = Path(ref_dir)
        self._lo: dict = {}
        self._cache: dict = {}
        self.n_ok = self.n_fail = 0
        try:
            from pyliftover import LiftOver  # noqa: F401
            self._has = True
        except ImportError:
            self._has = False

    def _get(self, src, dst):
        key = (src, dst)
        if key in self._lo:
            return self._lo[key]
        obj = None
        if self._has:
            from pyliftover import LiftOver
            name = {("hg19", "GRCh38"): "hg19ToHg38.over.chain.gz",
                    ("GRCh38", "hg19"): "hg38ToHg19.over.chain.gz"}.get(key)
            chain = self.ref_dir / name if name else None
            if chain is not None and chain.exists():
                try:
                    obj = LiftOver(str(chain))
                except Exception as e:
                    print(f"  ! could not load {chain.name}: {e}")
        self._lo[key] = obj
        return obj

    def point(self, chrom, pos, src="hg19", dst="GRCh38"):
        if src == dst:
            return chrom, int(pos)
        lo = self._get(src, dst)
        if lo is None:
            return None
        key = (chrom, int(pos) // 1000, src, dst)
        if key in self._cache:
            return self._cache[key]
        try:
            res = lo.convert_coordinate(chrom, int(pos))
        except Exception:
            res = None
        out = (res[0][0], int(res[0][1])) if res else None
        self._cache[key] = out
        return out

    def interval(self, chrom, start, end, src="hg19", dst="GRCh38"):
        if src == dst:
            return chrom, int(start), int(end)
        a = self.point(chrom, start, src, dst)
        b = self.point(chrom, max(int(end) - 1, int(start)), src, dst)
        if a is None or b is None or a[0] != b[0]:
            self.n_fail += 1
            return None
        lo_, hi_ = sorted((a[1], b[1]))
        if hi_ <= lo_:
            hi_ = lo_ + 1
        self.n_ok += 1
        return a[0], lo_, hi_ + 1

    def report(self):
        tot = self.n_ok + self.n_fail
        if tot == 0:
            return "no cross-assembly lookups were needed"
        return (f"{self.n_ok:,}/{tot:,} intervals lifted "
                f"({100 * self.n_fail / tot:.2f}% failed and returned no signal)")


class HistoneExtractor:
    def __init__(self, hist_map, marks, stat="mean", lifter=None,
                 native="hg19", tissue_ids=None):
        import pyBigWig
        self._pyBigWig = pyBigWig
        self.hist_map = hist_map
        self.marks = list(marks)
        self.stat = stat
        self.lifter = lifter
        self.native = native
        self._open: dict = {}
        self._cache: dict = {}
        self._avg_cache: dict = {}
        self.tissue_ids = list(tissue_ids) if tissue_ids is not None else list(hist_map.keys())
        self.available = {
            sid: {mk: len(hist_map.get(sid, {}).get(mk, [])) > 0 for mk in self.marks}
            for sid in hist_map
        }
        self.n_direct = self.n_imputed = self.n_unavailable = 0

    def has(self, sid, mark):
        return self.available.get(sid, {}).get(mark, False)

    def mask_vector(self, sid):
        return np.array([1.0 if self.has(sid, mk) else 0.0 for mk in self.marks],
                        dtype=np.float32)

    def _bw(self, path):
        p = str(path)
        if p not in self._open:
            try:
                self._open[p] = self._pyBigWig.open(p)
            except Exception as e:
                print(f"  ! cannot open {Path(p).name}: {e}")
                self._open[p] = None
        return self._open[p]

    def _raw_mark_signal(self, sid, mark, chrom, start, end):
        entries = self.hist_map.get(sid, {}).get(mark, [])
        if not entries:
            return None
        vals = []
        for path, asm in entries:
            bw = self._bw(path)
            if bw is None:
                continue
            if asm == self.native:
                q = (chrom, int(start), int(end))
            else:
                q = self.lifter.interval(chrom, start, end,
                                         src=self.native, dst=asm) if self.lifter else None
                if q is None:
                    continue
            c, s, e = q
            try:
                chroms = bw.chroms()
                if c not in chroms:
                    alt = c[3:] if c.startswith("chr") else "chr" + c
                    if alt not in chroms:
                        continue
                    c = alt
                clen = chroms[c]
                s, e = max(0, int(s)), min(int(e), clen)
                if e <= s:
                    continue
                v = bw.stats(c, s, e, type=self.stat)[0]
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
            except Exception:
                continue
        return float(np.mean(vals)) if vals else None

    def _cross_tissue_average(self, sid, mark, chrom, start, end):
        key = (mark, chrom, int(start), int(end))
        if key in self._avg_cache:
            return self._avg_cache[key]
        vals = []
        for other in self.tissue_ids:
            if other == sid or not self.has(other, mark):
                continue
            v = self._raw_mark_signal(other, mark, chrom, start, end)
            if v is not None:
                vals.append(v)
        out = float(np.mean(vals)) if vals else None
        self._avg_cache[key] = out
        return out

    def signal(self, sid, chrom, start, end):
        key = (sid, chrom, start, end)
        if key in self._cache:
            return self._cache[key], self.mask_vector(sid)
        out = np.zeros(len(self.marks), dtype=np.float32)
        for i, mk in enumerate(self.marks):
            if self.has(sid, mk):
                v = self._raw_mark_signal(sid, mk, chrom, start, end)
                if v is not None:
                    out[i] = v
                    self.n_direct += 1
            else:
                avg = self._cross_tissue_average(sid, mk, chrom, start, end)
                if avg is not None:
                    out[i] = avg
                    self.n_imputed += 1
                else:
                    self.n_unavailable += 1
        self._cache[key] = out
        return out, self.mask_vector(sid)

    def imputation_report(self):
        tot = self.n_direct + self.n_imputed + self.n_unavailable
        if tot == 0:
            return "no signal lookups yet"
        return (f"{self.n_direct:,} direct + {self.n_imputed:,} imputed "
                f"(cross-tissue avg) + {self.n_unavailable:,} unavailable = "
                f"{tot:,} lookups ({100 * self.n_imputed / tot:.1f}% imputed)")

    def close(self):
        for bw in self._open.values():
            if bw is not None:
                try:
                    bw.close()
                except Exception:
                    pass


class HistoneTransform:
    """log1p -> clip at the 99.5th percentile -> standardise, fitted per mark."""

    def __init__(self, marks):
        self.marks = list(marks)
        self.clip = np.ones(len(self.marks), dtype=np.float32)
        self.mean = np.zeros(len(self.marks), dtype=np.float32)
        self.std = np.ones(len(self.marks), dtype=np.float32)

    def fit(self, raw: np.ndarray):
        raw_log = np.log1p(np.clip(raw, 0, None))
        clip = np.percentile(raw_log, 99.5, axis=0)
        clip[clip <= 0] = 1.0
        self.clip = clip.astype(np.float32)
        clipped = raw_log.clip(0, self.clip)
        self.mean = clipped.mean(0).astype(np.float32)
        self.std = (clipped.std(0) + 1e-6).astype(np.float32)
        return self

    def __call__(self, v):
        return ((np.log1p(np.clip(v, 0, None)).clip(0, self.clip) - self.mean)
                / self.std).astype(np.float32)
# --------------------------------------------------------------------------- #
#  Graph construction  (node F = 4 + 2*N_MARKS = 18; edge F = 4, PCHi-C only)
# --------------------------------------------------------------------------- #
def build_graphs(loops_df, gene_info, genes_ok, hist, transform, cfg, rng_seed=0):
    from torch_geometric.data import Data
    import torch

    rng = np.random.default_rng(rng_seed)
    graphs, meta = [], []
    marks = list(hist.marks)
    grouped = loops_df.groupby(["sample_id", "gene"], sort=False)
    total = grouped.ngroups
    t0 = time.time()

    for i, ((sid, gene), g) in enumerate(grouped):
        if gene not in genes_ok:
            continue
        g = g.sort_values("score", ascending=False).head(cfg.max_neighbours)
        if len(g) < 2:
            continue

        tissue = g["tissue"].iloc[0]
        tss = int(gene_info.at[gene, "tss"])
        b_chr = g["bait_chr"].iloc[0]
        b_s, b_e = int(g["bait_start"].iloc[0]), int(g["bait_end"].iloc[0])

        hmask = hist.mask_vector(sid)

        hv, _ = hist.signal(sid, b_chr, b_s, b_e)
        nodes = [np.concatenate([
            [1.0, np.log10(max(b_e - b_s, 1)), 0.0, 1.0],
            transform(hv), hmask]).astype(np.float32)]
        node_pos = [(b_s + b_e) // 2]

        e_src, e_dst, e_attr = [], [], []
        for _, r in g.iterrows():
            o_chr = r["other_chr"]
            o_s, o_e = int(r["other_start"]), int(r["other_end"])
            mid = (o_s + o_e) // 2
            same = float(r["same_chrom"])
            d = (mid - tss) if same else 0.0
            sgn_log_d = np.sign(d) * np.log10(abs(d) + 1)

            hv, _ = hist.signal(sid, o_chr, o_s, o_e)
            nodes.append(np.concatenate([
                [0.0, np.log10(max(o_e - o_s, 1)), sgn_log_d, same],
                transform(hv), hmask]).astype(np.float32))
            node_pos.append(mid)

            attr = np.array([
                np.log1p(float(r["score"])), np.log1p(float(r["N_reads"])),
                sgn_log_d, same], dtype=np.float32)
            j = len(nodes) - 1
            e_src += [0, j]
            e_dst += [j, 0]
            e_attr += [attr, attr]

        n_nodes = len(nodes)
        n_real_edges = len(e_src)

        if cfg.add_backbone_edges and n_nodes > 2:
            order = sorted(range(1, n_nodes), key=lambda k: node_pos[k])
            for a, b in zip(order[:-1], order[1:]):
                gap = abs(node_pos[a] - node_pos[b])
                attr = np.array([0.0, 0.0, np.log10(gap + 1), 1.0], dtype=np.float32)
                e_src += [a, b]
                e_dst += [b, a]
                e_attr += [attr, attr]

        edge_index = np.array([e_src, e_dst], dtype=np.int64)
        edge_attr = np.stack(e_attr).astype(np.float32)

        # distance-matched link-prediction supervision (gene-stable RNG)
        seed = (rng_seed + (hash((sid, gene)) & 0xFFFFFFFF)) % (2 ** 32)
        grng = np.random.default_rng(seed)

        real_pairs = [(0, j) for j in range(1, n_nodes)]
        n_hold = max(1, int(round(cfg.link_holdout_frac * len(real_pairs))))
        n_hold = min(n_hold, len(real_pairs) - 1)
        hold_j = grng.choice(np.arange(1, n_nodes), size=n_hold, replace=False)
        hold_set = set(int(j) for j in hold_j)

        keep = []
        for k in range(edge_index.shape[1]):
            a, b = int(edge_index[0, k]), int(edge_index[1, k])
            if k < n_real_edges and (a in hold_set or b in hold_set) and 0 in (a, b):
                continue
            keep.append(k)
        keep = np.array(keep, dtype=np.int64)

        pos_pairs = np.array([[0, j] for j in sorted(hold_set)], dtype=np.int64).T

        cand = [j for j in range(1, n_nodes) if j not in hold_set]
        neg_pairs = []
        if cand:
            pos_d = np.array([abs(node_pos[j] - node_pos[0]) for j in sorted(hold_set)])
            cand_d = np.array([abs(node_pos[j] - node_pos[0]) for j in cand])
            for pd_ in pos_d:
                order = np.argsort(np.abs(np.log10(cand_d + 1) - np.log10(pd_ + 1)))
                pick = cand[int(order[grng.integers(0, min(3, len(order)))])]
                neg_pairs.append([0, pick])
        if not neg_pairs:
            neg_pairs = [[0, int(grng.integers(1, n_nodes))]]
        neg_pairs = np.array(neg_pairs, dtype=np.int64).T

        import torch
        d = Data(
            x=torch.from_numpy(np.stack(nodes)),
            edge_index=torch.from_numpy(edge_index[:, keep]),
            edge_attr=torch.from_numpy(edge_attr[keep]),
            pos_edge_label_index=torch.from_numpy(pos_pairs),
            neg_edge_label_index=torch.from_numpy(neg_pairs),
        )
        d.num_nodes = n_nodes
        graphs.append(d)
        meta.append({"gene": gene, "sample_id": sid, "tissue": tissue,
                     "chrom": b_chr, "n_nodes": n_nodes,
                     "n_edges": int(keep.size), "n_pos": pos_pairs.shape[1],
                     "mean_score": float(g["score"].mean())})

        if (i + 1) % 20000 == 0:
            print(f"  built {i + 1:,}/{total:,} groups "
                  f"({time.time() - t0:.0f}s)")

    return graphs, pd.DataFrame(meta)


# --------------------------------------------------------------------------- #
#  Labels: H2 expression, H3 tau, H4 tissue, H1 link (from the graphs)
# --------------------------------------------------------------------------- #
def build_labels(graphs, meta, tpm_by_sid, cfg):
    """
    tpm_by_sid: gene x sample_id TPM matrix (mean over RNA-seq replicates),
    exactly as the training notebook builds it (by_tissue[sid]).
    """
    import torch

    genes = sorted(set(meta["gene"]))
    sids = sorted(meta["sample_id"].unique())
    tpm = tpm_by_sid.reindex(index=genes)
    tpm = tpm.reindex(columns=[s for s in sids if s in tpm.columns])
    expressed = (tpm > cfg.tpm_threshold).astype(int)

    logtpm = np.log2(tpm + 1)
    mx = logtpm.max(axis=1)
    valid = mx > 0
    tau = pd.Series(np.nan, index=tpm.index)
    n_t = logtpm.shape[1]
    if n_t > 1:
        tau[valid] = ((1 - logtpm[valid].div(mx[valid], axis=0)).sum(axis=1) / (n_t - 1))
    restricted = (tau > cfg.tau_threshold).astype(int)

    tissue_list = sorted(meta["tissue"].unique())
    tissue2idx = {t: i for i, t in enumerate(tissue_list)}

    n_expr = n_tau = 0
    y_expr, y_tau, y_tissue = [], [], []
    for i, d in enumerate(graphs):
        m = meta.iloc[i]
        g, sid, t = m["gene"], m["sample_id"], m["tissue"]
        ye = -1
        if g in expressed.index and sid in expressed.columns:
            ye = int(expressed.at[g, sid]); n_expr += 1
        yt = -1
        if g in restricted.index and not np.isnan(tau.get(g, np.nan)):
            yt = int(restricted.at[g]); n_tau += 1
        y_expr.append(ye)
        y_tau.append(yt)
        y_tissue.append(tissue2idx[t])
        d.y_expr = torch.tensor([ye], dtype=torch.float32)
        d.y_tau = torch.tensor([yt], dtype=torch.float32)
        d.y_tissue = torch.tensor([tissue2idx[t]], dtype=torch.long)

    meta = meta.copy()
    meta["y_expr"] = y_expr
    meta["y_tau"] = y_tau
    meta["y_tissue"] = y_tissue
    return meta, tissue_list, expressed, tau, restricted, n_expr, n_tau
# --------------------------------------------------------------------------- #
#  Model: GATv2 structure tower + 4 task heads (GAT-only arm)
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


class StructureEncoder(nn.Module):
    """GATv2 stack over promoter subgraphs -> (node states, graph embedding)."""

    def __init__(self, f_in, f_edge, d_hidden, d_out, layers, heads,
                 dropout, attn_dropout):
        super().__init__()
        if d_hidden % heads != 0:
            raise ValueError(f"d_hidden ({d_hidden}) not divisible by heads ({heads})")
        per_head = d_hidden // heads
        d_edge = max(8, d_hidden // 2)

        self.in_proj = nn.Linear(f_in, d_hidden)
        self.edge_proj = nn.Sequential(nn.Linear(f_edge, d_edge), nn.GELU())

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            last = (i == layers - 1)
            self.convs.append(GATv2Conv(
                d_hidden, d_hidden if last else per_head, heads=heads,
                concat=not last, dropout=attn_dropout, edge_dim=d_edge,
                add_self_loops=True))
            self.norms.append(nn.LayerNorm(d_hidden))

        self.drop = nn.Dropout(dropout)
        self.pool_gate = nn.Linear(d_hidden, 1)
        self.out = nn.Sequential(nn.Linear(2 * d_hidden, d_out), nn.GELU(),
                                 nn.Linear(d_out, d_out))

    def forward(self, x, edge_index, edge_attr, batch, anchor_index):
        h = self.in_proj(x)
        e = self.edge_proj(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = self.drop(F.gelu(norm(conv(h, edge_index, e) + h)))
        gate = pyg_softmax(self.pool_gate(h), batch)
        pooled = global_add_pool(gate * h, batch)
        anchor = h[anchor_index]
        return h, self.out(torch.cat([anchor, pooled], dim=-1)), None


class PangenomicGAT(nn.Module):
    """Structure-tower model with the four supervised heads (GAT-only arm)."""

    def __init__(self, cfg, f_node, f_edge, n_tissues):
        super().__init__()
        d = cfg.d_embed
        self.struc = StructureEncoder(f_node, f_edge, cfg.d_hidden, d,
                                      cfg.gat_layers, cfg.gat_heads,
                                      cfg.dropout, cfg.attn_dropout)
        self.head_expr = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                       nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        self.head_tau = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                      nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        self.head_tissue = nn.Linear(d, n_tissues)
        d_h = cfg.d_hidden
        self.link_bilinear = nn.Parameter(torch.eye(d_h) + 0.01 * torch.randn(d_h, d_h))

    def encode(self, batch):
        anchor = batch.ptr[:-1]
        h, z, _ = self.struc(batch.x, batch.edge_index, batch.edge_attr,
                             batch.batch, anchor)
        return z, h

    def score_edges(self, h, pair_index):
        a = h[pair_index[0]]
        b = h[pair_index[1]]
        return ((a @ self.link_bilinear) * b).sum(-1)


def masked_bce(logit, target):
    m = target >= 0
    if m.sum() == 0:
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[m], target[m])


# --------------------------------------------------------------------------- #
#  Loader + metric suite (same definitions as the training notebook)
# --------------------------------------------------------------------------- #
def make_loader(graphs, indices, bs, shuffle, num_workers=0):
    return DataLoader([graphs[i] for i in indices], batch_size=bs,
                      shuffle=shuffle, num_workers=num_workers)


def binary_metrics(y_true, y_prob, prefix=""):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve, precision_recall_fscore_support,
                                 accuracy_score, balanced_accuracy_score,
                                 matthews_corrcoef, cohen_kappa_score, jaccard_score,
                                 brier_score_loss, log_loss, confusion_matrix)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    m = y_true >= 0
    y_true, y_prob = y_true[m].astype(int), y_prob[m]
    keys = ("auroc", "auprc", "auprc_baseline", "accuracy", "balanced_accuracy",
            "precision", "recall", "specificity", "npv", "f1", "f2", "mcc", "kappa",
            "jaccard", "brier", "logloss", "best_threshold", "f1_at_best",
            "precision_at_best", "recall_at_best")
    out = {f"{prefix}n": len(y_true),
           f"{prefix}pos_rate": float(y_true.mean()) if len(y_true) else np.nan}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {**out, **{f"{prefix}{k}": np.nan for k in keys}}
    out[f"{prefix}auroc"] = roc_auc_score(y_true, y_prob)
    out[f"{prefix}auprc"] = average_precision_score(y_true, y_prob)
    out[f"{prefix}auprc_baseline"] = float(y_true.mean())

    def at(th):
        yp = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true, yp, average="binary", zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan
        f2 = (5 * pre * rec / (4 * pre + rec)) if (4 * pre + rec) else 0.0
        return dict(accuracy=accuracy_score(y_true, yp),
                    balanced_accuracy=balanced_accuracy_score(y_true, yp),
                    precision=pre, recall=rec, specificity=spec, npv=npv,
                    f1=f1, f2=f2, mcc=matthews_corrcoef(y_true, yp),
                    kappa=cohen_kappa_score(y_true, yp),
                    jaccard=jaccard_score(y_true, yp, zero_division=0),
                    tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

    for k, v in at(0.5).items():
        out[f"{prefix}{k}"] = v
    pr, rc, th = precision_recall_curve(y_true, y_prob)
    f1s = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    bi = int(np.nanargmax(f1s[:-1])) if len(th) else 0
    out[f"{prefix}best_threshold"] = float(th[bi]) if len(th) else 0.5
    out[f"{prefix}f1_at_best"] = float(f1s[bi])
    out[f"{prefix}precision_at_best"] = float(pr[bi])
    out[f"{prefix}recall_at_best"] = float(rc[bi])
    out[f"{prefix}brier"] = brier_score_loss(y_true, y_prob)
    out[f"{prefix}logloss"] = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7),
                                       labels=[0, 1])
    return out


def multiclass_metrics(y_true, y_prob, prefix="tissue_"):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, precision_recall_fscore_support,
                                 roc_auc_score, average_precision_score)
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return {}
    y_pred = y_prob.argmax(1)
    present = np.unique(y_true)
    out = {
        f"{prefix}n": len(y_true),
        f"{prefix}n_classes_present": len(present),
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}kappa": cohen_kappa_score(y_true, y_pred),
    }
    for avg in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        out[f"{prefix}precision_{avg}"] = p
        out[f"{prefix}recall_{avg}"] = r
        out[f"{prefix}f1_{avg}"] = f
    try:
        out[f"{prefix}auroc_macro_ovr"] = roc_auc_score(
            y_true, y_prob[:, present], multi_class="ovr", average="macro", labels=present)
    except Exception:
        out[f"{prefix}auroc_macro_ovr"] = np.nan
    aps = []
    for c in present:
        try:
            aps.append(average_precision_score((y_true == c).astype(int), y_prob[:, c]))
        except Exception:
            pass
    out[f"{prefix}auprc_macro_ovr"] = float(np.mean(aps)) if aps else np.nan
    return out


def all_metrics(pred):
    m = {}
    m.update(binary_metrics(pred["link_y"], pred["link_p"], "link_"))
    m.update(binary_metrics(pred["expr_y"], pred["expr_p"], "expr_"))
    m.update(binary_metrics(pred["tau_y"], pred["tau_p"], "tau_"))
    m.update(multiclass_metrics(pred["tis_y"], pred["tis_p"]))
    return m
# --------------------------------------------------------------------------- #
#  One epoch (train or eval) and one CV fold
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, cfg, optim=None, device="cpu"):
    train = optim is not None
    model.train() if train else model.eval()
    tot = {k: 0.0 for k in ("loss", "link", "expr", "tau", "tissue")}
    n_batches = 0
    store = {"expr_p": [], "expr_y": [], "tau_p": [], "tau_y": [],
             "tis_p": [], "tis_y": [], "link_p": [], "link_y": []}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            z, h = model.encode(batch)
            logit_expr = model.head_expr(z).squeeze(-1)
            logit_tau = model.head_tau(z).squeeze(-1)
            logit_tissue = model.head_tissue(z)

            pos = batch.pos_edge_label_index
            neg = batch.neg_edge_label_index
            s_pos = model.score_edges(h, pos)
            s_neg = model.score_edges(h, neg)
            s = torch.cat([s_pos, s_neg])
            y = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
            l_link = F.binary_cross_entropy_with_logits(s, y)

            l_expr = masked_bce(logit_expr, batch.y_expr)
            l_tau = masked_bce(logit_tau, batch.y_tau)
            l_tissue = F.cross_entropy(logit_tissue, batch.y_tissue)

            loss = (cfg.w_link * l_link + cfg.w_expr * l_expr
                    + cfg.w_tau * l_tau + cfg.w_tissue * l_tissue)

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

            for k, v in zip(("loss", "link", "expr", "tau", "tissue"),
                            (loss, l_link, l_expr, l_tau, l_tissue)):
                tot[k] += float(v)
            n_batches += 1

            if not train:
                store["expr_p"].append(torch.sigmoid(logit_expr).cpu())
                store["expr_y"].append(batch.y_expr.cpu())
                store["tau_p"].append(torch.sigmoid(logit_tau).cpu())
                store["tau_y"].append(batch.y_tau.cpu())
                store["tis_p"].append(torch.softmax(logit_tissue, -1).cpu())
                store["tis_y"].append(batch.y_tissue.cpu())
                store["link_p"].append(torch.sigmoid(s).cpu())
                store["link_y"].append(y.cpu())

    avg = {k: v / max(n_batches, 1) for k, v in tot.items()}
    cat = {k: (torch.cat(v).numpy() if v else np.array([])) for k, v in store.items()}
    return avg, cat


def train_one_fold(fold, train_idx, test_idx, graphs, meta, tissue_list, cfg,
                   device, quiet=False):
    from sklearn.model_selection import GroupKFold

    g_tr = meta.iloc[train_idx]["gene"].values
    n_groups = len(np.unique(g_tr))
    n_splits = max(2, min(10, n_groups))
    inner = GroupKFold(n_splits=n_splits)
    sub_tr, sub_va = next(iter(inner.split(np.arange(len(train_idx)), groups=g_tr)))
    tr_idx = train_idx[sub_tr]
    va_idx = train_idx[sub_va]

    model = PangenomicGAT(cfg, 18, 4, len(tissue_list)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    sched = CosineAnnealingLR(optim, T_max=cfg.epochs)

    dl_tr = make_loader(graphs, tr_idx, cfg.batch_size, True, cfg.num_workers)
    dl_va = make_loader(graphs, va_idx, cfg.batch_size, False, cfg.num_workers)
    dl_te = make_loader(graphs, test_idx, cfg.batch_size, False, cfg.num_workers)

    best = {"loss": np.inf, "state": None, "epoch": -1}
    history = []
    bad = 0
    for ep in range(cfg.epochs):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, dl_tr, cfg, optim, device)
        va_loss, _ = run_epoch(model, dl_va, cfg, None, device)
        sched.step()
        history.append({"fold": fold, "epoch": ep,
                        **{f"train_{k}": v for k, v in tr_loss.items()},
                        **{f"val_{k}": v for k, v in va_loss.items()},
                        "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        flag = ""
        if va_loss["loss"] < best["loss"] - 1e-4:
            best = {"loss": va_loss["loss"],
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    "epoch": ep}
            bad = 0
            flag = "  <- best"
        else:
            bad += 1
        if not quiet:
            print(f"  f{fold} ep{ep:>3}  train {tr_loss['loss']:.4f}  "
                  f"val {va_loss['loss']:.4f}  "
                  f"[link {va_loss['link']:.3f} expr {va_loss['expr']:.3f} "
                  f"tau {va_loss['tau']:.3f} tis {va_loss['tissue']:.3f}]  "
                  f"{time.time() - t0:.0f}s{flag}")
        if bad >= cfg.patience:
            if not quiet:
                print(f"  f{fold} early stop at epoch {ep} (best {best['epoch']})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    _, pred = run_epoch(model, dl_te, cfg, None, device)
    return model, pred, pd.DataFrame(history), best["epoch"]


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-genes", type=int, default=0,
                    help="subsample to N genes (0 = all)")
    ap.add_argument("--max-neighbours", type=int, default=10000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    set_all_seeds(args.seed)
    cfg = Config(seed=args.seed, epochs=args.epochs, n_folds=args.folds,
                 batch_size=args.batch_size, max_neighbours=args.max_neighbours,
                 num_workers=args.num_workers)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   epochs: {cfg.epochs}   folds: {cfg.n_folds}")

    # ---- inventory -------------------------------------------------------
    tissue_df = list_loop_tissues()
    hist_map = build_hist_map()
    rna_by_tissue = rna_columns_by_tissue()

    sel = tissue_df[tissue_df["sample_id"].isin(hist_map.keys())
                    & tissue_df["sample_id"].isin(rna_by_tissue.keys())].copy()
    sel = sel.sort_values("sample_id").reset_index(drop=True)
    print(f"\nselected {len(sel)} tissues (histone bigWig + RNA-seq + PCHi-C):")
    print(sel[["sample_id", "tissue"]].to_string(index=False))
    if len(sel) == 0:
        print("ERROR: no tissue has histone bigWigs + RNA-seq + loops.")
        return 1

    # ---- histone transform (fit on sampled bait fragments) ----------------
    lifter = CoordLifter(REF_DIR)
    hist = HistoneExtractor(hist_map, HISTONE_MARKS, "mean", lifter=lifter,
                            native=GENOME_BUILD, tissue_ids=sel["sample_id"].tolist())
    avail = pd.DataFrame(hist.available).T.reindex(sel["sample_id"]).fillna(False)
    avail.index = sel.set_index("sample_id")["tissue"]
    print("\nhistone availability (tissues x marks, 1 = measured):")
    print(avail.astype(int).to_string())
    frac = float(avail.values.mean())
    print(f"  {frac:.1%} of (tissue x mark) tracks measured directly; "
          f"the rest imputed cross-tissue (mask=0)")

    # ---- load loops ------------------------------------------------------
    frames = []
    for _, r in sel.iterrows():
        d, n0 = load_loops(r["path"], r["sample_id"], r["tissue"], cfg.min_chicago)
        frames.append(d)
        print(f"  {r['sample_id']} {r['tissue']:<38} {n0:>7,} loops -> {len(d):>7,} rows")
    loops = pd.concat(frames, ignore_index=True)
    del frames
    print(f"\nparsed {len(loops):,} (gene, bait, prey) rows over "
          f"{loops['tissue'].nunique()} tissues, {loops['gene'].nunique():,} genes")

    gene_info = load_gene_info()
    genes_ok = set(loops["gene"]) & set(gene_info.index)
    loops = loops[loops["gene"].isin(genes_ok)].reset_index(drop=True)
    print(f"genes resolvable to a TSS: {len(genes_ok):,}; "
          f"{len(loops):,} rows kept")

    # ---- optional gene subsample -----------------------------------------
    if args.max_genes and loops["gene"].nunique() > args.max_genes:
        keep_genes = sorted(loops["gene"].unique())[:args.max_genes]
        loops = loops[loops["gene"].isin(keep_genes)].reset_index(drop=True)
        print(f"subsampled to {len(keep_genes):,} genes -> {len(loops):,} rows")

    # ---- fit histone transform on a sample of bait fragments --------------
    samp = loops.sample(min(5000, len(loops)), random_state=cfg.seed)
    raw = np.array([hist.signal(r["sample_id"], r["bait_chr"],
                                int(r["bait_start"]), int(r["bait_end"]))[0]
                    for _, r in samp.iterrows()], dtype=np.float32)
    transform = HistoneTransform(HISTONE_MARKS).fit(raw)
    print("\nhistone transform (log1p -> clip p99.5 -> z-score):")
    for i, mk in enumerate(HISTONE_MARKS):
        print(f"  {mk:<9} clip={transform.clip[i]:.3f} "
              f"mean={transform.mean[i]:.3f} std={transform.std[i]:.3f}")
    print(f"  signal provenance: {hist.imputation_report()}")
    if n_g38 := lifter.report() != "no cross-assembly lookups were needed":
        print(f"  {lifter.report()}")

    # ---- graphs ----------------------------------------------------------
    print("\nbuilding graphs...")
    graphs, meta = build_graphs(loops, gene_info, genes_ok, hist, transform, cfg, cfg.seed)
    print(f"built {len(graphs):,} graphs; node F={graphs[0].x.shape[1]}, "
          f"edge F={graphs[0].edge_attr.shape[1]}")

    # ---- labels ----------------------------------------------------------
    tpm_cols = sorted({c for sid in sel["sample_id"] for c in rna_by_tissue.get(sid, [])})
    tpm_raw = pd.read_csv(TPM_FILE, sep="\t", usecols=["gene_name"] + tpm_cols,
                          low_memory=False)
    tpm_raw = tpm_raw.dropna(subset=["gene_name"]).groupby("gene_name").max()
    tpm_by_sid = {}
    for sid in sel["sample_id"]:
        cols = [c for c in rna_by_tissue.get(sid, []) if c in tpm_raw.columns]
        if cols:
            tpm_by_sid[sid] = tpm_raw[cols].mean(axis=1)
    tpm_by_sid = pd.DataFrame(tpm_by_sid)

    meta, tissue_list, expressed, tau, restricted, n_expr, n_tau = build_labels(
        graphs, meta, tpm_by_sid, cfg)
    print(f"\nlabels over {len(graphs):,} graphs, {len(tissue_list)} tissue classes")
    print(f"  H2 expression: {n_expr:,} labelled, pos rate "
          f"{meta.loc[meta.y_expr >= 0, 'y_expr'].mean():.1%}")
    print(f"  H3 tau       : {n_tau:,} labelled, pos rate "
          f"{meta.loc[meta.y_tau >= 0, 'y_tau'].mean():.1%}")
    print(f"  H4 tissue    : {len(graphs):,} labelled")
    print(f"  H1 link      : {int(meta['n_pos'].sum()):,} positives + "
          f"{int(meta['n_pos'].sum()):,} distance-matched negatives")

    # ---- gene-grouped folds ----------------------------------------------
    from sklearn.model_selection import GroupKFold
    genes = meta["gene"].values
    n_groups = len(np.unique(genes))
    n_folds = max(2, min(cfg.n_folds, n_groups))
    gkf = GroupKFold(n_splits=n_folds)
    folds = list(gkf.split(np.arange(len(graphs)), groups=genes))
    print(f"\n{n_folds}-fold gene-grouped CV: "
          f"{[int(len(te)) for _, te in folds]} test graphs per fold")

    # ---- train -----------------------------------------------------------
    preds, metrics, hists, models = [], [], [], []
    t_start = time.time()
    for k, (tr, te) in enumerate(folds):
        print(f"\n{'=' * 74}\nFOLD {k + 1}/{n_folds}  train={len(tr):,}  test={len(te):,}\n{'=' * 74}")
        model, pred, hist_df, best_ep = train_one_fold(
            k, tr, te, graphs, meta, tissue_list, cfg, device)
        met = all_metrics(pred)
        met["fold"] = k
        met["best_epoch"] = best_ep
        preds.append(pred)
        metrics.append(met)
        hists.append(hist_df)
        models.append(model)
        print(f"  fold {k} test: link AUPRC {met.get('link_auprc', np.nan):.3f} | "
              f"expr AUPRC {met.get('expr_auprc', np.nan):.3f} | "
              f"tau AUPRC {met.get('tau_auprc', np.nan):.3f} | "
              f"tissue acc {met.get('tissue_accuracy', np.nan):.3f}")

    met_df = pd.DataFrame(metrics)
    met_df.to_csv(out_dir / "metrics_per_fold.csv", index=False)
    pd.concat(hists, ignore_index=True).to_csv(out_dir / "training_history.csv", index=False)
    meta.to_csv(out_dir / "graph_metadata.csv", index=False)
    for k, pred in enumerate(preds):
        np.savez_compressed(out_dir / f"predictions_fold{k}.npz", **pred)
        torch.save({"state_dict": models[k].state_dict(), "config": asdict(cfg),
                    "n_node_feat": 18, "n_edge_feat": 4, "tissues": tissue_list},
                   out_dir / f"model_fold{k}.pt")

    summary_cols = ["link_auprc", "link_auroc", "expr_auprc", "expr_auroc",
                    "expr_f1", "tau_auprc", "tau_auroc", "tau_f1",
                    "tissue_accuracy", "tissue_f1_macro", "tissue_auroc_macro_ovr"]
    summary = met_df[summary_cols].agg(["mean", "std"]).T
    summary.to_csv(out_dir / "metrics_summary.csv")
    print(f"\n{'=' * 74}\nSUMMARY over {n_folds} gene-grouped folds "
          f"({(time.time() - t_start) / 60:.1f} min)\n{'=' * 74}")
    print(summary.round(4).to_string())
    print(f"\noutputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
