#!/usr/bin/env python
"""
fetch_histone_encode.py
=======================
Fill the gaps in `histone/` by pulling matching bigWig signal tracks from the
ENCODE portal.

Why this exists
---------------
The `histone/` folder that shipped with this project covers only a handful of
sample IDs (0056, 0060, 0061, 0063-0066) while `interection_data/` has PCHi-C
loops for ~46 tissues. The model wants the same six marks for every tissue, so
this script tries to source the remainder from ENCODE.

How it matches
--------------
Tissue directory names look like ``0077_Liver_merged_loop.txt``. We take the
human-readable part ("Liver"), normalise it, and look it up in TISSUE_TO_ENCODE
below, which maps to an ENCODE ``biosample_ontology.term_name``. For each mark
we query the ENCODE search API for a released Histone-ChIP-seq (or TF-ChIP-seq
for CTCF) experiment on that biosample, then take the ``fold change over
control`` bigWig for the requested assembly.

Assembly fallback (hg19 -> GRCh38 -> liftOver)
----------------------------------------------
ENCODE has been reprocessing everything on GRCh38 and retiring hg19 outputs, so
for many biosamples the only available bigWig is GRCh38 while our PCHi-C
fragments are hg19. This script therefore tries, in order:

1. an **hg19** bigWig  -> saved as ``{sid}.{mark}.bw``
2. a **GRCh38** bigWig -> saved as ``{sid}.{mark}.GRCh38.bw``

**By default we do NOT convert GRCh38 tracks to hg19.** Converting a
whole-genome bigWig means bigWigToBedGraph -> liftOver -> merge -> bedGraphToBigWig,
which spends 10-30 GB of scratch and several minutes *per track* to rebuild
signal at ~3 billion positions, when the notebook only ever queries a few
hundred thousand restriction fragments. The cheap and equivalent move is to
leave the track in GRCh38 and lift the *query coordinates* hg19 -> GRCh38 at
read time, which the notebook does with ``pyliftover``. The manifest records
each file's assembly so the notebook knows which files need lifted queries.

If you do want genuine hg19 tracks on disk (e.g. for a genome browser), pass
``--liftover-tracks``. That path is fully implemented and requires
``ucsc-liftover``, ``ucsc-bigwigtobedgraph``, ``ucsc-bedgraphtobigwig`` and
``bedtools`` (all installed by setup_env.sh), plus the hg38->hg19 chain file
that download_data.sh fetches.

Honesty about coverage
----------------------
ENCODE does not have all six marks for all 46 GTEx-style tissues, and several
of the entries here are approximate matches (e.g. "Blood - Naive CD4+ Cells" ->
"naive thymus-derived CD4-positive, alpha-beta T cell"). Every mapping decision
and every failure is written to ``histone/DOWNLOAD_manifest.csv`` with
``match_quality`` and ``assembly`` columns. **Read that file before trusting the
features.** Anything not downloaded is left absent; the notebook masks absent
marks rather than imputing them, so a gap costs you information but never
invents data.

Usage
-----
    python fetch_histone_encode.py --interactions interection_data --out histone
    python fetch_histone_encode.py --dry-run              # print the plan only
    python fetch_histone_encode.py --marks H3K27ac H3K4me3
    python fetch_histone_encode.py --liftover-tracks      # rebuild real hg19 bigWigs
    python fetch_histone_encode.py --assemblies hg19      # no GRCh38 fallback
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("requests is required:  conda activate pangenomic-gat && pip install requests")

ENCODE = "https://www.encodeproject.org"
HEADERS = {"accept": "application/json"}

MARKS = ["H3K4me1", "H3K4me3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K9me3", "CTCF"]

# Preference order. ENCODE labels the two builds "hg19" and "GRCh38".
ASSEMBLY_PREF = ["hg19", "GRCh38"]

# bigWig output types we accept, best first
OUTPUT_PREF = ["fold change over control", "signal p-value", "signal"]

# ---------------------------------------------------------------------------
# Tissue name  ->  ENCODE biosample_ontology.term_name
#
# match_quality:
#   exact       the ENCODE biosample is the same tissue
#   close       a defensible proxy (same cell type, different donor/protocol)
#   proxy       a related but distinct biosample; treat features with caution
#   none        no reasonable ENCODE equivalent; will be left missing + masked
# ---------------------------------------------------------------------------
TISSUE_TO_ENCODE: dict[str, tuple[str | None, str]] = {
    "adipose - subcutaneous":                    ("subcutaneous adipose tissue", "exact"),
    "adrenal gland":                             ("adrenal gland", "exact"),
    "artery - aorta":                            ("aorta", "exact"),
    "bladder":                                   ("urinary bladder", "close"),
    "bone marrow":                               ("bone marrow", "exact"),
    "brain - frontal cortex (ba9)":              ("dorsolateral prefrontal cortex", "close"),
    "brain - hippocampus":                       ("layer of hippocampus", "close"),
    "cells - astrocyte":                         ("astrocyte", "exact"),
    "cells - cultured fibroblasts":              ("fibroblast of dermis", "close"),
    "cells - embryonic stem cell":               ("H1", "close"),
    "cells - endothelial cell":                  ("endothelial cell of umbilical vein", "proxy"),
    "cells - leukocyte":                         ("common myeloid progenitor, CD34-positive", "proxy"),
    "cells - lymphoblastoid cell":               ("GM12878", "exact"),
    "cells - mesenchymal stem cell":             ("mesenchymal stem cell", "exact"),
    "cells - neuroectodermal cell":              ("neural progenitor cell", "proxy"),
    "cells - neuron":                            ("neural cell", "proxy"),
    "cervix - ectocervix":                       (None, "none"),
    "embryo - mesendoderm":                      ("mesendoderm", "exact"),
    "embryo - trophoblast":                      ("trophoblast cell", "close"),
    "heart - left ventricle":                    ("heart left ventricle", "exact"),
    "liver":                                     ("liver", "exact"),
    "lung":                                      ("upper lobe of left lung", "close"),
    "muscle - skeletal":                         ("skeletal muscle tissue", "exact"),
    "nerve - tibial":                            ("tibial nerve", "exact"),
    "ovary":                                     ("ovary", "exact"),
    "pancreas":                                  ("body of pancreas", "close"),
    "skin - not sun exposed (suprapubic)":       ("suprapubic skin", "exact"),
    "small intestine - terminal ileum":          ("Peyer's patch", "close"),
    "spleen":                                    ("spleen", "exact"),
    "stomach":                                   ("stomach", "exact"),
    "thymus":                                    ("thymus", "exact"),
    "uterus":                                    ("uterus", "exact"),
    # ---- haematopoietic ----------------------------------------------------
    "blood - activated cd4+ cells":              ("activated CD4-positive, alpha-beta T cell", "close"),
    "blood - erythroblast":                      ("erythroblast", "exact"),
    "blood - lymphoblast":                       ("K562", "proxy"),
    "blood - macrophages":                       ("CD14-positive monocyte", "proxy"),
    "blood - megakaryocytes":                    ("CD34-positive mobilized cell", "proxy"),
    "blood - monocytes":                         ("CD14-positive monocyte", "exact"),
    "blood - naive b cells":                     ("naive B cell", "exact"),
    "blood - naive cd4+ cells":                  ("naive thymus-derived CD4-positive, alpha-beta T cell", "exact"),
    "blood - naive cd8+ cells":                  ("naive thymus-derived CD8-positive, alpha-beta T cell", "exact"),
    "blood - neutrophils":                       ("neutrophil", "exact"),
    "blood - non-activated cd4+ cells":          ("CD4-positive, alpha-beta T cell", "close"),
    "blood - total b cells":                     ("B cell", "exact"),
    "blood - total cd4+ cells":                  ("CD4-positive, alpha-beta T cell", "exact"),
    "blood - total cd8+ cells":                  ("CD8-positive, alpha-beta T cell", "exact"),
}


def norm(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def list_tissues(interactions_dir: Path) -> list[tuple[str, str]]:
    """Return [(sample_id, tissue_name)] parsed from the loop file names."""
    out = []
    for p in sorted(interactions_dir.glob("*_merged_loop.txt")):
        stem = p.name.replace("_merged_loop.txt", "")
        sid, _, tissue = stem.partition("_")
        out.append((sid, tissue))
    return out


def encode_get(path: str, params: dict, tries: int = 4):
    """GET the ENCODE API with linear backoff. Returns parsed JSON or None."""
    for attempt in range(tries):
        try:
            r = requests.get(f"{ENCODE}{path}", params=params, headers=HEADERS, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"      ! ENCODE query failed after {tries} tries: {exc}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def find_bigwig(term_name: str, mark: str,
                assemblies: list[str]) -> tuple[str, str, str] | None:
    """Return (download_url, experiment_accession, assembly) for the best bigWig.

    Assemblies are tried in the order given, so passing ["hg19", "GRCh38"] means
    "native hg19 if it exists anywhere, otherwise GRCh38".
    """
    assay = "TF ChIP-seq" if mark == "CTCF" else "Histone ChIP-seq"
    hits = encode_get(
        "/search/",
        {
            "type": "Experiment",
            "assay_title": assay,
            "target.label": mark,
            "biosample_ontology.term_name": term_name,
            "status": "released",
            "format": "json",
            "limit": "8",
        },
    )
    if not hits or not hits.get("@graph"):
        return None

    # Collect every candidate file across every matching experiment first, then
    # choose once. Deciding per-experiment would return a GRCh38 file from the
    # first experiment even when the second has the hg19 one we prefer.
    candidates = []            # (assembly_rank, output_rank, href, accession, assembly)
    for exp in hits["@graph"]:
        acc = exp["accession"]
        detail = encode_get(f"/experiments/{acc}/", {"format": "json"})
        if not detail:
            continue
        for f in detail.get("files", []):
            if (f.get("file_format") != "bigWig"
                    or f.get("status") != "released"
                    or f.get("output_type") not in OUTPUT_PREF):
                continue
            asm = f.get("assembly")
            if asm not in assemblies:
                continue
            candidates.append((assemblies.index(asm),
                               OUTPUT_PREF.index(f["output_type"]),
                               f["href"], acc, asm))
    if not candidates:
        return None
    candidates.sort()
    _, _, href, acc, asm = candidates[0]
    return ENCODE + href, acc, asm


# ---------------------------------------------------------------------------
# Optional: rebuild a genuine hg19 bigWig from a GRCh38 one.
# ---------------------------------------------------------------------------
def _have(*tools) -> bool:
    return all(shutil.which(t) for t in tools)


def liftover_bigwig(src: Path, dest: Path, chain: Path,
                    chrom_sizes: Path) -> bool:
    """GRCh38 bigWig -> hg19 bigWig via bedGraph round trip.

    Expensive: a whole-genome fold-change track expands to tens of GB of text.
    Only invoked with --liftover-tracks.

    The merge step is not optional. liftOver maps intervals independently and
    can emit overlapping output when the two assemblies disagree about local
    order; bedGraphToBigWig rejects overlaps outright, so we collapse them with
    bedtools merge, averaging the signal.
    """
    need = ["bigWigToBedGraph", "liftOver", "bedGraphToBigWig", "bedtools"]
    if not _have(*need):
        missing = [t for t in need if not shutil.which(t)]
        print(f"      ! cannot liftOver, missing tools: {', '.join(missing)}")
        print("        conda install -c bioconda ucsc-liftover "
              "ucsc-bigwigtobedgraph ucsc-bedgraphtobigwig bedtools")
        return False
    if not chain.exists():
        print(f"      ! chain file missing: {chain}  (run download_data.sh)")
        return False

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bg, lifted = td / "src.bedGraph", td / "lifted.bedGraph"
        srt, merged = td / "sorted.bedGraph", td / "merged.bedGraph"
        unmapped = td / "unmapped.bed"
        steps = [
            (["bigWigToBedGraph", str(src), str(bg)], "bigWig -> bedGraph"),
            (["liftOver", str(bg), str(chain), str(lifted), str(unmapped)],
             "liftOver GRCh38 -> hg19"),
        ]
        for cmd, label in steps:
            print(f"      . {label}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"      ! {label} failed: {r.stderr.strip()[:200]}")
                return False
        print("      . sort + merge overlapping intervals")
        with open(srt, "w") as fh:
            r = subprocess.run(["sort", "-k1,1", "-k2,2n", str(lifted)],
                               stdout=fh, stderr=subprocess.PIPE, text=True,
                               env={**os.environ, "LC_ALL": "C"})
        if r.returncode != 0:
            print(f"      ! sort failed: {r.stderr.strip()[:200]}")
            return False
        with open(merged, "w") as fh:
            r = subprocess.run(["bedtools", "merge", "-i", str(srt),
                                "-c", "4", "-o", "mean"],
                               stdout=fh, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print(f"      ! bedtools merge failed: {r.stderr.strip()[:200]}")
            return False
        print("      . bedGraph -> bigWig")
        r = subprocess.run(["bedGraphToBigWig", str(merged), str(chrom_sizes),
                            str(dest)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"      ! bedGraphToBigWig failed: {r.stderr.strip()[:200]}")
            return False

        n_unmapped = sum(1 for _ in open(unmapped)) // 2 if unmapped.exists() else 0
        print(f"      = lifted to hg19 ({n_unmapped:,} intervals dropped as unmappable)")
    return True


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"      = already have {dest.name}")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            got = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = 100 * got / total
                        print(f"\r      . {dest.name}  {pct:5.1f}%", end="", flush=True)
        print()
        tmp.rename(dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n      ! download failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interactions", default="interection_data", type=Path)
    ap.add_argument("--out", default="histone", type=Path)
    ap.add_argument("--assemblies", nargs="*", default=ASSEMBLY_PREF,
                    help="assembly preference order (default: hg19 then GRCh38)")
    ap.add_argument("--marks", nargs="*", default=MARKS)
    ap.add_argument("--liftover-tracks", action="store_true",
                    help="rebuild GRCh38 tracks as real hg19 bigWigs "
                         "(slow, needs tens of GB of scratch)")
    ap.add_argument("--chain", type=Path,
                    default=Path("reference/hg38ToHg19.over.chain.gz"))
    ap.add_argument("--chrom-sizes", type=Path,
                    default=Path("reference/hg19.chrom.sizes"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and what is already on disk, download nothing")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\n  assembly preference: {' -> '.join(args.assemblies)}")
    if args.liftover_tracks:
        print("  --liftover-tracks: GRCh38 hits will be converted to hg19 on disk")
    else:
        print("  GRCh38 hits are kept as-is (suffix .GRCh38.bw); the notebook")
        print("  lifts query coordinates instead of rebuilding whole tracks")
    tissues = list_tissues(args.interactions)
    if not tissues:
        print(f"No *_merged_loop.txt under {args.interactions}", file=sys.stderr)
        return 1

    have = {p.name for p in args.out.glob("*.bw")}
    print(f"\n  {len(tissues)} tissues, {len(args.marks)} marks "
          f"-> {len(tissues)*len(args.marks)} target files")
    print(f"  {len(have)} bigWigs already on disk\n")

    rows = []
    n_have = n_got = n_miss = n_lifted = 0
    by_assembly: dict[str, int] = {}

    for sid, tissue in tissues:
        key = norm(tissue)
        term, quality = TISSUE_TO_ENCODE.get(key, (None, "none"))
        print(f"  [{sid}] {tissue}")
        if term is None:
            print(f"      - no ENCODE equivalent mapped (quality={quality})")
        else:
            print(f"      -> ENCODE biosample '{term}' (match={quality})")

        for mark in args.marks:
            hg19_name = f"{sid}.{mark}.bw"
            existing = [n for n in (hg19_name, f"{sid}.{mark}.GRCh38.bw")
                        if n in have or (args.out / n).exists()]
            if existing:
                n_have += 1
                asm = "hg19" if existing[0] == hg19_name else "GRCh38"
                by_assembly[asm] = by_assembly.get(asm, 0) + 1
                rows.append([sid, tissue, mark, term or "", quality,
                             "present", asm, existing[0], ""])
                continue
            if term is None:
                n_miss += 1
                rows.append([sid, tissue, mark, "", quality, "unmapped", "", "", ""])
                continue
            if args.dry_run:
                rows.append([sid, tissue, mark, term, quality,
                             "would-query", "|".join(args.assemblies), "", ""])
                continue

            hit = find_bigwig(term, mark, args.assemblies)
            if hit is None:
                print(f"      x {mark}: no bigWig on ENCODE in "
                      f"{'/'.join(args.assemblies)}")
                n_miss += 1
                rows.append([sid, tissue, mark, term, quality,
                             "not-on-encode", "", "", ""])
                continue

            url, acc, asm = hit
            if asm == "hg19":
                dest = args.out / hg19_name
                note = ""
            elif args.liftover_tracks:
                dest = args.out / f"{sid}.{mark}.GRCh38.tmp.bw"
                note = "lifted GRCh38->hg19"
            else:
                dest = args.out / f"{sid}.{mark}.GRCh38.bw"
                note = "native GRCh38; notebook lifts query coords"

            print(f"      + {mark}: {asm} ({acc})")
            if not download(url, dest):
                n_miss += 1
                rows.append([sid, tissue, mark, term, quality,
                             "download-failed", asm, "", acc])
                continue

            final_asm, final_name = asm, dest.name
            if asm != "hg19" and args.liftover_tracks:
                out_bw = args.out / hg19_name
                if liftover_bigwig(dest, out_bw, args.chain, args.chrom_sizes):
                    dest.unlink(missing_ok=True)
                    final_asm, final_name = "hg19", out_bw.name
                    n_lifted += 1
                else:
                    keep = args.out / f"{sid}.{mark}.GRCh38.bw"
                    dest.rename(keep)
                    final_name = keep.name
                    note = "liftOver failed; kept GRCh38"

            n_got += 1
            by_assembly[final_asm] = by_assembly.get(final_asm, 0) + 1
            rows.append([sid, tissue, mark, term, quality, "downloaded",
                         final_asm, final_name, acc + (f" | {note}" if note else "")])

    manifest = args.out / "DOWNLOAD_manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "tissue", "mark", "encode_biosample",
                    "match_quality", "status", "assembly", "filename", "notes"])
        w.writerows(rows)

    total = len(tissues) * len(args.marks)
    print(f"\n  already present : {n_have}")
    print(f"  downloaded      : {n_got}")
    print(f"  lifted to hg19  : {n_lifted}")
    print(f"  still missing   : {n_miss}")
    print(f"  coverage        : {(n_have+n_got)}/{total} "
          f"({100*(n_have+n_got)/total:.1f}%)")
    if by_assembly:
        print(f"  by assembly     : "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_assembly.items())))
    print(f"\n  manifest -> {manifest}")
    print("  Columns to read before trusting a feature:")
    print("    match_quality  'proxy' means a related but DIFFERENT biosample")
    print("    assembly       'GRCh38' files are queried with lifted coordinates")
    print("  Missing marks are masked in the notebook (a has_mark indicator")
    print("  channel), never silently zero-filled.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
