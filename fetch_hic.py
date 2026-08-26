#!/usr/bin/env python
"""
fetch_hic.py
============
Download real Juicer ``.hic`` contact matrices for the tissues in
``interection_data/``, trying the ENCODE portal first and then the 4D Nucleome
(4DN) data portal.

Why two sources
---------------
ENCODE hosts intact/in-situ Hi-C for a moderate number of tissues and cell
lines. 4DN hosts a larger and more varied Hi-C collection (including many
primary cells) and publishes ``.hic`` files as processed-file outputs. Between
them they cover a decent fraction of a GTEx-style tissue panel, but neither
covers all 46, and that gap is reported rather than papered over.

Assembly, and why we do not liftOver the matrices
-------------------------------------------------
Almost all current ENCODE and 4DN Hi-C is **GRCh38**, while our PCHi-C fragments
are hg19. A ``.hic`` file is a binary, multi-resolution, normalised contact
matrix — there is no meaningful "liftOver" of one. Remapping would require going
back to the raw reads and rerunning Juicer, which is a pipeline in its own right
and not something a download script should pretend to do.

The correct cheap move is the same one used for GRCh38 bigWigs: **lift the query
coordinates, not the data.** The notebook holds hg19 fragment coordinates,
converts each end to GRCh38 with ``pyliftover``, and asks the GRCh38 matrix for
that bin pair. Each file's assembly is recorded in ``hic/HIC_manifest.csv`` so
the notebook knows which files need lifted queries.

Placeholders
------------
Tissues with no ``.hic`` anywhere keep the placeholder stub written by
download_data.sh. The notebook detects the stub, sets ``has_hic = 0`` for those
edges, and carries on — the feature is masked, exactly like a missing histone
mark.

Usage
-----
    python fetch_hic.py --interactions interection_data --out hic
    python fetch_hic.py --dry-run                 # show what would be fetched
    python fetch_hic.py --sources encode          # skip 4DN
    python fetch_hic.py --max-gb 5                # skip files larger than 5 GB
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("requests is required:  conda activate pangenomic-gat && pip install requests")

ENCODE = "https://www.encodeproject.org"
FOURDN = "https://data.4dnucleome.org"
HEADERS = {"accept": "application/json"}

# Reuse the tissue -> biosample mapping from the histone fetcher so the two
# scripts cannot drift apart.
try:
    from fetch_histone_encode import TISSUE_TO_ENCODE, list_tissues, norm
except ImportError:
    sys.exit("fetch_hic.py expects fetch_histone_encode.py beside it "
             "(it reuses the tissue -> biosample mapping).")

# 4DN uses its own biosample vocabulary; these are the terms that differ enough
# from ENCODE's that a literal reuse would miss.
FOURDN_ALIASES = {
    "GM12878": "GM12878",
    "H1": "H1-hESC",
    "K562": "K562",
    "liver": "liver",
    "spleen": "spleen",
    "lung": "lung",
    "ovary": "ovary",
    "thymus": "thymus",
    "aorta": "aorta",
    "heart left ventricle": "left cardiac atrium",
    "skeletal muscle tissue": "skeletal muscle myoblast",
    "endothelial cell of umbilical vein": "HUVEC",
    "fibroblast of dermis": "IMR-90",
    "CD14-positive monocyte": "CD14+ monocyte",
    "B cell": "B-cell",
}


def api_get(base: str, path: str, params: dict, tries: int = 4):
    for attempt in range(tries):
        try:
            r = requests.get(f"{base}{path}", params=params,
                             headers=HEADERS, timeout=90)
            if r.status_code in (404, 403):
                return None
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == tries - 1:
                print(f"      ! query failed after {tries} tries: {exc}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


# --------------------------------------------------------------------------- ENCODE
def find_encode_hic(term_name: str) -> tuple[str, str, str, int] | None:
    """-> (url, accession, assembly, size_bytes) for the best ENCODE .hic file."""
    hits = api_get(ENCODE, "/search/", {
        "type": "Experiment",
        "assay_title": ["Hi-C", "intact Hi-C", "in situ Hi-C"],
        "biosample_ontology.term_name": term_name,
        "status": "released",
        "format": "json",
        "limit": "8",
    })
    if not hits or not hits.get("@graph"):
        return None

    # Prefer the mapping-quality-30 matrix; it is the standard analysis product.
    pref = ["mapping quality thresholded contact matrix", "contact matrix"]
    best = None
    for exp in hits["@graph"]:
        acc = exp["accession"]
        detail = api_get(ENCODE, f"/experiments/{acc}/", {"format": "json"})
        if not detail:
            continue
        for f in detail.get("files", []):
            if f.get("file_format") != "hic" or f.get("status") != "released":
                continue
            ot = f.get("output_type", "")
            rank = pref.index(ot) if ot in pref else len(pref)
            cand = (rank, -(f.get("file_size") or 0), ENCODE + f["href"],
                    f.get("accession", acc), f.get("assembly", "unknown"),
                    f.get("file_size") or 0)
            if best is None or cand[:2] < best[:2]:
                best = cand
    if best is None:
        return None
    return best[2], best[3], best[4], best[5]


# --------------------------------------------------------------------------- 4DN
def find_4dn_hic(term_name: str) -> tuple[str, str, str, int] | None:
    """-> (url, accession, assembly, size_bytes) for the best 4DN .hic file."""
    term = FOURDN_ALIASES.get(term_name, term_name)
    hits = api_get(FOURDN, "/search/", {
        "type": "ExperimentSetReplicate",
        "experiments_in_set.experiment_type.display_title": "in situ Hi-C",
        "experiments_in_set.biosample.biosource.individual.organism.name": "human",
        "q": term,
        "status": "released",
        "format": "json",
        "limit": "10",
    })
    if not hits or not hits.get("@graph"):
        return None

    for es in hits["@graph"]:
        for pf in es.get("processed_files", []):
            href = pf.get("href", "")
            if not href.endswith(".hic"):
                continue
            return (FOURDN + href,
                    pf.get("accession", es.get("accession", "4DN")),
                    pf.get("genome_assembly", "GRCh38"),
                    pf.get("file_size") or 0)
        # some sets expose files only on the nested experiments
        for exp in es.get("experiments_in_set", []):
            for pf in exp.get("processed_files", []):
                href = pf.get("href", "")
                if href.endswith(".hic"):
                    return (FOURDN + href,
                            pf.get("accession", "4DN"),
                            pf.get("genome_assembly", "GRCh38"),
                            pf.get("file_size") or 0)
    return None


# --------------------------------------------------------------------------- io
def is_placeholder(p: Path) -> bool:
    try:
        return p.read_bytes()[:24].startswith(b"DUMMY_HIC_PLACEHOLDER")
    except Exception:
        return False


def is_real_hic(p: Path) -> bool:
    # Juicer .hic files start with the magic string "HIC"
    try:
        return p.exists() and p.read_bytes()[:3] == b"HIC"
    except Exception:
        return False


def download(url: str, dest: Path, expected: int = 0) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", expected or 0))
            got = 0
            t0 = time.time()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(4 << 20):
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        mb = got / 1e6
                        print(f"\r      . {dest.name}  {100*got/total:5.1f}%  "
                              f"{mb:,.0f} MB  {mb/max(time.time()-t0,1):.1f} MB/s",
                              end="", flush=True)
        print()
        if tmp.read_bytes()[:3] != b"HIC":
            print("      ! downloaded file is not a Juicer .hic archive; discarding")
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n      ! download failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interactions", default="interection_data", type=Path)
    ap.add_argument("--out", default="hic", type=Path)
    ap.add_argument("--sources", nargs="*", default=["encode", "4dn"],
                    choices=["encode", "4dn"])
    ap.add_argument("--max-gb", type=float, default=12.0,
                    help="skip files larger than this (Hi-C archives get big)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tissues = list_tissues(args.interactions)
    if not tissues:
        print(f"No *_merged_loop.txt under {args.interactions}", file=sys.stderr)
        return 1

    print(f"\n  {len(tissues)} tissues; sources: {', '.join(args.sources)}")
    print(f"  size cap: {args.max_gb} GB per file")
    print("  NOTE: ENCODE/4DN Hi-C is almost all GRCh38. We keep the native")
    print("        assembly and lift QUERY COORDINATES in the notebook, because")
    print("        a .hic contact matrix cannot be lifted without re-running")
    print("        Juicer from the raw reads.\n")

    rows = []
    n_have = n_got = n_miss = n_skip = 0

    for sid, tissue in tissues:
        stem = f"{sid}_{tissue}"
        dest = args.out / f"{stem}.hic"
        term, quality = TISSUE_TO_ENCODE.get(norm(tissue), (None, "none"))
        print(f"  [{sid}] {tissue}")

        if is_real_hic(dest):
            n_have += 1
            print("      = real .hic already present")
            rows.append([sid, tissue, "present", "", "", "", "", quality])
            continue
        if term is None:
            n_miss += 1
            print("      - no biosample mapping; keeping placeholder")
            rows.append([sid, tissue, "unmapped", "", "", "", "", quality])
            continue

        hit = source = None
        for src in args.sources:
            finder = find_encode_hic if src == "encode" else find_4dn_hic
            hit = finder(term)
            if hit:
                source = src
                break

        if hit is None:
            n_miss += 1
            print(f"      x no .hic for '{term}' on {'/'.join(args.sources)}; "
                  "keeping placeholder")
            rows.append([sid, tissue, "not-found", "", "", "", term, quality])
            continue

        url, acc, assembly, size = hit
        gb = size / 1e9 if size else 0.0
        print(f"      + {source}: {acc}  {assembly}  "
              f"{gb:.1f} GB" if size else f"      + {source}: {acc}  {assembly}")

        if size and gb > args.max_gb:
            n_skip += 1
            print(f"      ~ skipped: {gb:.1f} GB exceeds --max-gb {args.max_gb}")
            rows.append([sid, tissue, "too-large", source, acc, assembly,
                         term, quality])
            continue
        if args.dry_run:
            rows.append([sid, tissue, "would-download", source, acc, assembly,
                         term, quality])
            continue

        if download(url, dest, size):
            n_got += 1
            rows.append([sid, tissue, "downloaded", source, acc, assembly,
                         term, quality])
        else:
            n_miss += 1
            rows.append([sid, tissue, "download-failed", source, acc, assembly,
                         term, quality])

    manifest = args.out / "HIC_manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "tissue", "status", "source", "accession",
                    "assembly", "biosample", "match_quality"])
        w.writerows(rows)

    print(f"\n  already present : {n_have}")
    print(f"  downloaded      : {n_got}")
    print(f"  skipped (size)  : {n_skip}")
    print(f"  unavailable     : {n_miss}  (placeholder retained, feature masked)")
    print(f"  coverage        : {n_have+n_got}/{len(tissues)} "
          f"({100*(n_have+n_got)/len(tissues):.1f}%)")
    print(f"\n  manifest -> {manifest}")
    print("  The notebook reads the 'assembly' column: GRCh38 files are queried")
    print("  with hg19->GRCh38 lifted coordinates via pyliftover.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
