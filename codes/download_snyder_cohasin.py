#!/usr/bin/env python3
"""
Download every file associated with an ENCODE *publication* record.

Default target:
  "Landscape of Cohesin-Mediated Chromatin Loops in the Human Genome"
  Grubert et al., PMID:32728247
  https://www.encodeproject.org/publications/8d853642-45b4-47cf-ada6-f32c3058a39d/

Pipeline
  publication record -> list of dataset accessions (ENCSR...)
                     -> per-dataset File objects (ENCODE search API)
                     -> filtered -> parallel resumable download -> md5 check

Nothing is hard-coded except the publication UUID, so the same script works
for any ENCODE publication page.

Usage
  # see what you'd get, download nothing
  python encode_pub_download.py --dry-run

  # everything, 6 parallel streams
  python encode_pub_download.py -o ./encode_grubert -j 6

  # just the processed hg38 signal + peak files (skips the huge fastqs)
  python encode_pub_download.py --assembly GRCh38 \
      --file-format bigWig bigBed bed \
      --exclude-format fastq

  # only raw reads
  python encode_pub_download.py --file-format fastq

  # resume an interrupted run: just re-run the same command
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.encodeproject.org"
PUB_UUID = "8d853642-45b4-47cf-ada6-f32c3058a39d"  # Grubert et al. 2020

# Fields we ask the API to return for each File object. Keeping this list tight
# makes the search calls fast and the responses small.
FILE_FIELDS = [
    "accession", "href", "md5sum", "file_size", "file_format", "file_format_type",
    "file_type", "output_type", "assembly", "status", "biological_replicates",
    "technical_replicates", "dataset", "restricted", "no_file_available",
    "assay_term_name", "target.label", "biosample_ontology.term_name",
    "preferred_default",
]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------- #
# HTTP session
# --------------------------------------------------------------------------- #
def make_session(user: str | None = None, pwd: str | None = None) -> requests.Session:
    """ENCODE session. Auth is only needed for embargoed/restricted files;
    public data works anonymously."""
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "encode-pub-downloader/1.0 (python-requests)",
    })
    retry = Retry(
        total=8, connect=5, read=5, backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    # ENCODE uses HTTP basic auth with an access key ID / secret from your profile
    user = user or os.environ.get("ENCODE_KEY")
    pwd = pwd or os.environ.get("ENCODE_SECRET")
    if user and pwd:
        s.auth = (user, pwd)
    return s


def get_json(session: requests.Session, url: str, **params) -> dict:
    params.setdefault("format", "json")
    r = session.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# 1. publication -> dataset accessions
# --------------------------------------------------------------------------- #
def get_publication(session: requests.Session, uuid: str) -> dict:
    return get_json(session, f"{BASE}/publications/{uuid}/")


def dataset_ids(pub: dict) -> list[str]:
    """The `datasets` field can hold bare accessions ('ENCSR478BMT') or
    @ids ('/experiments/ENCSR478BMT/'); normalise to accessions."""
    out: list[str] = []
    for d in pub.get("datasets", []) or []:
        if isinstance(d, dict):
            d = d.get("accession") or d.get("@id", "")
        d = str(d).strip("/")
        out.append(d.split("/")[-1])
    # some publications also attach supplementary FileSet / Annotation objects
    for d in pub.get("supplementary_data", []) or []:
        if isinstance(d, dict):
            for k in ("dataset", "@id", "accession"):
                if d.get(k):
                    out.append(str(d[k]).strip("/").split("/")[-1])
                    break
    seen, uniq = set(), []
    for a in out:
        if a and a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


# --------------------------------------------------------------------------- #
# 2. dataset -> File objects
# --------------------------------------------------------------------------- #
def files_for_dataset(session: requests.Session, accession: str) -> list[dict]:
    """Ask the search API for every File whose dataset is this accession.
    Works for Experiment, Annotation, FunctionalCharacterization, ...
    ENCODE search returns HTTP 404 when a query matches zero objects."""
    params = [("type", "File"), ("dataset", f"/experiments/{accession}/"),
              ("limit", "all"), ("format", "json")]
    params += [("field", f) for f in FILE_FIELDS]
    r = session.get(f"{BASE}/search/", params=params, timeout=180)
    if r.status_code == 404:
        # not an Experiment -- resolve the object's real @id and retry
        try:
            obj = get_json(session, f"{BASE}/{accession}/")
        except requests.HTTPError:
            return []
        at_id = obj.get("@id")
        if not at_id:
            return obj.get("files", []) or []
        params[1] = ("dataset", at_id)
        r = session.get(f"{BASE}/search/", params=params, timeout=180)
        if r.status_code == 404:
            return []
    r.raise_for_status()
    return r.json().get("@graph", [])


def dataset_meta(session: requests.Session, accession: str) -> dict:
    try:
        o = get_json(session, f"{BASE}/{accession}/")
    except requests.HTTPError:
        return {}
    tgt = o.get("target") or {}
    bios = o.get("biosample_ontology") or {}
    return {
        "assay": o.get("assay_title") or o.get("assay_term_name") or o.get("annotation_type", ""),
        "target": tgt.get("label", "") if isinstance(tgt, dict) else str(tgt),
        "biosample": bios.get("term_name", "") if isinstance(bios, dict) else str(bios),
        "description": (o.get("description") or "").replace("\t", " ").replace("\n", " "),
        "lab": ((o.get("lab") or {}).get("title", "") if isinstance(o.get("lab"), dict)
                else str(o.get("lab") or "")),
    }


# --------------------------------------------------------------------------- #
# 3. filtering
# --------------------------------------------------------------------------- #
def keep(f: dict, args) -> tuple[bool, str]:
    if f.get("no_file_available"):
        return False, "no_file_available"
    if f.get("restricted") and not args.include_restricted:
        return False, "restricted (needs authorised access)"
    if args.status and f.get("status") not in args.status:
        return False, f"status={f.get('status')}"
    if not f.get("href"):
        return False, "no href"
    fmt = (f.get("file_format") or "").lower()
    if args.file_format and fmt not in {x.lower() for x in args.file_format}:
        return False, f"file_format={fmt}"
    if args.exclude_format and fmt in {x.lower() for x in args.exclude_format}:
        return False, f"excluded format={fmt}"
    if args.output_type and f.get("output_type") not in args.output_type:
        return False, f"output_type={f.get('output_type')}"
    if args.assembly:
        asm = f.get("assembly")
        # raw files (fastq) have no assembly -- keep them unless the user
        # explicitly restricts to processed output
        if asm is not None and asm not in args.assembly:
            return False, f"assembly={asm}"
        if asm is None and args.strict_assembly:
            return False, "unassembled file"
    if args.preferred_default_only and not f.get("preferred_default"):
        return False, "not preferred_default"
    return True, ""


# --------------------------------------------------------------------------- #
# 4. download
# --------------------------------------------------------------------------- #
def md5_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_one(session: requests.Session, f: dict, outdir: Path, args) -> dict:
    url = BASE + f["href"]
    dest = outdir / os.path.basename(f["href"])
    part = dest.with_suffix(dest.suffix + ".part")
    expect_md5 = f.get("md5sum")
    expect_size = f.get("file_size")
    rec = {"accession": f.get("accession"), "path": str(dest), "url": url,
           "status": "", "bytes": 0}

    # already complete?
    if dest.exists():
        if expect_size and dest.stat().st_size != expect_size:
            dest.unlink()
        elif args.verify_existing and expect_md5 and md5_of(dest) != expect_md5:
            log(f"  ! md5 mismatch on existing {dest.name}; re-downloading")
            dest.unlink()
        else:
            rec.update(status="skipped (present)", bytes=dest.stat().st_size)
            return rec

    resume_from = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    mode = "ab" if resume_from else "wb"

    for attempt in range(1, args.retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 600)) as r:
                if resume_from and r.status_code == 200:      # server ignored Range
                    resume_from, mode = 0, "wb"
                r.raise_for_status()
                with open(part, mode) as out:
                    for chunk in r.iter_content(chunk_size=4 << 20):
                        if chunk:
                            out.write(chunk)
            break
        except (requests.RequestException, OSError) as e:
            if attempt == args.retries:
                rec["status"] = f"FAILED: {e}"
                return rec
            wait = min(60, 2 ** attempt)
            log(f"  retry {attempt}/{args.retries} for {dest.name} in {wait}s ({e})")
            time.sleep(wait)
            resume_from = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            mode = "ab" if resume_from else "wb"

    if expect_size and part.stat().st_size != expect_size:
        rec["status"] = f"FAILED: size {part.stat().st_size} != {expect_size}"
        return rec
    if expect_md5 and not args.no_md5:
        got = md5_of(part)
        if got != expect_md5:
            part.unlink(missing_ok=True)
            rec["status"] = f"FAILED: md5 {got} != {expect_md5}"
            return rec
    part.rename(dest)
    rec.update(status="ok", bytes=dest.stat().st_size)
    return rec


def human(n: int | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return str(n)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(
        description="Download all ENCODE files for a publication record.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--uuid", default=PUB_UUID, help="ENCODE publication UUID")
    p.add_argument("-o", "--outdir", default="encode_publication_data")
    p.add_argument("-j", "--jobs", type=int, default=4,
                   help="parallel downloads (be polite: <=8)")
    p.add_argument("--flat", action="store_true",
                   help="all files in one dir instead of one subdir per dataset")
    p.add_argument("--file-format", nargs="*", default=None,
                   help="keep only these formats, e.g. bigWig bed bigBed fastq bam")
    p.add_argument("--exclude-format", nargs="*", default=None,
                   help="drop these formats, e.g. fastq bam")
    p.add_argument("--output-type", nargs="*", default=None,
                   help='e.g. "signal p-value" "IDR thresholded peaks" "reads"')
    p.add_argument("--assembly", nargs="*", default=None,
                   help="e.g. GRCh38 hg19")
    p.add_argument("--strict-assembly", action="store_true",
                   help="also drop files with no assembly (fastq/raw) when --assembly is set")
    p.add_argument("--status", nargs="*", default=["released"],
                   help="file statuses to keep; pass empty to keep all")
    p.add_argument("--preferred-default-only", action="store_true",
                   help="only ENCODE's preferred default file per output type")
    p.add_argument("--include-restricted", action="store_true")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--no-md5", action="store_true", help="skip md5 verification")
    p.add_argument("--verify-existing", action="store_true",
                   help="md5-check files already on disk")
    p.add_argument("--dry-run", action="store_true",
                   help="write the manifest, download nothing")
    p.add_argument("--user", default=None, help="ENCODE access key ID (or $ENCODE_KEY)")
    p.add_argument("--password", default=None, help="ENCODE secret (or $ENCODE_SECRET)")
    args = p.parse_args()

    session = make_session(args.user, args.password)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Publication {args.uuid}")
    pub = get_publication(session, args.uuid)
    log(f"  {pub.get('title', '?')}")
    log(f"  {pub.get('journal', '')} {pub.get('date_published', '')} "
        f"{' '.join(pub.get('identifiers', []) or [])}")

    accessions = dataset_ids(pub)
    log(f"  {len(accessions)} datasets: {', '.join(accessions)}\n")

    # ---- gather file records ------------------------------------------------
    todo: list[tuple[dict, Path]] = []
    manifest_rows: list[dict] = []
    skipped = 0
    total_bytes = 0

    for i, acc in enumerate(accessions, 1):
        meta = dataset_meta(session, acc)
        files = files_for_dataset(session, acc)
        label = " | ".join(x for x in (meta.get("assay"), meta.get("target"),
                                       meta.get("biosample")) if x)
        kept = []
        for f in files:
            ok, why = keep(f, args)
            if not ok:
                skipped += 1
                continue
            kept.append(f)
            dest_dir = outdir if args.flat else outdir / acc
            todo.append((f, dest_dir))
            total_bytes += f.get("file_size") or 0
            manifest_rows.append({
                "dataset": acc,
                "assay": meta.get("assay", ""),
                "target": meta.get("target", ""),
                "biosample": meta.get("biosample", ""),
                "file_accession": f.get("accession"),
                "file_format": f.get("file_format"),
                "file_format_type": f.get("file_format_type", ""),
                "output_type": f.get("output_type"),
                "assembly": f.get("assembly", ""),
                "biological_replicates": ",".join(map(str, f.get("biological_replicates") or [])),
                "file_size": f.get("file_size"),
                "md5sum": f.get("md5sum"),
                "url": BASE + f["href"],
                "local_path": str((dest_dir / os.path.basename(f["href"]))),
            })
        log(f"[{i:>2}/{len(accessions)}] {acc}  {label:<55.55} "
            f"{len(kept):>4} files kept / {len(files)} total")

    log(f"\n{len(todo)} files to fetch, {human(total_bytes)} "
        f"({skipped} filtered out)")

    # ---- manifest -----------------------------------------------------------
    man = outdir / "manifest.tsv"
    if manifest_rows:
        with open(man, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(manifest_rows)
        log(f"manifest -> {man}")

    if args.dry_run:
        log("dry run; nothing downloaded")
        return 0
    if not todo:
        log("nothing to do")
        return 0

    for _, d in todo:
        d.mkdir(parents=True, exist_ok=True)

    # ---- download -----------------------------------------------------------
    done = failed = 0
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(download_one, session, f, d, args): f for f, d in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            done += 1
            if rec["status"].startswith("FAILED"):
                failed += 1
                log(f"[{done}/{len(todo)}] {rec['accession']}  {rec['status']}")
            else:
                log(f"[{done}/{len(todo)}] {rec['accession']}  "
                    f"{rec['status']}  {human(rec['bytes'])}")

    with open(outdir / "download_log.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["accession", "status", "bytes", "path", "url"],
                           delimiter="\t")
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in results)

    log(f"\ndone: {done - failed} ok, {failed} failed -> {outdir}")
    if failed:
        log("re-run the same command to retry only the failures (completed files are skipped)")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\ninterrupted -- re-run to resume")
        sys.exit(130)