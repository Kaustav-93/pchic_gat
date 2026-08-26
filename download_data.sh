#!/usr/bin/env bash
# =============================================================================
#  download_data.sh — fetch every external asset the notebook needs.
#
#      reference/hg19.2bit              UCSC reference sequence (~800 MB)
#      reference/refGene_hg19.txt       RefSeq gene models -> canonical TSS
#      reference/hg19.chrom.sizes       chromosome lengths
#      reference/hg38ToHg19.over.chain.gz  UCSC liftOver chains, both directions
#      reference/hg19ToHg38.over.chain.gz
#      histone/*.bw                     ENCODE bigWigs (hg19, else GRCh38)
#      hic/*.hic                        real Hi-C from ENCODE / 4DN, else stubs
#
#  Assembly policy
#  ---------------
#  Our PCHi-C fragments are hg19, but ENCODE has largely retired hg19 outputs.
#  So: prefer hg19; fall back to GRCh38; and rather than rebuilding whole
#  genome-wide tracks, keep GRCh38 files as-is and let the notebook lift the
#  QUERY COORDINATES with pyliftover. That is exact for point/interval lookups
#  and costs nothing. Use --liftover-tracks if you want genuine hg19 bigWigs on
#  disk anyway (slow, tens of GB of scratch per track).
#
#  A .hic contact matrix cannot be lifted at all without re-running Juicer from
#  raw reads, so Hi-C is always queried with lifted coordinates.
#
#  Usage:
#      bash download_data.sh                  # everything
#      bash download_data.sh --skip-genome    # genome already present
#      bash download_data.sh --skip-histone   # don't touch ENCODE histone
#      bash download_data.sh --skip-hic       # don't fetch real Hi-C
#      bash download_data.sh --liftover-tracks  # convert GRCh38 bigWigs to hg19
#      bash download_data.sh --hic-max-gb 5   # size cap per .hic file
#      bash download_data.sh --no-dummy-hic   # don't create placeholder stubs
#
#  Run this from inside the conda env created by setup_env.sh:
#      conda activate pangenomic-gat && bash download_data.sh
# =============================================================================
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${ROOT}/reference"
HIST_DIR="${ROOT}/histone"
HIC_DIR="${ROOT}/hic"
INT_DIR="${ROOT}/interection_data"

SKIP_GENOME=0; SKIP_HISTONE=0; SKIP_HIC=0; DUMMY_HIC=1
LIFTOVER_TRACKS=0; HIC_MAX_GB=1200; HIC_SOURCES="encode 4dn"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-genome)     SKIP_GENOME=1; shift ;;
    --skip-histone)    SKIP_HISTONE=1; shift ;;
    --skip-hic)        SKIP_HIC=1; shift ;;
    --liftover-tracks) LIFTOVER_TRACKS=1; shift ;;
    --hic-max-gb)      HIC_MAX_GB="$2"; shift 2 ;;
    --hic-sources)     HIC_SOURCES="$2"; shift 2 ;;
    --dummy-hic)       DUMMY_HIC=1; shift ;;
    --no-dummy-hic)    DUMMY_HIC=0; shift ;;
    -h|--help)         sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
say()  { printf "\n${BOLD}==> %s${OFF}\n" "$*"; }
ok()   { printf "  ${GREEN}OK${OFF}    %s\n" "$*"; }
warn() { printf "  ${YELLOW}WARN${OFF}  %s\n" "$*"; }
die()  { printf "  ${RED}FAIL${OFF}  %s\n" "$*" >&2; exit 1; }

mkdir -p "${REF_DIR}" "${HIST_DIR}" "${HIC_DIR}"

# resumable download that skips work if the file is already complete
fetch() {  # fetch <url> <dest> [min_bytes]
  local url="$1" dest="$2" min="${3:-1024}"
  if [[ -s "$dest" ]]; then
    local sz; sz=$(wc -c < "$dest")
    if (( sz >= min )); then ok "$(basename "$dest") already present ($(numfmt --to=iec "$sz" 2>/dev/null || echo "$sz B"))"; return 0; fi
    warn "$(basename "$dest") looks truncated (${sz} B) — refetching"
  fi
  echo "  ..  $(basename "$dest")  <-  ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 3 -C - -o "${dest}.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${dest}.part" "$url"
  else
    die "neither curl nor wget is available"
  fi
  mv "${dest}.part" "$dest"
  ok "$(basename "$dest") ($(numfmt --to=iec "$(wc -c < "$dest")" 2>/dev/null || echo ok))"
}

# ----------------------------------------------------------------- 1. genome
if [[ $SKIP_GENOME -eq 0 ]]; then
  say "1/5  Reference genome (hg19 / GRCh37)"
  echo "  The PCHi-C fragment coordinates in interection_data/ are hg19 —"
  echo "  the notebook re-verifies this against refGene before using it."
  fetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit" \
        "${REF_DIR}/hg19.2bit" 700000000
  fetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.chrom.sizes" \
        "${REF_DIR}/hg19.chrom.sizes" 1000
else
  say "1/5  --skip-genome given"
fi

# ----------------------------------------------------------------- 1b. chains
say "2/5  liftOver chain files (both directions)"
echo "  hg38->hg19 converts downloaded GRCh38 tracks to hg19 (--liftover-tracks)."
echo "  hg19->hg38 converts our fragment coordinates when querying a GRCh38"
echo "  bigWig or .hic in the notebook. Both are small (~1-10 MB)."
fetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz" \
      "${REF_DIR}/hg38ToHg19.over.chain.gz" 100000
fetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz" \
      "${REF_DIR}/hg19ToHg38.over.chain.gz" 100000

# ----------------------------------------------------------------- 2. refGene
say "3/5  Gene models (RefSeq, hg19) for canonical TSS"
if [[ ! -s "${REF_DIR}/refGene_hg19.txt" ]]; then
  fetch "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/refGene.txt.gz" \
        "${REF_DIR}/refGene.txt.gz" 5000000
  gunzip -c "${REF_DIR}/refGene.txt.gz" > "${REF_DIR}/refGene_hg19.txt"
  ok "refGene_hg19.txt ($(wc -l < "${REF_DIR}/refGene_hg19.txt") rows)"
else
  ok "refGene_hg19.txt already present"
fi

# ----------------------------------------------------------------- 3. histone
if [[ $SKIP_HISTONE -eq 0 ]]; then
  say "4/5  Histone marks — filling the gaps from ENCODE"
  echo "  You currently have bigWigs for only a handful of sample IDs."
  echo "  fetch_histone_encode.py maps each tissue to an ENCODE biosample and"
  echo "  takes the best fold-change-over-control bigWig, preferring hg19 and"
  echo "  falling back to GRCh38. Results go to histone/DOWNLOAD_manifest.csv"
  echo "  with match_quality and assembly columns."
  HIST_ARGS=(--interactions "${INT_DIR}" --out "${HIST_DIR}"
             --assemblies hg19 GRCh38
             --chain "${REF_DIR}/hg38ToHg19.over.chain.gz"
             --chrom-sizes "${REF_DIR}/hg19.chrom.sizes")
  if [[ $LIFTOVER_TRACKS -eq 1 ]]; then
    HIST_ARGS+=(--liftover-tracks)
    warn "--liftover-tracks: expect ~10-30 GB scratch and minutes per track"
  fi
  python "${ROOT}/codes/fetch_histone_encode.py" "${HIST_ARGS[@]}"
else
  say "4/5  --skip-histone given"
fi

# ----------------------------------------------------------------- 4. hic
say "5/5  Hi-C contact matrices"

if [[ $SKIP_HIC -eq 0 ]]; then
  echo "  Querying ENCODE and the 4D Nucleome portal for real .hic files."
  echo "  These are large; --hic-max-gb ${HIC_MAX_GB} caps each download."
  echo "  Almost all are GRCh38: a .hic matrix cannot be lifted without"
  echo "  re-running Juicer, so the notebook lifts the query coordinates."
  # shellcheck disable=SC2086
  python "${ROOT}/codes/fetch_hic.py" \
      --interactions "${INT_DIR}" \
      --out "${HIC_DIR}" \
      --sources ${HIC_SOURCES} \
      --max-gb "${HIC_MAX_GB}" || warn "fetch_hic.py exited non-zero; continuing"
else
  echo "  --skip-hic given; not querying ENCODE/4DN"
fi
if [[ $DUMMY_HIC -eq 1 ]]; then
  echo
  echo "  Stubbing any tissue that still has no real .hic, so every tissue has a"
  echo "  file and the notebook exercises one code path. A stub is detected by"
  echo "  its header and masked (has_hic=0), never mistaken for zero contacts."
  n=0; real=0
  while IFS= read -r f; do
    base="$(basename "$f")"; base="${base%_merged_loop.txt}"
    target="${HIC_DIR}/${base}.hic"
    if [[ -s "$target" ]] && [[ "$(head -c 3 "$target")" == "HIC" ]]; then
      real=$((real+1)); continue          # genuine Juicer archive, leave alone
    fi
    if [[ ! -e "$target" ]]; then
      printf 'DUMMY_HIC_PLACEHOLDER\ttissue=%s\tnote=replace_with_real_hic\n' "$base" > "$target"
      n=$((n+1))
    fi
  done < <(find "${INT_DIR}" -name '*_merged_loop.txt' | sort)
  ok "hic/: ${real} real .hic, ${n} new stubs, $(find "${HIC_DIR}" -name '*.hic' | wc -l | tr -d ' ') total"
  cat > "${HIC_DIR}/README.txt" <<'EOF'
Hi-C directory.

Real Juicer archives start with the magic bytes "HIC" and were fetched from
ENCODE or 4DN by fetch_hic.py -- see HIC_manifest.csv for the source, accession
and, importantly, the genome ASSEMBLY of each one.

Files that begin with DUMMY_HIC_PLACEHOLDER are stubs for tissues where no .hic
exists on either portal. The notebook detects them, sets has_hic = 0 for those
edges and masks the feature, exactly as it does for a missing histone mark. It
never treats a stub as "zero contacts".

Assembly note
-------------
Nearly all public Hi-C is GRCh38 while our PCHi-C fragments are hg19. A .hic
file is a binary multi-resolution normalised contact matrix and cannot be
lifted between assemblies without re-running Juicer from the raw reads. The
notebook therefore converts each hg19 query coordinate to GRCh38 with
pyliftover before asking the matrix, which is exact and costs nothing.

To add your own data, drop a Juicer .hic named {sampleid}_{Tissue}.hic here and
add a row to HIC_manifest.csv giving its assembly.
EOF
else
  ok "--no-dummy-hic — leaving hic/ alone"
fi

# ----------------------------------------------------------------- summary
say "Inventory"
python - "$ROOT" <<'PY'
import sys, os, glob
root = sys.argv[1]
def count(pat): return len(glob.glob(os.path.join(root, pat)))
def size(p):
    p = os.path.join(root, p)
    if not os.path.exists(p): return "absent"
    b = os.path.getsize(p)
    for u in "B KB MB GB".split():
        if b < 1024: return f"{b:.0f} {u}"
        b /= 1024
    return f"{b:.1f} TB"
rows = [
    ("interaction files", count("interection_data/*_merged_loop.txt")),
    ("histone bigWigs",   count("histone/*.bw")),
    ("hic files",         count("hic/*.hic")),
    ("rna-seq matrices",  count("rna_seq/*.txt")),
]
for k, v in rows:
    print(f"  {k:<20} {v}")
for f in ("reference/hg19.2bit", "reference/refGene_hg19.txt",
          "reference/hg19.chrom.sizes"):
    print(f"  {f:<32} {size(f)}")
PY

cat <<EOF

  Assets ready. Launch the notebook from the same env:

      conda activate pangenomic-gat
      jupyter lab pangenomic_gat_pipeline.ipynb

EOF
