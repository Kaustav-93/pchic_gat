#!/usr/bin/env bash
# =============================================================================
#  setup_env.sh — build the conda environment for the Pangenomic-AI
#                 promoter-centred graph attention pipeline.
#
#  Usage:
#      bash setup_env.sh                 # create env, install everything, verify
#      bash setup_env.sh --cpu           # force CPU-only PyTorch
#      bash setup_env.sh --verify-only   # skip install, just re-check the env
#      bash setup_env.sh --name myenv    # custom env name
#
#  After it finishes:
#      conda activate pangenomic-gat
#      jupyter lab notebooks/pangenomic_gat_pipeline.ipynb
#      # then pick the kernel "Python (pangenomic-gat)"
# =============================================================================
set -Eeuo pipefail

ENV_NAME="pangenomic-gat"
PY_VER="3.10"
FORCE_CPU=0
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)        ENV_NAME="$2"; shift 2 ;;
    --cpu)         FORCE_CPU=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help)     sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# ----------------------------------------------------------------- pretty print
BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
say()  { printf "\n${BOLD}==> %s${OFF}\n" "$*"; }
ok()   { printf "  ${GREEN}OK${OFF}    %s\n" "$*"; }
warn() { printf "  ${YELLOW}WARN${OFF}  %s\n" "$*"; }
die()  { printf "  ${RED}FAIL${OFF}  %s\n" "$*" >&2; exit 1; }

trap 'echo; echo "${RED}setup_env.sh failed on line $LINENO${OFF}" >&2' ERR

# ----------------------------------------------------------------- 0. conda
say "0/7  Locating conda"
if ! command -v conda >/dev/null 2>&1; then
  die "conda not found on PATH. Install Miniconda first:
        https://docs.conda.io/en/latest/miniconda.html"
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
ok "conda $(conda --version | awk '{print $2}') at ${CONDA_BASE}"

# mamba is dramatically faster; use it when present
if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"; ok "mamba found — using it as the solver"
else
  SOLVER="conda"; warn "mamba not found — falling back to conda (slower)"
fi

# ----------------------------------------------------------------- 1. hardware
say "1/7  Detecting hardware"
ACCEL="cpu"
if [[ $FORCE_CPU -eq 0 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    ACCEL="cuda"
    ok "NVIDIA GPU detected: $(nvidia-smi -L | head -1)"
  elif [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    ACCEL="mps"
    ok "Apple Silicon detected — PyTorch will use the MPS backend"
  else
    ok "No GPU detected — installing CPU build"
  fi
else
  ok "--cpu given — installing CPU build"
fi

# ----------------------------------------------------------------- 2. env
if [[ $VERIFY_ONLY -eq 0 ]]; then
  say "2/7  Creating conda env '${ENV_NAME}' (python ${PY_VER})"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    warn "env '${ENV_NAME}' already exists — reusing it (delete with: conda env remove -n ${ENV_NAME})"
  else
    $SOLVER create -y -n "${ENV_NAME}" "python=${PY_VER}" pip
    ok "env created"
  fi
else
  say "2/7  --verify-only: skipping env creation"
fi

conda activate "${ENV_NAME}"
ok "active env: ${CONDA_DEFAULT_ENV}  ($(python -c 'import sys;print(sys.version.split()[0])'))"

if [[ $VERIFY_ONLY -eq 0 ]]; then

  # --------------------------------------------------------------- 3. bioconda
  say "3/7  Installing genomics stack (bioconda / conda-forge)"
  # These are the packages that are painful via pip because they need C libraries
  # (zlib, curl, hdf5, htslib). Conda handles that for us.
  $SOLVER install -y -c conda-forge -c bioconda \
      pybigwig \
      py2bit \
      pyfaidx \
      pybedtools \
      bedtools \
      ucsc-liftover \
      ucsc-bigwigtobedgraph \
      ucsc-bedgraphtobigwig \
      ucsc-bigwigaverageoverbed \
      htslib samtools \
      hdf5 zlib curl
  ok "genomics stack installed (incl. UCSC liftOver toolchain)"

  # --------------------------------------------------------------- 4. pytorch
  say "4/7  Installing PyTorch (${ACCEL})"
  case "$ACCEL" in
    cuda)
      pip install --upgrade "torch>=2.2,<2.6" torchvision torchaudio \
          --index-url https://download.pytorch.org/whl/cu121 ;;
    mps|cpu)
      pip install --upgrade "torch>=2.2,<2.6" torchvision torchaudio ;;
  esac
  TORCH_VER="$(python -c 'import torch;print(torch.__version__)')"
  ok "torch ${TORCH_VER}"

  # --------------------------------------------------------------- 5. PyG
  say "5/7  Installing PyTorch Geometric"
  pip install --upgrade torch_geometric
  # torch-scatter / torch-sparse are optional for PyG >= 2.4 but make GATv2 and
  # the pooling ops considerably faster. They must match the torch build exactly,
  # so we install from the pinned wheel index and tolerate failure.
  TORCH_TAG="$(python - <<'PY'
import torch, re
v = re.match(r"(\d+\.\d+\.\d+)", torch.__version__).group(1)
cu = torch.version.cuda
print(f"{v}+{'cu'+cu.replace('.','') if cu else 'cpu'}")
PY
)"
  if pip install --no-cache-dir torch_scatter torch_sparse \
       -f "https://data.pyg.org/whl/torch-${TORCH_TAG}.html" 2>/dev/null; then
    ok "torch_scatter / torch_sparse installed (fast path enabled)"
  else
    warn "torch_scatter / torch_sparse unavailable for torch-${TORCH_TAG}"
    warn "PyG will fall back to its pure-torch kernels — slower but correct"
  fi
  ok "torch_geometric $(python -c 'import torch_geometric;print(torch_geometric.__version__)')"

  # --------------------------------------------------------------- 6. python
  say "6/7  Installing scientific / plotting / notebook stack"
  pip install --upgrade \
      "numpy>=1.24,<3"  "pandas>=2.0"   "scipy>=1.10" \
      "scikit-learn>=1.3" "statsmodels" \
      "matplotlib>=3.7" "seaborn>=0.13" "umap-learn" \
      "networkx>=3.0"   "tqdm"          "pyarrow" \
      "jupyterlab>=4"   "ipykernel"     "ipywidgets" "nbformat" \
      "hic-straw" \
      "pyliftover" \
      "requests"        "tenacity" \
      "gtfparse" \
      "shap"
  ok "python stack installed"

  # register the kernel so the notebook can only ever run inside this env
  python -m ipykernel install --user \
      --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"
  ok "Jupyter kernel registered as 'Python (${ENV_NAME})'"
fi

# ----------------------------------------------------------------- 7. verify
say "7/7  Verifying every dependency"
python - <<'PY'
import importlib, sys, platform

# (import name, human name, hard requirement?)
REQUIRED = [
    ("numpy",            "numpy",            True),
    ("pandas",           "pandas",           True),
    ("scipy",            "scipy",            True),
    ("sklearn",          "scikit-learn",     True),
    ("matplotlib",       "matplotlib",       True),
    ("seaborn",          "seaborn",          True),
    ("networkx",         "networkx",         True),
    ("tqdm",             "tqdm",             True),
    ("pyarrow",          "pyarrow",          True),
    ("torch",            "torch",            True),
    ("torch_geometric",  "torch_geometric",  True),
    ("pyBigWig",         "pyBigWig",         True),
    ("py2bit",           "py2bit",           True),
    ("pyfaidx",          "pyfaidx",          True),
    ("requests",         "requests",         True),
    ("nbformat",         "nbformat",         True),
    ("ipykernel",        "ipykernel",        True),
    # nice to have — pipeline degrades gracefully without these
    ("torch_scatter",    "torch_scatter",    False),
    ("torch_sparse",     "torch_sparse",     False),
    ("hicstraw",         "hic-straw",        False),
    ("pyliftover",       "pyliftover",       True),
    ("umap",             "umap-learn",       False),
    ("statsmodels",      "statsmodels",      False),
    ("pybedtools",       "pybedtools",       False),
    ("shap",             "shap",             False),
]

GREEN, YELLOW, RED, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[0m"
missing_hard, missing_soft = [], []

print(f"  python  {platform.python_version()}  ({sys.executable})")
for mod, name, hard in REQUIRED:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {GREEN}OK{OFF}    {name:<18} {ver}")
    except Exception as e:
        tag = f"{RED}FAIL{OFF}" if hard else f"{YELLOW}SKIP{OFF}"
        print(f"  {tag}  {name:<18} {type(e).__name__}: {e}")
        (missing_hard if hard else missing_soft).append(name)

# ---- command-line tools (needed for assembly conversion) --------------------
import shutil
print()
CLI = [("liftOver", True), ("bigWigToBedGraph", False),
       ("bedGraphToBigWig", False), ("bedtools", False)]
for tool, hard in CLI:
    path = shutil.which(tool)
    if path:
        print(f"  {GREEN}OK{OFF}    {tool:<18} {path}")
    else:
        tag = f"{RED}FAIL{OFF}" if hard else f"{YELLOW}SKIP{OFF}"
        print(f"  {tag}  {tool:<18} not on PATH")
        (missing_hard if hard else missing_soft).append(tool)
print("  (bigWigToBedGraph/bedGraphToBigWig/bedtools are only needed for")
print("   fetch_histone_encode.py --liftover-tracks)")

# ---- functional checks: importing is not the same as working -----------------
print()
try:
    import torch
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if getattr(torch.backends, "mps", None)
                         and torch.backends.mps.is_available()
           else "cpu")
    x = torch.randn(64, 8, device=dev)
    y = (x @ torch.randn(8, 8, device=dev)).sum()
    y.backward() if x.requires_grad else None
    print(f"  {GREEN}OK{OFF}    torch matmul on '{dev}' -> {tuple(x.shape)}")
except Exception as e:
    print(f"  {RED}FAIL{OFF}  torch runtime check: {e}")
    missing_hard.append("torch(runtime)")

try:
    import torch
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
    d = Data(x=torch.randn(5, 4),
             edge_index=torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]),
             edge_attr=torch.randn(4, 3))
    out = GATv2Conv(4, 6, heads=2, edge_dim=3)(d.x, d.edge_index, d.edge_attr)
    print(f"  {GREEN}OK{OFF}    GATv2Conv forward -> {tuple(out.shape)}")
except Exception as e:
    print(f"  {RED}FAIL{OFF}  PyG GATv2 check: {e}")
    missing_hard.append("torch_geometric(runtime)")

try:
    from pyliftover import LiftOver
    print(f"  {GREEN}OK{OFF}    pyliftover importable "
          f"(chain file is fetched by download_data.sh)")
except Exception as e:
    print(f"  {RED}FAIL{OFF}  pyliftover check: {e}")
    missing_hard.append("pyliftover")

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    import numpy as np
    yt = np.array([0, 0, 1, 1]); yp = np.array([.1, .4, .35, .8])
    print(f"  {GREEN}OK{OFF}    sklearn metrics  AUROC={roc_auc_score(yt,yp):.3f} "
          f"AUPRC={average_precision_score(yt,yp):.3f}")
except Exception as e:
    print(f"  {RED}FAIL{OFF}  sklearn check: {e}")
    missing_hard.append("scikit-learn(runtime)")

print()
if missing_hard:
    print(f"{RED}DEPENDENCY CHECK FAILED{OFF} — missing/broken: {', '.join(missing_hard)}")
    sys.exit(1)
if missing_soft:
    print(f"{YELLOW}Optional packages unavailable:{OFF} {', '.join(missing_soft)}")
    print("The notebook handles each of these with a documented fallback.")
print(f"{GREEN}All required dependencies present and functional.{OFF}")
PY

say "Done"
cat <<EOF

  Environment : ${ENV_NAME}
  Accelerator : ${ACCEL}

  Next steps
    1)  bash download_data.sh          # genome, TSS, chains, histone, Hi-C
    2)  conda activate ${ENV_NAME}
    3)  jupyter lab notebooks/pangenomic_gat_pipeline.ipynb
    4)  Kernel -> Change Kernel -> "Python (${ENV_NAME})"

  Re-check the env at any time with:
        bash setup_env.sh --verify-only

EOF
