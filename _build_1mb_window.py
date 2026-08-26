"""Build pangenomic_gat_pipeline_1mb_window_fast.ipynb and
pangenomic_gat_pipeline_1mb_window_full.ipynb: the PCHi-C-only pipeline with
a 1,000,000 bp (instead of 16,384 bp) promoter window, plus the same Part III
validation strategies as the *_validations notebook.

Two real engineering changes beyond the window size, both necessary for
tractability at this scale (documented in-notebook, not silent):

  1. stem_stride 4 -> 64. A dilated/implicit-conv tower still materialises
     O(batch x channels x L_eff) intermediate tensors; L_eff = L / stride.
     At stride 4, 1,000,000 bp gives L_eff=250,000 (infeasible). At stride
     64, L_eff=15,625 -- about 4x the original 16kb-window's L_eff=4,096,
     which is a defensible "longer but not 61x longer" tradeoff.
  2. The full-genome one-hot sequence bank is kept as uint8 end-to-end and
     cast to float32 only after per-batch indexing (inside the model's
     encode()), instead of once up front. At this window size the upfront
     float32 cast the base notebook does would need ~4x the RAM of the
     uint8 array for no benefit (only ~64 genes' worth of sequence is ever
     on the float side at once, inside one training batch).

fast:  fast_mode=True,  fast_n_tissues=5, fast_n_genes=8000 -> results_1mb_window_fast/
full:  fast_mode=False (all tissues/genes)                  -> results_1mb_window_full/
"""
from pathlib import Path
from _nb_tools import load_base, replace_cell, drop_cells, append_cells, build_ipynb
import _nb_edits as E

ROOT = Path(__file__).parent

PREFLIGHT_CODE = '''# ---- preflight: this window size makes sequence storage the binding
# resource, not compute -- print the expected footprint before spending
# minutes to hours extracting sequence so a bad config is caught here, not
# after a long run.
_n_genes_est = len(genes_needed)
_bytes_per_seq = 4 * L   # 4 channels (A/C/G/T), uint8 -> 1 byte per base per channel
_raw_gb = _n_genes_est * _bytes_per_seq / 1e9
print(f"promoter window      : {L:,} bp")
print(f"genes to extract      : {_n_genes_est:,}")
print(f"one-hot array (uint8) : ~{_raw_gb:.1f} GB in RAM "
      f"({_n_genes_est:,} x 4 x {L:,} bytes)")
print(f"  cached to disk as .npz (compressed, typically smaller); reloaded "
      f"instantly on any later run with the same window/gene count.")
print(f"  the model never sees this whole array as float32 at once -- only "
      f"a {CFG.batch_size}-gene batch is cast to float per training step "
      f"(see the sequence tower's encode() call).")
if _raw_gb > 40:
    print(f"  NOTE: {_raw_gb:.0f} GB is a lot of RAM. If this node does not "
          f"have it, lower fast_n_genes / CFG.promoter_window before "
          f"continuing rather than letting the extraction step OOM.")
'''

def build_variant(mode: str):
    assert mode in ("fast", "full")
    cells = load_base()

    banner = f'''# Variant note — 1,000,000 bp promoter window ({mode.upper()})

This notebook is `pangenomic_gat_pipeline_pchic_only.ipynb` with the
promoter window widened from 16,384 bp to **1,000,000 bp (1 Mb)**, plus the
same Part III validation strategies added in
`pangenomic_gat_pipeline_pchic_only_validations.ipynb` (chromosome-block CV,
leave-one-tissue-out CV, blood-as-test). Graph construction (PCHi-C loops,
histone/CTCF features, labels) is completely unchanged — only the DNA
window fed to the sequence tower, and what it costs to hold that window in
memory, changes.

**Two things had to change beyond the window number itself, both explained
in full at the point they happen below:**

1. **`stem_stride` 4 → 64.** The Hyena stem's stride sets the sequence
   tower's effective length (`L_eff = window / stride`). At the original
   stride of 4, a 1 Mb window gives `L_eff = 250,000` — every Hyena block
   would materialise batch x channels x 250,000 intermediate tensors,
   which is not a small-tweak amount of extra memory, it is infeasible on
   ordinary hardware. Stride 64 gives `L_eff = 15,625`: about 4x the
   original 16 kb window's `L_eff = 4,096`, a "longer context, not
   61x-longer context" tradeoff. Lower `CFG.stem_stride` yourself if your
   hardware can take it; the whole architecture is driven by config, not
   hardcoded to any particular length.
2. **The sequence bank stays `uint8` end-to-end**, cast to `float32` only
   after a batch's ~64 genes are indexed out of it inside the model, not
   once up front for every gene. At 1 Mb x {"8,000" if mode == "fast" else "~20,000"} genes the
   base notebook's upfront `.float()` cast would need roughly 4x the RAM of
   the raw array for zero benefit — nothing outside one training batch is
   ever used as float. A preflight cell prints the expected memory
   footprint before extraction starts.

**This variant ({mode}):** `fast_mode={"True" if mode == "fast" else "False"}`
{"-> 8,000 genes, 5 tissues" if mode == "fast" else "-> every gene and tissue in the loop files"}.
Results go to `results_1mb_window_{mode}/`; the promoter one-hot cache
lives in the shared `cache/` dir under its own filename (keyed on window
size and gene count, so it cannot collide with the 16 kb cache from the
other notebooks).
{"" if mode == "fast" else '''

**Full-mode runtime warning.** This is the most expensive notebook in the
set: a ~4x longer effective sequence than the 16 kb baseline, run across
5-fold gene-grouped CV x 3 ablation arms, then chromosome-block CV (5 more
full runs), leave-one-tissue-out CV (**one full run per tissue** — up to 46
of them), and the blood holdout. If that is more compute than you want to
spend in one go, the two knobs to trim first are `CFG.variants` (drop
`"gat"`/`"seq"` to keep only the fused arm) and `LOTO_TISSUES` in Part III
(edit it down to a handful of representative tissues instead of all of
them) — both are called out again at the point you'd edit them.'''}'''

    cells = [["markdown", banner]] + cells

    IDX = {"config": 3, "seq_extract": 15, "cv_split": 25, "model": 27,
           "build_graphs": 19, "cell36_equiv": 37}
    assert cells[IDX["config"]][0] == "code" and "RESULTS_DIR" in cells[IDX["config"]][1]
    assert "extracting" in cells[IDX["seq_extract"]][1]
    assert "SEQ_TENSOR" in cells[IDX["cv_split"]][1]
    assert "class PangenomicGAT" in cells[IDX["model"]][1]
    assert cells[IDX["build_graphs"]][0] == "code" and "build_graphs_parallel" in cells[IDX["build_graphs"]][1]
    assert cells[IDX["cell36_equiv"]][1].strip() == E.CELL36_OLD.strip()

    # ---- config edits --------------------------------------------------
    replace_cell(cells, IDX["config"],
                'promoter_window: int = 16384      # bp, centred on the canonical TSS',
                'promoter_window: int = 1_000_000  # bp, centred on the canonical TSS -- 1 Mb variant')
    replace_cell(cells, IDX["config"],
                'seq_len_fast: int = 4096          # shorter window in fast_mode',
                'seq_len_fast: int = 1_000_000     # same 1 Mb window in fast_mode too -- fast_mode\n'
                '                                       # only trims gene/tissue COUNT here, not window length')
    replace_cell(cells, IDX["config"],
                'stem_stride: int = 4              # >1 only to bound FFT cost; 1 = base resolution',
                'stem_stride: int = 64             # forced up from 4: keeps L_eff ~15.6k at a 1 Mb window\n'
                '                                       # (see the banner cell above for the full rationale)')
    replace_cell(cells, IDX["config"],
                'fast_n_tissues: int = 8           # tissues kept when fast_mode',
                'fast_n_tissues: int = 5           # 1 Mb variant: 5 tissues in fast_mode')
    replace_cell(cells, IDX["config"],
                'fast_n_genes: int = 2500          # genes kept when fast_mode',
                'fast_n_genes: int = 8000          # 1 Mb variant: 8,000 genes in fast_mode')
    replace_cell(cells, IDX["config"],
                'fast_mode: bool = False           # subsample so the notebook runs quickly',
                f'fast_mode: bool = {"True" if mode == "fast" else "False"}{"            " if mode == "fast" else "           "}# {mode} variant')
    replace_cell(cells, IDX["config"],
                'RESULTS_DIR    = PROJECT_ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True)',
                f'RESULTS_DIR    = PROJECT_ROOT / "results_1mb_window_{mode}"; RESULTS_DIR.mkdir(exist_ok=True)')

    # ---- memory-safe sequence bank --------------------------------------
    replace_cell(cells, IDX["seq_extract"], "L = CFG.seq_len\ncache_path",
                PREFLIGHT_CODE + "\nL = CFG.seq_len\ncache_path")

    replace_cell(cells, IDX["cv_split"],
                'SEQ_TENSOR = torch.from_numpy(SEQ_ONEHOT).float()\n'
                'print(f"\\nsequence tensor for the model: {tuple(SEQ_TENSOR.shape)} "\n'
                '      f"({SEQ_TENSOR.nbytes/1e6:.0f} MB float32)")',
                'SEQ_TENSOR = torch.from_numpy(SEQ_ONEHOT)   # uint8: cast to float lazily per-batch\n'
                'print(f"\\nsequence tensor for the model: {tuple(SEQ_TENSOR.shape)} "\n'
                '      f"({SEQ_TENSOR.nbytes/1e6:.0f} MB uint8; cast to float32 per-batch in encode(), "\n'
                '      f"never materialised whole)")')

    replace_cell(cells, IDX["model"],
                "seq = seq_bank[batch.seq_idx.to(seq_bank.device)].to(dev)",
                "seq = seq_bank[batch.seq_idx.to(seq_bank.device)].to(dev).float()  "
                "# late cast: seq_bank is uint8")

    # ---- graph cache + drop broken debug cells + Part III (same as validations nb) ----
    replace_cell(cells, IDX["build_graphs"], E.CELL18_OLD_BUILD_CALL, E.CELL18_NEW_BUILD_CALL)
    replace_cell(cells, IDX["cell36_equiv"], E.CELL36_OLD, E.CELL36_NEW)
    drop_cells(cells, [38, 39])

    append_cells(cells, [
        ("markdown", E.PART3_MD_INTRO),
        ("code", E.PART3_CODE_SETUP),
        ("markdown", E.PART3_MD_CHROM),
        ("code", E.PART3_CODE_CHROM),
        ("code", E.PART3_CODE_CHROM_TRAIN),
        ("markdown", E.PART3_MD_LOTO),
        ("code", E.PART3_CODE_LOTO),
        ("markdown", E.PART3_MD_BLOOD),
        ("code", E.PART3_CODE_BLOOD),
        ("markdown", E.PART3_MD_COMPARE),
        ("code", E.PART3_CODE_COMPARE),
    ])

    build_ipynb(cells, ROOT / f"pangenomic_gat_pipeline_1mb_window_{mode}.ipynb",
               title=f"Pangenomic AI - PCHi-C only, 1Mb window ({mode})")


for _mode in ("fast", "full"):
    build_variant(_mode)
