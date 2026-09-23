#!/usr/bin/env python3
"""
feature_readers.py

Extracted HistoneExtractor and HiCReader from the training pipeline
for use in inference without full notebook dependencies.

Usage:
    from feature_readers import HistoneExtractor, HiCReader, CoordLifter
"""
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pyBigWig
    HAS_PYBIGWIG = True
except ImportError:
    HAS_PYBIGWIG = False

try:
    import hicstraw
    HAS_HICSTRAW = True
except ImportError:
    HAS_HICSTRAW = False

try:
    from pyliftover import LiftOver
    HAS_PYLIFTOVER = True
except ImportError:
    HAS_PYLIFTOVER = False


class CoordLifter:
    """Cross-assembly coordinate lifter using pyliftover."""

    def __init__(self, chain_dir=None):
        self.chain_dir = Path(chain_dir) if chain_dir else None
        self._lifters = {}
        self._cache = {}
        self.n_ok = 0
        self.n_fail = 0

    def _get(self, src, dst):
        if not HAS_PYLIFTOVER:
            return None
        key = (src, dst)
        if key in self._lifters:
            return self._lifters[key]

        if self.chain_dir is None:
            return None

        chain = self.chain_dir / f"{src}To{dst.replace('GRCh', 'Hg')}.over.chain.gz"
        if not chain.exists():
            self._lifters[key] = None
            return None

        try:
            lo = LiftOver(str(chain))
            self._lifters[key] = lo
            return lo
        except Exception:
            self._lifters[key] = None
            return None

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
                f"({100*self.n_fail/tot:.2f}% failed and returned no signal)")


class HistoneExtractor:
    """
    Lazily opens bigWig files and caches per-(sample, fragment) mean signal.

    Args:
        hist_map: dict[sample_id, dict[mark, list[(path, assembly)]]]
        marks: tuple of histone mark names
        stat: statistic to compute ("mean", "max", etc.)
        lifter: CoordLifter instance for cross-assembly lookups
        native: native assembly (default "hg19")
    """

    def __init__(self, hist_map, marks, stat="mean", lifter=None, native="hg19"):
        self.hist_map, self.marks, self.stat = hist_map, list(marks), stat
        self.lifter, self.native = lifter, native
        self._open, self._cache = {}, {}
        self.available = {
            sid: {mk: len(hist_map.get(sid, {}).get(mk, [])) > 0 for mk in self.marks}
            for sid in hist_map
        }

    def has(self, sid, mark):
        return self.available.get(sid, {}).get(mark, False)

    def mask_vector(self, sid):
        return np.array([1.0 if self.has(sid, mk) else 0.0 for mk in self.marks],
                        dtype=np.float32)

    def _bw(self, path):
        p = str(path)
        if p not in self._open:
            try:
                self._open[p] = pyBigWig.open(p)
            except Exception as e:
                print(f"  ! cannot open {Path(p).name}: {e}")
                self._open[p] = None
        return self._open[p]

    def signal(self, sid, chrom, start, end):
        """
        Extract histone signal over a genomic interval.

        Returns:
            (values[n_marks], mask[n_marks]) where coordinates are in self.native
        """
        key = (sid, chrom, start, end)
        if key in self._cache:
            return self._cache[key], self.mask_vector(sid)

        out = np.zeros(len(self.marks), dtype=np.float32)
        for i, mk in enumerate(self.marks):
            entries = self.hist_map.get(sid, {}).get(mk, [])
            if not entries:
                continue
            vals = []
            for path, asm in entries:
                bw = self._bw(path)
                if bw is None:
                    continue
                if asm == self.native:
                    q = (chrom, int(start), int(end))
                else:
                    q = self.lifter.interval(chrom, start, end,
                                           src=self.native, dst=asm) \
                        if self.lifter else None
                    if q is None:
                        continue
                c, s, e = q
                try:
                    chroms = bw.chroms()
                    # GRCh38 files use 'chr1'; some use '1'
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
            if vals:
                out[i] = float(np.mean(vals))
        self._cache[key] = out
        return out, self.mask_vector(sid)

    def close(self):
        for bw in self._open.values():
            if bw is not None:
                try: bw.close()
                except Exception: pass


class HiCReader:
    """
    Reads .hic archives, lifting query coordinates when the file's assembly
    differs from fragment coordinates. Stubs and failures degrade to
    (0.0, has_hic=0) so the feature is masked rather than faked.

    Args:
        hic_dir: directory containing .hic files and HIC_manifest.csv
        resolution: bin resolution (e.g., 10000)
        lifter: CoordLifter instance
        native: native assembly (default "hg19")
        enabled: whether to actually open files (default True)
    """

    def __init__(self, hic_dir, resolution, lifter=None, native="hg19", enabled=True):
        self.dir, self.res = Path(hic_dir), resolution
        self.lifter, self.native = lifter, native
        self.enabled = enabled and HAS_HICSTRAW
        self._files, self._cache = {}, {}
        self.status, self.assembly = {}, {}

        # Read assembly from manifest
        man = self.dir / "HIC_manifest.csv"
        self._man = {}
        if man.exists():
            try:
                m = pd.read_csv(man)
                for _, r in m.iterrows():
                    self._man[f"{str(r['sample_id']).zfill(4)}_{r['tissue']}"] = {
                        "assembly": str(r.get("assembly") or "").strip() or "unknown",
                        "source": r.get("source", ""), "status": r.get("status", ""),
                    }
            except Exception as e:
                print(f"  ! could not read HIC_manifest.csv: {e}")

    def _open(self, stem):
        if stem in self._files:
            return self._files[stem]
        path = self.dir / f"{stem}.hic"
        obj, asm = None, self._man.get(stem, {}).get("assembly", "unknown")
        if not path.exists():
            self.status[stem] = "fallback (file absent)"
        elif not self.enabled:
            self.status[stem] = "fallback (hicstraw unavailable)"
        else:
            try:
                head = path.read_bytes()[:24]
                if head.startswith(b"DUMMY_HIC_PLACEHOLDER"):
                    raise ValueError("stub placeholder, no real data")
                if head[:3] != b"HIC":
                    raise ValueError("not a Juicer .hic archive")
                obj = hicstraw.HiCFile(str(path))
                if asm in ("", "unknown"):
                    asm = getattr(obj, "getGenomeID", lambda: "unknown")() or "unknown"
                    asm = {"GRCh38": "GRCh38", "hg38": "GRCh38",
                           "hg19": "hg19", "GRCh37": "hg19"}.get(asm, asm)
                src = self._man.get(stem, {}).get("source", "")
                self.status[stem] = (f"real .hic [{asm}]"
                                    + (f" from {src}" if src else ""))
            except Exception as e:
                self.status[stem] = f"fallback ({type(e).__name__}: {e})"
                obj = None
        self.assembly[stem] = asm
        self._files[stem] = obj
        return obj

    def count(self, stem, chrom, pos_a, pos_b):
        """
        Get Hi-C contact count between two positions.

        Returns:
            (value, has_hic) where coordinates are in self.native
        """
        f = self._open(stem)
        if f is None:
            return 0.0, 0.0
        asm = self.assembly.get(stem, "unknown")
        if asm not in ("hg19", "GRCh38"):
            asm = self.native

        if asm != self.native:
            a = self.lifter.point(chrom, pos_a, self.native, asm) if self.lifter else None
            b = self.lifter.point(chrom, pos_b, self.native, asm) if self.lifter else None
            if a is None or b is None or a[0] != b[0]:
                return 0.0, 0.0
            chrom, pos_a, pos_b = a[0], a[1], b[1]

        key = (stem, chrom, int(pos_a) // self.res, int(pos_b) // self.res)
        if key in self._cache:
            return self._cache[key], 1.0
        try:
            c = chrom[3:] if chrom.startswith("chr") else chrom
            mzd = f.getMatrixZoomData(c, c, "observed", "KR", "BP", self.res)
            lo, hi = sorted((int(pos_a), int(pos_b)))
            recs = mzd.getRecords(lo, lo + self.res, hi, hi + self.res)
            val = float(recs[0].counts) if recs else 0.0
        except Exception:
            val = 0.0
        self._cache[key] = val
        return val, 1.0
