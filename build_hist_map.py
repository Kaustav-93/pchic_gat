#!/usr/bin/env python3
"""
build_hist_map.py

Build hist_map from histone/DOWNLOAD_manifest.csv for feature_readers.

Returns:
    hist_map: dict[sample_id, dict[mark, list[(path, assembly)]]]
"""
from pathlib import Path
import csv

def build_hist_map(histone_dir="histone", manifest_name="DOWNLOAD_manifest.csv"):
    """
    Build hist_map from manifest.

    Args:
        histone_dir: directory containing bigWig files and manifest
        manifest_name: manifest CSV filename

    Returns:
        hist_map: {sample_id: {mark: [(path, assembly)]}}
    """
    histone_dir = Path(histone_dir)
    manifest = histone_dir / manifest_name

    if not manifest.exists():
        print(f"Warning: {manifest} not found")
        return {}

    hist_map = {}

    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row['sample_id']
            mark = row['mark']
            filename = row['filename']
            assembly = row.get('assembly', 'hg19')
            status = row.get('status', '')

            # Only include files marked as present
            if status != 'present':
                continue

            path = histone_dir / filename
            if not path.exists():
                continue

            if sid not in hist_map:
                hist_map[sid] = {}
            if mark not in hist_map[sid]:
                hist_map[sid][mark] = []

            hist_map[sid][mark].append((str(path), assembly))

    return hist_map


if __name__ == "__main__":
    hist_map = build_hist_map()

    print(f"Built hist_map for {len(hist_map)} samples")
    print(f"\nSample coverage:")
    for sid, marks in sorted(hist_map.items())[:10]:
        mark_list = ', '.join(sorted(marks.keys()))
        print(f"  {sid}: {len(marks)}/7 marks ({mark_list})")

    if len(hist_map) > 10:
        print(f"  ... and {len(hist_map) - 10} more samples")
