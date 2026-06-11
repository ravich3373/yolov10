#!/usr/bin/env python
"""Download + convert datasets into canonical labels + manifests.

  python scripts/build_datasets.py openalpr ccpd          # specific datasets
  python scripts/build_datasets.py all                    # everything registered
  python scripts/build_datasets.py --list                 # show registry
  --root data    data directory (default: <repo>/data)
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lpr.data.datasets import DATASETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="dataset keys, or 'all'")
    ap.add_argument("--root", default=str(REPO / "data"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.datasets:
        print(f"{'key':<18} {'tier':<10} {'eval-only'}")
        for key, cls in DATASETS.items():
            print(f"{key:<18} {cls.license_tier:<10} {cls.eval_only}")
        return

    keys = list(DATASETS) if args.datasets == ["all"] else args.datasets
    for key in keys:
        if key not in DATASETS:
            sys.exit(f"unknown dataset '{key}' — choices: {', '.join(DATASETS)}")

    for key in keys:
        print(f"\n=== {key} ===")
        ds = DATASETS[key](args.root)
        df = ds.build()
        n_neg = (df["n_plates"] == 0).sum()
        print(
            f"{key}: {len(df):,} images, {df['n_plates'].sum():,} plates, {n_neg:,} negatives, "
            f"{df['group_key'].n_unique():,} groups -> {ds.manifest_path}"
        )


if __name__ == "__main__":
    main()
