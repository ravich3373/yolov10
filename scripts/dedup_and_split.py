#!/usr/bin/env python
"""Stage 2+3 of the pipeline: pHash every image, cluster duplicates across ALL
built datasets, assign group-aware splits, purge train->eval leakage, and write
the combined corpus manifest.

  python scripts/dedup_and_split.py [--root data] [--radius 8] [--seed 0]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402

from lpr.data.manifest import load_all_manifests, write_manifest  # noqa: E402
from lpr.dedup import add_phashes, dedup  # noqa: E402
from lpr.split import assign_splits, purge_leakage, split_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data"))
    ap.add_argument("--radius", type=int, default=8, help="pHash Hamming radius for near-dups")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root)

    df = load_all_manifests(root)
    print(f"corpus: {len(df):,} images from {df['source'].n_unique()} datasets")

    print("computing pHashes...")
    df = add_phashes(df, root)
    print("clustering duplicates...")
    df = dedup(df, radius=args.radius)
    n_dup = df["is_duplicate"].sum()
    print(f"  {n_dup:,} duplicates across {df.filter(pl.col('is_duplicate'))['canonical_id'].n_unique():,} clusters")

    df = assign_splits(df, seed=args.seed)
    df = purge_leakage(df, radius=args.radius)

    out = root / "corpus.parquet"
    write_manifest(df, out)
    print(f"\nwrote {out}")
    print(split_report(df))


if __name__ == "__main__":
    main()
