"""Deterministic, group-aware train/val/test splits + train-vs-eval leakage purge.

Rules, in order:
  1. duplicates (is_duplicate from dedup) never enter any split ("drop")
  2. eval_only datasets (OpenALPR, CLPD, held-out Coram footage) go entirely to test
  3. optionally respect author splits for named datasets (e.g. UC3M-LP)
  4. everything else: the GROUP (same plate / same source image lineage), not the
     image, is hashed with a seed into train/val/test — near-identical frames of one
     vehicle can never straddle splits, and adding datasets never reshuffles
     existing assignments (the hash depends only on seed + group_key)
  5. leakage pass: any train image within pHash radius of any val/test image is
     evicted from train ("drop_leak") — public sets verifiably contain benchmark
     images, so this runs every time the corpus changes
"""

from __future__ import annotations

import hashlib

import polars as pl

from .dedup import hamming_pairs

FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}


def _group_bucket(group_key: str, seed: int) -> float:
    """Stable hash of (seed, group) -> [0, 1)."""
    h = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_splits(
    df: pl.DataFrame,
    seed: int = 0,
    fractions: dict[str, float] = FRACTIONS,
    respect_author_split: tuple[str, ...] = ("uc3m_lp",),
) -> pl.DataFrame:
    t, v = fractions["train"], fractions["val"]

    def split_of(row) -> str:
        if row["is_duplicate"]:
            return "drop"
        if row["eval_only"]:
            return "test"
        if row["source"] in respect_author_split and row["author_split"]:
            return {"valid": "val"}.get(row["author_split"], row["author_split"])
        x = _group_bucket(row["group_key"], seed)
        return "train" if x < t else "val" if x < t + v else "test"

    cols = ["is_duplicate", "eval_only", "source", "author_split", "group_key"]
    splits = [split_of(r) for r in df.select(cols).iter_rows(named=True)]
    return df.with_columns(pl.Series("split", splits))


def purge_leakage(df: pl.DataFrame, radius: int = 8) -> pl.DataFrame:
    """Evict train images that are near-duplicates of any val/test image."""
    train = df.with_row_index().filter(pl.col("split") == "train")
    eval_ = df.with_row_index().filter(pl.col("split").is_in(["val", "test"]))
    if train.is_empty() or eval_.is_empty():
        return df
    pairs = hamming_pairs(train["phash"].to_numpy(), eval_["phash"].to_numpy(), radius=radius)
    leaked = {int(train["index"][i]) for i, _ in pairs}
    if leaked:
        print(f"  leakage purge: {len(leaked)} train images within radius {radius} of eval — dropped")
    splits = df["split"].to_list()
    for i in leaked:
        splits[i] = "drop_leak"
    return df.with_columns(pl.Series("split", splits))


def split_report(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("source", "split")
        .agg(pl.len().alias("images"), pl.col("n_plates").sum().alias("plates"))
        .sort("source", "split")
    )
