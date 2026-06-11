"""Manifest = the single source of truth for every image in the corpus.

Each dataset stage emits one parquet at ``data/processed/<key>/manifest.parquet``;
dedup and split are pure transformations that add columns and re-write it. Nothing
downstream (dedup, splits, training) reads dataset directories directly — only
manifests. That is what makes every stage auditable, diffable, and re-runnable.

Label convention: one YOLO-format txt per image (``class cx cy w h``, normalized),
written under ``data/processed/<key>/labels/<image_id>.txt`` with class 0 = plate.
The remap to the model's class 80 happens in the training dataloader, not on disk.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# columns written by the dataset stage
BASE_SCHEMA = {
    "image_id": pl.Utf8,  # sha256[:16] of file bytes — stable, collision-safe key
    "source": pl.Utf8,  # dataset key, e.g. "ccpd"
    "subset": pl.Utf8,  # dataset-internal subset, e.g. "ccpd_weather", "us"
    "image_path": pl.Utf8,  # relative to data root
    "label_path": pl.Utf8,  # relative to data root
    "width": pl.Int32,
    "height": pl.Int32,
    "sha256": pl.Utf8,
    "n_plates": pl.Int32,  # 0 = verified negative image
    "group_key": pl.Utf8,  # split unit: same group never straddles train/val/test
    "license_tier": pl.Utf8,  # "clean" (MIT/CC-BY/CC0) | "research" (everything else)
    "eval_only": pl.Boolean,  # never allowed into train regardless of split config
    "author_split": pl.Utf8,  # split shipped by the dataset authors ("" if none)
}
# columns added by later stages: phash (dedup), is_duplicate, canonical_id (dedup), split


def write_manifest(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_manifest(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def load_all_manifests(data_root: Path) -> pl.DataFrame:
    """Concatenate every per-dataset manifest under data/processed/*/manifest.parquet."""
    paths = sorted(Path(data_root, "processed").glob("*/manifest.parquet"))
    if not paths:
        raise FileNotFoundError(f"no manifests under {data_root}/processed/*/")
    return pl.concat([pl.read_parquet(p) for p in paths], how="diagonal")
