"""Cross-dataset dedup: exact (sha256) + perceptual (64-bit pHash, Hamming search).

Why this exists (verified in the dataset audit): the public LP pool is full of
re-hosted copies — keremberke ⊂ trudk ⊂ rxg4e, OpenALPR images circulating inside
Roboflow sets, CCPD fragments re-uploaded — and Roboflow re-encodes/resizes on
export, so byte hashes alone miss most of it. pHash at Hamming radius ~8 catches
resize/re-encode copies; sha256 catches byte-identical re-hosting exactly.

Everything here is standalone (PIL + scipy + torch/numpy) — no imagededup (we don't
need the dep) and no fastdup (CC BY-NC-ND, unusable in a commercial pipeline).

Outputs: three manifest columns —
  phash         uint64 perceptual hash
  canonical_id  image_id of the cluster representative (priority source, then res)
  is_duplicate  True for every non-representative cluster member
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import torch

# When two copies of an image exist, keep the one from the earlier source here
# (own data would rank first once added). Unlisted sources rank last.
SOURCE_PRIORITY = ["uc3m_lp", "open_images_vrp", "rxg4e", "lhqow", "ccpd", "crpd", "ir_lpr", "kaggle_andrewmvd", "openalpr"]


# ---------------------------------------------------------------------------
# pHash
# ---------------------------------------------------------------------------


def phash_file(path: Path, hash_size: int = 8) -> int:
    """Classic 64-bit pHash: 32x32 grayscale -> 2D DCT -> top-left 8x8 low
    frequencies -> bit = coefficient > median. Robust to resize/re-encode."""
    import scipy.fft
    from PIL import Image

    size = hash_size * 4
    with Image.open(path) as im:
        pixels = np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    dct = scipy.fft.dctn(pixels)
    low = dct[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def add_phashes(df: pl.DataFrame, data_root: Path, workers: int = 16) -> pl.DataFrame:
    """Compute pHash for every row (skipped if the column already exists)."""
    if "phash" in df.columns:
        return df
    paths = [data_root / p for p in df["image_path"]]
    with ThreadPoolExecutor(workers) as pool:
        hashes = list(pool.map(phash_file, paths))
    return df.with_columns(pl.Series("phash", hashes, dtype=pl.UInt64))


# ---------------------------------------------------------------------------
# Hamming search (blockwise, GPU when available)
# ---------------------------------------------------------------------------


def _bits(hashes: np.ndarray) -> torch.Tensor:
    """(N,) uint64 -> (N, 64) float matrix of 0/1 bits."""
    bytes_ = hashes.astype(">u8").view(np.uint8).reshape(-1, 8)
    return torch.from_numpy(np.unpackbits(bytes_, axis=1).astype(np.float32))


def hamming_pairs(a: np.ndarray, b: np.ndarray | None = None, radius: int = 8, block: int = 4096) -> list[tuple[int, int]]:
    """All index pairs with Hamming(a[i], b[j]) <= radius. b=None means a-vs-a
    (returns i<j only). Distance via bit algebra: d = |a| + |b| - 2 a·b."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = _bits(a).to(device)
    B = A if b is None else _bits(b).to(device)
    a_pop, b_pop = A.sum(1, keepdim=True), B.sum(1, keepdim=True)
    pairs = []
    for s in range(0, len(A), block):
        d = a_pop[s : s + block] + b_pop.T - 2 * (A[s : s + block] @ B.T)
        hits = (d <= radius).nonzero()
        hits[:, 0] += s
        if b is None:
            hits = hits[hits[:, 0] < hits[:, 1]]  # upper triangle: skip self + mirrored
        pairs.extend((int(i), int(j)) for i, j in hits.cpu().numpy())
    return pairs


# ---------------------------------------------------------------------------
# Clustering + resolution
# ---------------------------------------------------------------------------


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def dedup(df: pl.DataFrame, radius: int = 8) -> pl.DataFrame:
    """Cluster duplicates (sha256 exact ∪ pHash<=radius) and pick one canonical
    image per cluster: eval-only first, then highest-priority source, then highest
    resolution.

    Eval-only first is load-bearing: benchmark images verifiably circulate inside
    public training sets. When a cluster spans both, the benchmark keeps the image
    and the train-source copy is the one dropped — never the other way around."""
    n = len(df)
    dsu = _DSU(n)

    # exact: identical bytes (pHash would also catch these, but sha256 is certain)
    for idxs in df.with_row_index().group_by("sha256").agg(pl.col("index"))["index"]:
        for i in idxs[1:]:
            dsu.union(int(idxs[0]), int(i))

    # perceptual
    for i, j in hamming_pairs(df["phash"].to_numpy(), radius=radius):
        dsu.union(int(i), int(j))

    prio = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    source = df["source"].to_list()
    area = (df["width"] * df["height"]).to_list()
    image_id = df["image_id"].to_list()
    eval_only = df["eval_only"].to_list()

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(dsu.find(i), []).append(i)

    canonical = list(image_id)
    is_dup = [False] * n
    for members in clusters.values():
        if len(members) == 1:
            continue
        rep = min(members, key=lambda i: (not eval_only[i], prio.get(source[i], len(prio)), -area[i]))
        for i in members:
            canonical[i] = image_id[rep]
            is_dup[i] = i != rep
    return df.with_columns(pl.Series("canonical_id", canonical), pl.Series("is_duplicate", is_dup))
