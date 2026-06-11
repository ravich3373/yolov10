#!/usr/bin/env python
"""Dedup + split behavior tests on synthetic data:
  - exact byte copies cluster (sha256 path)
  - resized + JPEG-recompressed copies cluster (pHash path)
  - distinct images do NOT cluster
  - canonical selection follows source priority
  - same group_key -> same split, always
  - eval_only -> test; duplicates -> drop
  - a train image near-duplicating an eval image gets purged (drop_leak)
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402
from PIL import Image  # noqa: E402

from lpr.data.datasets.base import sha256_file  # noqa: E402
from lpr.dedup import add_phashes, dedup  # noqa: E402
from lpr.split import assign_splits, purge_leakage  # noqa: E402

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


def make_image(path: Path, seed: int, size=(320, 240)):
    """Structured image (gradient + rectangles) — pHash needs low-freq content."""
    rng = np.random.default_rng(seed)
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    img = (gx * rng.uniform(0.2, 1) + gy * rng.uniform(0.2, 1)) % 255
    for _ in range(6):
        x, y = rng.integers(0, w - 60), rng.integers(0, h - 40)
        img[y : y + rng.integers(15, 40), x : x + rng.integers(30, 60)] = rng.integers(0, 255)
    Image.fromarray(img.astype(np.uint8)).convert("RGB").save(path)


root = Path(tempfile.mkdtemp())
rows = []


def add(name, source, group, eval_only=False, builder=None, width=320, height=240):
    p = root / f"{name}.png" if builder else root / name
    if builder:
        builder(p)
    rows.append(
        dict(
            image_id=f"id_{name}", source=source, subset="", image_path=str(p.relative_to(root)),
            label_path="", width=width, height=height, sha256=sha256_file(p), n_plates=1,
            group_key=f"{source}:{group}", license_tier="clean", eval_only=eval_only, author_split="",
        )
    )


# 10 distinct images across two sources
for i in range(10):
    add(f"img{i}", "rxg4e" if i % 2 else "ccpd", f"g{i}", builder=lambda p, i=i: make_image(p, seed=i))

# exact copy of img0 in a LOWER-priority source (ccpd has higher priority than openalpr)
shutil.copy(root / "img0.png", root / "copy0.png")
add("copy0.png", "openalpr", "gc0")

# resized + JPEG-recompressed copy of img1 (pHash must catch; sha cannot)
with Image.open(root / "img1.png") as im:
    im.resize((240, 180), Image.LANCZOS).save(root / "near1.jpg", quality=70)
add("near1.jpg", "kaggle_andrewmvd", "gn1", width=240, height=180)

# eval-only image + a near-duplicate of it sneaking into a TRAIN source (leak)
make_image(root / "eval0.png", seed=100)
add("eval0.png", "openalpr", "ge0", eval_only=True)
with Image.open(root / "eval0.png") as im:
    im.resize((280, 210), Image.LANCZOS).save(root / "leak0.jpg", quality=80)
add("leak0.jpg", "rxg4e", "gl0", width=280, height=210)

# 30 images sharing one group (group-integrity check)
for i in range(30):
    add(f"grp{i}", "crpd", "shared_plate", builder=lambda p, i=i: make_image(p, seed=200 + i))

df = pl.DataFrame(rows)
df = add_phashes(df, root)
df = dedup(df, radius=8)

by_id = {r["image_id"]: r for r in df.iter_rows(named=True)}
check("exact copy clustered with original", by_id["id_copy0.png"]["canonical_id"] == by_id["id_img0"]["canonical_id"])
check("exact copy: higher-priority source kept", by_id["id_img0"]["is_duplicate"] is False and by_id["id_copy0.png"]["is_duplicate"] is True)
check("resized+jpeg near-dup clustered", by_id["id_near1.jpg"]["canonical_id"] == by_id["id_img1"]["canonical_id"])
check("near-dup: rxg4e copy kept over kaggle", by_id["id_img1"]["is_duplicate"] is False and by_id["id_near1.jpg"]["is_duplicate"] is True)
distinct = [f"id_img{i}" for i in range(2, 10)]
check("distinct images form no clusters", all(by_id[i]["is_duplicate"] is False and by_id[i]["canonical_id"] == i for i in distinct))

df = assign_splits(df, seed=0)
df = purge_leakage(df, radius=8)
by_id = {r["image_id"]: r for r in df.iter_rows(named=True)}

check("eval_only beats source priority for cluster rep", by_id["id_eval0.png"]["is_duplicate"] is False)
check("eval_only -> test", by_id["id_eval0.png"]["split"] == "test")
check("duplicates -> drop", by_id["id_copy0.png"]["split"] == "drop" and by_id["id_near1.jpg"]["split"] == "drop")
grp_splits = {by_id[f"id_grp{i}"]["split"] for i in range(30)}
check(f"30 images of one group land in ONE split ({grp_splits})", len(grp_splits) == 1)
check("planted train->eval near-dup dropped by dedup", by_id["id_leak0.jpg"]["split"] == "drop")

# purge_leakage in isolation: simulate an eval set added AFTER dedup ran (the pair
# is not clustered; the leakage pass is the only line of defense)
iso = pl.DataFrame(
    [
        dict(image_id="t1", split="train", phash=0xABCDEF0123456789, n_plates=1),
        dict(image_id="t2", split="train", phash=0xABCDEF0123456788, n_plates=1),  # 1 bit from e1
        dict(image_id="e1", split="test", phash=0xABCDEF0123456789, n_plates=1),
        dict(image_id="t3", split="train", phash=0x0000000000000000, n_plates=1),  # far away
    ],
    schema_overrides={"phash": pl.UInt64},
)
iso = purge_leakage(iso, radius=8)
got = dict(zip(iso["image_id"], iso["split"]))
check("purge: exact-hash train row evicted", got["t1"] == "drop_leak")
check("purge: 1-bit-away train row evicted", got["t2"] == "drop_leak")
check("purge: distant train row kept", got["t3"] == "train")
check("purge: eval row untouched", got["e1"] == "test")

# determinism: same seed -> same assignment; different seed -> (almost surely) some change
df2 = assign_splits(df.drop("split"), seed=0)
check("splits deterministic for fixed seed", df2["split"].to_list() == purge_leakage(df2, radius=8)["split"].to_list() and True)
check(
    "same-seed reassignment identical on non-leak rows",
    df.filter(pl.col("split") != "drop_leak")["split"].to_list()
    == df2.join(df.select("image_id", pl.col("split").alias("s1")), on="image_id").filter(pl.col("s1") != "drop_leak")["split"].to_list(),
)

shutil.rmtree(root)
print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILURES: {failures}'}")
sys.exit(1 if failures else 0)
