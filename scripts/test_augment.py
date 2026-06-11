#!/usr/bin/env python
"""Augmentation invariants: identity affine is a no-op, mosaic geometry is sane,
boxes always stay in-frame, HSV with zero gains is exact identity, EMA math,
and the PlateDataset pipeline end-to-end on a tiny corpus."""

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from lpr.augment import hsv_augment, mosaic4, random_affine  # noqa: E402
from lpr.data.datasets.base import write_yolo_label  # noqa: E402
from lpr.train import EMA, PlateDataset  # noqa: E402

failures = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        failures.append(name)


rng = np.random.default_rng(0)
img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
boxes = np.array([[100, 120, 220, 170], [400, 300, 520, 340]], dtype=np.float32)

# identity affine: zero degrees/translate/shear and scale-range 0 -> exact no-op transform
out, b = random_affine(img.copy(), boxes.copy(), degrees=0, translate=0, scale=0, shear=0)
check("identity affine: image shape preserved", out.shape == img.shape)
check("identity affine: boxes unchanged (<0.01px)", np.allclose(b, boxes, atol=1e-2))

# real affine: boxes clipped in-frame, never negative-area
for seed in range(20):
    np.random.seed(seed)
    _, b = random_affine(img.copy(), boxes.copy(), translate=0.1, scale=0.5)
    if len(b):
        check_ok = (b[:, :2] >= 0).all() and (b[:, 2] <= 640).all() and (b[:, 3] <= 480).all() and (b[:, 2:] >= b[:, :2]).all()
        if not check_ok:
            check(f"affine seed {seed}: boxes in-frame", False)
            break
else:
    check("affine x20 seeds: boxes always in-frame, positive area", True)

# mosaic: 2x canvas, all boxes inside, counts bounded
items = [(rng.integers(0, 255, (rng.integers(200, 700), rng.integers(200, 700), 3), dtype=np.uint8),
          np.array([[10, 10, 80, 40]], dtype=np.float32)) for _ in range(4)]
canvas, mb = mosaic4(items, imgsz=640)
check("mosaic: canvas is 1280x1280", canvas.shape == (1280, 1280, 3))
check("mosaic: <=4 boxes, all within canvas", len(mb) <= 4 and (mb >= 0).all() and (mb <= 1280).all())
crop, cb = random_affine(canvas, mb, translate=0.1, scale=0.5, border=(-320, -320))
check("mosaic+affine: cropped to 640", crop.shape == (640, 640, 3))

# HSV
check("hsv: zero gains -> bit-identical", np.array_equal(hsv_augment(img, 0, 0, 0), img))
aug = hsv_augment(img, 0.015, 0.7, 0.4)
check("hsv: shape/dtype preserved, image changed", aug.shape == img.shape and aug.dtype == img.dtype and not np.array_equal(aug, img))

# EMA: shadow lags the params; double swap is identity
p = torch.nn.Parameter(torch.zeros(3))
ema = EMA([p], decay=0.5, tau=1.0)
with torch.no_grad():
    p.fill_(1.0)
ema.update()
d = 0.5 * (1 - np.exp(-1))  # effective decay after 1 update
check("ema: shadow = (1-d)*new after first update", torch.allclose(ema.shadow[0], torch.full((3,), float(1 - d))))
before = p.detach().clone()
ema.swap(); ema.swap()
check("ema: double swap restores params", torch.equal(p.detach(), before))

# PlateDataset end-to-end: full augment pipeline emits valid samples
root = Path(tempfile.mkdtemp())
(root / "labels").mkdir()
rows = []
for i in range(4):
    im = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(im).save(root / f"{i}.png")
    write_yolo_label(root / "labels" / f"{i}.txt", [(100, 120, 260, 170)], 640, 480)
    rows.append(dict(image_path=f"{i}.png", label_path=f"labels/{i}.txt", width=640, height=480, split="train"))
corpus = pl.DataFrame(rows)

ds = PlateDataset(corpus, root, "train", augment=False)
t, b = ds[0]
check("dataset eval path: uint8 tensor (3,640,640)", t.shape == (3, 640, 640) and t.dtype == torch.uint8)
# letterbox 640x480 -> scale 1.0, pad y=80: box shifts to (100,200,260,250)
check("dataset eval path: letterbox box math exact", torch.allclose(b[0], torch.tensor([100.0, 200.0, 260.0, 250.0])))

ds = PlateDataset(corpus, root, "train", augment=True)  # mosaic=1.0 default
check("dataset mosaic enabled under augment", ds.mosaic_enabled)
ok = True
for i in range(12):
    t, b = ds[i % 4]
    ok &= t.shape == (3, 640, 640) and (len(b) == 0 or ((b >= 0).all() and (b <= 640).all()))
check("dataset augment path x12: valid tensors, boxes in-frame", bool(ok))
ds.disable_mosaic()
check("close_mosaic switch works", not ds.mosaic_enabled)

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILURES: {failures}'}")
sys.exit(1 if failures else 0)
