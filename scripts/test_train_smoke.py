#!/usr/bin/env python
"""End-to-end training smoke test on synthetic data. Proves the whole Tier-0 loop:

  synthetic plates -> PlateDataset -> TAL assignment -> BCE on plate channel
  -> AdamW on the 12 new tensors -> AP50 rises -> COCO logits BIT-IDENTICAL after

Synthetic scenes: dark noisy background + dark distractor rectangles + 1-3 "plates"
(bright rectangle, dark border, horizontal dark stripes — text-like). A linear probe
on frozen COCO features must separate something this distinctive; if it can't, the
wiring is broken — which is exactly what this test exists to catch.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from lpr.data.datasets.base import write_yolo_label  # noqa: E402
from lpr.models.plate_head import add_plate_class  # noqa: E402
from lpr.models.yolov10 import YOLOv10  # noqa: E402
from lpr.train import PlateDataset, collate, evaluate_ap50, load_plate_head, save_plate_head, train_plate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

rng = np.random.default_rng(0)


def synth_scene(path: Path, n_plates: int) -> list[tuple[float, float, float, float]]:
    """640x640 scene; returns plate boxes in pixels."""
    img = rng.integers(20, 90, (640, 640, 3), dtype=np.uint8)  # dark noise
    for _ in range(rng.integers(2, 5)):  # dark distractor rectangles (NOT plates)
        x, y = rng.integers(0, 500, 2)
        w, h = rng.integers(40, 140, 2)
        img[y : y + h, x : x + w] = rng.integers(0, 60, 3)
    boxes = []
    for _ in range(n_plates):
        w = int(rng.integers(60, 160))
        h = max(20, int(w / rng.uniform(2.5, 4.0)))
        x, y = int(rng.integers(0, 640 - w)), int(rng.integers(0, 640 - h))
        img[y : y + h, x : x + w] = 235  # bright plate
        img[y : y + 2, x : x + w] = 30  # border
        img[y + h - 2 : y + h, x : x + w] = 30
        img[:, x : x + 2][y : y + h] = 30
        for sy in range(y + h // 4, y + h - h // 4, 6):  # text-like stripes
            img[sy : sy + 2, x + 6 : x + w - 6] = 40
        boxes.append((x, y, x + w, y + h))
    Image.fromarray(img).save(path)
    return boxes


def build_corpus(root: Path, n_train=96, n_val=24) -> pl.DataFrame:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    rows = []
    for i in range(n_train + n_val):
        img = root / "images" / f"{i:04d}.png"
        boxes = synth_scene(img, n_plates=int(rng.integers(1, 4)))
        label = root / "labels" / f"{i:04d}.txt"
        write_yolo_label(label, boxes, 640, 640)
        rows.append(
            dict(image_path=str(img.relative_to(root)), label_path=str(label.relative_to(root)),
                 width=640, height=640, split="train" if i < n_train else "val")
        )
    return pl.DataFrame(rows)


def main():
    failures = []

    def check(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            failures.append(name)

    root = Path(tempfile.mkdtemp())
    corpus = build_corpus(root)
    train_ds = PlateDataset(corpus, root, "train", augment=True)
    val_ds = PlateDataset(corpus, root, "val")

    weights = REPO / "weights" / "yolov10s.pt"
    if weights.exists():
        model = YOLOv10.from_ultralytics_pt(str(weights), "s")
        print("using pretrained yolov10s (real COCO features)")
    else:
        model = YOLOv10("s").eval()
        print("WARNING: random trunk — thresholds may not be meaningful")
    trainable = add_plate_class(model)

    # reference COCO logits BEFORE training (CPU for determinism)
    torch.manual_seed(0)
    probe = torch.rand(1, 3, 640, 640)
    with torch.inference_mode():
        feats = model.forward_features(probe)
        coco_before = model.head._forward_branch(feats, model.head.one2one_cv2, model.head.one2one_cv3)["scores"][:, :80].clone()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate, num_workers=2)
    model = model.to(device)
    ap_before = evaluate_ap50(model, val_loader, device)["ap50"]

    history = train_plate(model, trainable, train_ds, val_ds, epochs=8, batch_size=8, lr=1e-2, device=device, workers=2)
    ap_after = history[-1]["ap50"]

    check(f"loss decreased ({history[0]['loss']:.3f} -> {history[-1]['loss']:.3f})", history[-1]["loss"] < 0.5 * history[0]["loss"])
    check(f"AP50 rose from ~0 ({ap_before:.3f}) to >0.5 ({ap_after:.3f})", ap_before < 0.05 and ap_after > 0.5)

    # COCO must not have moved a single bit
    model = model.cpu()
    with torch.inference_mode():
        feats = model.forward_features(probe)
        coco_after = model.head._forward_branch(feats, model.head.one2one_cv2, model.head.one2one_cv3)["scores"][:, :80]
    check("COCO logits bit-identical after training", torch.equal(coco_before, coco_after))

    # checkpoint round-trip: save 12 tensors, load into a FRESH surgered model, same AP
    ckpt = root / "plate_head.pt"
    save_plate_head(model, ckpt, meta={"smoke": True})
    check(f"checkpoint is tiny ({ckpt.stat().st_size} bytes)", ckpt.stat().st_size < 100_000)
    fresh = YOLOv10.from_ultralytics_pt(str(weights), "s") if weights.exists() else YOLOv10("s").eval()
    add_plate_class(fresh)
    load_plate_head(fresh, ckpt)
    ap_fresh = evaluate_ap50(fresh.to(device), val_loader, device)["ap50"]
    check(f"reloaded head reproduces AP ({ap_fresh:.3f} vs {ap_after:.3f})", abs(ap_fresh - ap_after) < 1e-4)

    print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILURES: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
