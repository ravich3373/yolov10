#!/usr/bin/env python
"""Stage 4: train the plate head (Tier 0, frozen trunk) on the corpus manifest.

  python scripts/train_plate.py --corpus data/corpus.parquet --weights weights/yolov10s.pt \
      --epochs 20 --batch-size 32 --lr 5e-3

Outputs artifacts/plate_head.pt — ONLY the 12 new tensors (a few KB); the trunk is
the unmodified official checkpoint.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402
import torch  # noqa: E402

from lpr.models.plate_head import add_plate_class  # noqa: E402
from lpr.models.yolov10 import YOLOv10  # noqa: E402
from lpr.train import PlateDataset, evaluate_ap50, save_plate_head, train_plate  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from lpr.train import collate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(REPO / "data" / "corpus.parquet"))
    ap.add_argument("--root", default=str(REPO / "data"))
    ap.add_argument("--variant", default="s")
    ap.add_argument("--weights", default=str(REPO / "weights" / "yolov10s.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lrf", type=float, default=0.01, help="final lr fraction")
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--cos-lr", action="store_true", help="cosine decay (default linear)")
    ap.add_argument("--close-mosaic", type=int, default=10, help="disable mosaic for last N epochs")
    ap.add_argument("--mosaic", type=float, default=1.0, help="mosaic probability")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "plate_head.pt"))
    args = ap.parse_args()

    corpus = pl.read_parquet(args.corpus)
    train_ds = PlateDataset(corpus, args.root, "train", augment=True, hyp={"mosaic": args.mosaic})
    val_ds = PlateDataset(corpus, args.root, "val")
    test_ds = PlateDataset(corpus, args.root, "test")
    print(f"train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,}")
    if len(train_ds) == 0:
        sys.exit("no train rows — run build_datasets.py + dedup_and_split.py first")

    model = YOLOv10.from_ultralytics_pt(args.weights, args.variant)
    trainable = add_plate_class(model)
    print(f"trainable: {sum(p.numel() for p in trainable)} params (of {sum(p.numel() for p in model.parameters()):,})")

    history = train_plate(
        model, trainable, train_ds, val_ds or None,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lrf=args.lrf,
        warmup_epochs=args.warmup_epochs, cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic, use_ema=not args.no_ema,
        amp=not args.no_amp, workers=args.workers,
    )

    if len(test_ds):
        device = next(model.parameters()).device
        loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=4)
        print("test:", evaluate_ap50(model, loader, str(device)))

    save_plate_head(model, Path(args.out), meta={"variant": args.variant, "weights": args.weights, "history": history})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
