#!/usr/bin/env python
"""Stage 4: train the plate head (Tier 0, frozen trunk) on the corpus manifest.

  python scripts/train_plate.py --corpus data/corpus.parquet --weights weights/yolov10s.pt \
      --epochs 20 --batch-size 32 --lr 5e-3

Outputs artifacts/plate_head.pt — ONLY the 12 new tensors (a few KB); the trunk is
the unmodified official checkpoint.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402
import torch  # noqa: E402

from lpr.data.datasets.base import sha256_file  # noqa: E402
from lpr.experiment import ExperimentTracker  # noqa: E402
from lpr.models.plate_head import PlateT1Model, add_plate_class  # noqa: E402
from lpr.models.yolov10 import YOLOv10  # noqa: E402
from lpr.train import (  # noqa: E402
    PlateDataset,
    collate,
    evaluate_ap50,
    plate_loss,
    plate_t1_loss,
    save_plate_head,
    save_plate_t1,
    train_plate,
)
from torch.utils.data import DataLoader  # noqa: E402


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
    ap.add_argument("--tier", type=int, default=0, choices=(0, 1),
                    help="0: 774-param linear probe; 1: full parallel plate head (own box+cls branches, ~1.6M params)")
    ap.add_argument("--name", default="", help="experiment name (default: t<tier>); run dir = experiments/<name>")
    ap.add_argument("--out", default=str(REPO / "artifacts" / "plate_head.pt"))
    args = ap.parse_args()

    corpus = pl.read_parquet(args.corpus)
    train_ds = PlateDataset(corpus, args.root, "train", augment=True, hyp={"mosaic": args.mosaic})
    val_ds = PlateDataset(corpus, args.root, "val")
    test_ds = PlateDataset(corpus, args.root, "test")
    print(f"train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,}")
    if len(train_ds) == 0:
        sys.exit("no train rows — run build_datasets.py + dedup_and_split.py first")

    base = YOLOv10.from_ultralytics_pt(args.weights, args.variant)
    if args.tier == 0:
        model, trainable, loss_fn, save_fn = base, add_plate_class(base), plate_loss, save_plate_head
    else:
        model = PlateT1Model(base)
        trainable, loss_fn, save_fn = model.trainable_parameters(), plate_t1_loss, save_plate_t1
    n_trainable = sum(p.numel() for p in trainable)

    fingerprint = {
        "corpus": args.corpus,
        "corpus_sha256": sha256_file(Path(args.corpus)),
        "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "sources": sorted(corpus["source"].unique()),
    }
    tracker = ExperimentTracker(
        REPO / "experiments",
        args.name or f"t{args.tier}",
        config={**vars(args), "trainable_params": n_trainable},
        data_fingerprint=fingerprint,
    )
    tracker.log(f"tier {args.tier} | trainable: {n_trainable:,} params (of {sum(p.numel() for p in model.parameters()):,})")

    history = train_plate(
        model, trainable, train_ds, val_ds or None,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, lrf=args.lrf,
        warmup_epochs=args.warmup_epochs, cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic, use_ema=not args.no_ema,
        amp=not args.no_amp, workers=args.workers, loss_fn=loss_fn,
        tracker=tracker, save_fn=save_fn,
    )

    # end-of-run analysis: PR curve + prediction grid on val, final AP on test
    device = str(next(model.parameters()).device)
    extra = {"notes": f"tier{args.tier} {len(train_ds)}tr"}
    if len(val_ds):
        loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=4)
        res = evaluate_ap50(model, loader, device, return_pr=True)
        tracker.plot_pr_curve(*[__import__("numpy").array(a) for a in res["pr"]], res["ap50"])
        tracker.plot_val_predictions(model, val_ds, device)
    if len(test_ds):
        loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=4)
        test_res = evaluate_ap50(model, loader, device)
        tracker.log(f"test: {test_res}")
        extra["test_ap50"] = round(test_res["ap50"], 4)
    tracker.finish(extra)

    # keep the deploy artifact where export_onnx.py looks for it (= best checkpoint)
    out = Path(args.out)
    if args.tier == 1 and out == REPO / "artifacts" / "plate_head.pt":
        out = REPO / "artifacts" / "plate_head_t1.pt"
    best = tracker.dir / "best.pt"
    if best.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out)
        print(f"best checkpoint -> {out}")


if __name__ == "__main__":
    main()
