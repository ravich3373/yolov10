#!/usr/bin/env python
"""Box-level negative flips between two plate-head checkpoints (regression test
for model updates: which plates did the OLD model find that the NEW one loses?).

  python scripts/eval_flips.py OLD.pt NEW.pt [--split test] [--conf 0.25]

Checkpoint tier is auto-detected from its payload key (plate_head / plate_head_t1).
Output: per-source table + JSON next to the NEW checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from lpr.flips import count_flips  # noqa: E402
from lpr.models.plate_head import PlateT1Model, add_plate_class, extend_to_81_trainable  # noqa: E402
from lpr.models.yolov10 import YOLOv10  # noqa: E402
from lpr.train import PlateDataset, collate, load_plate_ft, load_plate_head, load_plate_t1  # noqa: E402


def load_model(ckpt_path: Path, weights: str, variant: str):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    base = YOLOv10.from_ultralytics_pt(weights, variant)
    if "plate_head_t1" in payload:
        model = PlateT1Model(base)
        load_plate_t1(model, ckpt_path)
        return model, "1"
    if "model_ft" in payload:
        extend_to_81_trainable(base)
        load_plate_ft(base, ckpt_path)
        return base.eval(), "ft"
    add_plate_class(base)
    load_plate_head(base, ckpt_path)
    return base, "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old", help="reference checkpoint (the deployed/previous model)")
    ap.add_argument("new", help="candidate checkpoint")
    ap.add_argument("--corpus", default=str(REPO / "data" / "corpus.parquet"))
    ap.add_argument("--root", default=str(REPO / "data"))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--weights", default=str(REPO / "weights" / "yolov10s.pt"))
    ap.add_argument("--variant", default="s")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    model_a, tier_a = load_model(Path(args.old), args.weights, args.variant)
    model_b, tier_b = load_model(Path(args.new), args.weights, args.variant)
    print(f"old: {args.old} (tier {tier_a}) | new: {args.new} (tier {tier_b})")

    ds = PlateDataset(pl.read_parquet(args.corpus), args.root, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = count_flips(model_a, model_b, loader, device, conf=args.conf)

    print(f"\n{'source':<20} {'n_gt':>6} {'det_old':>8} {'det_new':>8} {'neg_flips':>10} {'pos_flips':>10} {'NFR':>7}")
    for src, d in report.items():
        print(f"{src:<20} {d['n_gt']:>6} {d['det_a']:>8} {d['det_b']:>8} {d['neg_flips']:>10} {d['pos_flips']:>10} {d['nfr']:>7.4f}")

    out = Path(args.new).with_suffix(".flips.json")
    out.write_text(json.dumps({"old": args.old, "new": args.new, "split": args.split, "conf": args.conf, "report": report}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
