#!/usr/bin/env python
"""WiSE-FT baseline (arXiv:2109.01903; also the 1-task degenerate case of every
task-arithmetic/merging method): interpolate the naive-FT checkpoint with the
pretrained base, theta(alpha) = (1-alpha)*base + alpha*ft, and trace the
plate-AP-vs-COCO-retention tradeoff curve that the merging literature buys with.

  python scripts/wise_ft.py experiments/ft-run/best.pt --alphas 0.25 0.5 0.75 1.0

For the 81-channel cls convs (base has 80), the first 80 output rows interpolate
and the new plate row keeps the FT weights scaled by alpha (at alpha=0 the plate
logit collapses to its prior — the curve's honest left end). Each alpha saves a
model_ft-format checkpoint (works with eval_flips.py / eval_coco.py) and reports
plate AP on the corpus val split.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from lpr.models.plate_head import extend_to_81_trainable  # noqa: E402
from lpr.models.yolov10 import YOLOv10  # noqa: E402
from lpr.train import PlateDataset, collate, evaluate_ap50, load_plate_ft, save_plate_ft  # noqa: E402


def interpolate(base_sd: dict, ft_sd: dict, alpha: float) -> dict:
    out = {}
    for k, ft in ft_sd.items():
        b = base_sd.get(k)
        if b is None or b.shape != ft.shape:
            if b is not None and b.ndim == ft.ndim and ft.shape[0] == b.shape[0] + 1:
                # 81-vs-80 cls conv: lerp the shared 80 rows, scale the plate row by alpha
                merged = ft.clone()
                merged[:-1] = (1 - alpha) * b + alpha * ft[:-1]
                merged[-1] = alpha * ft[-1]
                out[k] = merged
            else:
                out[k] = ft  # FT-only tensors (none expected, but stay safe)
        else:
            out[k] = (1 - alpha) * b + alpha * ft
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ft_ckpt", help="naive-FT checkpoint (model_ft format)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--weights", default=str(REPO / "weights" / "yolov10s.pt"))
    ap.add_argument("--variant", default="s")
    ap.add_argument("--corpus", default=str(REPO / "data" / "corpus.parquet"))
    ap.add_argument("--root", default=str(REPO / "data"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = YOLOv10.from_ultralytics_pt(args.weights, args.variant)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}

    model = YOLOv10(args.variant)
    extend_to_81_trainable(model)
    load_plate_ft(model, Path(args.ft_ckpt))
    ft_sd = {k: v.clone() for k, v in model.state_dict().items()}

    val = PlateDataset(pl.read_parquet(args.corpus), args.root, "val")
    loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate, num_workers=8)

    print(f"{'alpha':>6} {'plate ap50':>11} {'recall':>8}   checkpoint")
    for alpha in args.alphas:
        model.load_state_dict(interpolate(base_sd, ft_sd, alpha))
        res = evaluate_ap50(model.to(device).eval(), loader, device)
        out = Path(args.ft_ckpt).with_name(f"wise_ft_a{alpha:g}.pt")
        save_plate_ft(model, out, meta={"alpha": alpha, "ft_ckpt": args.ft_ckpt, "val_ap50": res["ap50"]})
        print(f"{alpha:>6g} {res['ap50']:>11.4f} {res['recall']:>8.4f}   {out}")
    print("\nCOCO-retention per alpha: scripts/eval_coco.py --ckpt <wise_ft_aX.pt> --compare artifacts/coco_eval_base.json")


if __name__ == "__main__":
    main()
