#!/usr/bin/env python
"""COCO-retention evaluation: 80-class mAP on COCO val2017 for any checkpoint
(or the plain pretrained base). The retention axis of the ours-vs-theirs table —
frozen tiers should match the base bit-for-bit; fine-tuning baselines will not.

  python scripts/eval_coco.py                          # pretrained base (reference)
  python scripts/eval_coco.py --ckpt experiments/ft/best.pt
  python scripts/eval_coco.py --ckpt ... --compare artifacts/coco_eval_base.json

Downloads val2017 (~1GB) + annotations (~241MB) into data/coco/ on first run.
Writes <out>.json with overall + per-class AP; --compare prints deltas and the
worst per-class regressions.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from lpr.models.yolov10 import YOLOv10  # noqa: E402
from lpr.train import letterbox  # noqa: E402


def ensure_coco(root: Path) -> tuple[Path, Path]:
    from lpr.data.datasets.base import download_url, extract

    img_dir, ann = root / "val2017", root / "annotations" / "instances_val2017.json"
    if not img_dir.exists():
        extract(download_url("http://images.cocodataset.org/zips/val2017.zip", root / "val2017.zip"), root)
    if not ann.exists():
        extract(download_url("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", root / "ann.zip"), root)
    return img_dir, ann


@torch.inference_mode()
def run_detector(model, img_dir: Path, coco, device: str, imgsz: int = 640, conf_min: float = 1e-3) -> list[dict]:
    """COCO-format results: contiguous class i -> sorted COCO category ids (standard)."""
    import cv2

    cat_ids = sorted(coco.getCatIds())
    results = []
    for img_id in tqdm(coco.getImgIds(), desc="coco val", unit="img", mininterval=2, dynamic_ncols=True):
        info = coco.loadImgs(img_id)[0]
        img = cv2.cvtColor(cv2.imread(str(img_dir / info["file_name"])), cv2.COLOR_BGR2RGB)
        lb, scale, (px, py) = letterbox(img, imgsz)
        t = torch.from_numpy(lb).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255.0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            dets = model(t).float()[0]
        dets = dets[(dets[:, 4] >= conf_min) & (dets[:, 5] < 80)]  # COCO classes only (drop plate)
        for x1, y1, x2, y2, score, cls in dets.cpu().numpy():
            # invert letterbox to original pixel coords
            bx, by = (x1 - px) / scale, (y1 - py) / scale
            bw, bh = (x2 - x1) / scale, (y2 - y1) / scale
            results.append(
                dict(image_id=img_id, category_id=cat_ids[int(cls)],
                     bbox=[round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)], score=round(float(score), 5))
            )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="plate checkpoint (any tier); empty = plain pretrained base")
    ap.add_argument("--weights", default=str(REPO / "weights" / "yolov10s.pt"))
    ap.add_argument("--variant", default="s")
    ap.add_argument("--coco-root", default=str(REPO / "data" / "coco"))
    ap.add_argument("--out", default="", help="output json (default: artifacts/coco_eval_<name>.json)")
    ap.add_argument("--compare", default="", help="reference eval json to diff against")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    img_dir, ann = ensure_coco(Path(args.coco_root))
    coco = COCO(str(ann))

    if args.ckpt:
        sys.path.insert(0, str(REPO / "scripts"))
        from eval_flips import load_model

        model, tier = load_model(Path(args.ckpt), args.weights, args.variant)
        name = Path(args.ckpt).parent.name + f"-t{tier}"
    else:
        model, name = YOLOv10.from_ultralytics_pt(args.weights, args.variant), "base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    results = run_detector(model, img_dir, coco, device)
    res_file = Path(args.coco_root) / "_results_tmp.json"
    res_file.write_text(json.dumps(results))
    ev = COCOeval(coco, coco.loadRes(str(res_file)), "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # per-class AP@[.5:.95] from the precision tensor: (iou, recall, cls, area, maxdet)
    cat_ids = sorted(coco.getCatIds())
    names = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    per_class = {}
    for k, cid in enumerate(cat_ids):
        p = ev.eval["precision"][:, :, k, 0, -1]
        per_class[names[cid]] = round(float(np.mean(p[p > -1])) if (p > -1).any() else 0.0, 4)

    out = Path(args.out) if args.out else REPO / "artifacts" / f"coco_eval_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt or "base", "mAP": round(ev.stats[0], 4), "mAP50": round(ev.stats[1], 4), "per_class": per_class}, indent=2))
    print(f"\nwrote {out}: mAP {ev.stats[0]:.4f}  mAP50 {ev.stats[1]:.4f}")

    if args.compare:
        ref = json.loads(Path(args.compare).read_text())
        d = ev.stats[0] - ref["mAP"]
        print(f"\nvs {ref['ckpt']}: ΔmAP {d:+.4f} ({ref['mAP']:.4f} -> {ev.stats[0]:.4f})")
        regressions = sorted(((per_class[n] - ref["per_class"].get(n, 0), n) for n in per_class))[:10]
        print("worst per-class regressions:")
        for delta, n in regressions:
            print(f"  {n:<20} {ref['per_class'].get(n, 0):.4f} -> {per_class[n]:.4f}  ({delta:+.4f})")


if __name__ == "__main__":
    main()
