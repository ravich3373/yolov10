#!/usr/bin/env python
"""Prove the standalone YOLOv10 (src/lpr/models/yolov10.py) is numerically identical
to ultralytics: load official weights into both, run the same inputs, compare.

Checks per variant:
  1. strict state-dict load (any architecture mismatch raises)
  2. parameter-count parity
  3. max abs difference of final detections (B, 300, 6) on random inputs < 1e-4
  4. a real image (bus.jpg): same boxes/classes from both implementations
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ultralytics import YOLO  # noqa: E402

from lpr.models.yolov10 import YOLOv10  # noqa: E402

VARIANTS = sys.argv[1:] or ["s", "n"]
WEIGHTS_DIR = REPO / "weights"


def load_pair(variant):
    ref = YOLO(str(WEIGHTS_DIR / f"yolov10{variant}.pt")).model.float().eval()
    mine = YOLOv10(variant).float().eval()
    mine.load_ultralytics_state_dict(ref.state_dict())  # strict
    return ref, mine


def main():
    WEIGHTS_DIR.mkdir(exist_ok=True)
    failures = 0
    for variant in VARIANTS:
        ref, mine = load_pair(variant)

        n_ref = sum(v.numel() for v in ref.state_dict().values())
        n_mine = sum(v.numel() for v in mine.state_dict().values())

        torch.manual_seed(0)
        x = torch.rand(2, 3, 640, 640)
        with torch.inference_mode():
            y_ref = ref(x)
            y_ref = y_ref[0] if isinstance(y_ref, (tuple, list)) else y_ref
            y_mine = mine(x)
        diff = (y_ref - y_mine).abs().max().item()

        ok = n_ref == n_mine and diff < 1e-4
        failures += not ok
        print(
            f"yolov10{variant}: params ref={n_ref:,} mine={n_mine:,} | "
            f"max|Δ| over {tuple(y_ref.shape)} outputs = {diff:.2e} | "
            f"{'PASS' if ok else 'FAIL'}"
        )

    # Real-image sanity check on the first variant: do both implementations
    # produce the same detections on an actual photo?
    bus = Path("/tmp/ultralytics-src/ultralytics/assets/bus.jpg")
    if bus.exists():
        import numpy as np
        import cv2

        img = cv2.cvtColor(cv2.imread(str(bus)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        ref, mine = load_pair(VARIANTS[0])
        with torch.inference_mode():
            y_ref = ref(t)
            y_ref = y_ref[0] if isinstance(y_ref, (tuple, list)) else y_ref
            y_mine = mine(t)
        same = (y_ref - y_mine).abs().max().item()
        names = YOLO(str(WEIGHTS_DIR / f"yolov10{VARIANTS[0]}.pt")).names
        dets = y_mine[0]
        keep = dets[:, 4] > 0.5
        print(f"\nbus.jpg (yolov10{VARIANTS[0]}), max|Δ| ref-vs-mine = {same:.2e}; detections >0.5 conf:")
        for d in dets[keep]:
            x1, y1, x2, y2, conf, cls = d.tolist()
            print(f"  {names[int(cls)]:<12} conf={conf:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        failures += same >= 1e-4

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
