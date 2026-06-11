#!/usr/bin/env python
"""Export the (surgered) YOLOv10 to ONNX for TensorRT, with a numerical parity check.

The exported graph is the EVAL path: backbone -> neck -> one2one head -> decode ->
top-300. NMS-free end to end — the one2many branch never enters the trace (it is
training-only), and the SplitFinalConv pairs are fused back into single convs first.
Output: (batch, 300, 6) = [x1, y1, x2, y2, score, class_id], class 80 = license_plate.

  python scripts/export_onnx.py --weights weights/yolov10s.pt \
      [--plate-head artifacts/plate_head.pt] [--batch 1] [--imgsz 640]

Then build a TensorRT engine on the target GPU (or use `make engine`):
  trtexec --onnx=artifacts/yolov10s_plate.onnx --fp16 --saveEngine=...

TODO "to the bone" passes (深 optimization, after the first real training run):
  conv+BN folding and RepVGGDW reparameterization before export (TensorRT folds
  conv+BN itself, but the parallel 7x7+3x3 in RepVGGDW only merges if we do it),
  INT8 calibration on fleet frames, profile-guided layer precision.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lpr.models.plate_head import add_plate_class, fuse_plate_head  # noqa: E402
from lpr.models.yolov10 import YOLOv10  # noqa: E402
from lpr.train import load_plate_head  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO / "weights" / "yolov10s.pt"))
    ap.add_argument("--variant", default="s")
    ap.add_argument("--plate-head", default=str(REPO / "artifacts" / "plate_head.pt"))
    ap.add_argument("--no-plate", action="store_true", help="export the plain 80-class model")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model = YOLOv10.from_ultralytics_pt(args.weights, args.variant)
    suffix = ""
    if not args.no_plate:
        add_plate_class(model)
        suffix = "_plate"
        if Path(args.plate_head).exists():
            meta = load_plate_head(model, Path(args.plate_head))
            print(f"loaded plate head {args.plate_head} (meta keys: {list(meta)})")
        else:
            print(f"WARNING: {args.plate_head} not found — exporting an UNTRAINED plate channel")
        fuse_plate_head(model)  # SplitFinalConv pairs -> single convs (verified identical)
    model = model.eval()

    out = Path(args.out) if args.out else REPO / "artifacts" / f"yolov10{args.variant}{suffix}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(args.batch, 3, args.imgsz, args.imgsz)

    # torch.onnx.export runs a traced forward in train() mode, which UPDATES BN
    # running stats on the live model (verified on torch 2.12). The surgered model
    # is already immune (freeze_bn_stats guard); snapshot+restore covers --no-plate.
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["images"],
        output_names=["detections"],
        opset_version=17,
        dynamo=False,  # legacy tracer: battle-tested for this graph (topk/gather decode)
    )
    model.load_state_dict(state)
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # parity: onnxruntime vs torch, on a REAL image, comparing confident detections.
    # (On noise inputs every anchor's score is near-identical, the top-300 selection
    # is dominated by exact ties, and torch/ORT break ties differently — comparing
    # the tie-junk produces huge spurious diffs. Confident detections are the part
    # of the output that matters and is deterministic.)
    import numpy as np
    import onnxruntime as ort

    x = _test_image(args.batch, args.imgsz)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        y_torch = model(x).numpy()
    y_onnx = sess.run(None, {"images": x.numpy()})[0]

    t_conf = y_torch[y_torch[..., 4] >= 0.1]
    o_conf = y_onnx[y_onnx[..., 4] >= 0.1]
    t_conf = t_conf[np.argsort(-t_conf[:, 4])]
    o_conf = o_conf[np.argsort(-o_conf[:, 4])]
    ok = t_conf.shape == o_conf.shape and (len(t_conf) == 0 or np.allclose(t_conf, o_conf, atol=1e-3))
    diff = float(np.abs(t_conf - o_conf).max()) if ok and len(t_conf) else float("nan")
    print(f"onnxruntime parity: {len(t_conf)} detections >=0.1 conf, max|Δ| = {diff:.2e} -> {'PASS' if ok else 'FAIL'}")
    if len(t_conf):
        print("top detections (xyxy, conf, cls):")
        for row in t_conf[:5]:
            print(f"  {row[:4].round(1).tolist()} conf={row[4]:.3f} cls={int(row[5])}")
    sys.exit(0 if ok else 1)


def _test_image(batch: int, imgsz: int) -> torch.Tensor:
    """A real image from the built corpus if available (real detections make the
    parity meaningful); deterministic structured synthetic otherwise."""
    import numpy as np

    corpus_imgs = sorted((REPO / "data" / "raw" / "openalpr").rglob("endtoend/us/*.jpg"))
    if corpus_imgs:
        import cv2

        from lpr.train import letterbox

        img = cv2.cvtColor(cv2.imread(str(corpus_imgs[0])), cv2.COLOR_BGR2RGB)
        img, _, _ = letterbox(img, imgsz)
        print(f"parity image: {corpus_imgs[0].name}")
    else:
        rng = np.random.default_rng(0)
        img = rng.integers(20, 90, (imgsz, imgsz, 3), dtype=np.uint8)
        img[200:280, 100:340] = 230  # a bright box so SOMETHING fires
        print("parity image: synthetic (build openalpr for a real-image check)")
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).repeat(batch, 1, 1, 1)


if __name__ == "__main__":
    main()
