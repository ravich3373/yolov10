#!/usr/bin/env python
"""Prove the plate-class surgery (src/lpr/models/plate_head.py) cannot harm COCO:

  1. all 80 COCO class scores and all boxes are BIT-IDENTICAL before/after surgery,
     in both the one2many and one2one branches
  2. the new plate channel starts as a constant background prior
  3. exactly the 6 new convs (12 tensors) are trainable, nothing else
  4. even after (fake) training the new convs, COCO outputs remain bit-identical
  5. fused export head (single conv) is bit-identical to the split head
  6. the end-to-end eval path emits class id 80 without breaking

Uses pretrained weights when weights/yolov10s.pt exists, random init otherwise
(bit-identity is weight-independent).
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lpr.models.yolov10 import YOLOv10  # noqa: E402
from lpr.models.plate_head import add_plate_class, fuse_plate_head  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "s"


def branch_outputs(model, x):
    """Raw (boxes, scores) of both branches, with no postprocessing."""
    feats = model.forward_features(x)
    h = model.head
    o2m = h._forward_branch(feats, h.cv2, h.cv3)
    o2o = h._forward_branch(feats, h.one2one_cv2, h.one2one_cv3)
    return o2m, o2o


def main():
    weights = REPO / "weights" / f"yolov10{VARIANT}.pt"
    if weights.exists():
        model = YOLOv10.from_ultralytics_pt(str(weights), VARIANT)
        print(f"using pretrained {weights.name}")
    else:
        model = YOLOv10(VARIANT).eval()
        print("using random init (no weights file found)")

    torch.manual_seed(0)
    x = torch.rand(2, 3, 640, 640)
    with torch.inference_mode():
        (base_o2m, base_o2o) = branch_outputs(model, x)

    trainable = add_plate_class(model)  # in-place, freeze=True
    model.eval()

    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    with torch.inference_mode():
        (new_o2m, new_o2o) = branch_outputs(model, x)

    for tag, base, new in (("one2many", base_o2m, new_o2m), ("one2one", base_o2o, new_o2o)):
        check(f"{tag}: 80 COCO score channels bit-identical", torch.equal(new["scores"][:, :80], base["scores"]))
        check(f"{tag}: box outputs bit-identical", torch.equal(new["boxes"], base["boxes"]))
        plate = new["scores"][:, 80]
        check(f"{tag}: plate channel is a constant prior (~{plate.flatten()[0]:.2f}..)", plate.std() < 1e-6 or plate.unique().numel() <= 3)

    n_trainable = sum(p.requires_grad for p in model.parameters())
    n_total = sum(1 for _ in model.parameters())
    expect = len(trainable)
    check(f"exactly {expect} trainable tensors of {n_total} total (12 expected)", n_trainable == expect == 12)
    c3 = model.head.cv3[0][2].orig.in_channels
    n_params = sum(p.numel() for p in trainable)
    check(f"trainable params = 6*({c3}+1) = {n_params}", n_params == 6 * (c3 + 1))

    # fake training step: perturb the new convs hard, COCO must not move
    with torch.no_grad():
        for p in trainable:
            p.add_(torch.randn_like(p))
    with torch.inference_mode():
        (post_o2m, post_o2o) = branch_outputs(model, x)
    check("after training: COCO scores still bit-identical (one2many)", torch.equal(post_o2m["scores"][:, :80], base_o2m["scores"]))
    check("after training: COCO scores still bit-identical (one2one)", torch.equal(post_o2o["scores"][:, :80], base_o2o["scores"]))
    check("after training: plate channel actually changed", not torch.equal(post_o2o["scores"][:, 80], new_o2o["scores"][:, 80]))

    # BN-stat immunity: model.train() + forward must not move a single buffer.
    # This is what torch.onnx.export does internally during tracing (verified) —
    # without the freeze_bn_stats guard it silently corrupts the frozen trunk.
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    mean_before = bn.running_mean.clone()
    model.train()
    with torch.no_grad():
        model.forward_features(x)
    model.eval()
    check("BN stats immune to train()+forward (the onnx.export footgun)", torch.equal(bn.running_mean, mean_before))
    with torch.inference_mode():
        (guard_o2m, _) = branch_outputs(model, x)
    check("COCO scores still bit-identical after train()+forward", torch.equal(guard_o2m["scores"][:, :80], base_o2m["scores"]))

    # fused export head == split head
    with torch.inference_mode():
        split_out = model(x)
        fuse_plate_head(model)
        fused_out = model(x)
    check("fused single-conv head bit-identical to split head", torch.equal(split_out, fused_out))
    check("eval path emits 81-class detections (B,300,6)", fused_out.shape == (2, 300, 6) and fused_out[..., 5].max() <= 80)

    ok = all(c for _, c in checks)
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
