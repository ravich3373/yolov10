"""Tier-0 class-addition surgery: add `license_plate` as class 80 to a COCO-pretrained
YOLOv10 without changing a single COCO prediction.

The idea (see yolov10.py's docstring for why this is sufficient): the only per-class
parameters in YOLOv10 are the output channels of the final 1x1 conv of each
classification branch — ``head.cv3[i][2]`` and ``head.one2one_cv3[i][2]`` for the
three scales i. Everything else (backbone, neck, box regression, earlier cls-branch
layers) is shared across classes and stays frozen.

Surgery: each of those six nc-channel convs is replaced by a :class:`SplitFinalConv`
that runs the ORIGINAL conv (frozen) and a NEW 1-channel conv (trainable) on the same
input and concatenates logits 80 -> 81. Because the original conv's weights are
untouched and nothing upstream changes, COCO class scores and all boxes are
bit-identical before and after surgery — asserted by scripts/verify_plate_surgery.py.

Training notes
==============
- New conv weights are ZERO-initialized with a strongly negative "prior" bias, so the
  fresh plate channel starts out predicting "background everywhere" at the same prior
  COCO classes were born with (sigmoid(bias) ~ 1e-4..1e-5). Training is stable from
  step 0 and step-0 plate predictions are no-ops.
- Keep the model in eval() during training. The only trainable modules are plain
  convs (mode-independent), and eval() keeps every frozen BatchNorm's running stats
  untouched — the classic `requires_grad=False does not freeze BN stats` trap.
- No special no_grad handling is needed for speed: since every upstream parameter has
  requires_grad=False, autograd builds a graph only from the new convs onward.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from .yolov10 import YOLOv10, V10Detect

PLATE_CLASS_NAME = "license_plate"


def freeze_bn_stats(model: nn.Module) -> None:
    """Make every BatchNorm immune to train() — its running stats can never update.

    requires_grad=False does NOT protect BN buffers; anything that flips the model
    into train mode and runs a forward silently corrupts them. This is not
    hypothetical: torch.onnx.export does exactly that during tracing (verified —
    it cost us a day of ONNX 'parity failures' that were really a poisoned
    reference model). With this guard, train(True) is a no-op for BN modules."""
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
            m.train = types.MethodType(lambda self, mode=True: self, m)


class SplitFinalConv(nn.Module):
    """Frozen original nc-channel 1x1 conv ∥ trainable k-channel 1x1 conv, concat.

    Output channel order: [original nc classes..., new classes...] so existing class
    ids are preserved and the new class lands at index nc (80 for COCO).
    """

    def __init__(self, orig: nn.Conv2d, num_new: int, prior_bias: float):
        super().__init__()
        self.orig = orig.requires_grad_(False)
        self.new = nn.Conv2d(orig.in_channels, num_new, 1)
        # Zero weights + prior bias: the plate logit is a constant `prior_bias` at
        # init (independent of input), i.e. P(plate) ~ 1e-4 everywhere. Zero-init of
        # a final layer is fine for learning — the incoming activations provide the
        # gradient signal for the weights.
        nn.init.zeros_(self.new.weight)
        nn.init.constant_(self.new.bias, prior_bias)

    def forward(self, x):
        return torch.cat([self.orig(x), self.new(x)], 1)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """Merge back into a single (nc+k)-channel conv for export (TensorRT etc.)."""
        nc, k = self.orig.out_channels, self.new.out_channels
        fused = nn.Conv2d(self.orig.in_channels, nc + k, 1)
        fused.weight.copy_(torch.cat([self.orig.weight, self.new.weight], 0))
        fused.bias.copy_(torch.cat([self.orig.bias, self.new.bias], 0))
        return fused


def _prior_bias(nc: int, stride: float) -> float:
    """Same prior the COCO classes were initialized with (see V10Detect.bias_init):
    roughly 'expect 5 objects of ~nc classes in a 640px image at this stride'."""
    return math.log(5 / nc / (640 / stride) ** 2)


def add_plate_class(model: YOLOv10, num_new: int = 1, freeze: bool = True) -> list[nn.Parameter]:
    """In-place surgery. Returns the (small) list of trainable parameters.

    With ``freeze=True`` (Tier 0) every pre-existing parameter is frozen and only the
    six new convs (3 scales x {one2many, one2one}) remain trainable.
    """
    head: V10Detect = model.head
    if freeze:
        model.requires_grad_(False)  # params (conv weights, BN affine)
        freeze_bn_stats(model)  # BN buffers (running stats) — see freeze_bn_stats

    trainable: list[nn.Parameter] = []
    for branch in (head.cv3, head.one2one_cv3):
        for i, seq in enumerate(branch):
            assert not isinstance(seq[2], SplitFinalConv), "surgery already applied"
            seq[2] = SplitFinalConv(seq[2], num_new, _prior_bias(head.nc, head.stride[i].item()))
            trainable += [seq[2].new.weight, seq[2].new.bias]

    head.nc += num_new
    head.no += num_new
    model.nc = head.nc
    return trainable


def fuse_plate_head(model: YOLOv10) -> YOLOv10:
    """Replace every SplitFinalConv with its fused single conv (for export)."""
    for branch in (model.head.cv3, model.head.one2one_cv3):
        for seq in branch:
            if isinstance(seq[2], SplitFinalConv):
                seq[2] = seq[2].fuse()
    return model
