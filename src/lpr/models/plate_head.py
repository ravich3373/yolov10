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


def extend_to_81_trainable(model: YOLOv10, num_new: int = 1) -> list[nn.Parameter]:
    """BASELINE path (naive fine-tuning, the standard IOD lower bound): enlarge the
    REAL head's final cls convs 80 -> 81 and leave EVERYTHING trainable. Old-class
    weights are copied, the new channel starts at the background prior — identical
    init to our tiers, but nothing is frozen, no BN guard: COCO is expected to
    degrade, and quantifying that degradation (mAP delta + negative flips) against
    our frozen tiers is the point of the comparison."""
    head: V10Detect = model.head
    for branch in (head.cv3, head.one2one_cv3):
        for i, seq in enumerate(branch):
            old: nn.Conv2d = seq[2]
            new = nn.Conv2d(old.in_channels, head.nc + num_new, 1)
            with torch.no_grad():
                new.weight[: head.nc] = old.weight
                new.bias[: head.nc] = old.bias
                new.weight[head.nc :] = 0.0
                new.bias[head.nc :] = _prior_bias(head.nc, head.stride[i].item())
            seq[2] = new
    head.nc += num_new
    head.no += num_new
    model.nc = head.nc
    return list(model.parameters())


# ---------------------------------------------------------------------------
# Tier 1: full parallel detection head for the plate class
# ---------------------------------------------------------------------------


class PlateDetectHead(nn.Module):
    """A complete 1-class detection head (box + cls branches, one2many + one2one)
    running on the frozen neck features, mirroring V10Detect's structure.

    Tier 0's limitation is capacity AND localization: a 1x1 probe can only
    re-weight existing features, and plates must reuse the frozen class-agnostic
    box regression that never learned part-of-object extents. Here the plate
    class gets ~1.6M parameters of its own (yolov10s), including its own box
    branch — while COCO predictions still come from the untouched original head.

    Warm start: every layer whose shape matches the pretrained COCO head is
    copied from it (the whole box branch, the cls-branch feature layers); final
    cls convs are zero-init with the background prior, like Tier 0.
    """

    def __init__(self, ch: tuple, reg_max: int = 16, init_from: V10Detect | None = None):
        super().__init__()
        from .yolov10 import Conv, DFL  # local import to avoid cycle at module load

        self.nc = 1
        self.nl = len(ch)
        self.reg_max = reg_max
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], 100)  # match the COCO head's width so warm-start shapes line up

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        import copy

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self.dfl = DFL(reg_max)
        if init_from is not None:
            self._warm_start(init_from)

    @torch.no_grad()
    def _warm_start(self, coco_head: V10Detect):
        pairs = (
            (coco_head.cv2, self.cv2),
            (coco_head.one2one_cv2, self.one2one_cv2),
            (coco_head.cv3, self.cv3),
            (coco_head.one2one_cv3, self.one2one_cv3),
        )
        for src_branch, dst_branch in pairs:
            for src, dst in zip(src_branch, dst_branch):
                matching = {k: v for k, v in src.state_dict().items() if k in dst.state_dict() and dst.state_dict()[k].shape == v.shape}
                dst.load_state_dict(matching, strict=False)
        for branch in (self.cv3, self.one2one_cv3):
            for i, seq in enumerate(branch):
                nn.init.zeros_(seq[2].weight)
                nn.init.constant_(seq[2].bias, _prior_bias(80, self.stride[i].item()))

    def forward_branch(self, feats, one2one: bool):
        box_head = self.one2one_cv2 if one2one else self.cv2
        cls_head = self.one2one_cv3 if one2one else self.cv3
        bs = feats[0].shape[0]
        boxes = torch.cat([box_head[i](feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](feats[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=feats)


class PlateT1Model(nn.Module):
    """Frozen YOLOv10 + parallel PlateDetectHead. COCO detections come from the
    original head, plate detections (class 80) from the new one; eval merges the
    two candidate sets by score. The base cannot move: params frozen, BN guarded."""

    PLATE_CLASS = 80

    def __init__(self, base: YOLOv10):
        super().__init__()
        base.requires_grad_(False)
        freeze_bn_stats(base)
        self.base = base.eval()
        ch = tuple(base.head.cv2[i][0].conv.in_channels for i in range(base.head.nl))
        self.plate_head = PlateDetectHead(ch, init_from=base.head)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.plate_head.parameters())

    def forward(self, x):
        feats = self.base.forward_features(x)
        if self.training:
            return dict(
                feats=feats,
                o2m=self.plate_head.forward_branch(feats, one2one=False),
                o2o=self.plate_head.forward_branch([f.detach() for f in feats], one2one=True),
            )
        return self._inference(feats)

    def _inference(self, feats):
        from .yolov10 import dist2bbox, make_anchors, topk_detections

        coco = self.base.head(feats)  # (B, 300, 6), classes 0..79
        p = self.plate_head.forward_branch(feats, one2one=True)
        anchors, strides = (t.transpose(0, 1) for t in make_anchors(feats, self.plate_head.stride, 0.5))
        dbox = dist2bbox(self.plate_head.dfl(p["boxes"]), anchors.unsqueeze(0), xywh=False, dim=1) * strides
        y = torch.cat((dbox, p["scores"].sigmoid()), 1)  # (B, 5, A)
        plate = topk_detections(y.permute(0, 2, 1), nc=1, max_det=300)
        plate = torch.cat([plate[..., :5], torch.full_like(plate[..., 5:], self.PLATE_CLASS)], dim=-1)
        merged = torch.cat([coco, plate], dim=1)  # (B, 600, 6)
        keep = merged[..., 4].topk(coco.shape[1], dim=1).indices  # final top-300 by score
        return merged.gather(1, keep.unsqueeze(-1).repeat(1, 1, 6))
