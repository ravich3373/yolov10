"""Standalone YOLOv10 — a single-file, torch-only reimplementation you can actually read.

Numerically identical to ultralytics (verified by ``scripts/verify_yolov10_parity.py``:
official weights load into this model and outputs match to float precision). The
ultralytics version is assembled at runtime by a YAML parser from modules scattered
across four files; this file is the same network written down as plain code.

Architecture (yolov10s shapes at 640x640 input, batch B)
=========================================================

    input (B, 3, 640, 640)
      │
      │  BACKBONE — extracts features at 5 scales; only /8, /16, /32 leave the backbone
      ├─ stem      Conv 3x3 s2          (B,  32, 320, 320)   P1, /2
      ├─ down_2    Conv 3x3 s2          (B,  64, 160, 160)   P2, /4
      ├─ stage_2   C2f x1
      ├─ down_3    Conv 3x3 s2          (B, 128,  80,  80)   P3, /8
      ├─ stage_3   C2f x2 ──────────────────────────┐ (skip to neck)
      ├─ down_4    SCDown               (B, 256,  40,  40)   P4, /16
      ├─ stage_4   C2f x2 ──────────────────────┐   │ (skip to neck)
      ├─ down_5    SCDown               (B, 512,  20,  20)   P5, /32
      ├─ stage_5   C2fCIB x1                     │   │
      ├─ sppf      SPPF (pool pyramid)           │   │
      └─ psa       PSA  (self-attention)         │   │
            │                                    │   │
            │  NECK (PAN-FPN) — mixes scales so each level sees both fine detail
            │  and global context. Top-down first, then bottom-up.
            ├────────────── up 2x ── cat ◄───────┘   │
            │                 │                      │
            │              td_p4   C2f → f4 (B,256,40,40)
            │                 │                      │
            │                up 2x ── cat ◄──────────┘
            │                 │
            │              td_p3   C2f → f3 (B,128,80,80)    ──► head P3 (small objs)
            │                 │
            │            bu_conv_p3 (Conv s2) ── cat with f4
            │                 │
            │              bu_p4   C2f → g4 (B,256,40,40)    ──► head P4 (medium objs)
            │                 │
            │            bu_conv_p4 (SCDown) ─── cat with p5
            │                 │
            └──────────►   bu_p5   C2fCIB → g5 (B,512,20,20) ──► head P5 (large objs)

      HEAD (v10Detect) — one anchor point per feature-map cell:
      80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400 anchor points.
      Per scale, TWO independent branches (this is the part that matters for surgery):

        cv2[i] "box branch":  Conv 3x3 → Conv 3x3 → Conv2d 1x1 → 64 ch  (4 sides x 16-bin DFL)
        cv3[i] "cls branch":  DWConv+1x1 → DWConv+1x1 → Conv2d 1x1 → nc ch (80 logits)

      and BOTH branches exist twice (dual assignment, YOLOv10's key trick):
        one2many (cv2/cv3):           trained with "many anchors per GT" — rich gradients.
        one2one  (one2one_cv2/cv3):   trained with "best anchor per GT only" — learns to
                                      output a single confident box per object, which is
                                      what makes NMS unnecessary at inference.
      At inference ONLY the one2one branch is used: decode → top-300 by score → done.
      No NMS. The one2many branch can be deleted after training (`fuse()`).

WHERE THE PER-CLASS PARAMETERS LIVE (the fact that motivated this file)
=======================================================================
The ONLY parameters tied to a specific class are the output channels of the final
1x1 conv of each cls branch:  cv3[i][2]  and  one2one_cv3[i][2]  (an nc-channel
Conv2d, one output channel per class, at each of the 3 scales). Everything else —
backbone, neck, box branches, and all cls-branch layers before the last conv — is
shared across classes. Box regression (cv2) is CLASS-AGNOSTIC: one box per anchor
point, regardless of class.

So "add a license-plate class without touching COCO" means: freeze everything,
bolt a parallel 1-channel conv next to each of the six nc-channel convs
(3 scales x {one2many, one2one}), concat logits 80→81.

Training-vs-inference outputs
=============================
- model.train(): forward returns raw branch outputs for the loss:
  {'one2many': {'boxes': (B,64,8400), 'scores': (B,nc,8400), 'feats': [...]},
   'one2one':  {...same, computed on detached feats...}}
  (one2one consumes DETACHED neck features — its gradients only shape its own
  convs, never the trunk; the trunk is shaped by the richer one2many signal.)
- model.eval(): forward returns decoded detections (B, 300, 6) as
  [x1, y1, x2, y2, score, class_id] in input-pixel coordinates. No NMS needed.

License note: the module definitions follow ultralytics (AGPL-3.0) and the YOLOv10
paper (arXiv:2405.14458); this derived file inherits AGPL-3.0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Variant table. All six models share the exact topology above and differ only in:
#   depth  — multiplies the number of bottleneck repeats inside C2f blocks
#   width  — multiplies channel counts (rounded up to a multiple of 8)
#   max_ch — channel ceiling applied BEFORE the width multiplier
#   cib    — which stages swap C2f for the cheaper C2fCIB block (depthwise convs)
#   lk     — which of those CIB stages use a 7x7 large-kernel RepVGGDW
# Stage ids are the ultralytics YAML layer indices, kept for traceability:
#   6=stage_4, 8=stage_5, 13=td_p4, 19=bu_p4, 22=bu_p5.
# ---------------------------------------------------------------------------
VARIANTS = {
    "n": dict(depth=0.33, width=0.25, max_ch=1024, cib={22}, lk={22}),
    "s": dict(depth=0.33, width=0.50, max_ch=1024, cib={8, 22}, lk={8, 22}),
    "m": dict(depth=0.67, width=0.75, max_ch=768, cib={8, 19, 22}, lk=set()),
    "b": dict(depth=0.67, width=1.00, max_ch=512, cib={8, 13, 19, 22}, lk=set()),
    "l": dict(depth=1.00, width=1.00, max_ch=512, cib={8, 13, 19, 22}, lk=set()),
    "x": dict(depth=1.00, width=1.25, max_ch=512, cib={6, 8, 13, 19, 22}, lk=set()),
}


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round channel count up to the nearest multiple of `divisor` (GPU-friendly widths)."""
    return math.ceil(x / divisor) * divisor


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """'same'-shape padding for stride-1 convs (and the standard half-pad for stride 2)."""
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Conv(nn.Module):
    """The universal YOLO conv: Conv2d (no bias) + BatchNorm + SiLU.

    BN makes the conv bias redundant; at export time conv+BN fuse into one conv.
    `g=channels` turns it into a depthwise conv (used heavily in v10 to cut FLOPs).
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act: bool | nn.Module = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # eps/momentum match ultralytics' global BN override (initialize_weights),
        # NOT the PyTorch defaults — eps changes the normalization math, and a tiny
        # per-layer difference compounds across ~150 BN layers into real divergence.
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two 3x3 convs with a residual connection. The repeated unit inside C2f."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP-style stage block: split channels, run half through n bottlenecks,
    concatenating every intermediate result, then fuse with a 1x1 conv.

    The "grow a list of feature maps then fuse" pattern gives multi-depth gradient
    paths at low cost. This is the standard stage block inherited from YOLOv8.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels (half of c2)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # [identity half, working half]
        y.extend(m(y[-1]) for m in self.m)  # each bottleneck appends its output
        return self.cv2(torch.cat(y, 1))


class RepVGGDW(nn.Module):
    """Depthwise 7x7 + depthwise 3x3 in parallel (re-parameterizable: at export the
    3x3 can be zero-padded and added into the 7x7, leaving a single conv)."""

    def __init__(self, ed):
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))


class CIB(nn.Module):
    """Compact Inverted Block — v10's cheap replacement for Bottleneck.

    All spatial convs are depthwise; all channel mixing is 1x1 (MobileNet-style
    factorization): DW 3x3 → PW expand → DW 3x3 (or 7x7 RepVGGDW) → PW project → DW 3x3.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2f with its Bottlenecks swapped for CIBs. Same wiring, fewer FLOPs."""

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling (Fast): 3 chained 5x5 max-pools ≈ pooling at 5/9/13,
    concatenated. Gives the /32 features a large, multi-scale receptive field."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """Multi-head self-attention over the HxW grid, conv-flavored:
    QKV from a 1x1 conv, plus a depthwise 3x3 'positional encoding' on V."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)  # keys/queries are half-width
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        self.qkv = Conv(dim, dim + nh_kd * 2, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSA(nn.Module):
    """Partial Self-Attention: split channels in half, run attention+FFN on one half
    only, concat back. Global context at half the attention cost. Applied once, at
    the /32 scale only (20x20 = 400 tokens — attention is affordable there)."""

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class SCDown(nn.Module):
    """Spatial-Channel decoupled downsampling: 1x1 conv changes channels, then a
    depthwise stride-2 conv shrinks the map. Much cheaper than a full 3x3 s2 conv."""

    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        return self.cv2(self.cv1(x))


# ---------------------------------------------------------------------------
# Box decoding helpers
# ---------------------------------------------------------------------------


class DFL(nn.Module):
    """Distribution Focal Loss decoder. The box branch predicts each box side as a
    SOFTMAX DISTRIBUTION over 16 discrete distances (0..15 cells) instead of one
    number — regression as classification, which trains more stably. This frozen
    conv (weights = [0,1,...,15]) just computes the distribution's expected value."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1

    def forward(self, x):
        # (B, 64, A) -> 4 sides x 16 bins -> softmax over bins -> expectation -> (B, 4, A)
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """One anchor POINT (not box) per feature-map cell, at the cell center,
    in feature-map units. Returns (sum(HW), 2) points and (sum(HW), 1) strides."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:]
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """(left, top, right, bottom) distances from the anchor point -> box corners."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------


class V10Detect(nn.Module):
    """YOLOv10 detection head: dual-assignment branches, NMS-free inference.

    Attribute names (cv2/cv3/one2one_cv2/one2one_cv3/dfl) deliberately match
    ultralytics so official checkpoints load by simple key remapping.
    """

    max_det = 300  # top-k detections kept at inference (replaces NMS)

    def __init__(self, nc=80, ch=(), reg_max=16):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)  # number of scales (3: P3, P4, P5)
        self.reg_max = reg_max  # DFL bins per box side
        self.no = nc + reg_max * 4  # raw outputs per anchor point
        self.stride = torch.tensor([8.0, 16.0, 32.0])  # plain attr (not in state_dict)
        c2 = max(16, ch[0] // 4, reg_max * 4)  # box-branch width
        c3 = max(ch[0], min(nc, 100))  # cls-branch width

        # Box branch, per scale: plain 3x3 convs -> 64 channels (4 sides x 16 DFL bins).
        # CLASS-AGNOSTIC: one box per anchor point, shared by all classes.
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
        )
        # Cls branch, per scale: v10's "light" head — two depthwise+pointwise pairs,
        # then THE 1x1 conv whose nc output channels are the only per-class params.
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),  # <- per-class parameters live HERE
            )
            for x in ch
        )
        # The one2one twin: identical structure, independent weights.
        import copy

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self.dfl = DFL(reg_max)

    def _forward_branch(self, feats, box_head, cls_head):
        """Run one branch pair over all scales; flatten HxW into one anchor axis."""
        bs = feats[0].shape[0]
        boxes = torch.cat([box_head[i](feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](feats[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=feats)  # boxes (B,64,A) scores (B,nc,A)

    def forward(self, feats):
        # one2one sees DETACHED features: its gradients train only its own convs;
        # the shared trunk is trained by the richer one2many assignment.
        one2one = self._forward_branch([f.detach() for f in feats], self.one2one_cv2, self.one2one_cv3)
        if self.training:
            one2many = self._forward_branch(feats, self.cv2, self.cv3)
            return {"one2many": one2many, "one2one": one2one}
        return self._inference(one2one)

    def _inference(self, x):
        """Decode the one2one branch into final detections. No NMS anywhere."""
        anchors, strides = (t.transpose(0, 1) for t in make_anchors(x["feats"], self.stride, 0.5))
        # DFL expectation -> ltrb distances -> xyxy in feature units -> input pixels
        dbox = dist2bbox(self.dfl(x["boxes"]), anchors.unsqueeze(0), xywh=False, dim=1) * strides
        y = torch.cat((dbox, x["scores"].sigmoid()), 1)  # (B, 4+nc, A)
        return self.postprocess(y.permute(0, 2, 1))

    def postprocess(self, preds):
        """(B, A, 4+nc) -> (B, max_det, 6) [x1,y1,x2,y2,score,cls] by pure top-k."""
        boxes, scores = preds.split([4, self.nc], dim=-1)
        batch, anchors, nc = scores.shape
        k = min(self.max_det, anchors)
        # top-k anchor points by their best class score...
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)  # (B, k, 1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        # ...then top-k (anchor, class) pairs among those
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch)[..., None], index // nc]
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores[..., None], (index % nc)[..., None].float()], dim=-1)

    def bias_init(self):
        """Prior-probability init for fresh heads (call after strides are known).
        Cls bias starts strongly negative so an untrained head predicts 'background'
        almost everywhere — needed when we later add a fresh plate channel."""
        for branch_pair in ((self.cv2, self.cv3), (self.one2one_cv2, self.one2one_cv3)):
            for i, (box, cls) in enumerate(zip(*branch_pair)):
                box[-1].bias.data[:] = 2.0
                cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------

# ultralytics stores the net as an indexed nn.Sequential; this maps their layer
# indices to our named attributes (indices 11/14 are Upsample, 12/15/18/21 are
# Concat — parameter-free, expressed inline in our forward()).
ULTRA_LAYER_MAP = {
    0: "stem",
    1: "down_2",
    2: "stage_2",
    3: "down_3",
    4: "stage_3",
    5: "down_4",
    6: "stage_4",
    7: "down_5",
    8: "stage_5",
    9: "sppf",
    10: "psa",
    13: "td_p4",
    16: "td_p3",
    17: "bu_conv_p3",
    19: "bu_p4",
    20: "bu_conv_p4",
    22: "bu_p5",
    23: "head",
}


class YOLOv10(nn.Module):
    """YOLOv10, all variants (n/s/m/b/l/x), built as explicit named modules."""

    def __init__(self, variant: str = "s", nc: int = 80):
        super().__init__()
        cfg = VARIANTS[variant]
        self.variant, self.nc = variant, nc

        def ch(c):  # base channel count -> this variant's channel count
            return make_divisible(min(c, cfg["max_ch"]) * cfg["width"])

        def reps(n):  # base repeat count -> this variant's repeat count
            return max(round(n * cfg["depth"]), 1)

        def stage(idx, c1, c2, n, backbone=False):
            """C2f or C2fCIB depending on variant; ultralytics layer idx selects."""
            if idx in cfg["cib"]:
                return C2fCIB(c1, c2, n, shortcut=True, lk=idx in cfg["lk"])
            return C2f(c1, c2, n, shortcut=backbone)  # backbone C2f uses residuals, neck C2f doesn't

        c1, c2, c3, c4, c5 = ch(64), ch(128), ch(256), ch(512), ch(1024)

        # ---- backbone ----
        self.stem = Conv(3, c1, 3, 2)  # /2
        self.down_2 = Conv(c1, c2, 3, 2)  # /4
        self.stage_2 = stage(2, c2, c2, reps(3), backbone=True)
        self.down_3 = Conv(c2, c3, 3, 2)  # /8
        self.stage_3 = stage(4, c3, c3, reps(6), backbone=True)
        self.down_4 = SCDown(c3, c4, 3, 2)  # /16
        self.stage_4 = stage(6, c4, c4, reps(6), backbone=True)
        self.down_5 = SCDown(c4, c5, 3, 2)  # /32
        self.stage_5 = stage(8, c5, c5, reps(3), backbone=True)
        self.sppf = SPPF(c5, c5, 5)
        self.psa = PSA(c5, c5)

        # ---- neck (PAN-FPN) ----
        self.td_p4 = stage(13, c5 + c4, c4, reps(3))  # top-down: P5 ctx into P4
        self.td_p3 = stage(16, c4 + c3, c3, reps(3))  # top-down: P4 ctx into P3
        self.bu_conv_p3 = Conv(c3, c3, 3, 2)  # bottom-up: P3 detail to /16
        self.bu_p4 = stage(19, c3 + c4, c4, reps(3))
        self.bu_conv_p4 = SCDown(c4, c4, 3, 2)  # bottom-up: P4 detail to /32
        self.bu_p5 = stage(22, c4 + c5, c5, reps(3))

        # ---- head ----
        self.head = V10Detect(nc, ch=(c3, c4, c5))

    def forward_features(self, x):
        """Backbone + neck. Returns the three neck outputs the head consumes —
        also the natural place to cache features for frozen-trunk training."""
        x = self.down_2(self.stem(x))
        x = self.stage_2(x)
        p3 = self.stage_3(self.down_3(x))  # /8
        p4 = self.stage_4(self.down_4(p3))  # /16
        p5 = self.psa(self.sppf(self.stage_5(self.down_5(p4))))  # /32

        f4 = self.td_p4(torch.cat([F.interpolate(p5, scale_factor=2.0, mode="nearest"), p4], 1))
        f3 = self.td_p3(torch.cat([F.interpolate(f4, scale_factor=2.0, mode="nearest"), p3], 1))
        g4 = self.bu_p4(torch.cat([self.bu_conv_p3(f3), f4], 1))
        g5 = self.bu_p5(torch.cat([self.bu_conv_p4(g4), p5], 1))
        return [f3, g4, g5]

    def forward(self, x):
        return self.head(self.forward_features(x))

    # ---- weight loading -------------------------------------------------

    def load_ultralytics_state_dict(self, sd: dict):
        """Load an ultralytics DetectionModel.state_dict() (keys 'model.<idx>.<rest>')
        by remapping <idx> through ULTRA_LAYER_MAP. Strict: any architecture
        mismatch raises rather than loading silently wrong."""
        remapped = {}
        for k, v in sd.items():
            parts = k.split(".")
            assert parts[0] == "model", f"unexpected key {k}"
            remapped[".".join([ULTRA_LAYER_MAP[int(parts[1])], *parts[2:]])] = v
        self.load_state_dict(remapped, strict=True)

    @classmethod
    def from_ultralytics_pt(cls, weights_path: str, variant: str | None = None) -> "YOLOv10":
        """Build the model and load an official yolov10{n,s,m,b,l,x}.pt checkpoint."""
        from ultralytics import YOLO  # only needed to unpickle the checkpoint

        ref = YOLO(weights_path).model.float()
        variant = variant or weights_path.rsplit("yolov10", 1)[-1][0]
        model = cls(variant)
        model.load_ultralytics_state_dict(ref.state_dict())
        return model.eval()


if __name__ == "__main__":
    # Smoke test: build every variant, count params, run a forward pass.
    for v in VARIANTS:
        m = YOLOv10(v).eval()
        n_params = sum(p.numel() for p in m.parameters())
        with torch.inference_mode():
            y = m(torch.zeros(1, 3, 640, 640))
        print(f"yolov10{v}: {n_params / 1e6:6.2f}M params, eval output {tuple(y.shape)}")
