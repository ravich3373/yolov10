"""Tier-0 plate-head training: teach the 6 new convs what a plate is, touch nothing else.

The model stays in eval() the whole time (frozen BatchNorms keep their running
stats; the trainable modules are plain convs, mode-independent). Autograd builds a
graph only from the new convs onward because every other parameter has
requires_grad=False — so "training" costs roughly a forward pass.

Loss follows YOLOv8/v10 semantics reduced to one class:
  - decode predicted boxes from the FROZEN box branch (used for assignment only)
  - TaskAlignedAssigner picks positive anchors per GT plate:
    alignment = score^0.5 * IoU^6, top-k candidates among anchors whose center lies
    inside the GT box, conflicts resolved by IoU
  - BCE on the plate logit against the normalized alignment score
  - one2many branch uses topk=10 (rich supervision), one2one uses topk=1 (the
    NMS-free winner-take-all behavior), exactly like full YOLOv10 training
COCO channels appear in no loss term and their convs are frozen: they cannot move.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .augment import DEFAULT_HYP, hsv_augment, mosaic4, random_affine
from .models.yolov10 import YOLOv10, dist2bbox, make_anchors

PLATE_CLASS = 80  # index of the grafted class


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def letterbox(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect, pad to size x size with gray. Returns (img, scale, (px, py))."""
    import cv2

    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, scale, (px, py)


class PlateDataset(Dataset):
    """(image, plate boxes) pairs from a corpus manifest, with the ultralytics
    train-time augmentation recipe: mosaic -> random affine -> HSV -> flip.
    Eval (augment=False) is letterbox only."""

    def __init__(
        self,
        corpus: pl.DataFrame,
        data_root: Path,
        split: str,
        img_size: int = 640,
        augment: bool = False,
        hyp: dict | None = None,
    ):
        self.rows = corpus.filter(pl.col("split") == split).select("image_path", "label_path", "width", "height").to_dicts()
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment
        self.hyp = {**DEFAULT_HYP, **(hyp or {})}
        self.mosaic_enabled = augment and self.hyp["mosaic"] > 0

    def disable_mosaic(self):
        """close_mosaic: trainer calls this for the final epochs."""
        self.mosaic_enabled = False

    def __len__(self):
        return len(self.rows)

    def _load(self, i) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        from .data.datasets.base import read_yolo_label

        r = self.rows[i]
        img = cv2.cvtColor(cv2.imread(str(self.data_root / r["image_path"])), cv2.COLOR_BGR2RGB)
        boxes = np.array(read_yolo_label(self.data_root / r["label_path"], r["width"], r["height"]), dtype=np.float32).reshape(-1, 4)
        return img, boxes

    def __getitem__(self, i):
        hyp, size = self.hyp, self.img_size
        if self.mosaic_enabled and np.random.rand() < hyp["mosaic"]:
            idxs = [i, *np.random.randint(0, len(self), 3)]
            img, boxes = mosaic4([self._load(j) for j in idxs], size)
            # border=-size/2 crops the 2x mosaic canvas back to train size
            img, boxes = random_affine(
                img, boxes, hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], border=(-size // 2, -size // 2)
            )
        else:
            img, boxes = self._load(i)
            img, s, (px, py) = letterbox(img, size)
            if len(boxes):
                boxes = boxes * s + np.array([px, py, px, py], dtype=np.float32)
            if self.augment:  # close_mosaic phase keeps affine/HSV/flip, like ultralytics
                img, boxes = random_affine(img, boxes, hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"])
        if self.augment:
            img = hsv_augment(img, hyp["hsv_h"], hyp["hsv_s"], hyp["hsv_v"])
            if np.random.rand() < hyp["fliplr"]:
                img = np.ascontiguousarray(img[:, ::-1])
                if len(boxes):
                    boxes[:, [0, 2]] = size - boxes[:, [2, 0]]
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return t, torch.from_numpy(boxes.reshape(-1, 4))


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, [b[1] for b in batch]  # variable box counts -> list


def _worker_init(_):
    """cv2 spawns its own thread pool per dataloader worker — with many workers
    that oversubscribes every core and slows decoding down."""
    import cv2

    cv2.setNumThreads(0)


# ---------------------------------------------------------------------------
# Task-aligned assignment (standalone, single-class)
# ---------------------------------------------------------------------------


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(n,4) x (m,4) xyxy -> (n,m) IoU."""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def assign_targets(
    plate_prob: torch.Tensor,  # (B, A) sigmoid plate scores
    pred_boxes: torch.Tensor,  # (B, A, 4) decoded xyxy pixels (frozen branch, no grad)
    anchor_xy: torch.Tensor,  # (A, 2) anchor centers in pixels
    gt_list: list[torch.Tensor],  # per-image (n, 4) xyxy pixels
    topk: int,
    alpha: float = 0.5,
    beta: float = 6.0,
) -> torch.Tensor:
    """-> (B, A) soft target score per anchor (0 for background). YOLOv8 TAL
    semantics, fully batched — a per-image Python loop costs ~B sequential GPU
    launches per step and dominated the step time at real batch sizes."""
    B, A = plate_prob.shape
    device = plate_prob.device
    N = max((len(g) for g in gt_list), default=0)
    if N == 0:
        return torch.zeros(B, A, device=device)
    gt = torch.zeros(B, N, 4, device=device)
    valid = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b, g in enumerate(gt_list):
        if len(g):
            gt[b, : len(g)] = g
            valid[b, : len(g)] = True

    # candidates: anchor center strictly inside a (real) GT box  -> (B, A, N)
    ax, ay = anchor_xy[None, :, None, 0], anchor_xy[None, :, None, 1]
    inside = (ax > gt[:, None, :, 0]) & (ax < gt[:, None, :, 2]) & (ay > gt[:, None, :, 1]) & (ay < gt[:, None, :, 3])
    keep = inside & valid[:, None, :]

    # batched IoU (B, A, N)
    lt = torch.maximum(pred_boxes[:, :, None, :2], gt[:, None, :, :2])
    rb = torch.minimum(pred_boxes[:, :, None, 2:], gt[:, None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_p = (pred_boxes[:, :, 2:] - pred_boxes[:, :, :2]).prod(-1)
    area_g = (gt[:, :, 2:] - gt[:, :, :2]).prod(-1)
    iou = inter / (area_p[:, :, None] + area_g[:, None, :] - inter + 1e-9)

    align = plate_prob[:, :, None].pow(alpha) * iou.pow(beta) * keep

    # top-k candidate anchors per GT (by alignment), over the anchor dim
    k = min(topk, A)
    topk_idx = align.topk(k, dim=1).indices  # (B, k, N)
    mask = torch.zeros_like(keep)
    mask.scatter_(1, topk_idx, True)
    mask &= keep & (align > 0)

    # an anchor claimed by several GTs goes to the one it overlaps most
    claimed = mask.sum(-1) > 1  # (B, A)
    if claimed.any():
        only_best = torch.zeros_like(mask)
        only_best.scatter_(2, iou.argmax(-1, keepdim=True), True)
        mask = torch.where(claimed[..., None], mask & only_best, mask)

    # soft targets: alignment normalized per GT, scaled by that GT's best IoU
    align = align * mask
    pos_align = align.amax(1, keepdim=True)  # (B, 1, N) best alignment per gt
    pos_iou = (iou * mask).amax(1, keepdim=True)  # (B, 1, N) best IoU per gt
    return (align * pos_iou / (pos_align + 1e-9)).amax(-1)  # (B, A)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def plate_loss(
    model: YOLOv10, imgs: torch.Tensor, gt_boxes: list[torch.Tensor], amp: bool = True
) -> tuple[torch.Tensor, dict]:
    """BCE on the plate channel of both branches; everything else untouched.
    Forward runs under bf16 autocast (safe: the trunk is frozen, gradients touch
    only the 12 new conv tensors); assignment math is done in fp32."""
    h = model.head
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and imgs.is_cuda):
        feats = model.forward_features(imgs)
        o2m = h._forward_branch(feats, h.cv2, h.cv3)
        o2o = h._forward_branch([f.detach() for f in feats], h.one2one_cv2, h.one2one_cv3)

    anchors, strides = make_anchors([f.float() for f in feats], h.stride, 0.5)  # (A,2) feature units, (A,1)
    anchor_xy = anchors * strides  # pixel centers
    gts = [g.to(imgs.device, non_blocking=True) for g in gt_boxes]

    total, stats = 0.0, {}
    for name, branch, topk in (("o2m", o2m, 10), ("o2o", o2o, 1)):
        logits = branch["scores"][:, PLATE_CLASS].float()  # (B, A)
        with torch.no_grad():  # frozen box branch: assignment only
            dist = h.dfl(branch["boxes"].float())  # (B, 4, A)
            boxes = (dist2bbox(dist, anchors.T.unsqueeze(0), xywh=False, dim=1) * strides.T).permute(0, 2, 1)
            targets = assign_targets(logits.detach().sigmoid(), boxes, anchor_xy, gts, topk)  # (B, A)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / max(targets.sum().item(), 1.0)
        total = total + loss
        stats[name] = loss.item()
    return total, stats


# ---------------------------------------------------------------------------
# Eval: AP@0.5 for the plate class via the model's real NMS-free inference path
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_ap50(model: YOLOv10, loader: DataLoader, device: str, conf_min: float = 1e-3) -> dict:
    scored, n_gt = [], 0  # scored: (conf, is_true_positive)
    for imgs, gts in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            dets = model(imgs.to(device, non_blocking=True)).float()  # (B, 300, 6) xyxy, conf, cls
        for b in range(imgs.shape[0]):
            d = dets[b]
            d = d[(d[:, 5] == PLATE_CLASS) & (d[:, 4] >= conf_min)]
            gt = gts[b].to(device)
            n_gt += len(gt)
            matched = torch.zeros(len(gt), dtype=torch.bool, device=device)
            for row in d[d[:, 4].argsort(descending=True)]:
                if len(gt) == 0:
                    scored.append((row[4].item(), False))
                    continue
                ious = box_iou(row[None, :4], gt)[0]
                ious[matched] = 0
                best = ious.argmax()
                if ious[best] >= 0.5:
                    matched[best] = True
                    scored.append((row[4].item(), True))
                else:
                    scored.append((row[4].item(), False))
    if not scored or n_gt == 0:
        return {"ap50": 0.0, "recall": 0.0, "n_gt": n_gt}
    scored.sort(key=lambda x: -x[0])
    tp = np.cumsum([s[1] for s in scored])
    precision = tp / np.arange(1, len(scored) + 1)
    recall = tp / n_gt
    # all-point interpolated AP
    ap, prev_r = 0.0, 0.0
    for r, p in zip(recall, np.maximum.accumulate(precision[::-1])[::-1]):
        ap += (r - prev_r) * p
        prev_r = r
    return {"ap50": float(ap), "recall": float(recall[-1]), "n_gt": n_gt}


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------


class EMA:
    """Exponential moving average of the trainable params only (the frozen ones
    can't move, so this equals full-model EMA at a fraction of the cost).
    Ultralytics ramp: decay_eff = decay * (1 - exp(-updates / tau))."""

    def __init__(self, params: list[torch.nn.Parameter], decay: float = 0.9999, tau: float = 2000.0):
        self.params = params
        self.shadow = [p.detach().clone() for p in params]
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def swap(self):
        """Exchange live params <-> shadow. Call once to eval/save with EMA weights,
        call again to resume training from the live weights."""
        for s, p in zip(self.shadow, self.params):
            tmp = p.detach().clone()
            p.copy_(s)
            s.copy_(tmp)


def train_plate(
    model: YOLOv10,
    trainable: list[torch.nn.Parameter],
    train_ds: Dataset,
    val_ds: Dataset | None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 5e-3,
    lrf: float = 0.01,  # final lr fraction (ultralytics default)
    warmup_epochs: float = 3.0,
    cos_lr: bool = False,  # linear decay by default, like ultralytics
    close_mosaic: int = 10,  # mosaic off for the last N epochs
    use_ema: bool = True,
    amp: bool = True,  # bf16 autocast on the frozen forward
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    workers: int = 16,
) -> list[dict]:
    model = model.to(device).eval()  # eval() ALWAYS: frozen BN stats; new convs don't care
    opt = torch.optim.AdamW(trainable, lr=lr)
    ema = EMA(trainable) if use_ema else None
    loader_kw = dict(
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=_worker_init if workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, **loader_kw) if val_ds else None

    # schedule: per-epoch decay to lr*lrf (linear or cosine) + per-iter linear warmup
    if cos_lr:
        lf = lambda e: ((1 - math.cos(e * math.pi / epochs)) / 2) * (lrf - 1) + 1  # noqa: E731
    else:
        lf = lambda e: max(1 - e / epochs, 0) * (1 - lrf) + lrf  # noqa: E731
    nb = max(len(train_loader), 1)
    nw = max(round(warmup_epochs * nb), 100) if warmup_epochs > 0 else 0

    history, ni = [], 0
    for epoch in range(epochs):
        # exactly ultralytics semantics: if close_mosaic >= epochs, mosaic never turns off
        if close_mosaic and epoch == epochs - close_mosaic and hasattr(train_ds, "disable_mosaic"):
            train_ds.disable_mosaic()
            print(f"  epoch {epoch}: mosaic disabled (close_mosaic={close_mosaic})")
        losses = []
        for imgs, gts in train_loader:
            warm = min(ni / nw, 1.0) if nw else 1.0
            for g in opt.param_groups:
                g["lr"] = lr * lf(epoch) * warm
            loss, _ = plate_loss(model, imgs.to(device, non_blocking=True), gts, amp=amp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema:
                ema.update()
            losses.append(loss.item())
            ni += 1
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"]}
        if val_loader is not None:
            if ema:
                ema.swap()  # evaluate with EMA weights
            entry |= evaluate_ap50(model, val_loader, device)
            if ema:
                ema.swap()
        history.append(entry)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()))
    if ema:
        ema.swap()  # leave the EMA weights in the model (what gets saved/exported)
    return history


def save_plate_head(model: YOLOv10, path: Path, meta: dict | None = None) -> None:
    """Persist ONLY the 12 new tensors (a few KB) + metadata — the trunk is the
    unmodified official checkpoint, referenced, not copied."""
    state = {
        f"{branch}.{i}": getattr(model.head, branch)[i][2].new.state_dict()
        for branch in ("cv3", "one2one_cv3")
        for i in range(model.head.nl)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"plate_head": state, "meta": meta or {}}, path)


def load_plate_head(model: YOLOv10, path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key, sd in ckpt["plate_head"].items():
        branch, i = key.rsplit(".", 1)
        getattr(model.head, branch)[int(i)][2].new.load_state_dict(sd)
    return ckpt["meta"]
