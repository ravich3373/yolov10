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

from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
    """Reads (image, plate boxes) pairs from a corpus manifest for one split."""

    def __init__(self, corpus: pl.DataFrame, data_root: Path, split: str, img_size: int = 640, augment: bool = False):
        self.rows = corpus.filter(pl.col("split") == split).select("image_path", "label_path", "width", "height").to_dicts()
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import cv2

        from .data.datasets.base import read_yolo_label

        r = self.rows[i]
        img = cv2.cvtColor(cv2.imread(str(self.data_root / r["image_path"])), cv2.COLOR_BGR2RGB)
        boxes = np.array(read_yolo_label(self.data_root / r["label_path"], r["width"], r["height"]), dtype=np.float32).reshape(-1, 4)
        img, s, (px, py) = letterbox(img, self.img_size)
        boxes = boxes * s + np.array([px, py, px, py], dtype=np.float32)
        if self.augment and np.random.rand() < 0.5:  # horizontal flip is the one safe, free aug
            img = np.ascontiguousarray(img[:, ::-1])
            boxes[:, [0, 2]] = self.img_size - boxes[:, [2, 0]]
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t, torch.from_numpy(boxes)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, [b[1] for b in batch]  # variable box counts -> list


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
    plate_prob: torch.Tensor,  # (A,) sigmoid plate score for ONE image
    pred_boxes: torch.Tensor,  # (A, 4) decoded xyxy pixels (frozen branch, no grad)
    anchor_xy: torch.Tensor,  # (A, 2) anchor centers in pixels
    gt_boxes: torch.Tensor,  # (n, 4) xyxy pixels
    topk: int,
    alpha: float = 0.5,
    beta: float = 6.0,
) -> torch.Tensor:
    """-> (A,) soft target score per anchor (0 for background). YOLOv8 TAL semantics."""
    A = plate_prob.shape[0]
    targets = torch.zeros(A, device=plate_prob.device)
    if len(gt_boxes) == 0:
        return targets

    # candidates: anchor center strictly inside the GT box
    inside = (
        (anchor_xy[:, None, 0] > gt_boxes[None, :, 0]) & (anchor_xy[:, None, 0] < gt_boxes[None, :, 2])
        & (anchor_xy[:, None, 1] > gt_boxes[None, :, 1]) & (anchor_xy[:, None, 1] < gt_boxes[None, :, 3])
    )  # (A, n)
    iou = box_iou(pred_boxes, gt_boxes)  # (A, n)
    align = plate_prob[:, None].pow(alpha) * iou.pow(beta) * inside  # (A, n)

    # top-k candidate anchors per GT (by alignment)
    k = min(topk, A)
    topk_idx = align.topk(k, dim=0).indices  # (k, n)
    mask = torch.zeros_like(align, dtype=torch.bool)
    mask.scatter_(0, topk_idx, True)
    mask &= inside & (align > 0)

    # an anchor claimed by several GTs goes to the one it overlaps most
    claimed = mask.sum(1) > 1
    if claimed.any():
        best_gt = iou.argmax(1)
        only_best = torch.zeros_like(mask)
        only_best[torch.arange(A, device=mask.device), best_gt] = True
        mask[claimed] &= only_best[claimed]

    # soft targets: alignment normalized per GT, scaled by that GT's best IoU
    align = align * mask
    pos_align = align.amax(0, keepdim=True)  # (1, n) best alignment per gt
    pos_iou = (iou * mask).amax(0, keepdim=True)  # (1, n) best IoU per gt
    norm = align * pos_iou / (pos_align + 1e-9)  # (A, n)
    return torch.maximum(targets, norm.amax(1))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def plate_loss(model: YOLOv10, imgs: torch.Tensor, gt_boxes: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """BCE on the plate channel of both branches; everything else untouched."""
    h = model.head
    feats = model.forward_features(imgs)
    o2m = h._forward_branch(feats, h.cv2, h.cv3)
    o2o = h._forward_branch([f.detach() for f in feats], h.one2one_cv2, h.one2one_cv3)

    anchors, strides = make_anchors(feats, h.stride, 0.5)  # (A,2) feature units, (A,1)
    anchor_xy = anchors * strides  # pixel centers

    total, stats = 0.0, {}
    for name, branch, topk in (("o2m", o2m, 10), ("o2o", o2o, 1)):
        logits = branch["scores"][:, PLATE_CLASS]  # (B, A)
        with torch.no_grad():  # frozen box branch: assignment only
            dist = h.dfl(branch["boxes"])  # (B, 4, A)
            boxes = dist2bbox(dist, anchors.T.unsqueeze(0), xywh=False, dim=1) * strides.T  # (B,4,A) pixels
            targets = torch.stack(
                [
                    assign_targets(logits[b].sigmoid(), boxes[b].T, anchor_xy, gt_boxes[b].to(imgs.device), topk)
                    for b in range(imgs.shape[0])
                ]
            )  # (B, A)
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
        dets = model(imgs.to(device))  # (B, 300, 6) xyxy, conf, cls
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


def train_plate(
    model: YOLOv10,
    trainable: list[torch.nn.Parameter],
    train_ds: Dataset,
    val_ds: Dataset | None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 5e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    workers: int = 4,
) -> list[dict]:
    model = model.to(device).eval()  # eval() ALWAYS: frozen BN stats; new convs don't care
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=workers, drop_last=len(train_ds) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate, num_workers=workers) if val_ds else None

    history = []
    for epoch in range(epochs):
        losses = []
        for imgs, gts in train_loader:
            loss, _ = plate_loss(model, imgs.to(device), gts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        entry = {"epoch": epoch, "loss": float(np.mean(losses))}
        if val_loader is not None:
            entry |= evaluate_ap50(model, val_loader, device)
        history.append(entry)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()))
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
