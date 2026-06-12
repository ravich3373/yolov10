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
from tqdm import tqdm

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
        # uint8 on purpose: fp32 here would 4x the dataloader/pinned-RAM footprint
        # (32 workers x prefetch x batch of fp32 640px images ~ 80GB — it OOM-killed
        # a shell once). Normalization happens on the GPU in _to_device.
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()
        return t, torch.from_numpy(boxes.reshape(-1, 4))


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    return imgs, [b[1] for b in batch]  # variable box counts -> list


def _worker_init(_):
    """cv2 spawns its own thread pool per dataloader worker — with many workers
    that oversubscribes every core and slows decoding down."""
    import cv2

    cv2.setNumThreads(0)


def _to_device(imgs: torch.Tensor, device: str) -> torch.Tensor:
    """uint8 batch -> normalized fp32 on the GPU (cheap there, 4x cheaper on the bus)."""
    return imgs.to(device, non_blocking=True).float().div_(255.0)


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
    """-> (target_scores (B,A), target_boxes (B,A,4), fg_mask (B,A)). YOLOv8 TAL
    semantics, fully batched — a per-image Python loop costs ~B sequential GPU
    launches per step and dominated the step time at real batch sizes.
    target_boxes/fg_mask feed the Tier-1 box+DFL losses; Tier 0 ignores them."""
    B, A = plate_prob.shape
    device = plate_prob.device
    N = max((len(g) for g in gt_list), default=0)
    if N == 0:
        z = torch.zeros(B, A, device=device)
        return z, torch.zeros(B, A, 4, device=device), z.bool()
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
    target_scores = (align * pos_iou / (pos_align + 1e-9)).amax(-1)  # (B, A)

    fg = mask.any(-1)  # (B, A)
    gt_idx = mask.float().argmax(-1)  # (B, A) assigned gt per anchor (junk where !fg)
    target_boxes = gt.gather(1, gt_idx.unsqueeze(-1).expand(-1, -1, 4))
    return target_scores, target_boxes, fg


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
            targets, _, _ = assign_targets(logits.detach().sigmoid(), boxes, anchor_xy, gts, topk)  # (B, A)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / max(targets.sum().item(), 1.0)
        total = total + loss
        stats[name] = loss.item()
    return total, stats


def ciou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complete IoU between paired boxes (n,4) xyxy -> (n,). IoU minus center
    distance (normalized by enclosing-box diagonal) minus aspect-ratio term."""
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    wa, ha = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
    wb, hb = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    union = wa * ha + wb * hb - inter + 1e-9
    iou = inter / union
    cw = torch.maximum(a[:, 2], b[:, 2]) - torch.minimum(a[:, 0], b[:, 0])  # enclosing box
    chh = torch.maximum(a[:, 3], b[:, 3]) - torch.minimum(a[:, 1], b[:, 1])
    c2 = cw.pow(2) + chh.pow(2) + 1e-9
    rho2 = ((a[:, 0] + a[:, 2] - b[:, 0] - b[:, 2]).pow(2) + (a[:, 1] + a[:, 3] - b[:, 1] - b[:, 3]).pow(2)) / 4
    v = (4 / math.pi**2) * (torch.atan(wb / (hb + 1e-9)) - torch.atan(wa / (ha + 1e-9))).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + 1e-9)
    return iou - rho2 / c2 - alpha * v


def dfl_loss(dist_logits: torch.Tensor, target: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    """Distribution Focal Loss: (n,4,reg_max) logits vs (n,4) continuous distances
    in [0, reg_max-1] -> (n,). CE against the two adjacent bins, linearly weighted."""
    tl = target.floor().long()
    tr = (tl + 1).clamp(max=reg_max - 1)
    wl = tr.float() - target
    wr = 1 - wl
    logits = dist_logits.reshape(-1, reg_max)
    ce_l = F.cross_entropy(logits, tl.reshape(-1), reduction="none")
    ce_r = F.cross_entropy(logits, tr.reshape(-1), reduction="none")
    return (ce_l * wl.reshape(-1) + ce_r * wr.reshape(-1)).view(-1, 4).mean(-1)


def plate_t1_loss(
    model, imgs: torch.Tensor, gt_boxes: list[torch.Tensor], amp: bool = True
) -> tuple[torch.Tensor, dict]:
    """Full v8-style detection loss for the Tier-1 plate head (cls BCE + CIoU box
    + DFL, ultralytics gains 0.5/7.5/1.5), one2many topk=10 + one2one topk=1.
    Only the plate head has gradients; the trunk and COCO head are frozen."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and imgs.is_cuda):
        out = model(imgs)  # training dict: feats + both branches
    ph = model.plate_head
    anchors, strides = make_anchors([f.float() for f in out["feats"]], ph.stride, 0.5)
    anchor_xy = anchors * strides
    gts = [g.to(imgs.device, non_blocking=True) for g in gt_boxes]

    total, stats = 0.0, {}
    for name, topk in (("o2m", 10), ("o2o", 1)):
        branch = out[name]
        logits = branch["scores"][:, 0].float()  # (B, A) — single class
        raw = branch["boxes"].float()  # (B, 64, A)
        boxes = (dist2bbox(ph.dfl(raw), anchors.T.unsqueeze(0), xywh=False, dim=1) * strides.T).permute(0, 2, 1)
        with torch.no_grad():
            targets, tboxes, fg = assign_targets(logits.detach().sigmoid(), boxes.detach(), anchor_xy, gts, topk)
        t_sum = max(targets.sum().item(), 1.0)

        loss_cls = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum") / t_sum
        loss_box = boxes.new_zeros(())
        loss_dfl = boxes.new_zeros(())
        if fg.any():
            w = targets[fg]
            loss_box = ((1.0 - ciou(boxes[fg], tboxes[fg])) * w).sum() / t_sum
            # DFL targets are distances in FEATURE units (the 16 bins are cell counts)
            ltrb = torch.cat([anchor_xy[None] - tboxes[..., :2], tboxes[..., 2:] - anchor_xy[None]], -1)
            ltrb = (ltrb / strides.T.unsqueeze(-1)).clamp(0, ph.reg_max - 1 - 0.01)  # (B,A,4) / (1,A,1)
            dist_logits = raw.view(raw.shape[0], 4, ph.reg_max, -1).permute(0, 3, 1, 2)  # (B,A,4,16)
            loss_dfl = (dfl_loss(dist_logits[fg], ltrb[fg], ph.reg_max) * w).sum() / t_sum
        total = total + 0.5 * loss_cls + 7.5 * loss_box + 1.5 * loss_dfl
        stats[name] = (0.5 * loss_cls + 7.5 * loss_box + 1.5 * loss_dfl).item()
    return total, stats


def save_plate_t1(model, path: Path, meta: dict | None = None) -> None:
    """Tier-1 checkpoint: the whole plate head (~6.5MB) — trunk stays the official file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"plate_head_t1": model.plate_head.state_dict(), "meta": meta or {}}, path)


def load_plate_t1(model, path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.plate_head.load_state_dict(ckpt["plate_head_t1"])
    return ckpt["meta"]


# ---------------------------------------------------------------------------
# Eval: AP@0.5 for the plate class via the model's real NMS-free inference path
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_ap50(model: YOLOv10, loader: DataLoader, device: str, conf_min: float = 1e-3) -> dict:
    scored, n_gt = [], 0  # scored: (conf, is_true_positive)
    for imgs, gts in tqdm(loader, desc="  eval", unit="b", leave=False, mininterval=2, dynamic_ncols=True):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            dets = model(_to_device(imgs, device)).float()  # (B, 300, 6) xyxy, conf, cls
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
    loss_fn=plate_loss,  # plate_loss (Tier 0) or plate_t1_loss (Tier 1)
) -> list[dict]:
    # train() is SAFE for the frozen parts: their BatchNorms carry the
    # freeze_bn_stats guard (train() is a no-op on them). Tier 1's own head has
    # live BNs that genuinely need train mode for their running stats.
    model = model.to(device).train()
    opt = torch.optim.AdamW(trainable, lr=lr)
    ema = EMA(trainable) if use_ema else None
    loader_kw = dict(
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
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
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", unit="b", mininterval=2, dynamic_ncols=True)
        for imgs, gts in pbar:
            warm = min(ni / nw, 1.0) if nw else 1.0
            for g in opt.param_groups:
                g["lr"] = lr * lf(epoch) * warm
            loss, stats = loss_fn(model, _to_device(imgs, device), gts, amp=amp)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema:
                ema.update()
            losses.append(loss.item())
            ni += 1
            pbar.set_postfix(
                loss=f"{np.mean(losses):.3f}",
                o2m=f"{stats['o2m']:.3f}",
                o2o=f"{stats['o2o']:.3f}",
                lr=f"{opt.param_groups[0]['lr']:.1e}",
                refresh=False,
            )
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"]}
        if val_loader is not None:
            model.eval()
            if ema:
                ema.swap()  # evaluate with EMA weights
            entry |= evaluate_ap50(model, val_loader, device)
            if ema:
                ema.swap()
            model.train()
        history.append(entry)
        print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()))
    if ema:
        ema.swap()  # leave the EMA weights in the model (what gets saved/exported)
    model.eval()
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
