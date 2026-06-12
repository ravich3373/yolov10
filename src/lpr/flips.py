"""Box-level negative-flip measurement between two plate detectors.

PCT (arXiv:2011.09161) defines negative flips for classifiers: samples the old
model got right that the new model gets wrong. No detection analogue exists in
the literature (the deep-read digest flags this as the open gap across all 28
boundary papers) — this is ours, at GT-box granularity:

  negative flip:  ground-truth plate detected by model A, missed by model B
  positive flip:  missed by A, detected by B
  NFR = negative flips / GT boxes A detected   (regression rate among A's wins)

"Detected" = some prediction with conf >= threshold matches the GT at IoU >= 0.5,
greedy by confidence — identical matching to evaluate_ap50, so flip counts and AP
deltas are directly comparable. Reported overall and per source dataset, because
aggregate improvements routinely hide slice regressions.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .train import PLATE_CLASS, _to_device, box_iou


@torch.inference_mode()
def _detected_mask(model, imgs, gts, device: str, conf: float, iou_thr: float) -> list[torch.Tensor]:
    """Per image: bool tensor over its GT boxes — matched by a confident prediction?"""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        dets = model(_to_device(imgs, device)).float()
    out = []
    for b in range(imgs.shape[0]):
        gt = gts[b].to(device)
        matched = torch.zeros(len(gt), dtype=torch.bool, device=device)
        d = dets[b]
        d = d[(d[:, 5] == PLATE_CLASS) & (d[:, 4] >= conf)]
        for row in d[d[:, 4].argsort(descending=True)]:
            if len(gt) == 0:
                break
            ious = box_iou(row[None, :4], gt)[0]
            ious[matched] = 0
            best = ious.argmax()
            if ious[best] >= iou_thr:
                matched[best] = True
        out.append(matched)
    return out


def count_flips(
    model_a,
    model_b,
    loader: DataLoader,
    device: str,
    conf: float = 0.25,
    iou_thr: float = 0.5,
) -> dict:
    """-> {"overall": {...}, "<source>": {...}} with n_gt, det_a, det_b,
    neg_flips, pos_flips, nfr (negatives / A's detections)."""
    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()
    buckets: dict[str, dict] = {}
    for imgs, gts, srcs, _sparse in loader:
        det_a = _detected_mask(model_a, imgs, gts, device, conf, iou_thr)
        det_b = _detected_mask(model_b, imgs, gts, device, conf, iou_thr)
        for b in range(imgs.shape[0]):
            agg = buckets.setdefault(srcs[b], dict(n_gt=0, det_a=0, det_b=0, neg_flips=0, pos_flips=0))
            a, bb = det_a[b], det_b[b]
            agg["n_gt"] += len(a)
            agg["det_a"] += int(a.sum())
            agg["det_b"] += int(bb.sum())
            agg["neg_flips"] += int((a & ~bb).sum())
            agg["pos_flips"] += int((~a & bb).sum())

    def finalize(d: dict) -> dict:
        return {**d, "nfr": round(d["neg_flips"] / max(d["det_a"], 1), 4)}

    overall = dict(n_gt=0, det_a=0, det_b=0, neg_flips=0, pos_flips=0)
    for d in buckets.values():
        for k in overall:
            overall[k] += d[k]
    out = {"overall": finalize(overall)}
    if len(buckets) > 1:
        out |= {src: finalize(d) for src, d in sorted(buckets.items())}
    return out
