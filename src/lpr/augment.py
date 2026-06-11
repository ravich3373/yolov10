"""Training augmentations — the ultralytics detect recipe, standalone.

What ultralytics actually applies for detection training (v8/v10 defaults), and
what we port here:

  mosaic (p=1.0)          4 images on a 2x canvas, random center  -> PORTED
  random affine           scale 0.5 (=> x0.5..x1.5), translate 0.1,
                          degrees/shear/perspective = 0 by default -> PORTED
  HSV jitter              h 0.015, s 0.7, v 0.4                    -> PORTED
  horizontal flip p=0.5                                            -> PORTED
  close_mosaic            mosaic off for the last 10 epochs        -> PORTED (trainer)
  mixup / copy-paste      p=0.0 by default (OFF) in v8/v10 detect  -> skipped, same default
  albumentations blur/CLAHE (p=0.01)                               -> skipped (negligible)

Mosaic + affine-scale matter most for our domain gap: public LP datasets have big
close-range plates, surveillance has small far ones — mosaic halves object scale and
the affine adds x0.5..x1.5 jitter, which is exactly the missing scale diversity.

All box bookkeeping in pixel xyxy, matching the rest of the pipeline.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

DEFAULT_HYP = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    fliplr=0.5,
    mosaic=1.0,
)


def hsv_augment(img: np.ndarray, hgain: float, sgain: float, vgain: float) -> np.ndarray:
    """Random HSV gains via LUTs (RGB in, RGB out)."""
    if hgain == sgain == vgain == 0:
        return img
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def box_candidates(before: np.ndarray, after: np.ndarray, wh_thr=2, ar_thr=100, area_thr=0.1) -> np.ndarray:
    """Keep boxes that survived the affine: big enough, sane aspect, >10% of original area."""
    w1, h1 = before[:, 2] - before[:, 0], before[:, 3] - before[:, 1]
    w2, h2 = after[:, 2] - after[:, 0], after[:, 3] - after[:, 1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)


def random_affine(
    img: np.ndarray,
    boxes: np.ndarray,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    border: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Random rotation/scale/shear/translation. `border` < 0 crops a mosaic canvas
    down to the train size (canvas + 2*border), exactly like ultralytics."""
    h_out = img.shape[0] + border[0] * 2
    w_out = img.shape[1] + border[1] * 2

    C = np.eye(3)  # move canvas center to origin
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2
    R = np.eye(3)  # rotation + isotropic scale
    a = np.random.uniform(-degrees, degrees)
    s = np.random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    S = np.eye(3)  # shear
    S[0, 1] = math.tan(np.random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(np.random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3)  # translation (recenters onto the output frame)
    T[0, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * w_out
    T[1, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * h_out

    M = T @ S @ R @ C
    img = cv2.warpAffine(img, M[:2], dsize=(w_out, h_out), borderValue=(114, 114, 114))

    if len(boxes) == 0:
        return img, boxes
    n = len(boxes)
    corners = np.ones((n * 4, 3))
    corners[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # all 4 corners
    corners = (corners @ M.T)[:, :2].reshape(n, 8)
    xs, ys = corners[:, 0::2], corners[:, 1::2]
    new = np.stack((xs.min(1), ys.min(1), xs.max(1), ys.max(1)), axis=1)
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, w_out)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, h_out)
    keep = box_candidates(boxes * s, new)  # compare at matched scale, like ultralytics
    return img, new[keep].astype(np.float32)


def mosaic4(
    items: list[tuple[np.ndarray, np.ndarray]], imgsz: int = 640
) -> tuple[np.ndarray, np.ndarray]:
    """4x (image, boxes px) -> (2*imgsz canvas, shifted boxes). Each image is scaled
    so its long side = imgsz and pasted into a quadrant around a random center."""
    s = imgsz
    yc, xc = (int(np.random.uniform(s // 2, 2 * s - s // 2)) for _ in range(2))
    canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
    all_boxes = []
    for i, (img, boxes) in enumerate(items):
        h0, w0 = img.shape[:2]
        r = s / max(h0, w0)
        img = cv2.resize(img, (round(w0 * r), round(h0 * r)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        if i == 0:  # top-left of center
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b = 0, h - (y2a - y1a)
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * s)
            x1b, y1b = w - (x2a - x1a), 0
        else:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(yc + h, 2 * s)
            x1b, y1b = 0, 0
        canvas[y1a:y2a, x1a:x2a] = img[y1b : y1b + (y2a - y1a), x1b : x1b + (x2a - x1a)]
        if len(boxes):
            shifted = boxes * r + np.array([x1a - x1b, y1a - y1b, x1a - x1b, y1a - y1b], dtype=np.float32)
            all_boxes.append(shifted)
    boxes = np.concatenate(all_boxes, 0) if all_boxes else np.zeros((0, 4), dtype=np.float32)
    boxes = boxes.clip(0, 2 * s).astype(np.float32)
    return canvas, boxes
