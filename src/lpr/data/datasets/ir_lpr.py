"""IR-LPR (github.com/mut-deep/IR-LPR) — ~21k Iranian images incl. 4,122 NIGHT
images with bbox annotations; the only sizeable free night-domain detection set.
GPL-3.0 -> license_tier "research" pending counsel's view on GPL-on-data.

NOTE: the repo distributes several Google Drive archives whose exact inner layout
we haven't verified yet. download() therefore points at the repo and expects a
manual (or scripted, once IDs are pinned) drop into data/raw/ir_lpr/; the parser
then scans for the standard YOLO images/labels pairing and validates plausibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, image_size, read_yolo_label

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class IRLPR(LprDataset):
    key = "ir_lpr"
    license_tier = "research"  # GPL-3.0 on data — unusual; treat as research-tier

    def download(self) -> None:
        raise RuntimeError(
            "IR-LPR: download the detection archives from the Google Drive links at "
            "https://github.com/mut-deep/IR-LPR and extract them under "
            f"{self.raw_dir}/ (then re-run; the parser auto-discovers YOLO image/label pairs). "
            "TODO: pin the Drive file ids here once verified."
        )

    def iter_samples(self) -> Iterator[Sample]:
        n_labels = 0
        for img in sorted(self.raw_dir.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
                continue
            label = _matching_label(img)
            if label is None:
                continue
            n_labels += 1
            w, h = image_size(img)
            yield Sample(img, read_yolo_label(label, w, h), group_key=img.stem, subset=img.parent.name, width=w, height=h)
        if n_labels == 0:
            raise RuntimeError(f"ir_lpr: no image/label pairs found under {self.raw_dir} — check the extracted layout")


def _matching_label(img: Path) -> Path | None:
    sibling = img.with_suffix(".txt")  # labels next to images
    if sibling.exists():
        return sibling
    parts = list(img.parts)  # .../images/x.jpg -> .../labels/x.txt
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        cand = Path(*parts).with_suffix(".txt")
        if cand.exists():
            return cand
    return None
