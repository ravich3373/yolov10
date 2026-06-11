"""Roboflow Universe datasets — rxg4e (the canonical 10k LP set) and lhqow (462
genuinely-US images). CC BY 4.0. Requires a free account: set ROBOFLOW_API_KEY.

Hygiene baked in (from the dataset audit):
- download the RAW/base version (rxg4e v11), never the augmentation-inflated exports
- do NOT also ingest trudk/keremberke/mochoye — strict subsets of the same pool
- group_key strips the Roboflow ``name_jpg.rf.<hash>`` suffix so re-exported copies
  of the same source image share a group (their published splits leak; we re-split)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, image_size, read_yolo_label


def roboflow_group_key(stem: str) -> str:
    """'scan0001_jpg.rf.0a1b...' -> 'scan0001' (the original source image name)."""
    base = stem.split(".rf.")[0]
    for suffix in ("_jpg", "_jpeg", "_png", "_JPG", "_PNG", "_bmp"):
        base = base.removesuffix(suffix)
    return base


class _RoboflowDataset(LprDataset):
    workspace: str
    project: str
    version: int

    def download(self) -> None:
        try:
            from roboflow import Roboflow
        except ImportError as e:
            raise ImportError("pip install roboflow, then set ROBOFLOW_API_KEY") from e
        key = os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError("set ROBOFLOW_API_KEY (free account: app.roboflow.com -> settings -> API)")
        # Do NOT pre-create raw_dir: the Roboflow SDK treats an existing location
        # as "already downloaded" and silently skips (bit us: empty dir + success).
        rf = Roboflow(api_key=key)
        rf.workspace(self.workspace).project(self.project).version(self.version).download(
            "yolov8", location=str(self.raw_dir), overwrite=True
        )
        if not any(self.raw_dir.rglob("*.jpg")):
            raise RuntimeError(f"{self.key}: Roboflow SDK reported success but {self.raw_dir} has no images")

    def iter_samples(self) -> Iterator[Sample]:
        for split in ("train", "valid", "test"):
            for img in sorted((self.raw_dir / split / "images").glob("*")):
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                label = self.raw_dir / split / "labels" / (img.stem + ".txt")
                w, h = image_size(img)
                boxes = read_yolo_label(label, w, h) if label.exists() else []
                yield Sample(
                    img, boxes, group_key=roboflow_group_key(img.stem), subset=split,
                    author_split={"valid": "val"}.get(split, split), width=w, height=h,
                )


class RoboflowRXG4E(_RoboflowDataset):
    key = "rxg4e"
    license_tier = "clean"  # CC BY 4.0, Roboflow's own account
    workspace = "roboflow-universe-projects"
    project = "license-plate-recognition-rxg4e"
    version = 11  # "Base": 10,125 images, NO baked-in augmentations (v3/v4/v13 are inflated)


class RoboflowLHQOW(_RoboflowDataset):
    key = "lhqow"
    license_tier = "clean"  # CC BY 4.0
    workspace = "objects-in-the-wild"
    project = "license-plate-recognition-lhqow"
    version = 1  # 462 US (Central Florida) images
