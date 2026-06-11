"""Kaggle andrewmvd/car-plate-detection — 433 web images, VOC XML boxes, CC0.

Tiny but license-perfect. Needs the kaggle CLI configured (~/.kaggle/kaggle.json).
Partially overlaps the rxg4e/keremberke lineage — the dedup stage handles that.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, extract, run


def parse_voc_xml(text: str) -> list[tuple[float, float, float, float]]:
    root = ET.fromstring(text)
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        vals = [float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append(tuple(vals))
    return boxes


class KaggleAndrewMVD(LprDataset):
    key = "kaggle_andrewmvd"
    license_tier = "clean"  # CC0 public domain (uploader-asserted)

    def download(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "datasets", "download", "-d", "andrewmvd/car-plate-detection", "-p", str(self.raw_dir)])
        extract(self.raw_dir / "car-plate-detection.zip", self.raw_dir)

    def iter_samples(self) -> Iterator[Sample]:
        for xml in sorted(self.raw_dir.rglob("annotations/*.xml")):
            img = xml.parent.parent / "images" / (xml.stem + ".png")
            if not img.exists():
                continue
            yield Sample(img, parse_voc_xml(xml.read_text()), group_key=xml.stem)
