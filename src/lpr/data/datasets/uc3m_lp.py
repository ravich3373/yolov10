"""UC3M-LP — 1,975 Spanish-plate images, corner-polygon annotations, ~14% night.

License is contradictory across the author's two channels (Zenodo: CC BY 4.0,
GitHub: ODbL + "contact authors for commercial use") -> "research" until cleared.

Per-image JSON: {imagePath, imageHeight, imageWidth, lps: [{lp_id, poly_coord: 4
corner points, characters: [...]}]}. Trust the file PAIRING, not the imagePath
field (known renumbering mismatches, e.g. test/00390.json says 00396.jpg).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, download_url, extract

ZENODO_URL = "https://zenodo.org/records/17152029/files/UC3M-LP.zip"  # 4.55 GB


def parse_uc3m_json(text: str) -> tuple[list[tuple[float, float, float, float]], int, int]:
    """-> ([bbox xyxy from polygon min/max], width, height)."""
    d = json.loads(text)
    boxes = []
    for lp in d.get("lps", []):
        pts = lp.get("poly_coord", [])
        if len(pts) >= 3:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes, int(d.get("imageWidth", 0)), int(d.get("imageHeight", 0))


class UC3MLP(LprDataset):
    key = "uc3m_lp"
    license_tier = "research"  # conflicting CC BY 4.0 vs ODbL+contact-for-commercial

    def download(self) -> None:
        archive = download_url(ZENODO_URL, self.raw_dir / "UC3M-LP.zip")
        extract(archive, self.raw_dir)

    def iter_samples(self) -> Iterator[Sample]:
        for js in sorted(self.raw_dir.rglob("*.json")):
            img = js.with_suffix(".jpg")
            if not img.exists():
                continue
            boxes, w, h = parse_uc3m_json(js.read_text())
            author_split = next((p for p in js.parts if p in ("train", "test")), "")
            # authors state 2,547 distinct vehicles -> image-level grouping is safe
            yield Sample(img, boxes, group_key=img.stem, subset=author_split, author_split=author_split, width=w, height=h)
