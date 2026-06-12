"""openalpr/benchmarks endtoend subsets — EVAL-ONLY, never train.

Two reasons (both verified): only ONE plate is labeled per image even when others
are visible (training on it teaches plate suppression), and it is the de-facto
community benchmark — training on it destroys comparability. The ~186 US fixed-cam
720p frames are the closest public match to a video-security viewpoint.

Annotation: one txt per image, single line "filename x y width height plate_text".
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, run

REPO_URL = "https://github.com/openalpr/benchmarks"


def parse_openalpr_line(text: str) -> tuple[tuple[float, float, float, float], str] | None:
    """-> (bbox xyxy, plate_text) from 'file x y w h plate' (tab or space separated)."""
    parts = text.split()
    if len(parts) < 5:
        return None
    try:
        x, y, w, h = (float(v) for v in parts[1:5])
    except ValueError:
        return None
    plate = parts[5] if len(parts) > 5 else ""
    return (x, y, x + w, y + h), plate


class OpenALPR(LprDataset):
    key = "openalpr"
    license_tier = "research"  # repo is AGPL-3.0, image provenance unstated
    eval_only = True

    def download(self) -> None:
        run(["git", "clone", "--depth", "1", REPO_URL, str(self.raw_dir / "benchmarks")])

    def iter_samples(self) -> Iterator[Sample]:
        for region in ("us", "eu", "br"):
            for txt in sorted((self.raw_dir / "benchmarks" / "endtoend" / region).glob("*.txt")):
                parsed = parse_openalpr_line(txt.read_text(errors="replace").strip())
                if parsed is None:
                    continue
                box, plate = parsed
                img = _find_image(txt)
                if img is None:
                    continue
                # group by plate text: the US wts-* frames capture the same cars repeatedly
                # sparse: only ONE plate labeled per image even when others are visible
                yield Sample(img, [box], group_key=plate or img.stem, subset=region, sparse=True)


def _find_image(txt: Path) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg"):
        cand = txt.with_suffix(ext)
        if cand.exists():
            return cand
    return None
