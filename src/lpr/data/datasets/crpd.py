"""CRPD — Chinese Road Plate Dataset (~33.5k images from fixed elevated traffic cams).

The best capture-domain match in the public pool (real surveillance viewpoint,
1080p, multi-plate scenes, day/night). NO LICENSE stated by the authors ->
license_tier "research".

Layout: CRPD_<single|double|multi>/{train,val,test}/{imgs,labels}; one txt per image,
one plate per line: "x1 y1 x2 y2 x3 y3 x4 y4 type content" — four corner points in
INCONSISTENT order (issues #4/#8), so we take min/max, which is order-invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, extract, gdown_file

GDRIVE = {
    "CRPD_single.zip": "1IBBHlg4VXXYSzq6TJR5S-6i_hTyh-6dD",  # ~14 GB
    "CRPD_double.zip": "14zZ8FG0dnjzAO84Rl4v76GuhYN22bY4C",  # ~4.3 GB
    "CRPD_multi.zip": "1Ud1QB-y9kXCWf1J9pegpMUnW5wkPgvis",  # ~1.1 GB
}


def parse_crpd_label(text: str) -> list[tuple[tuple[float, float, float, float], str]]:
    """-> [(bbox xyxy, plate_string)] — bbox from corner min/max."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        xs, ys = [float(v) for v in parts[0:8:2]], [float(v) for v in parts[1:8:2]]
        content = " ".join(parts[9:]) if len(parts) > 9 else ""
        out.append(((min(xs), min(ys), max(xs), max(ys)), content))
    return out


class CRPD(LprDataset):
    key = "crpd"
    license_tier = "research"  # no license at all; commercial-use question unanswered since 2022

    def download(self) -> None:
        for name, file_id in GDRIVE.items():
            archive = gdown_file(file_id, self.raw_dir / name)
            extract(archive, self.raw_dir)

    def iter_samples(self) -> Iterator[Sample]:
        for img in sorted(self.raw_dir.rglob("*.jpg")):
            if "imgs" not in img.parts and "images" not in img.parts:
                continue
            label = _sibling_label(img)
            plates = parse_crpd_label(label.read_text(encoding="utf-8", errors="replace")) if label else []
            subset = next((p for p in img.parts if p.startswith("CRPD_")), "")
            author_split = next((p for p in img.parts if p in ("train", "val", "test")), "")
            # group by plate string: the same vehicle re-captured by the same camera
            # shares its plate; multi-plate images group by the joined set
            key = "|".join(sorted(p for _, p in plates)) or f"empty:{img.stem}"
            yield Sample(img, [b for b, _ in plates], group_key=key, subset=subset, author_split=author_split)


def _sibling_label(img: Path) -> Path | None:
    for dirname in ("labels", "label", "txts"):
        cand = img.parent.parent / dirname / (img.stem + ".txt")
        if cand.exists():
            return cand
    return None
