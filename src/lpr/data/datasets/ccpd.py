"""CCPD — Chinese City Parking Dataset (CCPD2019 + CCPD2020 green-plate EVs).

~366k handheld 720x1160 images, one labeled plate per image, MIT license.
All annotations are encoded in the FILENAME, 7 dash-separated fields:

  025-95_113-154&383_386&473-386&473_..._363&402-0_0_22_27_27_33_16-37-15.jpg
  [0] area ratio   [1] tilt h_v   [2] bbox x1&y1_x2&y2   [3] 4 corner vertices
  [4] plate char indices (the plate IDENTITY -> our group_key)   [5] brightness  [6] blur

`ccpd_np` contains cars with NO plate (filenames don't follow the scheme) — kept as
verified negative images. Subsets (db/fn/rotate/tilt/weather/challenge/blur) are
recorded in the manifest so the training mix can weight them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, extract, gdown_file

# Small archive first: when the big file is quota-blocked (Drive's shared download
# quota, common for CCPD2019), the retry only has to fetch what's missing.
GDRIVE = {
    "CCPD2020.zip": "1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR",  # green EV plates, ~908 MB
    "CCPD2019.tar.xz": "1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc",  # ~13 GB
}
IMAGE_W, IMAGE_H = 720, 1160  # fixed portrait resolution (verified per-image anyway)


def parse_ccpd_filename(stem: str) -> tuple[list[tuple[float, float, float, float]], str] | None:
    """-> ([bbox xyxy], plate_identity) or None if the name doesn't follow the scheme."""
    fields = stem.split("-")
    if len(fields) != 7:
        return None
    try:
        tl, br = fields[2].split("_")
        x1, y1 = (float(v) for v in tl.split("&"))
        x2, y2 = (float(v) for v in br.split("&"))
    except ValueError:
        return None
    return [(x1, y1, x2, y2)], fields[4]


class CCPD(LprDataset):
    key = "ccpd"
    license_tier = "clean"  # MIT (LICENSE file, README, and the ECCV'18 paper all agree)

    def download(self) -> None:
        for name, file_id in GDRIVE.items():
            archive = gdown_file(file_id, self.raw_dir / name)
            extract(archive, self.raw_dir)

    def iter_samples(self) -> Iterator[Sample]:
        for img in sorted(self.raw_dir.rglob("*.jpg")):
            subset = img.parent.name if img.parent.name.startswith("ccpd_") else img.parent.parent.name
            parsed = parse_ccpd_filename(img.stem)
            if parsed is None:
                if subset == "ccpd_np":  # negatives: car, no plate — empty label is the point
                    yield Sample(img, [], group_key=f"np:{img.stem}", subset=subset)
                continue
            boxes, plate_id = parsed
            # CCPD2020 green subset stores author splits as train/val/test dirs
            author_split = img.parent.name if img.parent.name in ("train", "val", "test") else ""
            # group by plate identity: the same physical car photographed twice can
            # never straddle train/val (the 19% near-dup leakage in the original split)
            yield Sample(img, boxes, group_key=plate_id, subset=subset, author_split=author_split)
