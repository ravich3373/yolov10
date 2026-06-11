"""IR-LPR (github.com/mut-deep/IR-LPR) — 20,967 Iranian car images incl. ~4.1k NIGHT
shots with plate bboxes; the only sizeable free night-domain detection set.
GPL-3.0 -> license_tier "research" pending counsel's view on GPL-on-data.

We pull only the "Car Image" archives (full scenes + VOC XML). The "License Plate"
archives are pre-cropped plates for OCR — useless for detection.

Format notes (verified by inspection):
- each split dir holds day_NNNNN.jpg / night_NNNNN.jpg with a sibling .xml
- the XML contains ONE plate box named "کل ناحیه پلاک" ("whole plate region") PLUS
  one box PER CHARACTER (names like "4", "و") — parse must filter by name or every
  digit becomes a plate
- the XML <filename> field does not match the actual file name; trust the pairing
- no <size> tag; image dims read from the JPEG header
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, gdown_archives, parse_voc_xml

GDRIVE = {  # smallest first (quota-friendly banking)
    "car_img-validation.zip": "1hwz6X-Zp7JpJL35K6P3z7k6O_PTXhUcT",  # 1.19 GB
    "car_img-test.zip": "1pe4_HgXb9dctFGJXVNlyNcKSXZeht0lX",  # 2.33 GB
    "car_img-train.zip": "1XtZ-XQ8ImNFf40D-bFqTm0UVFqNKhbLi",  # 8.28 GB
}
PLATE_NAME = "کل ناحیه پلاک"  # "whole plate region"; all other object names are characters


class IRLPR(LprDataset):
    key = "ir_lpr"
    license_tier = "research"  # GPL-3.0 on data — unusual; treat as research-tier

    def download(self) -> None:
        gdown_archives(GDRIVE, self.raw_dir)

    def iter_samples(self) -> Iterator[Sample]:
        for img in sorted(self.raw_dir.rglob("*.jpg")):
            xml = img.with_suffix(".xml")
            if not xml.exists():
                continue
            objs = parse_voc_xml(xml.read_text(encoding="utf-8"))
            boxes = [b for name, b in objs if name == PLATE_NAME]
            split_dir = img.parent.name  # train / validation / test
            yield Sample(
                img,
                boxes,
                group_key=img.stem,
                subset="night" if img.stem.startswith("night") else "day",
                author_split={"validation": "val"}.get(split_dir, split_dir),
            )
