"""Open Images V7 — "Vehicle registration plate" class only (8,157 images / 11,682
boxes; annotations CC BY 4.0). The best permissive North-America-heavy source, and
its images co-carry vehicle boxes (useful later for joint-class work).

Downloads through FiftyOne's zoo (pip install fiftyone), which fetches just the
images containing the requested class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import LprDataset, Sample, image_size

CLASS_NAME = "Vehicle registration plate"


class OpenImagesVRP(LprDataset):
    key = "open_images_vrp"
    license_tier = "clean"  # annotations CC BY 4.0; images CC BY 2.0 (per-image unverified)

    def download(self) -> None:
        try:
            import fiftyone.zoo as foz
        except ImportError as e:
            raise ImportError("pip install fiftyone (needed once, for the Open Images downloader)") from e
        for split in ("train", "validation", "test"):
            # No dataset_dir kwarg: load_zoo_dataset forwards extra kwargs into its
            # importer which also receives dataset_dir internally -> "multiple values
            # for keyword argument". The zoo cache lives in fiftyone's default dir
            # (~/fiftyone); only the YOLO export below needs to be in our raw_dir.
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                classes=[CLASS_NAME],
                label_types=["detections"],
            )
            ds.export(
                export_dir=str(self.raw_dir / f"yolo_{split}"),
                dataset_type=__import__("fiftyone").types.YOLOv5Dataset,
                label_field="ground_truth",
                classes=[CLASS_NAME],
            )

    def iter_samples(self) -> Iterator[Sample]:
        from .base import read_yolo_label

        for split in ("train", "validation", "test"):
            root = self.raw_dir / f"yolo_{split}"
            img_dir = root / "images" / "val" if (root / "images" / "val").exists() else root / "images"
            for img in sorted(img_dir.rglob("*.jpg")):
                label = _label_for(root, img)
                w, h = image_size(img)
                boxes = read_yolo_label(label, w, h) if label and label.exists() else []
                author_split = {"validation": "val"}.get(split, split)
                yield Sample(img, boxes, group_key=img.stem, subset=split, author_split=author_split, width=w, height=h)


def _label_for(root: Path, img: Path) -> Path | None:
    rel = img.relative_to(root / "images")
    return root / "labels" / rel.with_suffix(".txt")
