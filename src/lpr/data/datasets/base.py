"""Base class every dataset module implements: download() + iter_samples().

The base handles everything else uniformly — hashing, image-size reading, YOLO label
writing, manifest assembly — so each dataset module is only its download recipe and
its annotation parser.
"""

from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import polars as pl

from ..manifest import write_manifest


@dataclass
class Sample:
    """One image as yielded by a dataset parser. Boxes in PIXEL xyxy."""

    image_path: Path
    boxes_xyxy: list[tuple[float, float, float, float]]
    group_key: str
    subset: str = ""
    author_split: str = ""
    # width/height optional: parser may know them for free (e.g. fixed-size datasets);
    # otherwise the base reads them from the image header.
    width: int = 0
    height: int = 0


class LprDataset:
    key: str = ""  # registry name + directory name
    license_tier: str = ""  # "clean" | "research"
    eval_only: bool = False

    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw" / self.key
        self.out_dir = self.data_root / "processed" / self.key
        self.labels_dir = self.out_dir / "labels"
        self.manifest_path = self.out_dir / "manifest.parquet"

    # -- to be implemented per dataset --------------------------------------
    def download(self) -> None:
        raise NotImplementedError

    def iter_samples(self) -> Iterator[Sample]:
        raise NotImplementedError

    # -- uniform machinery ---------------------------------------------------
    def is_downloaded(self) -> bool:
        return (self.raw_dir / ".complete").exists()

    def mark_downloaded(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / ".complete").touch()

    def build(self, workers: int = 16) -> pl.DataFrame:
        """download (if needed) -> parse all samples -> labels + manifest parquet."""
        if not self.is_downloaded():
            self.download()
            self.mark_downloaded()
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(workers) as pool:  # hashing + header reads are IO-bound
            rows = [r for r in pool.map(self._process_sample, self.iter_samples()) if r]
        if not rows:
            raise RuntimeError(f"{self.key}: parser produced no samples — check {self.raw_dir}")
        df = pl.DataFrame(rows)
        write_manifest(df, self.manifest_path)
        return df

    def _process_sample(self, s: Sample) -> dict | None:
        try:
            sha = sha256_file(s.image_path)
            w, h = (s.width, s.height) if s.width and s.height else image_size(s.image_path)
        except Exception as e:  # unreadable/corrupt image: drop with a note, don't abort 300k-file builds
            print(f"  [skip] {s.image_path}: {e}")
            return None
        image_id = sha[:16]
        label_path = self.labels_dir / f"{image_id}.txt"
        write_yolo_label(label_path, s.boxes_xyxy, w, h)
        return dict(
            image_id=image_id,
            source=self.key,
            subset=s.subset,
            image_path=str(s.image_path.relative_to(self.data_root)),
            label_path=str(label_path.relative_to(self.data_root)),
            width=w,
            height=h,
            sha256=sha,
            n_plates=len(s.boxes_xyxy),
            group_key=f"{self.key}:{s.group_key}",  # namespace groups by dataset
            license_tier=self.license_tier,
            eval_only=self.eval_only,
            author_split=s.author_split,
        )


# ---------------------------------------------------------------------------
# helpers shared by dataset modules
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def image_size(path: Path) -> tuple[int, int]:
    """Width/height from the file header only — no pixel decode."""
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def write_yolo_label(path: Path, boxes_xyxy, w: int, h: int) -> None:
    """Pixel xyxy -> normalized 'class cx cy bw bh' lines, clipped to the image.
    An empty file is meaningful: a verified negative (plate-free) image."""
    lines = []
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:  # degenerate after clipping
            continue
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_yolo_label(path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Normalized YOLO label file -> pixel xyxy boxes (inverse of write_yolo_label)."""
    boxes = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def gdown_file(file_id: str, dest: Path) -> Path:
    """Scripted Google Drive download (handles the quota/virus-scan interstitials).

    Invoked as a module of THIS interpreter (never a PATH binary, which may belong
    to another env). No --fuzzy flag: gdown 6.x removed it (fuzzy is the default);
    a bare uc?id= URL parses on every version anyway."""
    import sys

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        # --continue: resume partial multi-GB downloads instead of restarting.
        # NOTE: popular files hit Drive's shared download quota ("Too many users
        # have viewed or downloaded this file recently") — nothing to fix locally,
        # retry after some hours or pull a mirror.
        run([sys.executable, "-m", "gdown", "--continue", f"https://drive.google.com/uc?id={file_id}", "-O", str(dest)])
    return dest


def gdown_archives(archives: dict[str, str], raw_dir: Path) -> None:
    """Download + extract a set of Drive archives, banking whatever is available.

    Popular datasets routinely have some archives behind Drive's shared download
    quota ("Too many users...") while others fetch fine. Failing fast would waste
    the available ones: instead every archive is attempted, completed downloads
    and extractions are marked (so retries fetch ONLY the gaps), and the error at
    the end lists exactly what is still missing."""
    failed = []
    for name, file_id in archives.items():
        marker = raw_dir / f".extracted_{name}"
        if marker.exists():
            continue
        try:
            archive = gdown_file(file_id, raw_dir / name)
            extract(archive, raw_dir)
            marker.touch()
        except Exception as e:
            failed.append(name)
            print(f"  [unavailable, will retry] {name}: {e}")
    if failed:
        raise RuntimeError(
            f"{len(failed)} archive(s) unavailable (likely Drive download quota, clears within ~24h): "
            f"{failed} — everything else is banked; re-run the same command to fetch only what's missing"
        )


def download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        run(["curl", "-L", "--fail", "--retry", "3", "-o", str(dest), url])
    return dest


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    a = str(archive)
    if a.endswith((".tar.xz", ".tar.gz", ".tar")):
        run(["tar", "xf", a, "-C", str(dest)])
    elif a.endswith(".zip"):
        run(["unzip", "-q", "-o", a, "-d", str(dest)])
    else:
        raise ValueError(f"unknown archive type: {archive}")
