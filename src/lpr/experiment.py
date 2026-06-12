"""Per-run experiment tracking: every training run gets a self-contained folder.

    experiments/<name>/
      config.json      resolved config (CLI args + dataset stats)
      manifest.json    provenance: command, git commit/dirty/diff, env versions, data fingerprint
      log.txt          timestamped run log
      results.csv      per-epoch metrics (one row per epoch, ultralytics-style)
      history.json     same, as JSON list
      tb/              tensorboard events (losses, lr, val metrics)
      last.pt, best.pt checkpoints (best = highest val AP50)
      analysis/
        results.png    loss/AP/lr curves
        pr_curve.png   final precision-recall curve on val
        train_batch*.jpg  augmented batches as the model sees them
        val_preds.jpg  grid of val images: GT (green) vs predictions (red)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


class ExperimentTracker:
    def __init__(self, root: Path, name: str, config: dict, data_fingerprint: dict | None = None):
        self.dir = _unique_dir(Path(root) / name)
        self.name = self.dir.name
        self.dir.mkdir(parents=True)
        (self.dir / "analysis").mkdir()
        self.history: list[dict] = []
        self.best_metric = -float("inf")
        self._csv_keys: list[str] | None = None

        (self.dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        (self.dir / "manifest.json").write_text(json.dumps(_manifest(data_fingerprint), indent=2))
        self._log_file = open(self.dir / "log.txt", "a")
        self.log(f"experiment '{self.name}' -> {self.dir}")

        from torch.utils.tensorboard import SummaryWriter

        self.tb = SummaryWriter(log_dir=str(self.dir / "tb"))

    # ---------------- logging ----------------

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        self._log_file.write(line + "\n")
        self._log_file.flush()

    def log_epoch(self, entry: dict) -> None:
        self.history.append(entry)
        for k, v in entry.items():
            if isinstance(v, (int, float)) and k != "epoch":
                group = "val" if k in ("ap50", "recall", "n_gt") else "train"
                self.tb.add_scalar(f"{group}/{k}", v, entry.get("epoch", len(self.history)))
        # incremental CSV: header from the first epoch's keys
        if self._csv_keys is None:
            self._csv_keys = list(entry.keys())
            (self.dir / "results.csv").write_text(",".join(self._csv_keys) + "\n")
        with open(self.dir / "results.csv", "a") as f:
            f.write(",".join(_fmt(entry.get(k, "")) for k in self._csv_keys) + "\n")
        self.log("  ".join(f"{k}={_fmt(v)}" for k, v in entry.items()))

    # ---------------- checkpoints ----------------

    def save_checkpoint(self, model, save_fn, entry: dict, monitor: str = "ap50") -> bool:
        """save_fn(model, path, meta) — the tier-appropriate saver. Returns is_best."""
        meta = {"experiment": self.name, "entry": entry}
        save_fn(model, self.dir / "last.pt", meta=meta)
        metric = entry.get(monitor, -entry.get("loss", 0.0))
        if metric > self.best_metric:
            self.best_metric = metric
            shutil.copy2(self.dir / "last.pt", self.dir / "best.pt")
            return True
        return False

    # ---------------- end of run ----------------

    def finish(self, extra: dict | None = None) -> None:
        (self.dir / "history.json").write_text(json.dumps(self.history, indent=2))
        if self.history:
            _plot_results(self.history, self.dir / "analysis" / "results.png")
        self.tb.close()
        summary = {
            "best": self.best_metric if self.best_metric != -float("inf") else None,
            "epochs": len(self.history),
            **(extra or {}),
        }
        self.log(f"finished: {summary}")
        self._log_file.close()

    # ---------------- analysis artifacts ----------------

    def plot_pr_curve(self, recall: np.ndarray, precision: np.ndarray, ap50: float) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision)
        ax.set(xlabel="recall", ylabel="precision", title=f"plate PR @0.5 IoU (AP={ap50:.3f})", xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.dir / "analysis" / "pr_curve.png", dpi=120)
        plt.close(fig)

    def plot_train_batch(self, imgs, gt_list, tag, max_imgs: int = 16) -> None:
        """Tile a TRAIN batch exactly as the model sees it (post mosaic/affine/HSV/
        flip), GT boxes in green -> analysis/train_batch<tag>.jpg. The fastest way
        to catch a broken box transform is to look at one of these."""
        import cv2

        cells = []
        for i in range(min(len(imgs), max_imgs)):
            img = imgs[i].permute(1, 2, 0).numpy().copy()  # uint8 RGB straight from the loader
            for x1, y1, x2, y2 in gt_list[i].numpy().astype(int):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cells.append(cv2.resize(img, (320, 320)))
        if not cells:
            return
        cols = int(np.ceil(np.sqrt(len(cells))))
        rows = int(np.ceil(len(cells) / cols))
        grid = np.full((rows * 320, cols * 320, 3), 114, dtype=np.uint8)
        for i, cell in enumerate(cells):
            r, c = divmod(i, cols)
            grid[r * 320 : (r + 1) * 320, c * 320 : (c + 1) * 320] = cell
        cv2.imwrite(str(self.dir / "analysis" / f"train_batch{tag}.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    def plot_val_predictions(self, model, dataset, device: str, n: int = 16, conf: float = 0.25) -> None:
        """Grid of val images: GT plates green, predictions red (with conf)."""
        import cv2
        import torch

        from .train import PLATE_CLASS, _to_device

        model = model.eval()
        cells = []
        for i in range(min(n, len(dataset))):
            img_t, gt = dataset[i]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                dets = model(_to_device(img_t.unsqueeze(0), device)).float()[0]
            img = img_t.permute(1, 2, 0).numpy().copy()
            for x1, y1, x2, y2 in gt.numpy().astype(int):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for d in dets[(dets[:, 5] == PLATE_CLASS) & (dets[:, 4] >= conf)].cpu().numpy():
                x1, y1, x2, y2 = d[:4].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 50, 50), 2)
                cv2.putText(img, f"{d[4]:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 1)
            cells.append(cv2.resize(img, (480, 480)))
        if not cells:
            return
        cols = int(np.ceil(np.sqrt(len(cells))))
        rows = int(np.ceil(len(cells) / cols))
        grid = np.full((rows * 480, cols * 480, 3), 114, dtype=np.uint8)
        for i, cell in enumerate(cells):
            r, c = divmod(i, cols)
            grid[r * 480 : (r + 1) * 480, c * 480 : (c + 1) * 480] = cell
        cv2.imwrite(str(self.dir / "analysis" / "val_preds.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while (cand := path.with_name(f"{path.name}-{i}")).exists():
        i += 1
    return cand


def _fmt(v) -> str:
    return f"{v:.5g}" if isinstance(v, float) else str(v)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _manifest(data_fingerprint: dict | None) -> dict:
    import torch

    return {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "diff_stat": _git("diff", "--stat") or None,
        },
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": data_fingerprint or {},
    }


def _plot_results(history: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("loss", "o2m", "o2o", "ap50", "recall", "lr") if any(k in h for h in history)]
    epochs = [h.get("epoch", i) for i, h in enumerate(history)]
    cols = 3
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        ax.plot(epochs, [h.get(key, np.nan) for h in history], marker="o", ms=3)
        ax.set_title(key)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(keys) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


