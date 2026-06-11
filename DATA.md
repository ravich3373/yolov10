# Dataset acquisition — instructions & status

How to fetch every dataset in the registry, what each one's archives contain, and
where things stand. For *why* these datasets (licenses, quality, overlap audit) see
[datasets.md](datasets.md). Status last updated: **2026-06-11 ~15:00 PT**.

All commands run from the repo root. Downloads are **resumable and idempotent**:
every archive is banked individually (`.extracted_*` markers in `data/raw/<key>/`),
so re-running the same command fetches only what's missing.

```bash
make data DATASETS="ccpd crpd"     # any subset of keys
make data-all                      # everything registered
python scripts/build_datasets.py --list   # registry + license tiers
```

## Status at a glance

| dataset | images | how to get | status | blocker |
|---|---|---|---|---|
| openalpr | 444 | `make data DATASETS=openalpr` (git clone) | ✅ **built** (manifest done, eval-only) | — |
| ccpd | ~366k | `make data DATASETS=ccpd` | ⚠️ partial: CCPD2020 (11,776 green-plate imgs) banked; **CCPD2019 (13 GB) pending** | Drive download quota; auto-retry armed |
| crpd | ~33.5k | `make data DATASETS=crpd` | ⚠️ partial: multi (1,585) + double (~6k) banked; **single (14 GB, ~26k) pending** | Drive download quota; auto-retry armed |
| uc3m_lp | 1,975 | `make data DATASETS=uc3m_lp` | ⬜ not started | none — direct Zenodo URL (4.55 GB) |
| rxg4e | 10,125 | `pip install roboflow`, set `ROBOFLOW_API_KEY`, `make data DATASETS=rxg4e` | ⬜ not started | needs free Roboflow account/API key |
| lhqow | 462 | same as rxg4e | ⬜ not started | needs Roboflow API key |
| open_images_vrp | 8,157 | `pip install fiftyone`, `make data DATASETS=open_images_vrp` | ⬜ not started | needs fiftyone (heavy dep, installs mongodb) |
| kaggle_andrewmvd | 433 | kaggle CLI + `~/.kaggle/kaggle.json`, `make data DATASETS=kaggle_andrewmvd` | ⬜ not started | needs Kaggle credentials |
| ir_lpr | ~21k | manual for now — see below | ⬜ blocked | Drive file ids not pinned yet (TODO) |
| CLPD | 1,200 | not in registry | ❌ skipped | BaiduYun-only (needs CN-phone account); eval-only design anyway |

Manifests land in `data/processed/<key>/manifest.parquet`; only **complete** datasets
get one (all-or-nothing per dataset, so a manifest never describes partial data).
Currently built: `openalpr` only.

## The Google Drive quota problem (why ccpd/crpd "failed")

CCPD and CRPD are distributed **only** via the authors' personal Google Drive links.
Drive caps how much a single shared file can be downloaded by everyone worldwide
per ~24h rolling window; popular big archives sit at that cap much of the time and
return *"Too many users have viewed or downloaded this file recently."*

- The cap is on the **file**, not on you — VPN/IP changes don't help.
- Small archives slip through (that's how CCPD2020 + CRPD multi/double banked).
- It clears on its own within ~24h. A background retry loop re-runs
  `make data DATASETS="ccpd crpd"` every 3h (log: `/tmp/lpr_data_retry.log`) and
  gives up with a message after 48h.
- **Fallback if it drags**: with any Google account, "Make a copy" of the file into
  your own Drive — the copy has a fresh quota — and download it with rclone or the
  browser, dropping the archive at `data/raw/<key>/<archive-name>` before re-running
  `make data`. File ids: CCPD2019 `1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc`,
  CRPD_single `1IBBHlg4VXXYSzq6TJR5S-6i_hTyh-6dD`.

## Per-dataset notes

### ccpd — two archives, one dataset
- `CCPD2020.zip` (908 MB): green new-energy plates, 11,776 imgs — ✅ banked
- `CCPD2019.tar.xz` (13 GB): the main ~355k corpus with internal condition subsets
  (`ccpd_base/db/fn/rotate/tilt/weather/challenge/blur` + `ccpd_np` plate-free
  negatives) — ⏳ quota-walled
- Annotations are filename-encoded; the parser groups by plate identity so the
  documented 19% train/test near-dup leakage of the author split can't recur.

### crpd — three archives by plates-per-image
- `CRPD_multi.zip` (1.1 GB, 3+ plates/img, 1,585 imgs) — ✅ banked
- `CRPD_double.zip` (4.3 GB, 2 plates/img, ~6k) — ✅ banked
- `CRPD_single.zip` (14 GB, 1 plate/img, ~26k) — ⏳ quota-walled
- Fixed elevated traffic cameras — the best capture-domain match in the registry.
  No license stated → `license_tier=research`.

### uc3m_lp
Direct download from Zenodo (the original e-cienciaDatos host is bot-walled — don't
use it). Corner polygons → bboxes via the parser. License ambiguity (CC BY 4.0 on
Zenodo vs ODbL+contact-authors on GitHub) → `research` tier until cleared.

### rxg4e / lhqow (Roboflow)
Get a free key: app.roboflow.com → Settings → API. The module pins **rxg4e v11**
("Base", un-augmented) deliberately — v3/v4/v13 exports have augmented copies baked
in. Don't also ingest keremberke/trudk/mochoye: same images (see datasets.md).

### open_images_vrp
FiftyOne zoo downloads just the "Vehicle registration plate" class (8,157 imgs /
11,682 boxes). The dep is heavy; install it in the `lpr` env when needed.

### kaggle_andrewmvd
`kaggle datasets download andrewmvd/car-plate-detection` under the hood. Needs
`~/.kaggle/kaggle.json` (kaggle.com → Account → Create API token).

### ir_lpr (TODO)
~21k Iranian images incl. 4,122 night — the only sizeable free night set. The repo
(github.com/mut-deep/IR-LPR) links several Drive archives whose exact ids/layout we
haven't pinned; download manually into `data/raw/ir_lpr/` and the parser will
auto-discover YOLO image/label pairs, or pin the ids in
`src/lpr/data/datasets/ir_lpr.py`.

## After data completes

```bash
make prep    # pHash dedup across all built datasets + group-aware splits + leakage purge
make train   # plate head on the frozen trunk; evals on OpenALPR-US (auto: eval-only → test)
```
Re-run `make prep` every time a new dataset finishes building.
