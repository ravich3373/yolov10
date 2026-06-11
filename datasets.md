# LPR Detection Datasets

Mode: **use-all for research/benchmarking**. Every dataset carries a `license_tier` in the
manifest (`clean` = MIT/CC-BY/CC0, `research` = unlicensed/AGPL/NC/ambiguous) so a
commercial-safe training subset can be regenerated later with a single filter.

All facts below verified 2026-06-11 (links tested, several archives downloaded and counted).

## Training pool

| Dataset | Region | Images | Where / how | License (tier) | Notes |
|---|---|---|---|---|---|
| CCPD (2019+2020) | China | ~366k (incl. 11.8k green EV) | GitHub `detectRecog/CCPD` → Google Drive via `gdown` | MIT (**clean**) | Handheld close-range, 1 plate/img, filename-encoded bbox+corners. 19% test near-dup leakage — use CCPD-Fair-style resplit. Subsample base; keep weather/db/blur/tilt + `ccpd_np` negatives |
| CRPD | China | ~33.5k (26k single / 6k double / 1.6k multi) | GitHub `yxgong0/CRPD` → Google Drive (anonymous, verified) | none stated (**research**) | Best surveillance-domain match: fixed elevated traffic cams, 1080p, multi-plate (mean 3.2/img in multi), day/night. Quad corners → bbox. ~0.4% empty label files |
| Roboflow `rxg4e` **v11 base** | Mixed (US minority) | 10,125 | Roboflow API, free account (`roboflow-universe-projects/license-plate-recognition-rxg4e`) | CC BY 4.0 (**clean**) | NOT v3/v4/v13 (augmentation-inflated). Contains trudk/keremberke entirely. Documented cross-split contamination → dedup + group-aware resplit |
| Open Images V7 — "Vehicle registration plate" | Global, NA-heavy | 8,157 imgs / 11,682 boxes | FiftyOne zoo, one call (`classes=["Vehicle registration plate"]`) | CC BY 4.0 ann. (**clean**) | Co-annotated Car/Truck/Bus boxes — useful for COCO-class preservation |
| IR-LPR | Iran | ~21k (4.1k night) | GitHub `mut-deep/IR-LPR` → Google Drive | GPL-3.0 (**research**) | Largest free night-domain detection set. Persian plates fine for detection, useless for OCR |
| UC3M-LP | Spain/EU | 1,975 (2,547 plates) | Zenodo record 17152029 (4.55 GB zip; original e-cienciaDatos host is bot-walled) | Zenodo: CC BY 4.0 / GitHub: ODbL + "contact for commercial" (**research** until cleared) | Corner polygons, ~14% night, 20–32 MP stills. Mixed BGR/RGB saves — normalize on ingest |
| Roboflow `lhqow` | US (Central Florida) | 462 | Roboflow API (`objects-in-the-wild/license-plate-recognition-lhqow`) | CC BY 4.0 (**clean**) | Genuinely US parking-lot imagery; small but on-domain |
| Kaggle andrewmvd | Mixed | 433 | `kaggle datasets download andrewmvd/car-plate-detection` | CC0 (**clean**) | VOC XML. Partially overlaps rxg4e/keremberke lineage — dedup will catch |

## Eval-only (never train)

| Dataset | Region | Images | Where / how | License (tier) | Why eval-only |
|---|---|---|---|---|---|
| OpenALPR benchmarks | US/EU/BR | 445 (US 222 / EU 108 / BR 115) | `git clone openalpr/benchmarks`, `endtoend/` | AGPL-3.0 repo (**research**) | Only ONE plate labeled per image — trains suppression of unlabeled plates. De-facto community benchmark; US subset (~186 fixed-cam 720p frames) is the closest public match to our viewpoint. Day-only |
| CLPD | China | 1,200 | BaiduYun only (code `dt11`) — needs CN-phone Baidu account; mirrors don't exist | none stated (**research**) | Designed by authors as a test set; corner annotations in CLPD.csv (GBK encoding). Grab if access works, skip otherwise |
| Coram night/holdout set | US | TBD | own fleet | ours | No public set covers US fixed-cam night/IR — must come from our footage |

## Skipped, with reasons

- **keremberke/license-plate-object-detection (HF)** — exact mirror of Roboflow `trudk` v1, which is fully contained in `rxg4e`; ~94% Vietnamese (one garage cam + motorbikes), verified cross-split leakage. Adds only duplicates.
- **UFPR-ALPR / RodoSol-ALPR / AOLP / SSIG-SegPlate / LSV-LP / UFPR-SR-Plates** — signed-agreement access from academic accounts; RodoSol (20k static toll-cam day/night) is the best domain match in the literature if access ever works out.
- **PKUData, KarPlate, GLPD** — distribution links dead / withdrawn.
- **fastdup** (tooling, not data) — CC BY-NC-ND; use imagededup + SSCD + FAISS instead.

## Cross-dataset hygiene (baked into the pipeline)

1. Provenance first: only raw/v1 exports from Roboflow; skip slug-forks (`*-trudk-*`, `*rxg4e-*`, `ccpd-fts9k*`).
2. Dedup: SHA-256 → pHash Hamming ≤10 (`imagededup`) → SSCD embeddings + FAISS flat, cosine ~0.90.
3. Run the same pass between train pool and EVERY eval set — OpenALPR images verifiably circulate inside Roboflow datasets.
4. Group-aware splits (by source scene/camera, not by image); never trust shipped Roboflow splits.
5. Label-conflict handling: pseudo-label plates on COCO replay images; ignore COCO-class loss on plate-only images.
