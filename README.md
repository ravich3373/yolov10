# LPR — license-plate detection on YOLOv10, without touching COCO

End-to-end reproducible pipeline: dataset download → dedup → group-aware splits →
frozen-trunk plate-head training → (next: TensorRT export). Built on a standalone,
readable YOLOv10 reimplementation verified bit-exact against ultralytics.

## The approach (Tier 0)

YOLOv10's only per-class parameters are the final 1×1 conv channels of each
classification branch. We freeze the entire pretrained model and bolt a parallel
**1-channel conv** next to each of the six final convs (3 scales × {one2many,
one2one}), concatenating logits 80→81. `license_plate` becomes class 80.

- **COCO cannot regress**: frozen weights + bit-identity asserted in tests
- **774 trainable parameters** (yolov10s) — a checkpoint is ~8 KB
- no COCO replay, no pseudo-labeling, no loss masking needed (nothing shared moves)
- fallback ladder if frozen features prove insufficient: Tier 1 = parallel full
  plate head (own box branch); Tier 2 = partial unfreeze + replay machinery

## Pipeline

```bash
# 0. once: conda env `lpr` has torch + ultralytics (+ gdown; roboflow/fiftyone/kaggle as needed)

python scripts/build_datasets.py --list          # registry + license tiers
python scripts/build_datasets.py openalpr ccpd   # download + convert + manifest
python scripts/dedup_and_split.py                # pHash dedup + splits + leakage purge
python scripts/train_plate.py --epochs 20        # train the 6 new convs
```

Stages communicate ONLY through manifests (`data/processed/*/manifest.parquet` →
`data/corpus.parquet`): every image row carries source, sha256, pHash, license_tier
(`clean`/`research` — filter to regenerate a commercial-safe corpus), group_key,
eval_only, split. See `datasets.md` for the audited dataset list.

Key data rules (each one traces to a verified problem — see datasets.md):
- groups, not images, are split (same plate/source-image never straddles splits)
- eval-only sets (OpenALPR…) are protected: a benchmark image found duplicated in a
  training source keeps the benchmark copy and drops the train copy
- post-split leakage purge evicts any train image within pHash radius of any eval image

## Verification suite (all runnable now, no big downloads needed)

| script | proves |
|---|---|
| `scripts/verify_yolov10_parity.py` | standalone YOLOv10 == ultralytics, bit-exact (s, n) |
| `scripts/verify_plate_surgery.py` | surgery leaves COCO bit-identical; 12 trainable tensors; fused export head identical |
| `scripts/test_parsers.py` | every annotation parser against real-format fixtures |
| `scripts/test_dedup_split.py` | dedup clusters exact/near copies; split invariants; leakage purge |
| `scripts/test_augment.py` | mosaic/affine/HSV/EMA invariants + dataset pipeline |
| `scripts/test_train_smoke.py` | full training loop on synthetic plates: AP50 0→0.88, COCO bit-identical after |

## Layout

```
src/lpr/models/yolov10.py     standalone YOLOv10 (read its docstring first)
src/lpr/models/plate_head.py  the class-addition surgery
src/lpr/data/datasets/        one module per dataset (download + parser)
src/lpr/data/manifest.py      manifest schema
src/lpr/dedup.py              pHash + Hamming clustering (GPU-accelerated)
src/lpr/split.py              group-aware splits + leakage purge
src/lpr/augment.py            ultralytics train recipe: mosaic, affine, HSV, flip
src/lpr/train.py              TAL assignment + tier-0/1 losses + schedules/EMA + AP50 eval
src/lpr/experiment.py         per-run tracking: config/manifest/log/csv/tensorboard/ckpts/plots
```

## Status / TODO

- [x] datasets audited (licenses, sizes, overlap) — `datasets.md`
- [x] standalone YOLOv10, surgery, dedup, splits, training loop — all verified
- [ ] run full downloads (CCPD ~13 GB, CRPD ~19 GB; Roboflow needs `ROBOFLOW_API_KEY`,
      Kaggle needs `~/.kaggle/kaggle.json`, Open Images needs `pip install fiftyone`)
- [ ] IR-LPR: pin Google Drive ids + verify layout (`src/lpr/data/datasets/ir_lpr.py`)
- [ ] first real training run + eval on OpenALPR-US
- [ ] TensorRT export (fp16/int8) + benchmark harness
- [ ] Coram fleet footage: pseudo-label + night/IR eval set
```
