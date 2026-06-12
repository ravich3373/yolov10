# Papers → code map

Which techniques from the detector-class-extension reading corpus
(`~/Documents/formalisms/detector-class-extension/`) live where in this repo —
both ingredients we adopt AND literature methods implemented as BASELINES for the
ours-vs-theirs comparison. Updated 2026-06-12.

## Baselines for the comparison table (`--tier` / scripts)

Protocol per method: plate AP50 on corpus val/test (per-source) · COCO retention
(`eval_coco.py --compare artifacts/coco_eval_base.json`) · box-level negative
flips vs base (`eval_flips.py`).

| Method | Papers | Run as | Status |
|---|---|---|---|
| Ours T0: frozen trunk + cls probe | side-tuning lineage | `make train TIER=0` | done |
| Ours T1: frozen trunk + parallel head | side-tuning lineage | `make train TIER=1` | done |
| Naive full fine-tune (IOD lower bound) | every IOD paper's baseline | `make train TIER=ft` | done |
| WiSE-FT weight interpolation (merging family) | `2109.01903`, `2212.04089`, MagMax (1-task) | `scripts/wise_ft.py <ft-ckpt>` | done |
| FT + old-model COCO pseudo-labels | BPF `2407.11499`, MMA, PseDet | needs multi-class GT plumbing | queued next |
| FT + COCO replay (1-10%) | RICO `2508.13878` ("replay beats everything") | needs COCO train subset + plumbing | queued next |
| FT + output distillation (LwF/ERD-style) | `2204.01620` lineage | distill term on plate images | queued |

## Implemented

| Technique | Papers | Where |
|---|---|---|
| Side networks on a frozen trunk (zero-regression class addition by construction) | Side-Tuning `1912.13503` | `models/plate_head.py` Tier 0/1 — predates this map, validated by the corpus |
| Sparse-annotation background down-weighting (unlabeled-object suppression) | PU detection `2002.04672` (mechanism evidence), SparseDet `2201.04620`, MMA `2204.08766` | `Sample.sparse` flag (CCPD: 1 plate/img by construction; OpenALPR) → `sparse` manifest column → `_bce_weights` in both tier losses; `--sparse-bg-weight` (default 0.1) |
| Box-level negative-flip metric (no detection analogue existed) | PCT `2011.09161` (classifier NFR), AMC `2305.04135` | `src/lpr/flips.py`, `scripts/eval_flips.py` — per-GT flips between two checkpoints, per source; NFR = neg flips / old model's detections |
| EMA of trainable params | standard; also SWA-OD `2012.12645` context | `train.py` EMA (eval + checkpoints use EMA weights) |
| Ultralytics train recipe (mosaic/affine/HSV/flip, close_mosaic, optimizer auto-rule, decay groups, warmups) | — engineering, not paper-derived | `augment.py`, `train.py` |

## Queued (agreed valuable, not yet built)

- **Checkpoint soups** (model soups `2203.05482`, WiSE-FT `2109.01903`): per-epoch
  head snapshots are tiny; greedy-soup on val AP50 stacks with EMA. Small.
- **Pseudo-label stage** (PseDet, WBF `1910.13302`, Data Distillation `1712.04440`,
  OWLv2 weak-filtering): TTA + weighted-box-fusion + per-source adaptive thresholds.
  Two uses: densify sparse sources (CCPD's unlabeled plates → ignore regions or
  labels) and auto-label fleet footage. Open decision: teacher = own best model vs
  an open-vocab detector (OWLv2/Grounding-DINO — check weights' license first).
  PseDet's key lesson: aggressive filtering lowers pseudo-label mAP but improves
  the trained student.
- **Vehicle→plate routing** ("Tier 1.5", `2010.14266`): condition plate localization
  on the frozen COCO head's vehicle boxes (+7 AP@0.75 on small plates in-paper).
  Revisit after Tier-1 results on the full corpus.

## Not implementable as baselines here (with reason)

- **DETR-specific machinery** (Q-MCMF matching, DyQ query isolation, CL-DETR):
  wrong architecture family for YOLOv10's NMS-free TAL pipeline; comparing across
  detector families confounds the method with the architecture.
- **SDDGR generative replay**: needs Stable Diffusion + GLIGEN conditioned on old
  annotations — its own ablation shows pseudo-labeling does most of the work
  (38.6 of 40.9 AP), which our queued pseudo-label baseline covers directly.
- **TIES/MagMax multi-task merging**: degenerates to WiSE-FT interpolation for a
  single added task (covered above).
