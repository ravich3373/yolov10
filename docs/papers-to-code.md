# Papers → code map

Which techniques from the detector-class-extension reading corpus
(`~/Documents/formalisms/detector-class-extension/`) live where in this repo,
and what was deliberately deferred. Updated 2026-06-12.

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

## Deliberately not implementing (with reason)

- **Task arithmetic / model merging** (`2212.04089`, MagMax, DuET, TIES): only
  meaningful when the trunk fine-tunes (Tier 2 contingency). For one task it
  degenerates to weight interpolation; our frozen tiers have nothing to merge.
- **Distillation-based forgetting control** (ERD, CL-DETR lineage, SDDGR
  generative replay): solves a problem our architecture removes by construction;
  RICO `2508.13878` additionally shows 1% replay beats all of it.
- **DETR-specific machinery** (Q-MCMF matching, DyQ query isolation): wrong
  architecture family for YOLOv10's NMS-free TAL pipeline.
- **COCO-replay pseudo-labeling for old classes** (BPF bridge-the-past, MMA):
  needed only if shared parameters unfreeze (Tier 2 plan B, documented in README).
