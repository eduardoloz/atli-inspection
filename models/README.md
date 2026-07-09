# models/ — trained model registry

Every trained model that gets shared or referenced in results MUST have a card in this folder
(standard defined in `GIT.md`). A card contains: training details, the exact reproduction
commands, dataset + split with per-class annotation counts, the full augmentation
configuration, per-class test metrics (seed-mean ± std), provenance, and known limitations.
No card, no share.

Weights live on the training server under `~/atli/runs/<run>/weights/best.pt` (paths in each
card). To distribute weights, attach them to a GitHub Release (tag `vX.Y-<name>`) together
with the card — never commit TensorRT `.engine` files (device/version-specific; deployers
build their own with `optimization/jetson/build_engine_nano.sh` on `opt/jetson-nano`).

## Layout

Each model lives in its own folder: `models/<name>/README.md` is the card (GitHub renders it
when you open the folder), and any distributed artifacts for that model (ONNX, sample
predictions, confusion matrices) belong in the same folder next to it. Weights stay on the
server / GitHub Releases per the rules above.

## Cards

| card | mAP@0.5 (clean test) | role |
|---|---|---|
| [champ_v11n_1280_cplid_restore](champ_v11n_1280_cplid_restore/README.md) | **0.802 ± 0.015** | best benchmark model |
| [champ_v11n_1280_clean](champ_v11n_1280_clean/README.md) | 0.784 ± 0.011 | clean-data champion (reference) |
| [champ_v11n_768_deploy](champ_v11n_768_deploy/README.md) | 0.769 ± 0.009 | edge/drone deployment model |
| [obb_champ_v11n_1280](obb_champ_v11n_1280/README.md) | 0.765 ± 0.015 (OBB) | oriented-box variant |

## How augmentation works in this project (applies to every card)

**No synthetic data is used anywhere in training.** Two mechanisms only:

1. **Online augmentation** — nothing is added to the dataset on disk. Each real image passes
   through a fresh random transform every time it is loaded, so across a 250-epoch run the
   model never sees identical pixels twice, but every pixel originates in a real photo.
   The exact transform set and strengths are listed in each card; the only value we override
   from Ultralytics defaults is `scale=0.9` (ablation-validated sweet spot; default 0.5).
2. **Image-level oversampling** — rare-class training images are duplicated on disk
   (e.g. ×3 for every image containing a Defective_Damper box, script
   `data/build_oversample_native.py` pattern). The copies are byte-identical; they become
   useful because mechanism 1 gives each copy different random views every epoch. Val/test
   splits are never oversampled.

Evidence behind these choices (2026-07 ablation, 30 runs, 3 seeds/arm, logged in CLAUDE.md):
oversampling is worth +0.07 Defective_Damper AP over a no-oversampling control; ×6
duplication overfits (×3 is the recipe); rotation augmentation hurts in detection mode
(degrees=10: DD −0.10); mosaic must stay on (off: −0.030 mAP); scale=0.9 beats 0.5/0.7.
Copy-paste augmentation (quasi-synthetic composites) was tested earlier and rejected
(hurts DD). Known gap: no blur augmentation yet, and the model is blur-fragile
(−0.246 mAP under mild synthetic motion blur) — open experiment.

## Dataset the cards refer to

`ATLI_target_tightNI_noCPLID` — ATLI after the 2026-07-07 CPLID decontamination
(`results/cplid_purge_roboflow.md`), with hand-tightened Normal_Insulators boxes.
7 classes, original paper split. All cards evaluate on its identical 120-image test split
unless stated otherwise, so their test metrics are directly comparable (exception: the OBB
card, which uses its own 107-image split).

Per-category image counts before vs. after augmentation (oversampling), with the split and
methods spelled out for outside readers: `results/dataset_augmentation_summary.md`.

| split | images | annotations | Birdnest | Broken_Ins | Def_Damper | Flashover | Normal_Damper | Normal_Ins | Self-Exploded |
|---|---|---|---|---|---|---|---|---|---|
| train | 561 | 2,344 | 162 | 140 | 82 | 302 | 833 | 604 | 221 |
| val | 116 | 506 | 37 | 28 | 16 | 60 | 207 | 125 | 33 |
| test | 120 | 467 | 36 | 26 | 12 | 72 | 144 | 127 | 50 |

Caveat that applies to every card: the test split has only **12 Defective_Damper instances**,
so DD numbers carry ±0.05–0.08 seed spread and are directional; 5-fold CV on the clean pool
is the planned hardening step.

**eduardos-annotated-photos: NOT included in any current card.** The 300 user-annotated
images (151 Defective_Damper, 819 Normal_Damper, 992 Normal_Insulators, 152 generic
Defective_Insulators) were part of the old pre-purge 1,343-image merged pool but are excluded
from the clean dataset and every model above. They are the largest untapped
Defective_Damper source (would take the DD pool from 110 to 261 instances); a leakage-gated
train-only merge is the planned experiment. Every future card must state this field
explicitly (see the "eduardos-annotated-photos" line in each model folder's README).
