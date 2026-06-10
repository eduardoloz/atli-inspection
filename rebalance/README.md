# Rebalancing: add ~100 zoomed-out defective insulators to the target

**Goal:** the target `merged_ATLI_target` is imbalanced — the rare classes are
`Defective_Damper` (113) and `Broken_Insulator` (194), vs `Normal_Damper` (1460) and
`Normal_Insulators` (873). This adds ~100 **zoomed-out** (small-in-frame) defective-insulator
images from `ATLI_source_dataset` into the target's **train split only**, so evaluation
stays honest.

**Decisions baked in** (from our discussion):
- Source = `ATLI_source_dataset`, filtered to `Defective_Insulators` boxes, ranked by
  smallest normalized bbox area = most zoomed out.
- Destination = `merged_ATLI_target`, **train split only** (val/test untouched).
- Labels = the source's *generic* `Defective_Insulators` does **not** map cleanly to the
  target's `Broken_/Flashover_/Self-Exploded_Insulator`, so **you assign the subtype by hand**.
- Roboflow has no native "move", so this is export → relabel → upload-with-annotations.

## Why it's not a plain file move (read before running)
1. **Taxonomy:** source = one generic class; target = 3 subtypes. Phase 1 marks each
   defective box `DEFECTIVE_INSULATOR_TODO`; you resolve it; Phase 2 refuses to upload
   until none remain.
2. **Leakage/dups:** ATLI overlaps with public sets, so a source image could already be
   in target val/test. Phase 1 perceptual-hashes all target images and drops any match.
3. **Co-occurring boxes:** selected source images often also contain normal insulators/
   dampers — those are **kept** (matching target labels) so they don't become false
   negatives; towers/conductors are **dropped** (the target taxonomy excludes them).
4. **Version generation:** after upload you must generate the new version with
   **splits preserved (no rebalance)** and **augmentation OFF**, or Roboflow will reshuffle
   the new images into val/test and silently undo the leakage guard.

## Setup
```bash
cd /Users/eddie/Research/Vegas
python3 -m venv .venv && source .venv/bin/activate     # if you don't have one
pip install roboflow imagehash Pillow PyYAML tqdm python-dotenv
# .env already has ROBOFLOW_API_KEY + ROBOFLOW_WORKSPACE=tl-target-set-focus
```

## Phase 1 — select + stage  (no writes to Roboflow)
```bash
python rebalance/phase1_select.py            # 100 imgs; tune with --num --area-ceil --area-floor
```
Outputs `datasets/staging/` (images, YOLO labels, classes.txt, data.yaml, manifest.csv).
Review `manifest.csv` (`max_def_area` ascending = most zoomed out) and delete any image+label
pair you don't want.

## Phase 1b — assign subtypes  (manual; the part only you can do)
Open the staging set and change every `DEFECTIVE_INSULATOR_TODO` box to
`Broken_Insulator`, `Flashover_Insulator`, or `Self-Exploded_Insulator`:
- **labelImg:** `labelImg datasets/staging/images datasets/staging/classes.txt` (YOLO mode), or
- **Roboflow:** upload `datasets/staging/` to a *temporary* project, relabel in the UI, export
  back to `datasets/staging/` as YOLOv8.

## Phase 2 — upload  (the only step that changes Roboflow)
```bash
python rebalance/phase2_upload.py            # DRY-RUN: prints exactly what it will add
python rebalance/phase2_upload.py --yes      # actually upload (tagged 'zoomout_defective_v1')
```
**Undo:** in Roboflow, filter images by tag `zoomout_defective_v1` and bulk-delete.

## Phase 3 — version + retrain  (in Roboflow UI)
Generate a new version with **Train/Test split = preserve existing**, **augmentation = none**,
then retrain and compare against the current model (mAP 36.6) — watch `Broken_Insulator`
recall in particular.

## Reuse for the rarest class
`Defective_Damper` (113) is actually the most starved class. To stage it too, rerun Phase 1
with the source class swapped:
```bash
# edit phase1_select.py: SOURCE_DEFECTIVE = "Defective_Damper"  (and adjust SUBTYPES/labels)
```
…or tell me and I'll parameterize it via a `--source-class` flag.
