# ATLI target (original split) + tightened-NI labels − CPLID — build report

Built 2026-07-07 on `ai.ee.unlv.edu`. Output dataset: `~/atli/ATLI_target_tightNI_noCPLID/`.

## Interpretation / decisions

- **"my annotations from merged_ATLI_target of normal insulator"** = the tightened Normal_Insulators boxes in **merged_atli_target v5**, taken from the existing detection export `~/atli/downloads/target_v5` (yolov5pytorch). The v5 labels are used **for all classes** (v5 is the annotation source of record; only NI was re-drawn).
- **"the dataset within ATLI_TARGET"** = the paper's original 70/15/15 split, recovered from the split-wise Roboflow projects. The original split **was recoverable**: clean, unaugmented versions existed and were downloaded read-only:
  - `atli_target-train` v5 ("clean version", 732 imgs) → `~/atli/downloads/atli_target_train_v5/`
  - `atli_target-val` v3 ("clean ver", 157 imgs) → `~/atli/downloads/atli_target_val_v3/`
  - `atli_target-test` v3 ("clean ver", 157 imgs) → `~/atli/downloads/atli_target_test_v3/`
  All are 640×640 stretch, auto-orient — the same preprocessing as the v5 merged export, so pHash joins exactly.
- **"minus the CPLID"** = drop any target image whose `imagehash.phash` is within Hamming distance ≤ 4 of any of the 848 CPLID images (`~/atli/downloads/cplid_github/{Normal,Defective}_Insulators/images`, 600 + 248).
- **Join key** split-wise ↔ v5: pHash. Roboflow renames files per project, so filenames don't match; pHash does.
- For each kept image the **v5 image file + v5 label file** are copied together (guaranteed geometric alignment), placed into the split dictated by the split-wise project. Read-only on Roboflow; no existing directory touched.
- Note: a Roboflow project `atli_target-minus-the-cplid` (1,037 imgs, 0 versions) already exists in the workspace — someone previously started this idea there. It was not used or touched.

## Join quality

- **All 1,046 split-wise images matched a v5 image at pHash distance 0** (distance histogram: {0: 1046}). Zero unmatched.
- 4 v5 images were matched by two split-wise images each (intra-dataset duplicates; 2 pairs inside train, 1 train↔valid, 1 valid↔test — pre-existing duplicates in ATLI itself, kept with `_dup2` suffixes where needed within a split).

## CPLID removal

Threshold used for dropping: **d ≤ 4**. Counts per split:

| split | before | dropped d=0 | dropped d≤4 (used) | would drop at d≤8 | after |
|---|---|---|---|---|---|
| train | 732 | 154 | 171 | 171 | **561** |
| valid | 157 | 38 | 41 | 41 | **116** |
| test | 157 | 32 | 37 | 37 | **120** |
| total | 1,046 | 224 | **249** | 249 | **797** |

Sanity: 249 is inside the expected 150–250 band (paper says ~200 CPLID-seeded images). No extra matches appear between d=5 and d=8, i.e. the match set is stable/clean. Spot-checked dropped pairs even share CPLID's numeric filename stems (e.g. `001_jpg.rf.09e8…` ↔ CPLID `001.jpg`, `000_jpg.rf.5e56…` ↔ `000.jpg`), confirming genuine CPLID provenance rather than pHash coincidence.

The heavy hit to Self-Exploded_Insulator (554 → 304 instances) is expected: CPLID's `Defective_Insulators` set (248 imgs) is synthetic and is exactly where ATLI's self-exploded images came from.

## Final dataset — per-class instance counts (v5 tightened labels)

| class (index) | train | valid | test | total |
|---|---|---|---|---|
| Birdnest (0) | 162 | 37 | 36 | 235 |
| Broken_Insulator (1) | 140 | 28 | 26 | 194 |
| Defective_Damper (2) | 82 | 16 | 12 | 110 |
| Flashover_Insulator (3) | 302 | 60 | 72 | 434 |
| Normal_Damper (4) | 833 | 207 | 144 | 1,184 |
| Normal_Insulators (5) | 604 | 125 | 127 | 856 |
| Self-Exploded_Insulator (6) | 221 | 33 | 50 | 304 |

797 images, 797 labels, 0 empty-label images. Every kept image carries v5 (tightened-NI) labels — 100% matched, none fell back to old labels.

## Class-index alignment for a future annotation push (IMPORTANT)

- `merged_atli_target` v5 export, `atli_target-train` v5 export, and `atli_target-val` v3 export all use the identical 7-name list in identical order: `[Birdnest, Broken_Insulator, Defective_Damper, Flashover_Insulator, Normal_Damper, Normal_Insulators, Self-Exploded_Insulator]`.
- **`atli_target-test` is the exception:** the project's classes on Roboflow are literally named **"0"–"6"** (confirmed via the project API, not just the export yaml). The numeric names correspond positionally to the same order (per-class counts line up: 38≈Birdnest, 180≈Normal_Damper, …). Any push to `atli_target-test` must map name→digit-string (`Birdnest`→`"0"`, … `Self-Exploded_Insulator`→`"6"`) or first fix the class names in that project.

## Artifacts

On the server (`$ATLI_SERVER`):
- Dataset: `~/atli/ATLI_target_tightNI_noCPLID/{train,valid,test}/{images,labels}` + `data.yaml` (7 classes, v5 order, resolved absolute paths).
- Build script (reproducible, refuses to overwrite): `~/atli/build_target_tightni_nocplid.py`.
- CPLID match list + summary: `~/atli/cplid_vs_target_matches.json` (per-image nearest-CPLID and nearest-v5 distances, drop flags, duplicate-use diagnostics).
- Reusable join mapping for the annotation push: `~/atli/target_join_mapping.json` — 1,046 records `{project, splitwise_file, v5_file, v5_label, phash_dist, is_cplid}`; 0 null matches, 249 flagged `is_cplid`.
- Split-wise raw exports (new, read-only downloads): `~/atli/downloads/atli_target_{train_v5,val_v3,test_v3}/`.

Local: this report (`results/atli_minus_cplid_build.md`).
