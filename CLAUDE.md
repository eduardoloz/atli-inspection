# Vegas Research — Project Notes

## Key Paper: APET2025_T0388.pdf

**Title:** Autonomous Transmission Line Inspection using Transfer Learning Enhanced Deep Learning Models

**Authors:** Oscar Tsai (Texas A&M, Dept. of Computer Science); UNLV ECE faculty

**Venue / ID:** APET 2025, paper T0388

**Funding:** NSF grant no. 1950872 and UNLV AI SUSTEIN Seed Grant

### One-line summary
Uses **transfer learning** (source = Microsoft COCO) to boost lightweight **YOLO** object detectors for detecting transmission-line defects from UAV/RGB imagery, especially when training data is limited. Best result: **mAP@0.5 = 78.9%** (YOLOv5n, all layers unfrozen), with reduced training time vs. training from scratch.

### Problem / Motivation
- US has frequent power outages; 2018–2020 saw 231,000+ outages > 1 hr (17,484 lasting ≥8 hr). West coast hit by frequent wildfire-season outages.
- Overhead transmission lines have many defect-prone components (towers, conductors, fittings, insulators, dampers, spacers).
- UAV + deep-learning inspection is promising but hampered by **scarce annotated datasets**, limited bandwidth, and difficulty running heavy CNNs in real time.

### Datasets
- **ATLI (Aerial Transmission Line Inspection)** — NEW small dataset introduced by this paper. Target dataset.
  - 732 images, **no augmentations**. ~200 images from CPLID + other datasets; remainder from the internet.
  - Unlike other datasets, it labels the **defective spot** (not just whole component).
  - 7 categories (instance counts): Birdnest 241 · Broken Insulator 194 · Defective Damper 113 · Flashover Insulator 434 · Normal Damper 1460 · Normal Insulator 873 · Self-Exploded Insulator 554.
  - Conductors and towers excluded (already well studied).
- **Source dataset for transfer learning: Microsoft COCO.**
- Related public datasets cited: CPLID (848 insulator imgs), RSIn-dataset, FINet (13,700), TTPLA (1,234), STN PLAD (2,409).

### Models
- Focus model: **YOLOv5n** (backbone = modules 0–9, neck = 10–23, head = module 24).
- Also compared: **YOLOv8n** and **YOLOv11n** (lightweight nano variants, Ultralytics).
- Table II compares params(M)/FLOPs(B) across YOLO v5/6/8/9/10/11.

### Transfer Learning Method (3 stages)
1. Train from scratch on source (COCO).
2. **Feature Extraction** — freeze layers, train on target (ATLI) starting from best source weights.
3. **Fine-Tuning** — unfreeze all layers, continue training at much lower LR.
- Two FE configs tested: freeze first 10 layers vs. unfreeze all.

### Experiment Setup
- Server: Python 3.8, Ubuntu 18.04.4, PyTorch 1.4.0, CUDA 10.0.130, **Quadro RTX 6000** (8 installed), Intel Xeon Gold 5218 @ 2.30 GHz.
- Hyperparameters: batch size 8/GPU, image size 640, train/val/test split **70/15/15**.
  - lr0/lrf (scratch & feature extraction) = 0.01; initial lr0 (fine-tuning) = 0.00334; final OneCycle lrf (fine-tuning) = 0.1535.
- Epoch budgets: scratch = 150/300/600; TL = 150 (feature extraction) / 100 (fine-tuning).

### Evaluation Metrics
Precision, Recall, F1, **mAP@0.5** (IoU threshold = 0.5).

### Key Results (Table V)
- Transfer learning **significantly** improves mAP@0.5 vs. scratch at 150 epochs.
- Largest gains on low-instance classes (broken insulator, defective damper): **>20%** improvement.
- Best mAP@0.5 per model (TL, all layers unfrozen): **YOLOv5n = 78.9%**, YOLOv8n = 77.1%, YOLOv11n = 76.1%.
- TL reaches comparable/better accuracy in **less training time** than scratch.
- **YOLOv5n selected as best overall** — best mAP@0.5 (unfreezing all layers) with the fewest parameters; outperforms v8n/v11n at 150 & 300 epochs. YOLOv8 was better at detecting objects against similar backgrounds.

### Conclusions / Future Work
- UAV + DL is a promising, more accurate/efficient inspection method; TL with a general source dataset (COCO) boosts accuracy and efficiency on small datasets.
- Further training on small datasets helps but does not generalize equally across models.
- Future: use **more similar source datasets**, incorporate **infrared imagery**, optimize YOLO for **real-time**, and evaluate on **edge computing platforms**.

### Relevance to this project
Directly tied to the UNLV transmission-line / wildfire-season power-reliability research focus and the edge-deployment angle (see auto-memory: vegas-research-focus, unlv-wildfire-edge-refs). The "edge computing platforms" + "limited annotated data" framing is what connects this work to a Roboflow-based dataset/training pipeline.

---

## Roboflow workspace (`tl-target-set-focus`)
- Access via REST/SDK using `ROBOFLOW_API_KEY` in the `.gitignore`'d `.env` (no MCP server is connected in-session). `data/explore_roboflow.py` is a read-only inventory dump. **Never paste the key in chat.**
- **Mutations require an explicit user go-ahead** and should be done via a prepared script the user runs (not auto-executed), kept reversible (tagged batches). Earlier "do not modify anything" note now superseded per-task.
- Inventory (10 projects; key ones):
  - **Source** `atli_source_dataset` — was 11,294 imgs (v4); **2026-07-07: 596 CPLID-duplicate imgs deleted** (CPLID's 600 real normal-insulator photos were wholesale inside it) → now 10,698. v4 snapshot predates the purge. Generic classes incl. `Defective_Insulators` (4,369), `Defective_Damper` (704), plus towers/lines.
  - `no-close-up-public-data-...` — 12,135 imgs; non-close-up (zoomed-out) curation, `defect insulator` 4,224.
  - **Target = the paper's ATLI:** `merged_atli_target` — 1,046 imgs, split 772/137/137, trained model **mAP 36.6**. (Also stored split-wise: `atli_target-train`/`-val`/`-test`; **2026-07-07: these three were rewritten** — every annotation replaced with the v5 tightened-NI labels and all 249 CPLID dups deleted → now 561/116/120. Pre-change snapshots: versions train v5 / val v3 / test v3. `atli_target-test`'s classes are literally named "0"–"6". ⚠ new test set ≠ historical benchmark. Full log: `results/cplid_purge_roboflow.md`, incl. Roboflow API delete/search gotchas. Same treatment applied to `atli_target-minus-the-cplid` → 796 imgs, merged-split layout, pre-change annotations backed up to `~/atli/backup_minus_cplid_annotations.jsonl` since it has no versions.)
  - **Imbalance:** rare = `Defective_Damper` (113) & `Broken_Insulator` (194); dominant = `Normal_Damper` (1,460) & `Normal_Insulators` (873).
  - **Taxonomy mismatch:** source has one generic `Defective_Insulators`; target splits it into `Broken_/Flashover_/Self-Exploded_Insulator` — so harvested defective-insulator labels need a manual subtype assignment.

## Rebalancing task (`rebalance/`)
- Add ~100 **zoomed-out** defective-insulator images (smallest normalized bbox area) from `atli_source_dataset` into `merged_atli_target`'s **train split only** (val/test kept clean), to fix the imbalance above.
- `rebalance/phase1_select.py` (read-only): export source+target, filter+dedupe (pHash leakage guard) and stage to `datasets/staging/` with each defective box marked `DEFECTIVE_INSULATOR_TODO`. `rebalance/phase2_upload.py` (only writer; dry-run unless `--yes`): convert YOLO→VOC and upload tagged `zoomout_defective_v1`. See `rebalance/README.md`.

---

## Active thread: the `Defective_Damper` class problem
The current research focus. `Defective_Damper` is heavily confused with `Normal_Damper` in `merged_atli_target` — it's rare and dominated (pool 113 / **177 train instances** vs Normal_Damper 1460 / **1283 train**, ~12.9:1). This is the field's hardest class (worst/near-worst class in every multi-class damper study reviewed). Three workstreams done so far:

### (a) Literature review — `damper_lit_review.md`
- Digest of the PI's paper folder (`ATLI/*.pdf`) on how the literature treats damper defects, plus a web-search extension. Memory: `damper-defect-literature`.
- Key levers identified: more defective data, minority-targeted augmentation past parity, **detect-then-classify** (PA-DETR: sibling defect classes cost ~3 mAP), resolution/tiling (STN PLAD damper AP **0.21→0.84** with 4×4 tiling — dampers are a small-object problem), shape-aware attention. Nobody uses focal/class-weighted loss for the normal-vs-defective imbalance.
- No *paper-published* dataset ships labeled defective dampers; Roboflow Universe has ~3,800 immediately-exportable images with binary damper-defect labels (unvetted).

### (b) Dataset vetting — `damper_dataset_vetting.md`
- `rebalance/vet_universe_dampers.py` (read-only) pHash-vetted candidate Universe damper sets vs `merged_atli_target` train + val/test.
- **DVDI rejected** — ~50% already inside ATLI (40/300 in val/test, 109/300 in train; ATLI's "internet" images recycled DVDI). Harvesting it would have leaked eval images into train.
- **`wangbo/damper-o5wo3` (zero leakage, cleanest) and `yolov11-tasks/damper-defect-detection` (usable after dropping 4 leaked imgs + intra-dups) passed** and were uploaded to the `universe-damper-staging` Roboflow project (2,175 imgs, 1,430 tagged `has_defective`). `samiksha-gadhave` excluded (defect-spot labels, different semantics).

### (c) Benchmark experiment suite — `results/universe_benchmark_results.md`
- 20-run sweep on the UNLV server comparing baseline B vs universe-augmented training (Ucap1/Ucap2/Uw/Uto + Ufull in-domain control) × {v8n, v11n} × {150+100, 300+100}, recipe `train/run_config_v3.sh`, identical 199-img native test split (39 DefDamper instances). Memory: `universe-damper-benchmark`.
- **One-line conclusion:** raw community data did **not** improve `Defective_Damper` AP at any dose — the failure is a **recall drop from a domain/label-style mismatch** (precision holds), not quantity; Ufull's ≈0.91 in-domain DD confirms the labels are learnable but the transfer fails. Best baseline DD AP50 = **0.708** (`B_v11_150p100`).
- **Current direction:** automated experiments needing no manual labels — pretrain(universe)→finetune(native) (`train/run_config_pf.sh`), native oversampling (`data/build_oversample_native.py`, 3×/6×), and hi-res imgsz=1280 (a DD-crop audit, `results/dd_audit/`, showed defects are mostly small-object, not unlearnable rust). Manual box-fixing in `universe-damper-staging` and detect-then-classify (PA-DETR) are **deferred**.

---

## Experiment log — all conditions run (updated 2026-06-17; 113 runs since 06-10, 154 all-time)
Per-run metrics: `results/config_benchmark.csv`. Full writeup: `results/paper_results_section.md`. Running notes: `results/dd_recall_investigation.md`. PI email + confusion matrices: `results/email_to_PI.md`, `results/figures/confusion_matrices/CURRENT_*`. Memory: `universe-damper-benchmark`. All on the native 199-img test split (39 DefDamper instances) unless noted; numbers are seed-averaged DD AP@0.5.

**★ Verified champion: `HROaug_v11`** = YOLOv11n, COCO-init 2-stage TL, **imgsz 1280 + ×3 native oversampling of DefDamper images + scale-down aug (`scale=0.9`)**, 150+100 ep. Data-free (no external data). Drivers: `train/run_config_ext.sh` (`MODEL=yolo11n.pt EXTRA="scale=0.9" DATA=<OS3 yaml> ... 1280 16`); datasets via `data/build_oversample_native.py`.

**5-fold cross-validation** (⚠ old leaked CV — val=test, builder `data/build_cv.py`, eval `eval/eval_cv.py`; **superseded by the proper-CV block below**):
| metric | baseline B_v11 | champion HROaug_v11 | Δ |
|---|---|---|---|
| mAP@0.5 | 0.760 | **0.802** | +0.042 |
| DefDamper AP | 0.727 ± 0.089 | 0.745 ± 0.079 | +0.018 (within fold noise) |
| DefDamper recall | 0.680 | **0.728** | +0.048 |
| Normal_Damper AP | 0.733 | **0.789** | +0.056 |

**Single-split config means (seed-averaged), best→worst on DefDamper AP:**
| condition | recipe | seeds | DD AP | mAP |
|---|---|---|---|---|
| HROaug_OS6 | 1280 + ×6 oversample + scale | 2 | 0.751 | 0.744 |
| HRObird | champion + clean niaochao birdnest data | 1 | 0.763* | 0.750 |
| HROaug_v8 | YOLOv8n, 1280 + ×3 + scale | 2 | 0.715 | 0.738 |
| HRO_v11 | 1280 + ×3 (no scale-aug) | 2 | 0.709 | 0.724 |
| **HROaug_v11 (champion)** | 1280 + ×3 + scale | 11 | **0.699 ± 0.027** | 0.747 |
| HROaug_sc07/08/085/095 | hi-res scale-strength sweep | 1–3 | 0.67–0.70 | ~0.74 |
| OSaug_v11 | 640 + ×3 + scale (no hi-res) | 5 | 0.661 ± 0.058 | 0.67 |
| B_v11_300p100 | 300-ep stage-1 (longer) | 1 | 0.646 | 0.686 |
| HROfrz10 | freeze 10 backbone layers | 1 | 0.649 | 0.696 |
| USMcap / USM | scale-matched universe (1:1 / 4:1) | 3 | 0.63–0.65 | 0.66 |
| **baseline B_v11** | 640 native, 150+100 | 11 | **0.641 ± 0.036** | 0.691 |
| OS6aug(640) / OSaug95 / OSaug_v8 | over-aug / wrong-model variants | 3 | 0.61–0.64 | 0.65 |
| Uto / Uw / Ucap | universe mixed in, 1:1→11.7:1 | 8 | 0.58–0.65 | 0.67 |
| P2 / CP / PF | P2-head / copy-paste / pretrain→finetune | 6 | 0.59–0.69 | ~0.68 |
| HROfrz20 | freeze 20 layers — **collapsed** | 1 | 0.397 | 0.448 |
| (control) Ufull | universe data, scored in-domain | 4 | ~0.91 | 0.79 |
\* HRObird also raised Birdnest 0.76→0.90 but hurt insulators (single-class images → unlabeled co-objects); 1 seed.

**Key findings:** (1) **Native augmentation wins; external/community data never improved DefDamper** at any dose — label/domain mismatch causes a recall drop (Ufull's ≈0.91 *in-domain* proves the data is learnable but doesn't transfer). (2) **Overfitting:** 300-epoch stage-1 and ×6-oversample-at-640 *memorized* the rare class and underperformed the 150-ep/×3 setup. (3) **Freezing the backbone hurts** (freeze-20 collapsed) → full fine-tuning required (matches the paper's unfreeze-all). (4) Scale-aug sweet spot 0.85–0.9; copy-paste hurt DefDamper; SAHI inapplicable (640²/512² images). (5) **Test set too small for per-class precision** — baseline DefDamper AP swings **0.61–0.88** across CV folds → always report multi-seed / CV means, never single runs. (6) **Leakage hazard:** CPLID + DVDI + PTL-AI Furnas exact-duplicate into the ATLI test set (`results/classvet_results.md`, `dataset_and_technique_leads.md`) → a pHash+filename leakage gate is mandatory before any external data is added.

**SOTA 5-fold CV (2026-07-01, MMDetection, same `Merged_CV_proper` folds as YOLO CV — `results/sota_cv_results.md`):**
| model | params | mAP@0.5 | DD AP@0.5 |
|---|---|---|---|
| DINO-4scale | 47M | **0.785 ± 0.032** | 0.789 ± 0.110 |
| RTM-DET Tiny | 4.8M | 0.747 ± 0.023 | 0.774 ± 0.052 |
| Dynamic-RCNN | 41M | 0.718 ± 0.023 | 0.699 ± 0.065 |
| (ref) YOLO champ v11n | 2.6M | **0.785 ± 0.023** | **0.812 ± 0.080** |

**Champion YOLOv11n ties DINO on mAP and wins Defective_Damper at ~18× fewer params** — edge-deployment case strengthened. All three SOTA models score higher under CV than the original single split (pessimistic draw). Infra: `~/atli/env_mmdet`, runs in `~/atli/runs_mmdet/SOTA_*`, collector `~/atli/collect_sota_cv.py` → `results/sota_cv_summary.json`.

**Proper 5-fold CV — FINAL (2026-07-02 re-eval, all 45 runs incl. re-queued champ_v11 + osall — `results/cv_proper_results.md`):** best condition = **champ v11n: mAP 0.785 ± 0.023, DD AP 0.812 ± 0.080, DD recall 0.764** (its 06-25 crashes were the competing Ollama process, not the recipe). osall (all-defect ×3 oversample) is a wash on mAP but adds Flashover AP (+0.01–0.04) and Self-Exploded recall (+0.02–0.07); best for v5n. Champion recipe beats the APET paper's mAP on every model under stricter eval (78.5 vs 76.8 on v11n). The 06-25 partial CV numbers are superseded.

**OBB 5-fold CV — FINAL (2026-07-02 — `results/cv_obb_results.md`):** best = **osall v11n OBB, mAP 0.817 ± 0.017**; under OBB the *baseline* v11 has the best DD AP (0.798 ± 0.024) — champ/osall gains shift to Broken Insulator (+0.10–0.16) and defect recall. OBB folds come from a different export than the detection CV (not cross-comparable); the det-aligned control shows task mode is a wash overall (OBB helps NI box tightness 38%, ND 4.5%).

**Convention going forward:** log new experiment conditions in this section (seed-averaged, with the recipe), update `results/config_benchmark.csv`, and `git push` — so GitHub always reflects the full experiment record.

## Repo structure & key scripts
Repo was reorganized from a flat layout into `env/ data/ train/ eval/ analysis/ rebalance/ results/ scripts/` (see `README.md`). The old `.atli_*` hidden scripts were renamed and moved into these dirs. Key scripts:
- **Data builders** (`data/`): `build_dataset.py` (rebuild stratified merged dataset from Roboflow), `build_dataset_universe.py` (universe-augmented variants), `build_oversample_native.py` (duplicate native DD images N×), `build_ablation.py` / `build_condition_*.py` (ablation + per-condition variants), `explore_roboflow.py` (read-only inventory).
- **Train drivers** (`train/`): `run_config_v3.sh` (current 2-stage TL recipe), `run_config_pf.sh` (pretrain→finetune), plus `run_config*.sh` variants and `*_sweep.sh` multi-config sweeps.
- **Eval** (`eval/`): `collect_universe_results.py` (scrape runs → `epoch_map.csv` + `test_summary.csv`), `parse_results.py`, `eval_all.sh`.
- **Rebalance** (`rebalance/`): `vet_universe_dampers.py` (pHash vetting), `upload_universe_staging.py`, `phase1_select.py` / `phase2_upload.py`.
- **Monitor** (`scripts/`): `check_status_universe.sh`, `check_status.sh` (watch remote runs from the laptop).

## Pipeline & code index (quick-find for PI questions)

### YOLO task modes used in this project
| Task | Train command | Label format | Scripts |
|---|---|---|---|
| **Detection** (axis-aligned bbox) | `yolo detect train` | `class cx cy w h` (normalised) | `train/run_config_v3.sh:25`, `train/run_config_ext.sh:9` |
| **OBB** (rotated bbox) | `yolo obb train` | `class x1 y1 x2 y2 x3 y3 x4 y4` (4 corners, normalised) | `train/run_config_obb.sh:11`, `train/cv_obb_sweep.sh` |
| **Segmentation** (polygon masks) | `yolo segment train` | `class x1 y1 x2 y2 ... xN yN` (polygon vertices, normalised) | *(not yet used — annotations exist in Roboflow but exported as detection)* |

### Data pipelines — which script builds what
| Script | What it builds | Label type | Annotation source |
|---|---|---|---|
| `data/build_dataset.py` | `Merged_Dataset_Stratified/` (single split, seed 42, 70/15/15) | detection (cx cy w h) | Roboflow `yolov5` export |
| `data/build_dataset_v5.py` | `Merged_Dataset_v5/` + stratified + OSall (tightened NI boxes) | detection | Roboflow v5 `yolov5pytorch` |
| `data/build_cv.py` | `Merged_CV/` — old 5-fold CV (80/20, val=test, **leaked**) | detection | local `Merged_Dataset/` pool |
| `data/build_cv_proper.py` | `Merged_CV_proper/` — proper 5-fold CV (70/15/15, val≠test) | detection | local `Merged_Dataset/` pool |
| `data/build_cv_obb.py` | `Merged_CV_obb/` — 5-fold CV with OBB labels (rotated bboxes) | OBB (4 corners) | Roboflow segmentation export → `cv2.minAreaRect` conversion |
| `data/build_cv_extended.py` | extended CV variants | detection | local pool |
| `data/build_oversample_native.py` | `*_OS*/` dirs — duplicate DD images N× | detection | copies existing labels |
| `data/build_dataset_universe.py` | universe-augmented variants (Ucap/Uw/Uto) | detection | Roboflow universe datasets |

### Training scripts — which runner does what
| Script | YOLO task | Key command line | Notes |
|---|---|---|---|
| `train/run_config_v3.sh` | `yolo detect train` | 2-stage TL: SGD lr0=0.01 → lr0=0.00334 | Main detection recipe (single-split) |
| `train/run_config_ext.sh` | `yolo detect train` | Same 2-stage, supports `MODEL`, `EXTRA` env vars | Flexible detection runner (used by CV sweeps) |
| `train/run_config_obb.sh` | `yolo obb train` | Same 2-stage TL recipe, OBB mode | OBB runner |
| `train/run_config_pf.sh` | `yolo detect train` | pretrain(universe) → finetune(native) | Transfer experiment |
| `train/cv_proper_sweep.sh` | detect (via ext) | 45 runs: base/champ/osall × v5/v8/v11 × 5 folds | Proper CV sweep |
| `train/cv_obb_sweep.sh` | obb (via obb runner) | 45 runs: same conditions, OBB labels | OBB CV sweep |

### Evaluation scripts
| Script | YOLO task | What it evaluates |
|---|---|---|
| `eval/eval_cv_proper.py` | detect (`v.box`) | Proper CV: per-fold + mean±std for all 9 conditions |
| `eval/eval_cv_obb.py` | obb (`v.obb`) | OBB CV: same structure, uses OBB metrics |
| `eval/eval_cv.py` | detect | Old leaked CV (deprecated) |
| `eval/eval_cv_all.py` | detect | Extended CV evaluation |
| `eval/collect_universe_results.py` | detect | Scrape single-split runs → CSV |

### Annotation formats (how Roboflow exports map to YOLO)
- **Roboflow `yolov5pytorch`** → detection labels: `class cx cy w h` (bounding box, normalised). Polygons are collapsed to axis-aligned enclosing rectangles.
- **Roboflow `yolov5` (segmentation)** → polygon labels: `class x1 y1 x2 y2 ... xN yN` (polygon vertices, normalised).
- **OBB conversion** (`data/build_cv_obb.py:72-97`): reads segmentation polygons, runs `cv2.minAreaRect()` to get the minimum-area rotated rectangle, outputs 4-corner format for `yolo obb train`.
- **Tightened NI annotations**: drawn as polygons in Roboflow UI on `merged_atli_target` v5; exported as tighter bounding boxes via `yolov5pytorch`. The polygon geometry lives only on Roboflow's servers.

## Server
- Experiments run on `ssh $ATLI_SERVER` (8× Quadro RTX 6000).
- Working tree under `~/atli/`; conda env at `~/atli/env`; datasets under `~/atli/`; runs under `~/atli/runs/`; helper scripts mirrored under `~/atli/universe_bench/`.
