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
- Access via REST/SDK using `ROBOFLOW_API_KEY` in the `.gitignore`'d `.env` (no MCP server is connected in-session). `explore_roboflow.py` is a read-only inventory dump. **Never paste the key in chat.**
- **Mutations require an explicit user go-ahead** and should be done via a prepared script the user runs (not auto-executed), kept reversible (tagged batches). Earlier "do not modify anything" note now superseded per-task.
- Inventory (10 projects; key ones):
  - **Source** `atli_source_dataset` — 11,294 imgs, v4; generic classes incl. `Defective_Insulators` (4,369), `Defective_Damper` (704), plus towers/lines.
  - `no-close-up-public-data-...` — 12,135 imgs; non-close-up (zoomed-out) curation, `defect insulator` 4,224.
  - **Target = the paper's ATLI:** `merged_atli_target` — 1,046 imgs, split 772/137/137, trained model **mAP 36.6**. (Also stored split-wise: `atli_target-train` 732, `-val` 157, `-test` 157.)
  - **Imbalance:** rare = `Defective_Damper` (113) & `Broken_Insulator` (194); dominant = `Normal_Damper` (1,460) & `Normal_Insulators` (873).
  - **Taxonomy mismatch:** source has one generic `Defective_Insulators`; target splits it into `Broken_/Flashover_/Self-Exploded_Insulator` — so harvested defective-insulator labels need a manual subtype assignment.

## Rebalancing task (`rebalance/`)
- Add ~100 **zoomed-out** defective-insulator images (smallest normalized bbox area) from `atli_source_dataset` into `merged_atli_target`'s **train split only** (val/test kept clean), to fix the imbalance above.
- `phase1_select.py` (read-only): export source+target, filter+dedupe (pHash leakage guard) and stage to `datasets/staging/` with each defective box marked `DEFECTIVE_INSULATOR_TODO`. `phase2_upload.py` (only writer; dry-run unless `--yes`): convert YOLO→VOC and upload tagged `zoomout_defective_v1`. See `rebalance/README.md`.
