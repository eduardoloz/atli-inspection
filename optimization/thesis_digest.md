# Thesis Digest — Vazquez (UNLV MS-EE, May 2025)

**"An Edge Computing Device Optimized and Transfer Learning Enhanced Deep Learning Model for Detecting Wildfire Flame and Smoke"** — Giovanny Vazquez; same research group as our ATLI project (same group as our ATLI project). Composed from 3 first-author papers (SmartNets 2024, SusTech 2025, arXiv:2501.08639). Page references below are the printed thesis page numbers.

Source PDF: `/Users/eddie/Research/Vegas/Giovanny Vazquez Final Thesis (1).pdf` (95 PDF pages; thesis body pp. 1–82).

---

## 1. Task, datasets, hardware

- **Task:** 2-class object detection (fire, smoke) from aerial/UAV RGB imagery, for real-time wildfire early detection on a **CPU-only edge device** (no GPU/FPGA — cost + power argument, pp. 2–3).
- **Target dataset — AFSE** (Aerial Fire and Smoke Essential, new, IEEE Dataport doi 10.21227/h2zw-pq68): 282 images, no augmentations, 656 fire + 333 smoke instances + 25 nulls; 70/15/15 split (198/42/42) (p. 25, Table 3.1 p. 26). Deliberately tiny/diverse — the same "small target dataset" regime as our ATLI.
- **Source datasets:** FASDD (95,314 imgs — homogeneous fire/smoke source), D-Fire (21,527), COCO (heterogeneous), FLAME/FLAME2 (video-frame, low diversity; only used to seed 66 AFSE images) (pp. 17–18, Table 2.1).
- **Training server:** 6× Quadro RTX 6000, PyTorch 1.4, Python 3.8 (Table 3.2 p. 28) — the same UNLV server family we use.
- **Edge device:** **Raspberry Pi 5, 8 GB, Cortex-A76 quad @ 2.4 GHz, CPU-only** (Table 3.3 p. 28). Power measured with a FNIRSI FNB58 USB tester at 100 Hz; energy integrated by Simpson's rule (pp. 31–34, Fig. 3.4/3.5).
- **Metrics:** AP@0.5 / mAP@0.5, plus edge metrics: FPS, average inference power (mW), and **normalized Energy-Delay Product (EDP)** = (E/Emax)·(t/tmax) (Eqns. 3.6–3.7, pp. 31–32). Real-time bar defined as **25–30 FPS** for drone footage (p. 31).
- **Models:** YOLOv5n focus; YOLOv8n/YOLO11n comparators; SOTA comparison vs RTM-DET Tiny, Dynamic-RCNN, DINO-4scale via MMDetection (Table 3.6 p. 36, §4.1.4).

---

## 2. Enhancement #1 — Transfer-learning protocol (Ch. 4)

### 2.1 Source-dataset choice: homogeneous ≫ heterogeneous ≫ scratch (Table 4.2 p. 41, Fig. 4.1 p. 40)

YOLOv5n on AFSE test set (mAP@0.5, %):

| Init | Frozen layers | Epochs | Test mAP@0.5 |
|---|---|---|---|
| scratch | – | 150 | 45.7 |
| scratch | – | 300 | 61.9 |
| scratch | – | 600 | **69.2** |
| COCO TL (heterogeneous) | 0 | 300 | 64.8 |
| COCO TL | 5 | 300 | 58.6 |
| COCO TL | 10 | 300 | 49.1 |
| **FASDD TL (homogeneous)** | **0** | **150** | **79.2** |
| FASDD TL | 5 | 150 | 71.8 |
| FASDD TL | 10 | 150 | 64.2 |

- **Headline: homogeneous (in-domain) pretraining beats COCO pretraining by +14.4 mAP and beats even 600-epoch scratch by +10 mAP, at half the fine-tune epochs.** 600-epoch scratch *can* beat COCO TL but never FASDD TL (p. 40).
- **Freezing always hurts** — 0 frozen layers best in every configuration; freezing 10 (whole backbone) costs 15 mAP from FASDD weights (pp. 39–41). Freezing buys only marginal training-time savings.
- Same ranking holds for v8n/11n (Table 4.3 p. 41): FASDD-init test mAP v5n 79.2 / v8n 76.8 / 11n 77.5, vs COCO-init 64.8 / 70.9 / 73.4.
- Fine-tune hyperparameters (Table 4.1 p. 39): batch 16/GPU, imgsz 640; **fine-tune lr0 = 0.001 for v5n but 0.0001 for v8n/11n**, and fine-tune epochs 150 (v5n) but only **75 for v8n/11n** — explicitly chosen "to limit overfitting for each model" (p. 38).

### 2.2 TL improves generalizability (5-fold stratified CV, §4.1.2, Fig. 4.2 p. 43)

Stratified 5-fold CV: FASDD-TL vs 600-ep scratch vs 300-ep scratch — mAP mean 74.0 vs 66.9 vs 61.7; **mAP std 4.23 vs 3.85 vs 5.86**; AP_fire std 9.26 vs 8.04 vs 11.53. TL raises the mean and reduces variance vs 300-ep scratch (variance conclusion is strongest for the fire class; 600-ep scratch is comparably stable but 7 mAP worse).

### 2.3 Cascaded TL does NOT help; merged-source pretraining does (§4.1.3, Table 4.4 p. 46)

Three-stage chains (COCO→FASDD→AFSE; FASDD→D-Fire→AFSE) vs a single merged pretrain (FASDD+D-Fire → AFSE):

| Pipeline (best variant) | Test mAP@0.5 |
|---|---|
| COCO→FASDD(0 frz)→AFSE(0 frz) | 78.0 |
| FASDD→DFIRE(0 frz)→AFSE(0 frz) | 80.5 |
| FASDD→DFIRE(10 frz)→AFSE(0 frz) | 79.9 |
| **merged FASDD+DFIRE scratch → AFSE (0 frz)** | **80.3** |
| (single-stage FASDD→AFSE reference) | 79.2 |

Conclusions (pp. 44–45, 72): an extra TL stage yields *worse or similar* results vs one stage; **the best strategy for multiple large sources is to amalgamate them into one pretraining base** (also cheaper in total training time). Gains from adding D-Fire at all were marginal (~+1 mAP).

### 2.4 SOTA comparison (§4.1.4, Table 4.5 p. 49)

All models FASDD-pretrained then fine-tuned on AFSE. Test mAP@0.5: **YOLOv5n 79.3** > DINO-4scale 79.1 > RTM-DET Tiny 78.4 > YOLO11n 77.5 > YOLOv8n 76.8 > Dynamic-RCNN 74.9. A 1.8M-param nano YOLO ties a 47.7M DINO — the same conclusion our own SOTA-CV block reached on ATLI.

### 2.5 TL has no effect on edge metrics (§4.1.5, Tables 4.6–4.7 pp. 50–51, Fig. 4.5 p. 52)

- On the Pi 5, **YOLOv5n runs ~2× the FPS of v8n/11n** (6.1 vs 3.3 FPS PyTorch) and has the lowest EDP, regardless of init (scratch/COCO/FASDD ≈ identical FPS/power). Accuracy source doesn't change compute cost — so accuracy work and edge work decouple.
- Nano variants have 3–8× lower EDP than the small (s) variants (Fig. 4.4 p. 50).

---

## 3. Enhancement #2 — Lightweight architecture modifications (Ch. 5)

All ablations: YOLOv5n derivatives trained **from scratch** on AFSE, 300 ep, validation mAP (Table 5.6 p. 65; Table 5.7 p. 67).

| Modification | Params (M) | FLOPs (B) | Val mAP@0.5 |
|---|---|---|---|
| baseline YOLOv5n | 1.76 | 4.1 | **59.3** |
| MobileNetV3-Small backbone | 1.29 | 2.1 | 51.1 |
| ShuffleNetV2 backbone | **0.73** | **1.5** | 45.7 |
| GhostConv+C3Ghost backbone (Ghost-B) | 1.29 | 2.9 | 53.3 |
| Ghost backbone+neck (Ghost-BN) | 0.94 | 2.3 | 51.9 |
| MV3 + 3×3 bottleneck kernel + Ghost neck | 1.40 | 2.4 | 59.2 |
| **MG3x3-Half** (MV3 + 3×3 kernel + halved neck channels) | 0.84 | 1.3 | 55.7 |

- Every swap **cuts params/FLOPs 25–60% but loses 4–14 val mAP on the small dataset**. Best accuracy-recovery levers: **3×3 kernel in the Bottleneck's first conv (as in YOLOv8)** and Ghost convs in the neck; halving neck output channels cuts complexity cheaply (pp. 55, 66; conclusions 7–8 p. 72–73).
- **On the big FASDD dataset the picture changes** (Table 5.8 p. 69): Ghost-BN *matches* base v5n (test mAP 85.5 vs 85.5) and MG3x3-Half loses only ~3 (82.3) — i.e., compressed architectures are data-hungry; their penalty is largest exactly in the small-data regime.
- MG3x3-Half's weakness concentrated in **fire AP on AFSE test** (56.5 vs 70.0), i.e., the harder/smaller class degrades most (p. 69, Table 5.9).

---

## 4. Enhancement #3 — Post-training framework export (§5.5, Table 5.9 p. 70, Fig. 5.12)

PyTorch weights exported to **ONNX** and **OpenVINO**, inference on the Raspberry Pi 5 CPU:

| Model | Runtime | FPS | Avg power (mW) | Test mAP@0.5 |
|---|---|---|---|---|
| YOLOv5n | PyTorch | 6.1 | 6887 | 79.2 |
| YOLOv5n | ONNX | 8.3 | 7392 | 80.6 |
| **YOLOv5n** | **OpenVINO** | **17.4** | **6649** | **80.6** |
| Ghost-BN | OpenVINO | 20.9 | 6404 | 71.3 |
| MG3x3-Half | OpenVINO | **31.9** | **6101** | 71.2 |

- **Framework conversion is a free lunch: OpenVINO gives standard YOLOv5n a 2.9× FPS gain with *zero* accuracy loss** (79.2→80.6 mAP; the small gain is an export/eval artifact) and slightly lower power. ONNX gives 1.4× (at higher power).
- The headline "423% faster, −11.4% power, 31.9 FPS" abstract claim = MG3x3-Half + OpenVINO vs base PyTorch v5n — but it costs −8 test mAP; the *accuracy-neutral* option (plain v5n + OpenVINO) still reaches 17.4 FPS.
- Note: only architecture + runtime conversion were explored; **pruning, quantization, knowledge distillation explicitly NOT explored** (p. 14) and named as future work (p. 73).

---

## 5. Thesis conclusions relevant to us (pp. 72–73)

1. Homogeneous TL significantly improves accuracy, efficiency, generalizability.
2. Longer training broadens applicability but doesn't directly raise accuracy.
3. Cascaded TL adds complexity without benefit.
4. Amalgamate multiple large sources into one pretraining base.
5. TL has no cost at inference (accuracy work is free at the edge).
6. YOLOv5n is the best FPS/EDP model on CPU-only edge hardware (~2× v11n's FPS).
7–8. MobileNetV3 + halved neck channels ≈ ShuffleNetV2-level compression without its accuracy collapse; Ghost neck + 3×3 bottleneck kernels recover MobileNetV3's accuracy loss.
9. MG3x3-Half + OpenVINO = 31.9 FPS real-time CPU detection with minimal (task-dependent) accuracy impact.

## 6. Artifacts

- AFSE dataset: IEEE Dataport doi 10.21227/h2zw-pq68 (p. 74).
- Code (modified YOLOv5 YAMLs, power/energy calculator, CV builder): https://github.com/ms2025code/modifiedYOLOv5Files/tree/main and a Google Drive folder (p. 74).
