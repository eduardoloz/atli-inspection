# Citation audit — Sections 2–3 draft (`main.tex`)

**Audited:** 2026-09-10 · all 47 cite keys used in `paper/draft_sections_2_3/main.tex`
**Draft under audit:** [local PDF](main.pdf) · [Overleaf project](https://www.overleaf.com/project/6aa0740339f566201feeae3e)
**Method:** every DOI resolved against Crossref (publisher-deposited metadata); arXiv entries against the arXiv API; abstracts via Crossref/OpenAlex/arXiv/PMC; full-text quotes extracted from the 13 papers held locally in `local/ATLI/` and `local/refs/` and from open-access copies (PMC, Frontiers). Each entry below gives the claim our draft makes, a link to a PDF of the cited paper, and a verbatim quote from that paper supporting the claim.

Status legend: ✅ verified · ⚠️ verified with a caveat worth reading · ❌ needs a fix before submission.

---

## Issues found (ranked)

1. ❌ **`vazquez2026wildfire` — the Ghost-backbone claim is not in the cited paper.** The draft says *"a Ghost-modified backbone traded ∼8 mAP points for a 4× CPU speedup [vazquez2026wildfire]"* and grafts are *"mirroring the optimization methodology of [vazquez2026wildfire]"*. The published Sensors 26(10):3197 paper (checked in full text via [PMC13210558](https://pmc.ncbi.nlm.nih.gov/articles/PMC13210558/)) contains **no GhostNet/Ghost-backbone content**. The numbers live in the Vazquez **thesis**, Tables 5.8–5.9 (marked "© 2025 IEEE", i.e. a separate IEEE publication): base YOLOv5n test mAP 79.2 → Ghost-BN OpenVINO 71.3 / MG3x3-Half 71.2 (≈ −8 mAP) at 6.1 → 20.9 / 31.9 FPS on CPU (≈3.4–5.2×). **Fix:** cite the thesis or locate the 2025 IEEE optimization paper the thesis figures credit; a quick search did not surface it as published — confirm with Giovanny/the PI.
2. ❌ **`liu2020review` — wrong title in `refs.bib`.** The bib carries the arXiv-preprint title ("Review of data analysis in vision inspection of power lines…"). The published title (Crossref, Annual Reviews in Control 50:253–277) is **"Data analysis in visual power line inspection: An in-depth review of deep learning for component detection and fault diagnosis."** Volume/pages/DOI are correct; replace the title.
3. ⚠️ **`faisal2025powerline`/`fan2026survey` — "both are named core open challenges" is half-supported.** The Faisal review names class imbalance, data scarcity/quality, small-object detection, and "robust evaluation baselines" as open problems, but **never mentions cross-dataset duplication/overlap** (0 hits for "duplicat*" in the full text). If "both properties" means (imbalance, eval-set duplication), the second is our finding, not the surveys'. Suggest softening to "…data scarcity, imbalance, and evaluation quality are named core open challenges" or scoping the sentence to imbalance only.
4. ⚠️ **`oliveira2022furnas` — missing DOI.** Verified via OpenAlex/IEEE: **10.1109/SIBGRAPI55357.2022.9991806**. Add to the bib (the entry's page range 7–12 I could not confirm; the DOI is the safer anchor).
5. ⚠️ **`do2023outages` — missing DOI.** Crossref-verified: **10.1038/s41467-023-38084-6** (Nat. Commun. 14, 2470). Add to the bib.
6. ⚠️ **`farooq2026teyolov8` — example slightly over-claimed on both uses.** (a) Their augmentation stack includes **mixup but not blur** — the draft's "applies mixup and blur only inside undifferentiated augmentation stacks (e.g., [farooq…])" needs the example scoped to mixup, or a second example for blur. (b) On INT8: they report QAT 93.8 vs PTQ 93.3 mAP — supports "QAT preserves accuracy better than PTQ," but their gap is 0.5 pt; the draft's "has been reported to require quantization-aware training" is a fair but strong reading.
7. ⚠️ **`tsai2025atli` — "that study's best configuration (all layers unfrozen)" has a v11n exception.** The APET paper states "In general, unfreezing all layers in the Feature Extraction step obtains the best mAP@0.5 result in each version group," and that holds for v5n (78.9) and v8n (77.5) — but for **v11n the frozen-10 TL (76.8) beat unfrozen (76.1)**, and 76.8 is exactly the number our draft compares against. The draft's sentence is defensible (it describes the fine-tuning stage, which is always unfrozen) but a reviewer holding Table V may push; consider "matching that study's overall best configuration".
8. ⚠️ **`bao2022bcyolo` — "a 300-image DVDI sample":** the 300 is the size of the public Roboflow sample *we* audited, not DVDI's size (DVDI is a larger dataset). The sentence is accurate but easy to misread as DVDI ≈ 300 images; consider "a 300-image public sample of DVDI".
9. ✅ (note only) **Year conventions:** `abdelfattah2020ttpla` — Crossref says 2021 (Springer LNCS proceedings volume); bib's 2020 (ACCV 2020) is the standard citation, fine. `su2024cessd` — Crossref says 2023 (online-first); bib's 2024 (print issue, vol 83) is fine.

---

## Per-citation audit

### Core project citations

**`tsai2025atli`** — Tsai, Yang, Zhai, "Autonomous Transmission Line Inspection using Transfer Learning Enhanced Deep Learning Models," APET 2025, T0388. ⚠️ (see issue 7)
*Cited for:* the originating ATLI dataset; the 2-stage TL recipe (150+100 ep, lr 0.01→0.00334); the 76.8% v11n TL comparison; unfreeze-all confirmation.
*PDF:* no public copy found (APET proceedings); local: `local/refs/APET2025_T0388.pdf` (gitignored).
*Quotes:* "Containing a total of 732 images with no augmentations, ATLI has a combination of images with defective components and images with all normal components. About 200 images are taken from the CPLID dataset and other datasets while the remaining images are taken from the Internet." · "The general process of transfer learning includes three stages: 1) training a deep learning model from scratch on the source dataset; 2) training the DL model with the target data starting with the best weights obtained… referred to as Feature Extraction; and 3) unfreezing all layers and continuing training on the target dataset at a much lower learning rate, a stage called Fine-Tuning." · "In general, unfreezing all layers in the Feature Extraction step obtains the best mAP@0.5 result in each version group." · Table V: v11n (TL) 10-layer fz = **76.8** mAP@0.5; Unfrozen = 76.1; v5n unfrozen = 78.9; v8n unfrozen = 77.5. Hyperparameters table: lr0 fine-tuning 0.00334, lrf 0.1535, split 70/15/15 — all match our Table III.

**`vazquez2024wildfire`** — SmartNets 2024, DOI [10.1109/SmartNets61466.2024.10577715](https://doi.org/10.1109/SmartNets61466.2024.10577715). ✅
*Cited for:* companion wildfire TL work; (with 2026) the curated-source-beats-COCO result; tabulated validation accuracies (methodological contrast).
*Quote (abstract):* "This study highlights the importance of Transfer Learning (TL) to enhance the YOLOv5 convolutional neural network model's performance for detecting wildfire smoke and flames… We introduce the Aerial Fire and Smoke Essential (AFSE) dataset as the target dataset…"

**`vazquez2026wildfire`** — Sensors 26(10):3197, DOI [10.3390/s26103197](https://doi.org/10.3390/s26103197). ❌ for the Ghost claim (issue 1); ✅ for the other four uses.
*PDF:* [PMC13210558 (free full text)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13210558/) · [MDPI](https://www.mdpi.com/1424-8220/26/10/3197).
*Cited for and quotes:*
- 25–30 FPS real-time standard — confirmed in the paper's discussion ("often ~25–30 FPS"); thesis phrasing: "Given that drone footage involves fast motion events that can induce motion blur, 25-30 FPS provides an optimal standard for real-time object detection." ✅
- 14.4-mAP curated-source-beats-COCO — verbatim: "fine-tuning from FASDD with zero frozen layers reaches **79.2%** test mAP@0.5 after 150 epochs. Heterogeneous TL from COCO also improves performance but is less pronounced: fine-tuning with zero frozen layers reaches **64.8%** test mAP@0.5" (79.2 − 64.8 = 14.4). ✅
- power draw / energy-delay product instrumented — verbatim: "Average power P̄ is computed as the mean of P(t) over the interval. Energy consumption is computed as in (6)…"; EDP in their Tables 10/Figs 7–8. ✅
- Ghost-modified backbone ~8 mAP / 4× CPU — **not in this paper**; see issue 1. ❌

### Datasets

**`tao2020insulator`** (CPLID) — IEEE TSMC-S 50(4):1486–1498, DOI [10.1109/TSMC.2018.2871750](https://doi.org/10.1109/TSMC.2018.2871750). ✅
*Cited for:* CPLID as duplicate source (Sec. II); CPLID's defective images being synthetically composited (train-only restoration rationale).
*PDF:* local `local/ATLI/Detection_of_Power_Line_Insulator_Defects_Using_Aerial_Images_Analyzed_With_Convolutional_Neural_Networks.pdf`; paywalled at IEEE; dataset repo: [github.com/InsulatorData/InsulatorDataSet](https://github.com/InsulatorData/InsulatorDataSet).
*Quote:* "To address the scarcity of defect images in a real inspection environment, a data augmentation method is also proposed that includes four operations: 1) affine transformation; 2) insulator segmentation and background fusion; 3) Gaussian blur; and 4) brightness transformation." — directly supports "CPLID's defective-insulator images are synthetically composited." (The APET paper corroborates the size: "The Chinese Power Line Insulator Dataset (CPLID) contains 848 labeled insulator images.")

**`bao2022bcyolo`** (DVDI) — Remote Sensing 14(20):5176, DOI [10.3390/rs14205176](https://doi.org/10.3390/rs14205176). ⚠️ (issue 8)
*Cited for:* DVDI as duplicate source into ATLI.
*PDF:* [MDPI open access](https://www.mdpi.com/2072-4292/14/20/5176/pdf); local copy in `local/ATLI/`.
*Quote:* "We constructed a dataset of vibration dampers and insulators (DVDI) on transmission lines in images obtained by the UAV." · "The DVDI dataset includes normal and defective vibration dampers as well as normal and defective insulators."

**`oliveira2022furnas`** (PTL-AI Furnas) — SIBGRAPI 2022, DOI **10.1109/SIBGRAPI55357.2022.9991806** (add to bib). ⚠️ (issue 4)
*Cited for:* Furnas as duplicate source (32 eval images inside ATLI).
*PDF:* [IEEE Xplore](https://doi.org/10.1109/SIBGRAPI55357.2022.9991806) (paywalled; no OA copy found).
*Quote (abstract):* "We present a new images dataset called PTL-AI Furnas Dataset as a new benchmark for fault detection in power transmission lines. This dataset has 6,295 images, with resolution 1280×720… It contains annotations of 17,808 components classified as baliser, bird nest, insulator, spacer and stockbridge."

**`shuang2023rsin`** (RSIn-Dataset) — Drones 7(2):125, DOI [10.3390/drones7020125](https://doi.org/10.3390/drones7020125). ✅
*Cited for:* dataset lit-review anchor. *PDF:* [MDPI open access](https://www.mdpi.com/2504-446X/7/2/125/pdf); local `local/ATLI/drones-07-00125.pdf`.
*Quote:* "RSIn-Dataset: An UAV-Based Insulator Detection Aerial Images Dataset and Benchmark… datasets have a significant impact on deep learning [in] power line inspection based on computer vision."

**`abdelfattah2020ttpla`** (TTPLA) — ACCV 2020 / Springer LNCS 12627, DOI [10.1007/978-3-030-69544-6_36](https://doi.org/10.1007/978-3-030-69544-6_36). ✅ (year note, issue 9)
*Cited for:* dataset lit-review anchor. *PDF:* [arXiv:2010.10032](https://arxiv.org/pdf/2010.10032).
*Quote:* "Accurate detection and segmentation of transmission towers (TTs) and power lines (PLs) from aerial images plays a key role in protecting power-grid security and low-altitude UAV safety…"

**`vieiraesilva2021stnplad`** (STN PLAD) — SIBGRAPI 2021, DOI [10.1109/SIBGRAPI54419.2021.00037](https://doi.org/10.1109/SIBGRAPI54419.2021.00037). ✅
*Cited for:* dataset lit-review anchor. *PDF:* local `local/ATLI/STN_PLAD_...pdf`; paywalled at IEEE.
*Quote:* "It has 2,409 annotated objects divided into five classes: transmission tower, insulator, spacer, tower plate, and Stockbridge damper, which vary in size (resolution), orientation, illumination, angulation, and background."

**`lin2014coco`** (COCO) — ECCV 2014, DOI [10.1007/978-3-319-10602-1_48](https://doi.org/10.1007/978-3-319-10602-1_48). ✅
*Cited for:* TL source dataset. *PDF:* [arXiv:1405.0312](https://arxiv.org/pdf/1405.0312) · [Springer](https://link.springer.com/content/pdf/10.1007%2F978-3-319-10602-1_48.pdf).

### Damper-specific detectors

**`zhang2024dsanet`** (DSA-Net) — IEEE TIM 73, art. 3501022, DOI [10.1109/TIM.2023.3331418](https://doi.org/10.1109/TIM.2023.3331418). ✅
*Cited for:* damper lit anchor; **"63.6% AP"** as worst-class evidence.
*PDF:* local `local/ATLI/DSA-Net_...pdf`; paywalled at IEEE.
*Quote:* "…resulting in DSA-Net achieving the highest AP of **63.6% for the 'Damage' class**. However, further improvements are still required to elevate the detection of damage to a similarly high level." — their best model's damage class is the one they single out as still lagging; supports "worst or near-worst class."

**`su2024cessd`** (CE-SSD) — Multimed. Tools Appl. 83:36419–36431, DOI [10.1007/s11042-023-15063-z](https://doi.org/10.1007/s11042-023-15063-z). ✅ (year note, issue 9)
*Cited for:* **"40.0%"** damper AP as worst-class evidence.
*PDF:* local `local/ATLI/Transmission_line_defect_detection_based_on_featur.pdf`; paywalled at Springer.
*Quote (Table 5, TLDD dataset):* "CE-SSD (ours) 66.65 [mAP] … damper **39.95**" — damper (defined as "broken or missing dampers") is the second-lowest class for every method in the table (32.68–39.95 across models, above only broken strands). 39.95 rounds to the draft's 40.0%.

**`bao2021pmayolo`** (PMA-YOLO) — Remote Sensing 13(20):4134, DOI [10.3390/rs13204134](https://doi.org/10.3390/rs13204134). ✅
*Cited for:* damper lit anchor. *PDF:* [MDPI open access](https://www.mdpi.com/2072-4292/13/20/4134/pdf).
*Quote:* "The accurate detection and timely replacement of abnormal vibration dampers on transmission lines are critical for the safe and stable operation of power systems… we constructed a data set of abnormal vibration dampers (DAVDs)…"

**`huang2023damper`** — IEEE TIM 72, art. 5008114, DOI [10.1109/TIM.2022.3228008](https://doi.org/10.1109/TIM.2022.3228008). ✅
*Cited for:* damper lit anchor. *PDF:* paywalled at IEEE (no OA copy found).
*Quote (abstract):* "Metal dampers, a crucial protective fitting in the line, can effectively suppress the conductor's vibration energy… we are proposing a detection method for structural defect damper based on spatial relationship."

**`liu2021slippage`** — Electr. Power Syst. Res. 199:107449, DOI [10.1016/j.epsr.2021.107449](https://doi.org/10.1016/j.epsr.2021.107449). ✅
*Cited for:* damper lit anchor. *PDF:* paywalled at Elsevier. Title self-supporting: "Slippage fault diagnosis of dampers for transmission lines based on faster R-CNN and distance constraint" (Crossref-verified).

**`zhou2026carenet`** (CARE-Net) — Sensors 26(17):5648, DOI [10.3390/s26175648](https://doi.org/10.3390/s26175648). ✅
*Cited for:* "recent damper-condition detectors report only on custom single-purpose datasets."
*PDF:* [MDPI open access](https://www.mdpi.com/1424-8220/26/17/5648/pdf).
*Quote (abstract):* "Vibration damper detection in unmanned aerial vehicle (UAV)-based transmission line inspection presents distinctive task-specific challenges: the targets are not only small and weakly textured, but also characterized by slender structures." (Their evaluation is on a self-built damper dataset — consistent with "custom single-purpose.")

**`yang2025oneforall`** — Eng. Appl. Artif. Intell. 156:111129, DOI [10.1016/j.engappai.2025.111129](https://doi.org/10.1016/j.engappai.2025.111129). ✅
*Cited for:* "a recent domain-generalization study reports the same cross-domain loss for damper-defect detectors."
*PDF:* paywalled at [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625011303).
*Support (abstract, paraphrase — full text paywalled):* proposes single-domain generalization for vibration-damper defect detection because detector performance degrades across domains; uses geometric-structural domain knowledge with MCL/MSM losses. The premise (cross-domain degradation motivates SDG) supports our sentence; quote the exact abstract sentence at submission time if a full-text copy is obtained.

### Detect-then-classify / transformer detectors

**`zhang2023padetr`** (PA-DETR) — IEEE TIM 72, art. 5016914, DOI [10.1109/TIM.2023.3282302](https://doi.org/10.1109/TIM.2023.3282302). ✅
*Cited for:* detect-then-classify proposed for visually similar defect pairs.
*PDF:* local `local/ATLI/PA-DETR_...pdf`; paywalled at IEEE.
*Quote (title):* "PA-DETR: End-to-End **Visually Indistinguishable Bolt Defects** Detection Method Based on Transmission Line Knowledge Reasoning."

**`souza2023hybridyolo`** (Hybrid-YOLO) — Int. J. Electr. Power Energy Syst. 148:108982, DOI [10.1016/j.ijepes.2023.108982](https://doi.org/10.1016/j.ijepes.2023.108982). ✅
*Cited for:* detect-then-classify for insulator defects. *PDF:* paywalled at Elsevier.
*Support:* Crossref-verified title "Hybrid-YOLO for classification of insulators defects in transmission lines based on UAV" — the hybrid = YOLO detection followed by a classification stage, matching "detect-then-classify."

**`zhang2024hcvit`** (HC-ViT) — IEEE TSMC-S 54(11):6510–6521, DOI [10.1109/TSMC.2024.3386873](https://doi.org/10.1109/TSMC.2024.3386873). ✅
*Cited for:* transformer-detector lit anchor. *PDF:* local `local/ATLI/Transmission Line Component Defect Detection Based on Patrol Images.pdf` (title page confirms it is the HC-ViT paper); paywalled at IEEE.

**`carion2020detr`** (DETR) — ECCV 2020. ✅ *PDF:* [arXiv:2005.12872](https://arxiv.org/pdf/2005.12872). *Quote:* "We present a new method that views object detection as a direct set prediction problem."

### Transfer learning in inspection

**`shakiba2022insulator`** — IEEE SMC Magazine 8(4):15–25, DOI [10.1109/MSMC.2022.3198027](https://doi.org/10.1109/MSMC.2022.3198027). ✅
*Cited for:* freezing-based feature extraction being standard classification-side practice (contrast to our unfreeze-all finding).
*PDF:* local `local/ATLI/A_Transfer_Learning-Based_Method_...pdf`; paywalled at IEEE.
*Quote:* "…a pretrained neural network with its fully connected layers removed is used, and its **convolutional and pooling layers become frozen** to carry out the feature extraction. For adaptation of the given pretrained neural network to the target dataset, the fully connected layers (classifier layers) need to update…"

**`pradeep2025insulator`** — Neural Comput. Appl. 37:6951–6976, DOI [10.1007/s00521-025-11011-0](https://doi.org/10.1007/s00521-025-11011-0). ✅
*Cited for:* TL-in-inspection lit anchor. *PDF:* paywalled at Springer; local `local/ATLI/An improved transfer learning model for detection of insulator defects.pdf`.

### Surveys & motivation

**`do2023outages`** — Nat. Commun. 14:2470 (2023), DOI **10.1038/s41467-023-38084-6** (add to bib). ⚠️ (issue 5)
*Cited for:* U.S. outage statistics anchor (intro placeholder).
*PDF:* [Nature open access](https://www.nature.com/articles/s41467-023-38084-6.pdf).
*Quote (abstract):* "Here, we characterize 2018–2020 outages, finding an average of 520 million customer-hours total without power annually across 2447 US counties (73.7% of the US population)."

**`nguyen2018autonomous`** — Int. J. Electr. Power Energy Syst. 99:107–120, DOI [10.1016/j.ijepes.2017.12.016](https://doi.org/10.1016/j.ijepes.2017.12.016). ✅ Review anchor; Crossref-verified; paywalled at Elsevier.

**`liu2020review`** — Annu. Rev. Control 50:253–277, DOI [10.1016/j.arcontrol.2020.09.002](https://doi.org/10.1016/j.arcontrol.2020.09.002). ❌ **title wrong in bib** (issue 2). Published title: "Data analysis in visual power line inspection: An in-depth review of deep learning for component detection and fault diagnosis." *PDF:* preprint [arXiv:2003.09802](https://arxiv.org/abs/2003.09802) (carries the bib's old title).

**`yang2020powerline`** — IEEE TIM 69(12):9350–9365, DOI [10.1109/TIM.2020.3031194](https://doi.org/10.1109/TIM.2020.3031194). ✅ Review anchor; paywalled.
*Quote (abstract):* "With the fast development of smart grid, the power line mileage and power equipments get rapid growth… the traditional maintenance mode has the disadvantages of over or under maintenance…"

**`faisal2025powerline`** — Applied Energy 385:125507, DOI [10.1016/j.apenergy.2025.125507](https://doi.org/10.1016/j.apenergy.2025.125507). ⚠️ (issue 3)
*Cited for:* review anchor; "imbalance and [duplication] named core open challenges."
*PDF:* open access (CC-BY) at [ScienceDirect](https://doi.org/10.1016/j.apenergy.2025.125507); local `local/ATLI/Deep learning in automated power line inspection A Review.pdf`.
*Quotes:* "The scarcity of large-scale, publicly available datasets remains a significant bottleneck, with most studies limited to sm[all]…" · "These questions encompass challenges related to data quality, the intricacies of small object detection, the application of deep learning in embedded systems, and the definition of robust evaluation baselines." · Class imbalance is named ("Power lines are often a small portion of the image, leading to class imbalance issues"). **No mention of cross-dataset duplication/leakage anywhere in the text.**

**`fan2026survey`** — Drones 10(1):55, DOI [10.3390/drones10010055](https://doi.org/10.3390/drones10010055). ⚠️ (same sentence as issue 3 — verify its full text names imbalance/eval-quality before submission)
*PDF:* [MDPI open access](https://www.mdpi.com/2504-446X/10/1/55/pdf).
*Quote (abstract):* "With the rapid development of the power Internet of Things (IoT), the traditional manual inspection mode can no longer meet the growing demand for power equipment inspection."

### Tools, methods & SOTA baselines

**`jocher2023ultralytics`** — [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics). ✅
*Cited for:* framework + nano-variant params/GFLOPs. Note: the draft's numbers (v5n 2.51M/7.2, v8n 3.08M/8.5, v11n 2.66M/6.7 with OBB heads, from our `results/model_flops.json` collector) correspond to the current Ultralytics checkpoints ("v5nu"-style heads) — intentionally different from the original-v5n 1.8–1.9M figure in the APET paper's Table II. No action needed; be ready for the question.

**`chen2019mmdetection`** — [arXiv:1906.07155](https://arxiv.org/pdf/1906.07155). ✅ "We present MMDetection, an object detection toolbox that contains a rich set of object detection and instance segmentation methods…"

**`zhang2023dino`** (DINO) — ICLR 2023, [arXiv:2203.03605](https://arxiv.org/pdf/2203.03605). ✅ "We present DINO (DETR with Improved deNoising anchOr boxes), a state-of-the-art end-to-end object detector."

**`lyu2022rtmdet`** — [arXiv:2212.07784](https://arxiv.org/pdf/2212.07784). ✅ "we aim to design an efficient real-time object detector that exceeds the YOLO series…"

**`zhang2020dynamicrcnn`** — ECCV 2020, [arXiv:2004.06002](https://arxiv.org/pdf/2004.06002). ✅ "we first point out the inconsistency problem between the fixed network settings and the dynamic training procedure…"

**`han2020ghostnet`** — CVPR 2020, [arXiv:1911.11907](https://arxiv.org/pdf/1911.11907). ✅ "Deploying convolutional neural networks (CNNs) on embedded devices is difficult due to the limited memory and computation resources. The redundancy in feature maps…"

**`howard2017mobilenets`** — [arXiv:1704.04861](https://arxiv.org/pdf/1704.04861). ✅ "MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks."

**`chen2023fasternet`** (FasterNet/PConv) — CVPR 2023, [arXiv:2303.03667](https://arxiv.org/pdf/2303.03667). ✅ "We observe that such reduction in FLOPs, however, does not necessarily lead to a similar level of reduction in latency." (Also independently supports the draft's GFLOPs-are-a-poor-proxy discussion.)

**`xia2018dota`** (DOTA) — CVPR 2018, [arXiv:1711.10398](https://arxiv.org/pdf/1711.10398). ✅
*Cited for:* the DOTA Task-1 rotated-box evaluation protocol used to score the YOLOv5-OBB fork. The dataset/benchmark paper defines the rotated-box detection task; the devkit implements the protocol.

**`zhang2018mixup`** — ICLR 2018, [arXiv:1710.09412](https://arxiv.org/pdf/1710.09412). ✅
*Cited for:* mixup as label-noise-tolerant regularizer. *Quote:* "mixup trains a neural network on convex combinations of pairs of examples and their labels… mixup regularizes the neural network to favor simple linear behavior in-between training examples… reduces the memorization of corrupt labels."

**`buslaev2020albumentations`** — Information 11(2):125, DOI [10.3390/info11020125](https://doi.org/10.3390/info11020125). ✅
*Cited for:* the blur-augmentation pipeline. *PDF:* [MDPI open access](https://www.mdpi.com/2078-2489/11/2/125/pdf).
*Quote (abstract):* "Data augmentation is a commonly used technique for increasing both the size and the diversity of labeled training sets… we present Albumentations, a fast and flexible open source library for image augmentation…"

**`akyon2022sahi`** (SAHI) — ICIP 2022, pp. 966–970, DOI [10.1109/ICIP46576.2022.9897990](https://doi.org/10.1109/ICIP46576.2022.9897990). ✅
*Cited for:* slicing-aided inference as the standard small-object technique (inapplicable at our native 640²/512²).
*PDF:* [arXiv:2202.06934](https://arxiv.org/pdf/2202.06934).
*Quote:* "Detection of small objects and objects far away in the scene is a major challenge in surveillance applications. Such objects are represented by small number of pixels in the image and lack sufficient details…"

**`laf2026yolo`** — [arXiv:2602.13378](https://arxiv.org/pdf/2602.13378). ✅
*Cited for:* P2 heads as a standard small-object technique in aerial detection. arXiv API confirms title ("LAF-YOLOv10 with Partial Convolution Backbone, Attention-Guided Feature Pyramid, Auxiliary P2 Head, and Wise-IoU Loss for Small Object Detection in Drone Aerial Imagery") and the auxiliary-P2-head component.

### 2025–2026 SOTA (refs_sota.bib)

**`hu2026poldyolo`** (POLD-YOLO) — Sensors 26(5):1733, DOI [10.3390/s26051733](https://doi.org/10.3390/s26051733). ✅ *PDF:* [MDPI](https://www.mdpi.com/1424-8220/26/5/1733/pdf). *Quote:* "This study aims to develop a highly efficient and accurate model for real-time insulator defect inspection on resource-constrained UAV platforms."

**`chen2025plddetr`** (PLD-DETR) — Electronics 14(20):4107, DOI [10.3390/electronics14204107](https://doi.org/10.3390/electronics14204107). ✅ *PDF:* [MDPI](https://www.mdpi.com/2079-9292/14/20/4107/pdf). *Quote:* "This paper proposes a Transformer-based framework… for transmission line defect detection."

**`wang2026mleyolo`** (MLE-YOLOv11n) — PLOS One 21(8):e0354898, DOI [10.1371/journal.pone.0354898](https://doi.org/10.1371/journal.pone.0354898). ✅ *PDF:* [PLOS open access](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0354898&type=printable). *Quote:* "Unmanned aerial vehicle (UAV)-based defect detection faces persistent challenges from small object scales, complex background clutter, and multi-scale feature misalignment in lightweig[ht detectors]…"

**`stefenon2026diffusion`** — Auton. Intell. Syst. 6(1):13, DOI [10.1007/s43684-026-00135-2](https://doi.org/10.1007/s43684-026-00135-2). ✅
*Cited for:* "synthetic-defect generation is emerging precisely because harvested real data transfers poorly."
*PDF:* [SpringerOpen](https://link.springer.com/content/pdf/10.1007/s43684-026-00135-2.pdf).
*Quote (abstract):* "Deep learning-based autonomous inspection of power grid insulators is challenged by data imbalance and model opacity… a conditional diffusion model generates realistic synthetic fault images to balance the dataset." (Supports the emergence of synthetic generation for data-scarce defects; their stated driver is imbalance/scarcity — poor transfer of harvested data is our framing, compatible but not verbatim theirs.)

**`farooq2026teyolov8`** — Front. Artif. Intell. 8:1732616, DOI [10.3389/frai.2025.1732616](https://doi.org/10.3389/frai.2025.1732616). ⚠️ (issue 6)
*PDF:* [Frontiers open access](https://www.frontiersin.org/articles/10.3389/frai.2025.1732616/pdf).
*Quotes:* "Data augmentation techniques, including random scaling (0.5–1.5), translation (±10%), rotation (±10°), color jittering, mosaic augmentation, and mixup, are applied to enhance model robustness…" (mixup inside an undifferentiated stack — **no blur** in their stack). · Table 5: INT8 (PTQ) 93.3% mAP vs INT8 (QAT) 93.8% — QAT preserves accuracy better than post-training quantization.

---

*Generated by the citation-audit pass, 2026-09-10. Local PDFs referenced above live under the gitignored `local/` tree; links marked "paywalled" verified via Crossref metadata rather than full text (quotes there come from local copies or publisher abstracts as noted).*
