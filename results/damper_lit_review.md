# Damper-Defect Literature Review & Dataset Search

**Date:** 2026-06-10/11 · **Source:** paper folder shared by the PI (local copies in `ATLI/*.pdf`)
**Motivation:** `Defective_Damper` (113 instances) is heavily confused with `Normal_Damper` (1460) in `merged_atli_target` (12.9:1 imbalance). This review covers (a) how the literature treats damper defects and (b) which datasets exist that could supply additional defective-damper training data.

---

## 1. Dataset inventory

Availability and annotations **verified against the actual repos on 2026-06-11** (GitHub API / raw README checks), not just the papers' claims.

| Dataset | Source paper | Availability | Annotations included? | Damper content | Defect states? | Notes for ATLI |
|---|---|---|---|---|---|---|
| **DVDI** | "A Defect Detection Method Based on BC-YOLO for Transmission Line Components in UAV Remote Sensing Images" (Bao et al., *Remote Sensing* 2022) | ⚠️ **Partial** — [github.com/Emp-8/DVDI](https://github.com/Emp-8/DVDI) hosts **only the 300-image test split, images only** (no releases, no other branches) | ❌ **No.** Paper describes LabelImg/VOC XML annotations, but zero label files are in the repo | Paper's full set: 3,610 normal + 636 damaged dampers in 1,500 imgs (976 raw UAV); repo reality: 300 unlabeled test JPGs | ✅ in paper (normal vs damaged = shedding/serious bending); ❌ in repo | ❌ **REJECTED 2026-06-11 — pHash vetting found ~50% of DVDI is ALREADY IN ATLI** (40/300 match `merged_atli_target` val/test, 109/300 match train; 37 pixel-identical). ATLI's "internet" images recycled DVDI. Hand-labeling plan cancelled — it would have leaked eval images into train. See `damper_dataset_vetting.md`. Author email for the full set is also moot unless overlap is screened. |
| **STN PLAD** | "STN PLAD: A Dataset for Multi-Size Power Line Assets Detection in High-Resolution UAV Images" (Vieira-e-Silva et al., SIBGRAPI 2021) | ✅ Full — [github.com/andreluizbvs/PLAD](https://github.com/andreluizbvs/PLAD): `plad.zip` (Releases + Google Drive mirror), GPL-3.0 | ✅ **Yes** — `labels.zip`, **COCO JSON**, 5 train/test Monte-Carlo split files. Class names in Portuguese: `torre`, `cadeia de isoladores`, `espaçador`, **`amortecedor` (= damper)**, `placa de identificação` | **1,505 Stockbridge dampers** in 133 high-res imgs (5472 px wide) | ❌ component-presence only | No defective dampers. Useful only as extra `Normal_Damper` / hard negatives (not our bottleneck). Cited in APET as related dataset, not used as data. |
| **TLD** | "DSA-Net: An Attention-Guided Network for Real-Time Defect Detection of Transmission Line Dampers Applied to UAV Inspections" (Zhang et al., IEEE TIM 2024) | ❌ Private | n/a (COCO format internally) | 490 UAV imgs; post-aug labels: Deformation 1,650 / Damage 1,380 / Inversion 960 / Normal 6,860 (train) | ✅ 3 defect subtypes + normal | Best-matched taxonomy to ours, but unavailable. Could email authors (Xi'an Polytechnic Univ.). |
| **TLCD** | "Transmission Line Component Defect Detection Based on UAV Patrol Images: A Self-Supervised HC-ViT Method" (Zhang et al., IEEE TSMC 2024) | ❌ Private | n/a (LabelImg internally) | Shockproof hammer damage 184, intersection 353 (boxes) | ✅ defect-only classes (no normal class) | Unavailable; North China Electric Power Univ. |
| **TLDD** | "Transmission line defect detection based on feature enhancement" (Su & Liu, *Multimedia Tools and Applications* 2024) | ❌ Private (provincial grid O&M) | n/a | `damper` class = broken/missing dampers, **2,306 instances** | ✅ defect-only (no normal class) | Largest defective-damper count seen anywhere; unavailable. |
| **Felect** | "Deep learning for component fault detection in electricity transmission lines" (Maduako et al., *J. Big Data* 2022) | ❌ Data private (NEC restriction); code public ([github.com/EmekaKing/Felect](https://github.com/EmekaKing/Felect)) | n/a (VOC2007 XML internally) | "Missing knob" (broken damper) **783 instances** | ✅ defect-only | Nigerian grid; data restricted. |
| **VIBD** | "PA-DETR: End-to-End Visually Indistinguishable Bolt Defects Detection Method Based on Transmission Line Knowledge Reasoning" (Zhang et al., IEEE TIM 2023) | ❌ Private | n/a (COCO format internally) | Bolts not dampers (8,972 bolts; defects 584–1,271) | ✅ normal + 3 defect types | Methodology reference only. |
| **Break-ID-1632 / cable damage** | "An Enhanced SL-YOLOv8-Based Lightweight Remote Sensing Detection Algorithm for Identifying Broken Strands in Transmission Lines" (Zhang et al., *Applied Sciences* 2024) | ❌ "Via corresponding author" | n/a | None (broken conductor strands) | ✅ | Not damper-relevant. |
| **RSIn** | "RSIn-Dataset: An UAV-Based Insulator Detection Aerial Images Dataset and Benchmark" (Shuang et al., *Drones* 2023) | ❌ **Dead link** — `github.com/caigouyihao/Rsin-dataset` returns 404 (all capitalizations); no mirror found via GitHub search | ❌ Unobtainable (paper: LabelImg/VOC) | **None** — 4 insulator types only | ❌ | Already known from APET related work; repo has been removed since publication. |
| **CPLID** | "Detection of Power Line Insulator Defects Using Aerial Images Analyzed With Convolutional Neural Networks" (Tao et al., IEEE TSMC 2018 — also in the `ATLI/` folder) | ✅ Full — [github.com/InsulatorData/InsulatorDataSet](https://github.com/InsulatorData/InsulatorDataSet) | ✅ **Yes** — VOC2007 XML in `labels/` subdirs for both parts | None (insulators: 600 normal + 248 synthetic-defective) | partial (defects are synthetic composites) | Already a source of ~200 ATLI images. |

**Bottom line for the dataset search (revised after repo verification):** *no* public dataset currently ships labeled defective dampers. DVDI's repo — despite the paper's claim — contains only 300 unlabeled test images; the only fully-downloadable damper annotations are STN PLAD's (normal dampers only, COCO JSON). Getting labeled defective-damper data therefore requires either hand-labeling DVDI's 300 test images, or emailing authors (DVDI and TLD/DSA-Net are the two best targets — DVDI for matching binary taxonomy, TLD for subtype labels). *(See §1b below — Roboflow Universe softens this conclusion.)*

---

## 1b. Web-search extension (2026-06-11) — papers/datasets NOT in the PI's folder

A follow-up web sweep for UAV defective-damper datasets beyond the folder. Roboflow entries verified live via the Universe API.

### New papers with defective-damper datasets (all private)

| Dataset | Source paper | Availability | Damper defect content |
|---|---|---|---|
| **DAVD** | "Detection of Abnormal Vibration Dampers on Transmission Lines in UAV Remote Sensing Images with PMA-YOLO" (*Remote Sensing* 2021) | ❌ No public download found (no GitHub/Zenodo/Mendeley trace) | UAV imgs; 4 damper hardware types (FD/FDZ/FDY/FFH — same taxonomy as DVDI, adjacent group), each labeled **rusty / defective / normal**; mAP 94.3. Closest taxonomy match found anywhere → author email worthwhile. |
| **DamperDetSet** | "Transmission Line Vibration Damper Detection Using Deep Neural Networks Based on UAV Remote Sensing Image" (*Sensors* 2022) | ❌ Explicitly confidential (Yunnan Electric Power supplier restriction) | 3,000 UAV-cruise-video images, LabelMe boxes. Dead end. |
| — (synthesizes data) | "MC-GAN" damper paper (*Sensors* 2022) | ❌ No public data | Conditional-GAN synthesis of damper defects (missing-head, tilted) — a synthetic-augmentation precedent, not a dataset. |

### InsPLAD — STN PLAD's successor (public, labels included, but no damper defects)

"InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images" (*IJRS* 2023, [arXiv:2311.01619](https://arxiv.org/abs/2311.01619), [github.com/andreluizbvs/InsPLAD](https://github.com/andreluizbvs/InsPLAD), CC BY-NC 3.0). 10,607 UAV images, 17 asset classes incl. **Damper–Stockbridge (~5,700 train boxes) and Damper–Spiral (1,020)** — annotations ship in the zips. Its defect subset covers 5 asset types, **dampers not among them**. Largest labeled normal-damper source in existence; same Brazilian-hardware caveat as STN PLAD.

### Roboflow Universe (✅ verified via Universe API; annotations included by construction; CC BY 4.0)

| Dataset | Images | Classes | Verified status |
|---|---|---|---|
| [yolov11-tasks/damper-defect-detection](https://universe.roboflow.com/yolov11-tasks/damper-defect-detection) | **2,778** | `Broken_damper`, `Intact_damper` | ✅ v3 published, 2 trained YOLOv11 models attached → exportable now. **Best immediately-downloadable defective-damper source found anywhere.** |
| [wangbo/damper-o5wo3](https://universe.roboflow.com/wangbo/damper-o5wo3) | **998** | `defective`, `none_defective` | ✅ v1 published. Found via Universe API (missed by web search). |
| [samiksha-gadhave/defect-damper](https://universe.roboflow.com/samiksha-gadhave/defect-damper) | 1,015 | `defect location` (1 class) | ✅ v5 published. Defect-spot-style labels (like ATLI's philosophy). |
| [car-damage-ptojm/transmission-line-hrehz](https://universe.roboflow.com/car-damage-ptojm/transmission-line-hrehz) | 829 | `damper_defect`, `damper_good`, `insulator_defect`, `insulator_good`, `spacer_defect`, `spacer_good` | ⚠️ No published version → export may require contacting owner. ATLI-like class structure. |

**Caveats for all Universe sets:** community-uploaded — provenance, UAV-origin, and label quality unverified; may contain recycled public images (CPLID, DVDI, internet scrapes), so the **pHash leakage check against `merged_atli_target` val/test is mandatory** before any merge, and a manual quality pass (staging-project workflow) before trusting labels.

**Revised bottom line:** the academic claim "no public defective-damper data exists" holds for *paper-published* datasets, but Roboflow Universe has ~3,800 immediately-exportable images with binary damper-defect labels (top two sets) — unvetted, but instantly testable as a rebalance source.

---

## 2. How the literature treats damper defects

### Defect taxonomy used across papers
- **Shape-change defects (learnable):** missing/broken hammer head ("missing knob", "damage", shedding), deformation/serious bending, inversion (flipped), intersection (two dampers colliding), dislodgement.
- **Subtle defects (consistently punted or weak):** rust/corrosion levels (AGMNet rust mAP 75.4, "rust-level discrimination weak"; DSA-Net explicitly defers rust + slippage to future work).
- ⚠️ **Action item:** bucket our 113 `Defective_Damper` crops by defect mode. Shape-change defects are tractable; if many are rust-type, that's a scoping caveat for the paper, not a training problem.

### Label-design strategies (the central finding)
Most papers consider *normal + defective sibling detection classes* (our setup) an imbalance trap:

1. **Drop the normal class** — HC-ViT routes ~174k normal-component crops into self-supervised (SC-MAE) pretraining and detects only defect classes. Felect/TLDD also label defects only.
2. **Part decomposition + counting** — Energy Reports 2023 YOLOv5 paper labels `hammer_out` (assembly) + `hammer_in` (each body); defective = IoU-containment rule (2 bodies = normal, ≤1 = defective). Every damper trains both classes → imbalance disappears at the label level. Only covers dislodged-body defects.
3. **Detect-then-classify** — PA-DETR's ablation: putting defect states directly in the detector's class list cost ~3 mAP vs detecting the component and classifying the crop (at 15.8:1 imbalance, rarest class 397 instances → 76.3 AP). Strongest direct evidence relevant to our DD→ND confusion.
4. **Sibling classes CAN work (BC-YOLO)** — damaged damper was their *best* class (AP 92.7) with 636 instances at 5.7:1 imbalance and gross-geometry defect definitions. Their confusion was damper-vs-background (21% of normal dampers missed; 46% background FPs), NOT normal↔damaged (0.01–0.03). Suggests our failure mode is data quantity below threshold, not the label design per se.

### What actually moved damper numbers
- **More defective data / targeted augmentation past parity** — CE-SSD: augmenting the minority defective class beyond the normal class gave monotonic mAP gains (66.2 → 77.6). DSA-Net rebalanced to ~1:1.7 via field capture + augmentation.
- **Resolution preservation / tiling** — STN PLAD: damper AP 0.214 with naive full-image resize → **0.838** with a 4×4-tile small-object detector (+62 pts). Dampers are the smallest class in every dataset (~2.9k px² boxes in 5472-px images); defect cues are a handful of pixels at 640 input.
- **Shape-aware attention** — DSA-Net's strip-shaped attention sized to damper aspect ratio (2.6:1) gave +8.9 mAP over no attention; CA/BiFPN gave BC-YOLO +2.7.
- **Attribute/knowledge heads** — PA-DETR fuses subcomponent-presence scores into the defect classifier; defect classes gained the most (+8.9 to +14.7 AP).
- **NOT used by anyone:** focal loss, class-weighted loss, or resampling for the normal-vs-defective imbalance. (RSIn used focal loss, but for insulator-*type* imbalance.)

### Universal finding
Damper damage is the worst or near-worst class in every multi-class study regardless of architecture — DSA-Net's "Damage" class was lowest-AP for **all 14** detectors tested (0.47–0.64); HC-ViT's damper damage was worst of 6 classes (AP75 37.1) despite a 95.8M-param ViT; TLDD's damper AP topped out at 40 for every model. Our struggle is the field's struggle.

---

## 3. Recommended next steps

1. **Source defective-damper data** (revised — DVDI's repo turned out to be images-only, test split only):
   - **Email the DVDI authors** (Bao et al., Anhui Univ. / CAS Hefei Inst.) for the full labeled 1,500-image set — the paper states it is public, so the repo is likely just incomplete; and/or **email the DSA-Net authors** (Xi'an Polytechnic Univ.) for TLD, which has the richest defect-subtype labels.
   - **Fallback:** hand-label DVDI's 300 downloadable test images via the staging workflow used for `eduardos-annotated-photos` — UAV imagery with damaged dampers present, just unannotated.
   - Either way: pHash-dedupe against all of `merged_atli_target` (train AND val/test), stage for manual review, upload as a tagged batch to train split only (`rebalance/phase1_select.py` pattern).
2. **Audit the 113 defective crops** by defect mode (missing head / bent / rust / displaced) to know what's learnable.
3. **Consider a detect-then-classify condition** in the ablation framework: single `damper` detection class + second-stage crop classifier (PA-DETR evidence).
4. Check DVDI's "damaged" definition (shedding / serious bending) against ATLI's `Defective_Damper` semantics before merging — mismatched defect definitions would add label noise.

## 4. Provenance notes
- Verified via `pdftotext` against `APET2025_T0388.pdf`: BC-YOLO is cited only as related work ([14], generic in-text mentions); **DVDI and STN PLAD images were not used in ATLI** ("~200 images from CPLID and other datasets, remainder from the Internet"). Coincidental overlap with internet-sourced images can't be excluded → pHash guard stays mandatory.
- Full per-paper extraction notes are in the session digest (memory: `damper-defect-literature`); papers in `ATLI/` (git-ignored, copyrighted).
