# Dataset & Technique Leads (internet scour, 2026-06-15)

From 4 parallel research agents. **Nothing merged.** Downloadable candidates must pass the pHash
leakage gate (`rebalance/vet_universe_class.py`) vs ATLI source+target before any use — several
"insulator-defect" sets are CPLID re-uploads that leak into our test set (proven for giomartins/
uttam/trash-detection). See [[universe-insulator-birdnest-vetting]], [[damper-defect-literature]].

## Insulator datasets (NEW)
| Dataset | URL | License | imgs | classes | annot | access | CPLID risk |
|---|---|---|---|---|---|---|---|
| **IDID (EPRI)** ★ | ieee-dataport.org/competitions/insulator-defect-detection | free/req | 1,596 (7.5k box) | broken/flashed/good/insulator | Y bbox | email request | **low** (real EPRI) |
| **Light-ZhangTao** ★ | github.com/Light-ZhangTao/Insulator-Defect-Detection | CC BY 4.0 | 1,600 | insulator/pollution-flashover/broken | Y VOC/YOLO | direct DL | low-med |
| Kaggle Insulator-DET | kaggle.com/datasets/mazilishanglx/insulator-det | Kaggle | 2,150 | 9 (glass/polymer/broken-disc/flashover/snow…) | Y YOLO | direct DL | **med** (de-dupe!) |
| TLID-Dataset | github.com/Caughyhzd/TLID-Dataset | request | 712 | insulator defect (3 mat × 7 bg) | Y bbox | email | low |
| Insulator Mechanical Damage | ieee-dataport.org/documents/insulator-mechanical-damage-dataset | subscription | 8,886 | normal/bird-pecking/cracking/missing-cap | Y bbox | paid | low |

## Birdnest-on-tower datasets (NEW)
| Dataset | URL | License | imgs | classes | tower-context | access |
|---|---|---|---|---|---|---|
| **PTL-AI Furnas** ★ | github.com/freds0/PTL-AI_Furnas_Dataset | GPL-3.0 | 6,295 (17.8k box) | baliser/**bird nest**/insulator/spacer/stockbridge | Y (Brazil UAV) | GitHub+Drive |
| **niaochao** ★ | universe.roboflow.com/zhang-kpqhy/niaochao | CC BY 4.0 | 1,000 | nest | Y (verified: nest in lattice pylon) | RF |
| Power Transmission Tower (fyp) | universe.roboflow.com/fyp-workspace-wxnaw/power-transmission-tower | CC BY 4.0 | 2,000 | Bird Nest/Insulator/Kite/Balloon/… +6 YOLOv8 models | Y (mixed views) | RF |
| assetsdetection/power | universe.roboflow.com/assetsdetection/power-pldef | **none stated** | 9,400 | 38 incl Bird Nest + insulator/damper defects | Y (UAV) | RF — check license |
| RailFOD23 | figshare 10.6084/m9.figshare.24180738 | CC BY 4.0 | 14,615 | plastic-bag/bird-nest/balloon | railroad (not pylon); **many synthetic nests** | Figshare |

## Defective-damper datasets — NONE new+public (confirms native-aug is the path)
| Lead | status | note |
|---|---|---|
| **CDTLD** (PLD-DETR, Electronics 2025) | DOI **broken**, unverified | explicit "damaged vibration damper" class; email author for correct IEEE DataPort DOI |
| **DCP-YOLOv8** (State Grid Sichuan, Sensors 2024) | proprietary | richest taxonomy: damper shifting/deformation/breakage/corrosion — **top author-contact target** |
| **DSA-Net TLD** (IEEE TIM 2023) | unconfirmed | damper cross/displacement/corrosion; author-contact lead |
| Roboflow sabawoonwali / dbis2022 / rawabi | open CC BY | damper as **normal** only, no defect labels |

## Techniques to beat the paper (ranked, Ultralytics-ready)
1. **SAHI sliced fine-tune + inference** — magnifies the *defect* (unlike imgsz1280); +12–14 AP in lit. `pip install sahi`, slice dataset → train → `get_sliced_prediction`. **Top unbuilt pick.**
2. **P2 detection head** — `yolov8-p2.yaml` (running: P2_v8/P2OS_v8). RSP-YOLOv11n analog +2.5 mAP on UAV insulator defects.
3. **Copy-paste rare-class** — `copy_paste=0.4 copy_paste_mode=mixup` in fine-tune (running: CP_v11/CP_v8). +3.6 AP on rare LVIS; doesn't inflate ND.
4. **Loss reweight** — cheap `cls=0.7–1.0` (in CP runs); VFL/Slide-loss swap for more.
5. **close_mosaic=10 + scale jitter** — validated: **OSaug (scale=0.9) already won, DD 0.749** ✓.

## ⛔ PTL-AI Furnas — REJECTED (2026-06-16): ATLI was partly BUILT FROM it
pHash + **exact filename correspondence** (Furnas `LTADRVDP.ts-3101#032.jpg` ↔ ATLI test
`LTADRVDP_ts-3101-032_jpg.rf.*`): 32 exact dups in ATLI val/test, 79 in train, 137 in source.
So Furnas is the **2nd public dataset (after DVDI) that leaks into the ATLI benchmark** — using it
would train on the test set. Also its `stockbridge_nok` (defective damper) class has **0 instances**.
⇒ ATLI's "internet" provenance = CPLID + DVDI + Furnas recycled → a reproducibility hazard worth a
paper sentence. Artifacts: `~/atli/datasets/classvet3/`.

## Vetted 2026-06-15 (pHash vs ATLI, server `~/atli/datasets/classvet2/`)
- **niaochao** (birdnest, 1,000 imgs, tower nests): ✅ **CLEAN** — 0 leak at any threshold (nearest d=14).
  Genuine power-tower birdnest imagery → safe Birdnest source if/when that class is targeted.
- **Light-ZhangTao = CID** (4,719 imgs): ✅ clean of ATLI, BUT it's **catenary/railway** insulator,
  anomaly-style (defect-free train + categorized defect tests, test_sim synthetic) → domain-shifted
  from ATLI aerial TL; low direct value. (GitHub repo ships no images; agent pulled them from Drive.)

## Actionable (no merges)
- **Vet next** (downloadable, pHash vs ATLI): Light-ZhangTao + Kaggle Insulator-DET (insulator), PTL-AI Furnas + niaochao (birdnest). Furnas also has dampers (normal) → ND hard negatives.
- **Email leads** (for the user): EPRI/IDID (insulator, free), CDTLD/DCP-YOLOv8/DSA-Net authors (damper defects).
- **Build next**: SAHI sliced pipeline (the one unbuilt top technique).
