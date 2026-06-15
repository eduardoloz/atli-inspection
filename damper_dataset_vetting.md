# Universe Damper-Dataset Vetting — Results

**Date:** 2026-06-11 · **Script:** `rebalance/vet_universe_dampers.py` (read-only; ran on `ai.ee.unlv.edu`, data under `~/atli/datasets/vetting/`)
**Goal:** vet the Roboflow Universe damper-defect datasets (+ DVDI's 300 test images) as sources of **distant/UAV-style `Defective_Damper`** training images for `merged_atli_target` (v4, 772/137/137).
**Method:** pHash (64-bit, near-dup = Hamming ≤ 8) leakage check vs target val/test and train; cross-set duplicate matrix; normalized-bbox-area analysis of defective boxes; 40 most-zoomed-out defective images per set staged with boxes drawn for manual review.

---

## ⚠️ Headline finding: DVDI is already inside ATLI — do NOT harvest it

Of DVDI's 300 downloadable test images, **40 are near-duplicates of `merged_atli_target` VAL/TEST images** (37 at Hamming distance 0 = pixel-identical) and **109 more duplicate the train split**. Half of DVDI is already in ATLI.

Implications:
- ATLI's "remainder from the Internet" images evidently include recycled DVDI images. This is now a **known provenance fact** about the benchmark.
- The earlier fallback plan ("hand-label DVDI's 300 test images and add to train") is **cancelled** — it would have injected 40 evaluation images into training and silently invalidated every mAP number after the merge. This is exactly what the pHash gate is for.
- Any other "internet-sourced" public set must be assumed contaminated until checked.

## Verdicts per candidate set

| Set | Leak vs val/test | Dups vs train | Intra-set dups | Defective boxes (imgs) | Distance profile of defective boxes | Verdict |
|---|---|---|---|---|---|---|
| **yolov11-tasks/damper-defect-detection** (2,778 imgs, `Broken_damper`/`Intact_damper`) | **4 images** — excludable (see list below) | 29 | 680 exact (set contains `aug_*` augmented copies) | 3,306 (1,681 imgs) | Median box = **0.97%** of image; 1,707 boxes <1% → predominantly distant UAV shots (`DJI_*` filenames) | ✅ **Usable after exclusions + dedupe.** Largest distant-defective source. |
| **wangbo/damper-o5wo3** (998 imgs, `defective`/`none_defective`) | **0** | **0** | 62 | 931 (829 imgs) | Median 1.4%; 389 boxes <1%, 51 <0.2% | ✅ **Cleanest set** — zero overlap with ATLI anywhere. |
| **samiksha-gadhave/defect-damper** (2,439 imgs, `defect location`) | **0** | **0** | 1,399 exact (~57% redundancy) | 2,560 (2,433 imgs) | Median **0.023%** — these are defect-**spot** labels (tiny boxes on the defect, not the damper) | ⚠️ Different label semantics (spot vs whole-component). Visual check required to confirm subject matter; usable only with a deliberate taxonomy decision. |
| **DVDI test/** (300 imgs, unlabeled) | **40 images** | 109 | 16 | n/a (no labels) | n/a | ❌ **Rejected — already in ATLI** (see headline). |

**Cross-set overlaps: all zero** (yolov11tasks ∩ wangbo ∩ samiksha ∩ dvdi = 0 pairwise) — the three Universe sets are independent sources, not copies of each other or of DVDI.

### Exclusion list for yolov11-tasks (val/test leakage — drop before any merge)
```
288-3_DJI_0594_jpg.rf.3f703e52008b7947ff7a4c5de497f5a2.jpg   (d=2)
257-1_DJI_0095_jpg.rf.f8f5b9ed81928d06c8a7f4ce8c741ac3.jpg   (d=0)
aug_2_0_257-1_DJI_0095_jpg.rf.97fa50926b6e4ca6216c900d981895ad.jpg (d=0, aug variant)
269-1_DJI_0194_jpg.rf.eadb26ff108c2f4a99c98c89d055b34f.jpg   (d=0)
```
(Plus the 29 train-dups, which add nothing — the full ID lists are derivable by re-running the script; hashes cached on the server.)

Note the matched filenames are identical on both sides (`257-1_DJI_0095` etc.) — this Universe set and ATLI drew from a common public source, confirming the recycled-imagery pattern.

---

## [You] Manual quality pass — where to look

**Local (already on your Mac, 18 MB):** `datasets/vetting_review/review/<set>/` — the 40 most zoomed-out defective images per set, **boxes drawn on** (red = defective class, green = other). Filenames sort by box area, smallest first (`00_a0.0012_*.jpg` = most distant). Open the folder in Finder and Quick-Look through:
- `datasets/vetting_review/review/u_yolov11tasks/`
- `datasets/vetting_review/review/u_wangbo/`
- `datasets/vetting_review/review/u_samiksha/`

What to judge: (1) are these genuinely UAV/aerial transmission-line shots; (2) do the red boxes mark real damper defects (vs mislabeled normals); (3) does the defect definition match ATLI's `Defective_Damper`; (4) for samiksha — is the subject even dampers, and can spot-labels be widened to whole-damper boxes?

**Online (browse the full sets):**
- https://universe.roboflow.com/yolov11-tasks/damper-defect-detection
- https://universe.roboflow.com/wangbo/damper-o5wo3
- https://universe.roboflow.com/samiksha-gadhave/defect-damper

**Full machine-readable report:** `datasets/vetting_review/vetting_report.json` (local copy) or `~/atli/datasets/vetting/vetting_report.json` (server).

## Recommended path (pending your visual pass)

1. **[You]** Quick-Look the three review folders (~10 min).
2. **[Claude]** If yolov11tasks + wangbo pass: phase1-style selection script — dedupe (intra + vs ATLI train), drop the 4 leaked images, map `Broken_damper`/`defective` → `Defective_Damper` and `Intact_damper`/`none_defective` → `Normal_Damper`, select a distance band like the insulator rebalance, stage to a Roboflow staging project for your annotation review.
3. **[Together]** Decide samiksha's fate (spot labels) and the upload batch size; upload goes to **train split only**, tagged (e.g. `universe_damper_v1`), reversible — with your explicit go-ahead, per the usual mutation workflow.

Headroom: yolov11tasks (~1,650 usable defective imgs after dedupe/exclusions) + wangbo (829) ≈ **2,400+ candidate defective-damper images, mostly distant** — vs the current 113 instances. Even a curated 10% subset would roughly triple the class.

---

## Citations & original links (for the paper / lab notes)

**Roboflow Universe datasets (CC BY 4.0; accessed 2026-06-11):**
```bibtex
@misc{damper-defect-detection_dataset,
  title = {Damper Defect Detection Dataset},
  author = {{YOLOv11 tasks (Roboflow Universe workspace)}},
  howpublished = {\url{https://universe.roboflow.com/yolov11-tasks/damper-defect-detection}},
  note = {Roboflow Universe, v3. Accessed 2026-06-11}
}
@misc{damper_o5wo3_dataset,
  title = {damper Dataset},
  author = {{wangbo (Roboflow Universe workspace)}},
  howpublished = {\url{https://universe.roboflow.com/wangbo/damper-o5wo3}},
  note = {Roboflow Universe, v1. Accessed 2026-06-11}
}
@misc{defect-damper_dataset,
  title = {defect damper Dataset},
  author = {{Samiksha Gadhave (Roboflow Universe workspace)}},
  howpublished = {\url{https://universe.roboflow.com/samiksha-gadhave/defect-damper}},
  note = {Roboflow Universe, v5. Accessed 2026-06-11}
}
```

**Papers referenced by this vetting effort:**
```bibtex
@article{bao2022bcyolo,   % DVDI dataset — found to overlap ATLI
  title   = {A Defect Detection Method Based on BC-YOLO for Transmission Line Components in UAV Remote Sensing Images},
  author  = {Bao, Wenxia and Du, Xiang and Wang, Nian and Yuan, Mu and Yang, Xianjun},
  journal = {Remote Sensing}, volume = {14}, number = {20}, pages = {5176}, year = {2022},
  doi     = {10.3390/rs14205176}
}
% repo: https://github.com/Emp-8/DVDI  (test images only, no labels)

@article{zhang2021pmayolo,   % DAVD dataset — private; closest taxonomy (rusty/defective/normal); author-email target
  title   = {Detection of Abnormal Vibration Dampers on Transmission Lines in UAV Remote Sensing Images with PMA-YOLO},
  author  = {Zhang, Wenjie and others},
  journal = {Remote Sensing}, volume = {13}, number = {20}, pages = {4134}, year = {2021},
  doi     = {10.3390/rs13204134}
}
% page: https://www.mdpi.com/2072-4292/13/20/4134

@inproceedings{vieiraesilva2021stn,   % STN PLAD — public, normal dampers only
  title     = {STN PLAD: A Dataset for Multi-Size Power Line Assets Detection in High-Resolution UAV Images},
  author    = {Vieira-e-Silva, Andr{\'e} Luiz Buarque and others},
  booktitle = {2021 34th SIBGRAPI Conference on Graphics, Patterns and Images},
  pages     = {215--222}, year = {2021}, doi = {10.1109/SIBGRAPI54419.2021.00037}
}
% repo: https://github.com/andreluizbvs/PLAD  (COCO JSON labels included)

@article{vieiraesilva2023insplad,   % InsPLAD — public, ~6.7k labeled dampers, no damper defect labels
  title   = {InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images},
  author  = {Vieira-e-Silva, Andr{\'e} Luiz Buarque and others},
  journal = {International Journal of Remote Sensing}, year = {2023},
  note    = {arXiv:2311.01619}
}
% repo: https://github.com/andreluizbvs/InsPLAD  (CC BY-NC 3.0)
```

See `damper_lit_review.md` for the full literature/dataset survey this vetting follows from.
