# Universe Damper Benchmark — Results (updated 2026-06-12, evening)

20 runs on `ai.ee.unlv.edu`: `{B, Ucap1, Ucap2, Uw, Uto, Ufull} × {v8n, v11n} × {150+100, 300+100*}`,
recipe = `run_config_v3.sh` (s1 COCO-init SGD lr0=0.01 → s2 fine-tune lr0=0.00334 → test eval), imgsz 640, batch 32.
Methods summary: [README.md](README.md). Raw: `test_summary.csv`, `epoch_map.csv`.

**Conditions** (all share byte-identical val/test = 203/199 imgs, 48/39 DefDamper instances; universe data → train only):

| cond | universe data added to train | train DD instances | ratio universe:native |
|---|---|---|---|
| B | none (baseline) | 177 | — |
| Ucap1 | random cap, both sources (123 imgs) | 354 | 1:1 |
| Ucap2 | random cap, both sources (241 imgs) | 531 | 2:1 |
| Uw | all of wangbo (819 imgs) | 922 | 4.2:1 |
| Uto | everything (2,175 imgs) | 2,244 | 11.7:1 |
| Ufull | everything, fully re-stratified | — | (in-domain control; own test set*) |

## Test results — overall P / R / mAP and Defective_Damper P / R / AP50

| run | P | R | mAP50 | mAP50-95 | DD_P | DD_R | **DD_AP50** | ND_AP50 |
|---|---|---|---|---|---|---|---|---|
| B_v8_150p100 | 0.825 | 0.630 | **0.701** | 0.408 | 0.826 | 0.564 | 0.682 | 0.759 |
| B_v11_150p100 | 0.756 | 0.652 | 0.691 | 0.407 | 0.804 | 0.633 | **0.708** | 0.744 |
| B_v8_300p100 | 0.795 | 0.630 | 0.686 | 0.403 | 0.809 | 0.564 | 0.649 | 0.793 |
| B_v11_300p100 | 0.781 | 0.641 | 0.686 | 0.409 | 0.864 | 0.590 | 0.646 | 0.735 |
| Ucap1_v8_150p100 | 0.802 | 0.617 | 0.671 | 0.391 | 0.737 | 0.538 | 0.611 | 0.780 |
| Ucap1_v11_150p100 | 0.770 | 0.623 | 0.676 | 0.387 | 0.874 | 0.513 | 0.576 | 0.792 |
| Ucap2_v8_150p100 | 0.790 | 0.610 | 0.678 | 0.388 | 0.685 | 0.513 | 0.575 | 0.793 |
| Ucap2_v11_150p100 | 0.746 | 0.674 | 0.687 | 0.386 | 0.626 | 0.590 | 0.575 | 0.785 |
| Uw_v8_150p100 | 0.735 | 0.652 | 0.688 | 0.397 | 0.818 | 0.590 | 0.666 | 0.793 |
| Uw_v11_150p100 | 0.729 | 0.631 | 0.684 | 0.382 | 0.807 | 0.487 | 0.591 | 0.763 |
| Uw_v8_300p100 | 0.793 | 0.666 | 0.703 | 0.408 | 0.888 | 0.613 | 0.681 | 0.781 |
| Uw_v11_300p100 | 0.825 | 0.644 | **0.706** | 0.402 | 0.953 | 0.641 | 0.687 | 0.758 |
| Uto_v8_150p100 | 0.770 | 0.629 | 0.687 | 0.391 | 0.756 | 0.556 | 0.608 | 0.790 |
| Uto_v11_150p100 | 0.751 | 0.613 | 0.670 | 0.378 | 0.797 | 0.605 | 0.647 | 0.750 |
| Uto_v8_300p100 | 0.764 | 0.631 | 0.669 | 0.393 | 0.719 | 0.590 | 0.597 | 0.805 |
| Uto_v11_300p100 | 0.773 | 0.602 | 0.667 | 0.376 | 0.814 | 0.561 | 0.617 | 0.791 |
| Ufull_v8_150p100* | 0.839 | 0.705 | 0.755 | 0.461 | 0.907 | 0.863 | 0.903 | 0.928 |
| Ufull_v11_150p100* | 0.893 | 0.719 | 0.802 | 0.485 | 0.920 | 0.847 | 0.910 | 0.927 |
| Ufull_v8_300p100* | 0.871 | 0.741 | 0.789 | 0.483 | 0.891 | 0.873 | 0.914 | 0.926 |
| Ufull_v11_300p100* | 0.870 | 0.739 | 0.786 | 0.480 | 0.890 | 0.873 | 0.907 | 0.928 |

\* Ufull scored on its own universe-mixed test set (in-domain control) — not comparable to the other rows.

## Findings

1. **Ratio capping did not help.** Even the smallest dose (Ucap1, 1:1, 123 images) cut DefDamper AP50 to
   0.576–0.611 vs 0.646–0.708 baseline. The dose-response is flat-bad across 1:1 → 2:1 → 4.2:1 → 11.7:1
   (≈0.55–0.67 throughout): the problem is **content** (label convention / defect-definition mismatch),
   not quantity. This rules out swamping as the primary mechanism.
2. **The damage signature is recall, not precision.** DefDamper precision stays high under universe data
   (0.63–0.87) while recall falls (baseline 0.56–0.63 → capped/Uw/Uto mostly 0.49–0.61): universe-trained
   models *miss* native defects rather than hallucinate them — the "defective" concept is dragged toward
   the universe defect style.
3. **In-domain control (Ufull ≈ 0.91 DD)** confirms the labels are learnable; transfer is what fails.
4. **One condition recovered to baseline parity: Uw (all wangbo) + 300+100.** Final DD 0.681/0.687
   (vs B mean 0.671, B best 0.708) with the two highest overall mAP50 on the real benchmark
   (0.703/0.706) and very high DD precision (0.89/0.95). So the clean, single-source set plus a longer
   schedule neutralizes the damage — but still does not beat the best baseline. Elsewhere
   150+100 ≥ 300+100 for DefDamper (B: 0.682/0.708 vs 0.649/0.646).
5. **Statistical caveat for the journal:** the test split has only 39 DefDamper instances; single-run
   differences of ±0.05 AP are within noise (B itself spans 0.646–0.708). The robust pattern: 10 of 12
   universe-augmented runs fall below the baseline mean; the two at/above parity (Uw 300+100) are within
   noise of it.
6. **Normal_Damper benefits mildly** from universe data (0.75–0.81 vs 0.74–0.79 baseline).

**Implication:** raw community data — at any dose — does not transfer for the defective class. The
annotation-corrected subset (box-fixing pass in the `universe-damper-staging` Roboflow project,
2,175 imgs uploaded, 1,430 tagged `has_defective`) is the critical path; alternatives are
pretrain(universe)→finetune(native-only) and author-emails for in-domain data (DAVD, full DVDI).
