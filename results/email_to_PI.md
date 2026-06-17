# ATLI defect-detection — results update (for the PI & the lab)

**Attachments to include:** `CURRENT_baseline_Bv11_normalized.png` and
`CURRENT_champion_HROaug_normalized.png` (in `results/figures/confusion_matrices/`).

## Summary (email body)

Using a **data-free recipe** — hi-resolution training (1280 px) + moderate ×3 oversampling of the
rare defect classes + scale-down augmentation, on YOLOv11n with COCO transfer learning — we raised
**5-fold cross-validated mAP@0.5 from 0.760 → 0.802** and **Defective_Damper recall from 0.68 → 0.73**,
while *improving* (not sacrificing) the dominant Normal_Damper class (0.733 → 0.789) and adding no
external data. A key methodological finding is that **the ATLI test set is too small for reliable
per-class evaluation**: the same baseline's Defective_Damper AP swings from **0.61 to 0.88 across CV
folds** (only ~39–52 defective-damper instances per split), so we now report 5-fold CV means rather
than single-split numbers — under which the rare-class AP gain is modest (+0.02, within fold noise)
even though overall mAP, recall, and Normal_Damper improve clearly. **The configurations we judged to
be overfitting** were the **longer-schedule (300-epoch) runs** and the **heavy ×6-oversampling runs**,
both of which memorized the rare classes and *underperformed* the moderate 150-epoch / ×3 setup.
Finally, external community datasets did **not** help the defective classes at any dose (label/domain
mismatch), and several public sets — **CPLID, DVDI, and PTL-AI Furnas — actually leak into the ATLI
test set** via exact-duplicate images, so we added a perceptual-hash leakage gate before any use.

## Numbers (5-fold cross-validation, all ~1,343 images tested across folds)

| metric | baseline (B_v11) | our model (HROaug_v11) | Δ |
|---|---|---|---|
| mAP@0.5 | 0.760 | **0.802** | +0.042 |
| Defective_Damper AP | 0.727 ± 0.089 | 0.745 ± 0.079 | +0.018 (within noise) |
| Defective_Damper recall | 0.680 | **0.728** | +0.048 |
| Normal_Damper AP | 0.733 | **0.789** | +0.056 |

## Confusion-matrix evidence (single-split, attached PNGs)
The attached normalized confusion matrices show the concrete improvement on the damper confusion:
- Defective_Damper misclassified **as** Normal_Damper: **0.21 → 0.15**
- Spurious **Normal_Damper false positives** (background): **0.43 → 0.35**
- (One regression to flag honestly: Normal_Insulators false positives rose 0.33 → 0.44.)

## Which models overfit (detail)
- **300-epoch stage-1 runs**: consistently *worse* than 150-epoch on the rare Defective_Damper class
  (e.g. 0.65 vs 0.71 single-split) — longer training memorizes the ~180 rare-class training instances.
- **×6 oversampling at 640 px**: DD 0.64 vs ×3's 0.73 — over-duplicating rare images without enough
  augmentation diversity causes memorization. (×6 only helped *with* hi-res + scale-aug, and even then
  did not reliably beat ×3 across seeds.)
- Fix: moderate schedule + ×3 oversampling + scale-down augmentation adds diversity that counters the
  memorization, which is why it generalizes better under cross-validation.
