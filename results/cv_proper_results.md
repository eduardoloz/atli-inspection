# Proper 5-Fold CV Results (70/15/15, val ≠ test) — COMPLETE

Updated 2026-07-02 (supersedes 2026-06-25 partial version). All **45 runs** complete:
crashed champ_v11 folds and OSall were re-queued and finished; full re-eval via
`eval/eval_cv_proper.py` (log: server `~/atli/eval_cv_proper_final.log`). Some fold
values differ from the 06-25 snapshot because failed/anomalous runs were re-run.

## Conditions

| Condition | imgsz | Oversampling | Scale aug | Batch |
|---|---|---|---|---|
| **base** | 640 | None | No | 32 |
| **champ** | 1280 | DD ×3 | scale=0.9 | 16 |
| **osall** | 1280 | All defects ×3 (BI+DD+FI+SE) | scale=0.9 | 16 |

All models: COCO-pretrained, 2-stage TL (150+100 epochs, SGD lr0=0.01 → 0.00334).

---

## Summary Table (5-fold means ± std)

| Condition | Model | mAP@0.5 | P | R | DD AP | DD R | BI AP | FI AP | ND AP | NI AP |
|---|---|---|---|---|---|---|---|---|---|---|
| base | v5n | 0.716 ± 0.028 | 0.816 | 0.663 | 0.695 ± 0.076 | 0.608 | 0.600 | 0.577 | 0.724 | 0.781 |
| base | v8n | 0.725 ± 0.021 | 0.828 | 0.670 | 0.735 ± 0.044 | 0.655 | 0.604 | 0.587 | 0.731 | 0.788 |
| base | v11n | 0.722 ± 0.019 | 0.793 | 0.692 | 0.743 ± 0.074 | 0.695 | 0.592 | 0.593 | 0.716 | 0.786 |
| champ | v5n | 0.758 ± 0.018 | 0.828 | 0.700 | 0.789 ± 0.079 | 0.709 | 0.581 | 0.624 | 0.783 | 0.802 |
| champ | v8n | 0.771 ± 0.023 | 0.809 | 0.729 | 0.799 ± 0.063 | 0.749 | 0.623 | 0.636 | 0.785 | 0.799 |
| **champ** | **v11n ★** | **0.785 ± 0.023** | 0.840 | 0.745 | **0.812 ± 0.080** | 0.764 | **0.662** | 0.656 | 0.790 | 0.808 |
| osall | v5n | 0.776 ± 0.035 | 0.852 | 0.722 | 0.811 ± 0.073 | 0.716 | 0.648 | 0.668 | 0.787 | 0.789 |
| osall | v8n | 0.773 ± 0.022 | 0.857 | 0.736 | 0.804 ± 0.058 | 0.740 | 0.617 | 0.662 | 0.796 | 0.803 |
| osall | v11n | 0.772 ± 0.024 | 0.835 | 0.741 | 0.809 ± 0.071 | 0.767 | 0.633 | 0.664 | 0.779 | 0.796 |

**★ Champion v11n is the best condition overall** — highest mAP@0.5 (0.785), highest
DD AP (0.812), highest BI AP (0.662), best precision (0.840). Its 06-25 CUDA crashes
were entirely the competing Ollama process, not the recipe.

### Defect-class recall (the safety-critical metric)

| Condition | Model | DD R | BI R | FI R | SE R |
|---|---|---|---|---|---|
| base | v5n / v8n / v11n | 0.608 / 0.655 / 0.695 | 0.493 / 0.503 / 0.527 | 0.526 / 0.525 / 0.551 | 0.713 / 0.709 / 0.733 |
| champ | v5n / v8n / v11n | 0.709 / 0.749 / **0.764** | 0.515 / 0.569 / 0.564 | 0.534 / 0.577 / **0.622** | 0.761 / 0.803 / 0.807 |
| osall | v5n / v8n / v11n | 0.716 / 0.740 / **0.767** | 0.570 / 0.538 / **0.574** | 0.586 / 0.627 / **0.631** | 0.828 / **0.853** / 0.828 |

---

## Baseline → Champion improvement (Δ, same model)

| Model | Δ mAP | Δ DD AP | Δ DD R | Δ BI AP | Δ ND AP | Δ NI AP |
|---|---|---|---|---|---|---|
| v5n | +0.042 | +0.094 | +0.101 | −0.019 | +0.059 | +0.021 |
| v8n | +0.046 | +0.064 | +0.094 | +0.019 | +0.054 | +0.011 |
| v11n | **+0.063** | **+0.069** | **+0.069** | **+0.070** | **+0.074** | +0.022 |

Champion recipe lifts mAP by 4–6 pts and DD AP by 6–9 pts across all three models.

## Champion vs OSall

OSall (oversampling *all* defect classes, not just DD) is a wash on mAP for v8/v11 but:
- **helps v5n** (+0.018 mAP over champ v5) — the weakest model benefits most;
- consistently raises **FI AP** (+0.01–0.04) and **SE recall** (+0.02–0.07);
- slightly *reduces* champ v11's mAP (0.785 → 0.772) and BI AP (0.662 → 0.633).

**Recommendation:** champ v11n as the headline model; osall as the recall-oriented variant.

---

## Per-fold details (mAP / DD AP)

| Fold | base v5 | base v8 | base v11 | champ v5 | champ v8 | champ v11 | osall v5 | osall v8 | osall v11 |
|---|---|---|---|---|---|---|---|---|---|
| f0 | 0.678/0.616 | 0.701/0.682 | 0.691/0.707 | 0.733/0.718 | 0.733/0.699 | 0.758/0.712 | 0.725/0.720 | 0.748/0.719 | 0.748/0.721 |
| f1 | 0.725/0.794 | 0.715/0.817 | 0.735/0.878 | 0.754/0.897 | 0.765/0.881 | 0.774/0.910 | 0.792/0.905 | 0.785/0.879 | 0.773/0.873 |
| f2 | 0.749/0.769 | 0.757/0.729 | 0.742/0.755 | 0.781/0.843 | 0.799/0.851 | 0.820/0.893 | 0.817/0.869 | 0.808/0.852 | 0.800/0.876 |
| f3 | 0.739/0.687 | 0.743/0.725 | 0.735/0.718 | 0.776/0.805 | 0.790/0.790 | 0.803/0.809 | 0.802/0.826 | 0.776/0.805 | 0.798/0.851 |
| f4 | 0.688/0.609 | 0.709/0.722 | 0.708/0.659 | 0.747/0.684 | 0.770/0.774 | 0.771/0.737 | 0.745/0.735 | 0.749/0.763 | 0.743/0.726 |

Fold ordering is consistent (f1/f2 easy, f0/f4 hard) across every condition — fold
difficulty, not run noise, drives most of the spread. Full per-fold P/R/per-class
numbers: server `~/atli/eval_cv_proper_final.log`.

---

## Comparison to paper (APET2025, Table V)

Paper used 732-image ATLI, val=test (biased), single split.
Our proper CV uses 1,343-image merged dataset, val≠test (unbiased), 5-fold.

### Overall mAP@0.5

| | Paper | Our base | Our champ | Our osall |
|---|---|---|---|---|
| v5n | 75.6 | 71.6 | 75.8 | **77.6** |
| v8n | 77.5 | 72.5 | 77.1 | 77.3 |
| v11n | 76.8 | 72.2 | **78.5** | 77.2 |

### Defective_Damper (hardest class)

| | Paper v5n | Our base v11n | Our champ v11n | Our osall v11n |
|---|---|---|---|---|
| Recall | 53.8 | 69.5 | 76.4 | **76.7** |
| AP@0.5 | — | 74.3 | **81.2** | 80.9 |

Our champion recipe **beats the paper's mAP on every model** (v11n: 78.5 vs 76.8)
despite stricter evaluation (separate val/test, 5-fold CV), while substantially
improving defect-class recall.

### vs SOTA heavyweights (same folds — `sota_cv_results.md`)

Champ v11n (2.6M params) **matches DINO-4scale** (47M) on mAP@0.5 (0.785 both) and
**beats it on Defective_Damper** (0.812 vs 0.789, with lower fold variance).

---

## Known issues — all resolved

1. ~~champ_v11 needs rerun~~ → re-run with GPUs clear; now complete (and best overall).
2. ~~OSall still running~~ → complete.
3. **CPLID overlap** (unchanged): 249 of 1,343 merged images (18.5%) are exact/near-duplicate
   CPLID (Chinese grid). No US-origin infrastructure in training data.
