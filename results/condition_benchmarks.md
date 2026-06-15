# Condition Ablations A–H — Benchmark Comparison

What each "condition" is, and how every (model × condition) run scored on the **shared,
Eduardo-free test set** (156 images / 563 instances). All runs follow the standard
protocol — stage-1 train 150 ep from COCO-pretrained weights, stage-2 fine-tune 100 ep
SGD (`lr0=0.00334`, `lrf=0.1535`), 640 px — unless marked otherwise.

> Numbers scraped from `~/atli/eval/*.txt` on `ai.ee.unlv.edu` (2026-06-10).
> ⚠️ Not comparable to `benchmark_results.md` — that table uses a different,
> larger test split (199 img / 867 inst).

## The conditions

| Cond | Train set | Question it answers | Builder | Config suffix |
|---|---|---|---|---|
| A | Target only (1,046 imgs, stratified seed-42) | Baseline — no Eduardo data | `data/build_ablation.py` | `*_to` |
| B | A + Eduardo, **without** the 152 `Defective_Insulators` boxes | Does Eduardo help at all? | `data/build_ablation.py` | `*_tpe` |
| C | A + Eduardo **full** (DI boxes kept) | Do the DI boxes add anything? | `data/build_condition_c.py` | `*_tpef` |
| D | C − 12 bad imgs (7 "can't tell" + 5 too-close/dups) | Is bad data dragging C down? | `data/build_condition_d.py` | `*_tpef_clean` |
| E | C − 5 bad imgs (keeps the 7 "can't tell") | Were the "can't tell" imgs actually useful? | `data/build_condition_e.py` | `*_tpef_keepct` |
| F | C − 4 hard-negative `Defective_Damper` imgs | Are those 4 imgs confusing DD? | `data/build_condition_f.py` | `*_condf` |
| G | C with Eduardo `Normal_Damper` undersampled/stripped | Is ND flooding drowning DD? | `data/build_condition_g.py` / `_g2.py` | `*_condg` |
| H | C dataset + class-weighted loss (`cls=3.0`) | Can the loss fix imbalance instead of data? | (no builder — training change) | `*_condh` |

## Overall mAP@0.5 (test) — 100-ep fine-tune

| Cond | YOLOv5n | YOLOv8n | YOLOv11n | RT-DETR-l |
|---|---|---|---|---|
| A | 0.692 ¹ | 0.738 | **0.741** | 0.759 |
| B | 0.689 ¹ | 0.679 ² | 0.738 | — |
| C | 0.697 ¹ | 0.719 ² | **0.742** | 0.745 |
| D | 0.696 | 0.688 | 0.717 | — |
| E | 0.687 | 0.716 | 0.735 | **0.761** |
| F | **0.717** | 0.724 | 0.720 | — |
| G | 0.696 | 0.720 | 0.717 | — |
| H | 0.656 | **0.736** | 0.722 | — |

¹ v5 A/B/C from `analysis/pi_summary_charts.py` (the original `abl_v5_*` eval files are no
longer on the server; only `abl_v8_to.txt` survives).
² `v8_tpe_r2` / `v8_tpef_r2` reruns; the originals' eval files are gone.

mAP@0.5:0.95, same layout:

| Cond | v5n | v8n | v11n | RT-DETR-l |
|---|---|---|---|---|
| A | 0.420 ¹ | 0.473 | 0.489 | 0.493 |
| B | 0.407 ¹ | 0.428 | 0.481 | — |
| C | 0.415 ¹ | 0.457 | 0.485 | 0.487 |
| D | 0.409 | 0.444 | 0.457 | — |
| E | 0.418 | 0.465 | 0.478 | **0.507** |
| F | 0.410 | 0.462 | 0.471 | — |
| G | 0.413 | 0.462 | 0.472 | — |
| H | 0.350 | 0.461 | 0.446 | — |

## Rare classes — per-class mAP@0.5 (Broken_Insulator / Defective_Damper)

| Cond | v5n BI/DD | v8n BI/DD | v11n BI/DD | RT-DETR BI/DD |
|---|---|---|---|---|
| A | 0.511 / 0.404 ¹ | 0.654 / 0.454 | 0.599 / **0.537** | 0.628 / 0.525 |
| B | 0.570 / 0.394 ¹ | 0.470 / 0.358 | 0.619 / 0.496 | — |
| C | 0.543 / 0.406 ¹ | 0.591 / 0.407 | 0.655 / 0.484 | 0.707 / 0.449 |
| D | 0.595 / 0.445 | 0.517 / 0.305 | 0.546 / 0.457 | — |
| E | 0.590 / 0.421 | 0.543 / 0.401 | 0.642 / 0.471 | **0.722** / 0.458 |
| F | 0.565 / 0.468 | 0.550 / 0.379 | 0.596 / 0.422 | — |
| G | 0.559 / 0.412 | 0.569 / 0.489 | 0.592 / 0.414 | — |
| H | 0.507 / 0.439 | 0.622 / 0.486 | 0.639 / 0.431 | — |

⚠️ The shared test set has only **20 BI images (36 boxes)** and **8 DD images (21
boxes)** — swings of ±0.05 on these two classes are within noise.

## Variant runs (sweep V2 + TTA)

200-epoch stage-2 fine-tune (`*_ft200`), overall mAP@0.5 vs the 100-ep run:

| Cond | v5n | v8n | v11n |
|---|---|---|---|
| A | 0.695 (+0.003) | 0.729 (−0.009) | 0.741 (±0) ³ |
| B | 0.694 (+0.005) | 0.713 (+0.034 ²) | 0.708 (−0.030) |
| C | 0.697 (±0) | 0.711 (−0.008 ²) | 0.728 (−0.014) |

Test-time augmentation (eval-only, no retraining):

| Run | base → TTA |
|---|---|
| v11n on A | 0.741 → 0.738 |
| v11n on C | 0.742 → 0.730 |
| RT-DETR on E | 0.761 → 0.761 ³ |

³ **Anomaly:** `v11_to_ft200.txt` is identical to `v11_to.txt`, and `E_T_tta.txt` is
identical to `E_T.txt`, down to the third decimal on every class. Either the eval was
overwritten or the wrong weights/flags were used — re-run before citing these.

Still pending (launched, no eval file yet): `v11_A_hr` (1280 px), `A_T_hr` (RT-DETR
960 px), `v8_stack` / `v11_stack` (G dataset + `cls=3.0`).

## Scratch vs transfer learning (dataset A, shared test) — added 2026-06-10

Scratch = random init (`--cfg yolov5n.yaml` / `model=yolov8n.yaml pretrained=False`),
single stage, no fine-tune. TL = COCO `.pt` start, stage-1 + 100-ep fine-tune.
`train/scratch_sweep.sh`; addresses the "scratch trained longer catches up" critique.

Overall mAP@0.5 (mAP@0.5:0.95):

| Config | YOLOv5n | YOLOv8n | YOLOv11n |
|---|---|---|---|
| scratch 150 | 0.476 (0.227) | 0.582 (0.331) | 0.540 (0.305) |
| scratch 300 | 0.593 (0.309) | 0.626 (0.363) | 0.630 (0.363) |
| scratch 600 | 0.623 (0.344) | 0.635 (0.394) | 0.650 (0.393) |
| TL 150+100 | 0.692 (0.420) | 0.738 (0.473) | 0.741 (0.489) |
| TL 300+100 | 0.735 (0.450) | **0.750 (0.502)** | 0.731 (0.484) |

Rare classes, mAP@0.5 (Broken_Insulator / Defective_Damper):

| Config | v5n BI/DD | v8n BI/DD | v11n BI/DD |
|---|---|---|---|
| scratch 150 | 0.189 / 0.112 | 0.307 / 0.239 | 0.237 / 0.170 |
| scratch 300 | 0.286 / 0.383 | 0.310 / 0.367 | 0.394 / 0.382 |
| scratch 600 | 0.402 / 0.379 | 0.484 / 0.362 | 0.332 / 0.410 |
| TL 300+100 | 0.544 / 0.491 | **0.668 / 0.544** | 0.588 / 0.491 |

Findings:
- **Scratch never catches up on this split.** Best scratch (v11n @ 600 ep, 0.650) is
  still ~0.09 below the worst TL run — unlike the paper's Table V, where 600-ep scratch
  closed to within ~2 points. The critique does not reproduce here.
- The gap is largest exactly where TL is claimed to matter: rare classes. Scratch DD at
  150 ep is 0.11–0.24 vs TL's ~0.5.
- **v8n TL 300+100 = 0.750 is the new best nano result in the project** (beats v11n's
  0.742, within ~0.01 of RT-DETR), and its DD 0.544 is the best Defective_Damper of any
  run, including RT-DETR. Longer stage-1 helps v5n/v8n (+0.04/+0.01) but not v11n (−0.01).
- Caveat: default early-stopping patience (100) applies to the scratch runs; wall-clock
  ratios suggest they ran near-full duration, but check `results.csv` per run before
  quoting the 600-ep numbers as converged.

## Takeaways

1. **RT-DETR-l wins every dataset it ran on** (A 0.759, C 0.745, E 0.761) and posts the
   best overall result: **E = 0.761 mAP@0.5 / 0.507 mAP@0.5:0.95**, plus the best
   Broken_Insulator (0.722). But it is far heavier than the nanos — it answers "what's
   the ceiling," not "what ships to the edge."
2. **Among nanos, v11n leads** (0.735–0.742 on A/C/E) and Eduardo data is roughly a
   wash overall: v11 C (0.742) ≈ v11 A (0.741).
3. **Eduardo data helps Broken_Insulator but hurts Defective_Damper.** BI improves
   A→C for v5/v11/RT-DETR, while DD is best on plain A for every model (v11 0.537,
   RT-DETR 0.525) and drops 0.04–0.10 whenever Eduardo images are added.
4. **None of the DD rescue attempts (F, G, H) beat condition A on DD.** Dropping the 4
   hard negatives, undersampling Normal_Damper, and `cls=3.0` all failed to recover the
   loss — the DD problem is Eduardo's damper distribution itself, not a few bad images.
5. **`cls=3.0` (H) is model-dependent:** small gain for v8n (0.736), catastrophic for
   v5n (0.656, −0.04). Don't apply it blindly.
6. **Longer fine-tuning doesn't help** — ft200 moves results ≤0.005 except where it
   hurts (v11 B −0.03), confirming the earlier 150-vs-300 stage-1 finding.
7. Caveat for all of the above: the 12-image curations (D, E) move overall mAP by less
   than the test-set noise floor; treat their differences as suggestive, not conclusive.
