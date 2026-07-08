# Thesis-Enhancement Thread — Final Synthesis (checkpoint, 2026-07-08)

**Thread goal:** mine the group's wildfire-edge MS thesis (the thesis author, UNLV MS-EE May 2025) for levers applicable to our ATLI champion (YOLOv11n, COCO-init 2-stage TL, imgsz 1280, ×3 DD oversample, scale=0.9; clean-test 0.784 ± 0.011 mAP).
**Deliverables on this branch:** `thesis_digest.md` (page-referenced digest) → `thesis_enhancements_analysis.md` (T1–T8 transferability) → `round2_results_and_synthesis.md` (pilots + srcTL root-cause) → this synthesis.

## Lever scoreboard (final)

| # | thesis lever | verdict for ATLI | evidence (one line) |
|---|---|---|---|
| T1 | in-domain source pretraining (thesis: +14.4 mAP over COCO-init) | **DEAD** with available sources | srcTL pilot −7.5 mAP / −20 DD AP vs COCO-init control; root-caused as genuine negative transfer (below) |
| T2 | low stage-2 LR (lr0 = 1e-4, thesis Table 4.1) | **WASH — closed** | 3 seeds @768: mAP 0.769 ± 0.008 vs champion 0.766 ± 0.009; DD AP 0.659 ± 0.022 vs 0.664 ± 0.042; DD recall 0.623 vs 0.667. Keep lr0 = 0.00334 |
| T3 | zero frozen layers | **CONFIRMED** — already our practice | thesis Table 4.2 (freeze costs 7–15 mAP) + our frz10/frz20 (collapse at frz20); two independent within-group studies |
| T4 | no cascaded TL; merge sources | adopted as negative guidance | thesis §4.1.3; saved us a sweep we'd otherwise have run |
| T5 | lightweight backbone swaps | **DEAD** for accuracy at our data scale | thesis Tables 5.6/5.8: −4–14 mAP at small-data scale, loss concentrated on hard classes; v11n's C3k2 already has the 3×3 recovery trick |
| T6 | OpenVINO/ONNX export | descoped (deployment = Jetson/TensorRT workstream) | residue: conversion is accuracy-neutral; deployment imgsz ≈ 768 |
| T7 | EDP + power measurement | **ADOPT** (paper methodology) | ~$30 USB meter + Simpson-rule energy + normalized EDP quantifies our edge claims beyond params/FLOPs |
| T8 | variance as headline metric | **ADOPT** (paper methodology) | report seed/fold std as a generalizability claim; we already hold the 5-fold CV data |

## Result 1 — negative transfer from available in-domain sources (publishable negative result)

Pretraining on the leakage-gated `atli_source_dataset` (11,294 → 11,188 kept; 2 pHash / 104 filename drops) then fine-tuning with the champion recipe **lost 7.5 mAP** (0.702 vs 0.769 ± 0.009) and 20 DD AP points vs the COCO-init control, despite correct plumbing (all 499/499 tensors transferred) and a fully converged pretrain (source-val mAP 0.898). Causal chain, orchestrator-verified:

1. **Background-class supervision conflict** — 45.3% of source boxes (42,097 / 92,917) are transmission lines/towers, which the native task deliberately treats as background; the pretrain teaches firing on ubiquitous native background.
2. **Single-scene source** — visual audit: the 11k images are essentially one arid-corridor capture campaign (one terrain, lighting, tower and insulator type), vs the native set's internet-grade heterogeneity.
3. **COCO-diversity forgetting** — 150 ep at lr0 0.01 on the narrow source overwrites COCO's broad features; the largest drops hit exactly the classes absent from the source (Birdnest 0.994→0.875, Broken_Insulator 0.665→0.554).

With the earlier universe-damper co-training null (F1), **external power-line data is now 0-for-2 mechanisms on ATLI** (co-training and pretrain-init). The thesis's lever is not refuted in general — its precondition (large, heterogeneous, semantically aligned source, e.g. ~100k-image FASDD) simply does not exist in the public power-line inventory. Paper framing: *transfer-learning gains from in-domain sources are conditional on source diversity and taxonomy alignment; below that threshold, generic COCO initialization is strictly better.*

## Result 2 — the 768 deployment profile

Champion@768 = **0.766 ± 0.009 mAP (4 seeds)** vs 0.784 ± 0.011 @1280: ~1.8 mAP for ~2.8× fewer pixels — the operating point the Jetson workstream should quote. (Champion@640 baseline: 0.736 ± 0.015.)

## Adopted contributions for the paper

1. **EDP/power reporting (T7)** — adopt the thesis's normalized Energy-Delay Product protocol for all edge comparisons.
2. **Variance-as-metric (T8)** — promote seed/fold std from error bars to an explicit generalizability metric, with the precision caveat already established in-project (per-class AP on 12–39 test instances swings ±0.05–0.09 per seed).
3. **Within-group replication** — freezing hurts (T3) and cascaded TL is useless (T4) now hold across two independent studies in the group; cite both.

## Recommendation & thread status

**The champion recipe stands unmodified** — no thesis lever improved accuracy; two became paper methodology and one a publishable negative result. All pilots complete (nothing running on GPUs 6–7); thread complete pending main's direction.

## Run ledger (this thread, all on $ATLI_SERVER, test = noCPLID @768: 120 imgs / 467 inst / 12 DD)

| run(s) | recipe | mAP@0.5 | DD AP | DD recall |
|---|---|---|---|---|
| champion@768 ×4 (HR768nc s0–s2 + TH2_champ768_v11_s3) | stage-2 lr0 0.00334 | 0.766 ± 0.009 | 0.664 ± 0.042 | 0.667 |
| TH2_lowlr_v11_768 s0–s2 | stage-2 lr0 1e-4 | 0.769 ± 0.008 | 0.659 ± 0.022 | 0.623 |
| TH2_srcTL_v11_768_s0 | init from gated source pretrain | 0.702 | 0.524 | 0.500 |
| TH2_srcpre_v11_s1 | source pretrain itself (source-val) | 0.898 | 0.905 | — |
