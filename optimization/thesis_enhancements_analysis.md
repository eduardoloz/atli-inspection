# Transferability Analysis — Vazquez Thesis Techniques → Our ATLI Champion

**Our champion (reference recipe):** YOLOv11n, COCO-init 2-stage TL (150+100 ep, SGD lr0 0.01 → 0.00334, lrf 0.1535), imgsz 1280, ×3 native oversampling of Defective_Damper images, `scale=0.9`. Latest clean-test benchmark: mAP@0.5 0.784±0.011, DefDamper AP 0.622±0.073 (3 seeds).
**Known project findings referenced below:** (F1) external/community data mixed into training never improved DefDamper (recall drop); (F2) freezing backbone hurts; (F3) 300-ep stage-1 overfits the rare class; (F4) copy-paste aug hurt; (F5) scale-aug sweet spot 0.85–0.9; (F6) leakage gate (pHash+filename) mandatory before any external data.

Digest with page-referenced numbers: `optimization/thesis_digest.md`.

---

## Technique-by-technique assessment

### T1. Homogeneous-source pretraining (thesis §4.1.1: FASDD-init +14.4 mAP over COCO-init, +10 over 600-ep scratch)
- **Applies to us?** Yes — this is the thesis's single biggest accuracy lever and the one thing our pipeline has never properly tested. Our champion still initializes from **COCO** (heterogeneous). The thesis's direct analog for us: pretrain on a large *power-line-domain* source, then run our exact native fine-tune.
- **Crucial distinction vs our F1 (external data never helped):** F1 covers *mixing* external images into native training and our PF runs, which pretrained on the **small (~2.2k), single-class, label-style-mismatched** universe damper set. The thesis's mechanism is different: a **large, multi-class, in-domain** source used *only for initialization*, with the native set alone shaping the final decision boundaries. Label-style mismatch matters far less for initialization than for co-training — the fine-tune stage relabels the feature space. Our own Ufull control (≈0.91 DD AP in-domain) proves the source features are learnable.
- **What to use as source:** `atli_source_dataset` (11,294 imgs: towers, lines, generic `Defective_Insulators` 4,369, `Defective_Damper` 704) optionally merged with `no-close-up-public-data` (12,135) and `universe-damper-staging` (2,175) — thesis conclusion 4 says **amalgamate into one pretraining base, don't cascade**. F6 applies: run the pHash+filename leakage gate against the native val/test before building the source (CPLID/DVDI recycling means source images may duplicate ATLI test images — this alone could silently inflate results).
- **Expected benefit:** the thesis saw +14 mAP on a 282-img target; our target is 5× larger and our baseline much stronger, so scale expectations down: realistic **+1–4 mAP, with the biggest upside on rare classes** (thesis Fig. 4.1 shows in-domain init helps most early/on hard classes; our APET paper likewise saw >20% TL gains concentrated in Broken_Insulator/Defective_Damper). DefDamper AP is the high-variance target: plausible +0.03–0.08 if damper/tower features transfer.
- **Cost:** training-time only (one extra ~150-ep stage-1 on ~11–25k imgs, ~a few GPU-hours at 640). **Zero inference cost.**
- **Confidence: medium.** Strong, directly-analogous evidence from the same research group; main risks are (a) source label noise, (b) leakage, (c) our PF null result — but PF's source was 5–10× smaller and single-class, so it's not a clean refutation.

### T2. Fine-tune LR/epoch reduction for v8n/11n (thesis Table 4.1: fine-tune lr0 = 0.0001 and 75 epochs for v8n/11n vs 0.001/150 for v5n, "to limit overfitting")
- **Applies to us?** Yes. Our stage-2 uses lr0 0.00334 for **all** models — 33× higher than what the thesis found appropriate for v8n/11n on a small dataset. This meshes exactly with our F3 (longer stage-1 overfits the rare class) and the ×6-oversample overfit: our rare class is memorization-prone, and stage-2 LR is an untested knob.
- **Expected benefit:** modest mAP change (±1), but potentially **reduced DefDamper variance and better DD recall** (less late-stage memorization of the 110 rare instances). Also a cheap chance of +DD AP if 0.00334 is currently overshooting.
- **Cost:** none — same compute, arguably less (shorter stage 2).
- **Confidence: medium-high** that it's worth a sweep; low-medium that it beats the champion outright (Ultralytics OneCycle already decays LR; the thesis tuned per-model where we never did).

### T3. Zero frozen layers (thesis Table 4.2: freezing 5/10 layers costs 7–15 mAP)
- Already our practice; independently confirmed by our HROfrz10/HROfrz20 runs (F2). **No experiment needed** — but it strengthens the paper narrative: two independent studies in the group show freezing hurts small-target TL.

### T4. Cascaded TL is useless; merge sources instead (thesis §4.1.3)
- **Applies as negative guidance.** Do not build COCO→universe→source→native chains. If T1 is run, do it as **one merged source stage** (COCO-init → merged source → native), not multiple hops. Saves us from wasting a sweep we might otherwise have run.

### T5. Lightweight backbones (MobileNetV3/ShuffleNetV2/Ghost; thesis Ch. 5)
- **Applies to us?** Only for the edge/FPS story, **not** for accuracy. Every swap lost 4–14 val mAP on the small dataset (Table 5.6), and Table 5.8 shows the penalty shrinks only with FASDD-scale data — we have 1k images. Our hardest metric (DefDamper, small objects at 1280px) would bear the brunt: MG3x3-Half's loss concentrated on the hard fire class (70.0→56.5 AP).
- **Verdict: do not pursue for accuracy.** Revisit Ghost-BN-style compression only if a real drone FPS budget fails after T6, and then only with an in-domain-pretrain (T1) to offset the small-data penalty.
- Note: the thesis's accuracy-recovery trick (3×3 kernel in the bottleneck's first conv, borrowed from YOLOv8) is **already native to YOLOv11's C3k2 blocks** — one reason v11n needs no such surgery.

### T6. Post-training export to OpenVINO/ONNX (thesis §5.5: 2.9× FPS, zero mAP loss, on a Raspberry Pi 5 CPU)
- **Applies to us? Fully, and it's free.** Unmodified YOLOv5n went 6.1→17.4 FPS with mAP unchanged (79.2→80.6). This is the cheapest possible edge win for our champion: no retraining, no accuracy risk, one export command + a Pi-class benchmark.
- **Caveats for us:** (a) our champion runs **imgsz 1280** — 4× the pixels of the thesis's 640, so expect roughly ¼ of its FPS; an export benchmark should measure 1280 vs 640 to quantify the champion's true edge cost (possibly motivating a "640 edge profile" of the champion, mAP 0.736 baseline). (b) *[Supplement — Ultralytics docs, not from thesis]* export is one-liners: `yolo export model=best.pt format=openvino imgsz=1280 half=True` / `format=onnx simplify=True`; OpenVINO has an ARM CPU plugin (the thesis's own Pi 5 result demonstrates it works on ARM); INT8 post-training quantization (`int8=True`, unexplored by the thesis, flagged as its future work) is the natural next step.
- **Expected benefit:** 0 change to mAP/DD AP; 2–3× FPS and ~lowest power per the thesis. Directly serves the PI's 30-fps real-time goal.
- **Confidence: high** (mechanism is runtime graph optimization, demonstrated on identical model family + same class of hardware we target).

### T7. EDP + power measurement methodology (thesis §3.3)
- **Applies to us? Yes, as evaluation methodology.** ~$30 USB power meter + normalized Energy-Delay Product gives our edge-deployment claims quantitative teeth (currently we argue params/FLOPs only). Zero model impact; pairs with T6 for a deployment section in our paper. Confidence: high (it's instrumentation, not modeling).

### T8. TL-reduces-variance via stratified k-fold CV (thesis §4.1.2)
- Methodological confirmation of our F(5) "always report multi-seed/CV means." The thesis additionally uses **variance itself as a headline metric** — worth copying into our results tables (we already have the fold data; report std as a generalizability claim, not just error bars).

---

## Ranked shortlist — what to actually run

Ranked by expected DefDamper-AP/mAP impact per unit cost, respecting our known findings.

### 1. In-domain pretraining stage (T1 + T4) — "champ_srcTL"
The thesis's headline result, never properly tested here (PF used a tiny single-class source). Recipe deltas off champion:
1. Build merged source set (read-only export): `atli_source_dataset` v4 (+ optionally `universe-damper-staging`), **pHash+filename-gated against native val/test**; single `data_source.yaml` (its own classes — nc can differ; fine-tune resets the head).
2. Stage-0 (source pretrain, 640 is fine — cheap): `MODEL=yolo11n.pt DATA=~/atli/ATLI_source_merged/data.yaml bash ~/atli/run_config_ext.sh srcpre_v11 <GPU 0-2/6-7> 150 0 640 16` (COCO-init → source; skip stage 2 or set EP2=0).
3. Champion recipe from source weights: `MODEL=~/atli/runs/srcpre_v11/.../best.pt EXTRA="scale=0.9" DATA=~/atli/ATLI_noCPLID_OS3/data.yaml bash ~/atli/run_config_ext.sh champ_srcTL_s0 <GPU> 150 100 1280 16`, ≥3 seeds.
**Expected:** mAP +1–3, DefDamper AP +0.03–0.08 (medium confidence). **Cost:** ~+4–6 GPU-h once; inference unchanged.

### 2. Stage-2 LR/epoch sweep for v11n (T2) — "champ_lowlr"
`EXTRA="scale=0.9"` unchanged; override fine-tune lr0 via the runner's stage-2 args (or a variant runner): grid {lr0₂ = 0.001, 0.0005, 0.0001} × {EP2 = 100, 75}. E.g. `MODEL=yolo11n.pt EXTRA="scale=0.9" DATA=~/atli/ATLI_noCPLID_OS3/data.yaml LR2=0.001 bash ~/atli/run_config_ext.sh champ_lr2_1e3 <GPU> 150 75 1280 16` (add an `LR2` env passthrough to the stage-2 `lr0=` arg if not present). 2 seeds per cell, prune after first pass.
**Expected:** ±1 mAP; main win = DD variance/recall (medium confidence). **Cost:** ≤ champion cost per run (shorter stage 2).

### 3. OpenVINO/ONNX export + edge benchmark (T6, no training) — "edge_export"
`yolo export model=<champ best.pt> format=openvino imgsz=1280` and `imgsz=640`; verify test-set mAP parity with `yolo val`; benchmark FPS/power on a Pi-5-class ARM board (or server CPU as proxy). Then INT8 (`int8=True` with a calibration split) as the step the thesis left as future work.
**Expected:** identical accuracy, 2–3× CPU FPS (high confidence); produces the numbers the 30-fps real-time goal needs. **Cost:** ~zero.

### 4. EDP/power instrumentation + variance-as-metric reporting (T7 + T8)
Buy/borrow a FNIRSI FNB58-class USB meter, adopt the thesis's Simpson-rule energy + normalized EDP protocol for all edge comparisons; add fold-std as an explicit generalizability metric in our results tables. **Cost:** ~$30 + a script; strengthens the paper regardless of model outcomes.

### 5. (Conditional, deferred) Ghost-neck compression of v11n (T5)
Only if post-export FPS at the required imgsz still misses the real-time bar: Ghost/half-channel neck variant of v11n, **initialized via T1's in-domain pretrain** to offset the small-data penalty the thesis measured. Expect −2–4 mAP risk; do not run before 1–3 are settled.

### Explicit do-NOTs the thesis buys us
- No cascaded multi-hop TL chains (T4).
- No backbone swaps for accuracy on our data scale (T5, Table 5.6).
- No layer freezing anywhere (T3 — now confirmed twice within the group).
