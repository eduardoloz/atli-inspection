# README-OPTIMIZATION — Jetson-Nano Deployment of the ATLI Champion Detector

**Branch:** `opt/orchestrator` · **Owner:** orchestrator agent · **Started:** 2026-07-07

## Mission
Iteratively optimize the champion detector (**HROaug_v11** = YOLOv11n, 2.6M params, imgsz 1280, ×3 DefDamper oversample, scale=0.9 aug; clean-data mAP@0.5 **0.784 ± 0.011**, DefDamper AP **0.622 ± 0.073** on `ATLI_target_tightNI_noCPLID` 3-seed benchmark) until it plausibly runs *effectively* on an **original NVIDIA Jetson Nano** (4GB) mounted on an inspection drone. No physical Nano exists in-lab → every round is held to measurable **proxy gates** (below). If a gate is unreachable, document the best achievable trade-off frontier instead.

## Team topology
- **Orchestrator (this branch):** coordinates, verifies claims against primary sources, runs/queues server experiments, merges the two research threads. Only node that sees both threads.
- **Subagent A** — Jetson-Nano deployment research (branch `opt/jetson-nano`).
- **Subagent B** — thesis-mining for model enhancements (branch `opt/thesis-enhancements`).
- Threads are firewalled: neither subagent is told about the other. Findings merge only here.

## Proxy gates (v1, set Round 0 — revisable only with cited evidence)
| # | Gate | Criterion | Status |
|---|------|-----------|--------|
| G1 | Exportability | `best.pt` → ONNX (opset ≤ 13, simplified) → TensorRT **FP16** engine builds cleanly under TRT 8.x; engine-vs-PyTorch val parity spot-checked on RTX 6000 (functional proxy only, NOT a Nano latency measurement) | ⬜ untested |
| G2 | Throughput | Projected **≥ 30 fps batch-1** on original Nano in FP16, computed from published same-architecture Nano benchmarks scaled by GFLOPs ratio (anchor + method below), with sources cited | ⬜ fails at 1280 (projection ≈ 6 fps); open at ≤ 640 |
| G3 | Memory | Engine + runtime workspace + capture/pre/post pipeline < **2 GB**, leaving headroom in the Nano's 4 GB *unified* (CPU+GPU shared) memory, headless OS assumed | ⬜ untested (expected easy: nano-scale FP16 engine is ~5–10 MB; compute, not memory, is the binding constraint) |
| G4 | Accuracy | Deployed artifact (post-export, post-quantization, at deployment imgsz) retains **mAP@0.5 ≥ 0.745** (95% of 0.784) on the noCPLID test split, **and DefDamper AP@0.5 ≥ 0.59** (95% of 0.622) — the hard class may not be sacrificed disproportionately | ⬜ untested (baseline-640 mAP 0.736 already misses → resolution/accuracy frontier is the core experiment) |

### G2 projection method (anchor + scaling)
- **Anchor (primary source):** YOLOv8n TensorRT **FP16 = 19 fps** on original Jetson Nano (Qengineering C++ TensorRT implementation, github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano — a *favorable* C++ pipeline; Python pipelines run slower).
- **Scaling:** fps ≈ 19 × (8.7 GFLOPs / model GFLOPs at deployment imgsz). YOLOv8n@640 = 8.7 GFLOPs; YOLO11n@640 = 6.5 GFLOPs. GFLOPs scale ∝ (imgsz/640)².
- **Projections for YOLO11n (unpruned):** 640 → ~25 fps · 960 → ~11 fps · **1280 → ~6 fps** · 512 → ~40 fps. Caveat: v11's C2PSA attention block may be less TensorRT-efficient on Maxwell than pure-conv v8 layers; treat projections as upper bounds until Subagent A finds direct v11-on-Nano measurements.
- Consequence: **hitting 30 fps requires deployment imgsz ≈ ≤ 576–640, possibly plus pruning** — the champion's 1280 input cannot survive as-is. The central experiment is the resolution/accuracy frontier (train-hi/infer-lo, multi-scale training, imgsz ladder) with DefDamper AP as the sentinel metric.

## Round-0 expert baseline (verified, with sources)
Facts established by the orchestrator before any subagent input:

1. **Hardware:** original Jetson Nano = 128-core **Maxwell** GPU (compute capability 5.3), quad-core Cortex-A57 @ 1.43 GHz, **4 GB LPDDR4 shared** between CPU+GPU @ 25.6 GB/s, **472 GFLOPS (FP16)** peak, 5 W/10 W modes. [developer.nvidia.com/embedded/jetson-nano]
2. **Software ceiling:** Nano is frozen at **JetPack 4.6.x** (Ubuntu 18.04, Python 3.6, CUDA 10.2, **TensorRT 8.2**); no JetPack 5+; product EOL Jan 2027. Implications: (a) modern `ultralytics` cannot run natively on the Nano's Python 3.6 → deployment path is **ONNX (export on workstation) → `trtexec` engine build ON the Nano** (TRT engines are not portable across GPU/TRT versions) with custom C++/DeepStream-6.0 pre/post; (b) ONNX opset must stay within TRT 8.2 support (≤13 is safe).
3. **INT8 is a dead end on this device:** Maxwell (sm_53) lacks fast INT8 (DP4A/tensor-core int8) — measured result: "int8 models don't give any increase in FPS, while their mAP is significantly worse" [Qengineering, ibid.]. **FP16 is the operating precision.** Any INT8-quantization recommendation from research threads targeting the *original* Nano is to be rejected; INT8 only pays on Xavier/Orin.
4. **Beware Nano-vs-Orin-Nano conflation:** most recent "Jetson Nano" YOLO benchmarks (Ultralytics blog, Seeed, DeepStream docs) are actually **Orin Nano** (Ampere, 20–67 TOPS, ~10–40× faster). E.g., YOLO11n TRT FP16 ≈ 4.9 ms (~200 fps) is an *Orin* Nano number. All subagent-reported fps claims will be checked for which device they actually measured.
5. **Anchor benchmark:** YOLOv8n @ TRT FP16 = **19 fps** on original Nano [Qengineering]. This is the calibration point for all G2 projections.
6. **Server tooling gap (checked 2026-07-07):** `ai.ee.unlv.edu` has ultralytics but **no `trtexec`, no `tensorrt`/`onnx` pip packages** in `~/atli/env` → install `onnx onnxsim onnxruntime-gpu` (and optionally `tensorrt` pip wheel) before G1 can be exercised. GPUs 3–5 busy (unrelated OBB benchmark, finishing soon); 0–2, 6–7 idle.

## Candidate optimization levers (to be prioritized with subagent evidence)
1. **Resolution ladder** (biggest, cheapest): champion@1280 → deploy@640/704/768; includes *train-at-1280-infer-at-640* eval (Ultralytics allows imgsz override at val/export) vs retrain-at-640 with champion recipe. Known floor: baseline-640 mAP 0.736.
2. **FP16 TensorRT export** (mandatory, ~free): typically < 0.5 mAP cost.
3. **Structured pruning + fine-tune** (e.g., ~30–50% channel pruning) if 640 alone can't reach 30 fps.
4. **Distillation** from a larger teacher (v11s/m) into the deployed student — candidate accuracy-recovery lever at low res.
5. Architecture deltas from thesis thread (Subagent B) — TBD.
6. **Tiling/SAHI is OFF the table** for real-time (multiplies latency); small-object (DefDamper) sensitivity to downscaling is the key risk to measure per-class.

## Round log

### Round 0 (2026-07-07) — orchestrator baseline & setup
- Branch `opt/orchestrator` created from `origin/main` (cdc5537); this README added.
- Expert baseline built from primary sources (facts 1–6 above); proxy gates G1–G4 defined.
- Key strategic finding: **the 30 fps goal and the champion's 1280 px input are incompatible on the original Nano by ~5×** — the optimization campaign is therefore a *frontier search* (imgsz × pruning × recovery tricks), not a pure export exercise.
- Server capability audit done (fact 6). TEAM ROSTER received; Subagents A & B working on Round-1 deliverables.
- Gate scoreboard: G1 ⬜ · G2 ⬜ (fails at 1280 by projection) · G3 ⬜ · G4 ⬜.

### Round 0.5 (2026-07-07) — Experiment 1: train-hi/infer-lo resolution ladder ✅
Champion weights (3 seeds, trained @1280) evaluated at 6 inference sizes on the noCPLID test split (`results/optimization/res_ladder_infer_lo.csv`; runner `res_ladder_eval.py` on server, GPU 0). Seed-averaged:

| infer imgsz | mAP@0.5 | DefDamper AP | projected Nano fps* | G4 (mAP≥0.745 & DD≥0.59) |
|---|---|---|---|---|
| 512 | 0.674 ± 0.017 | 0.550 ± 0.051 | ~40 | ✗ |
| 640 | 0.717 ± 0.013 | 0.539 ± 0.093 | ~25 | ✗ |
| **768** | **0.754 ± 0.016** | **0.593 ± 0.077** | **~18** | **✓ (on the nose)** |
| 896 | 0.776 ± 0.009 | 0.599 ± 0.035 | ~13 | ✓ |
| 1024 | 0.783 ± 0.034 | 0.634 ± 0.143 | ~10 | ✓ |
| 1280 (native) | 0.783 ± 0.012 | 0.622 ± 0.073 | ~6 | ✓ |

\* G2 projection method above (anchor 19 fps YOLOv8n@640, ×8.7/6.5 GFLOPs, ×(640/sz)²).

**Findings:** (1) **Inference at 1024 is free** — identical mAP to 1280, DD actually up; instant 1.56× FLOPs cut with zero retraining. (2) **768 is the current G4/G2 frontier point**: passes both accuracy floors, projected ~18 fps → needs only ~1.7× more speedup (pruning / slimmer head / lighter arch) to reach 30 fps, vs 5× from 1280. (3) **Train-hi/infer-lo loses to retraining at 640** (0.717 vs the 640-retrained baseline's 0.736) — if we drop below 768, retrain; at 768+ the champion transfers well. (4) DD variance across seeds is large (±0.08–0.14) — per-seed, not single-run, decisions remain mandatory.

**Implication for strategy:** target deployment envelope is now **768–896 px + FP16 TensorRT + ~1.5–2× structural speedup**, or a 640/704 retrain with accuracy-recovery tricks (distillation, thesis-thread enhancements) to claw back the ~0.03 mAP gap to G4.

### Round 1 — pending subagent reports
(to be filled)

## Ledger of verification verdicts
| Round | Claim | Source of claim | Verdict | Evidence |
|---|---|---|---|---|
| 0 | INT8 gives no speedup on original Nano | orchestrator hypothesis | ✅ confirmed | Qengineering repo measurement; Maxwell sm_53 lacks DP4A |
| 0 | "Jetson Nano" fps numbers in recent posts | web at large | ⚠ mostly Orin Nano | Ultralytics blog/docs benchmark Orin devices only |
