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
| G1 | Exportability | `best.pt` → ONNX (opset ≤ 13, simplified) → TensorRT **FP16** engine builds cleanly under TRT 8.x; engine-vs-PyTorch val parity spot-checked on RTX 6000 (functional proxy only, NOT a Nano latency measurement) | ✅ **PASSED** (R3: TRT 8.6.1 FP16 engine 8.1 MB @768; parity −0.009 mAP, DD unchanged) |
| G2 | Throughput | Projected **≥ 30 fps batch-1** on original Nano in FP16, computed from published same-architecture Nano benchmarks scaled by GFLOPs ratio (anchor + method below), with sources cited | 🟧 ~18 fps @768 unpruned; prune grid (1.75× cut → ~30 fps projection) running; decision memo recommends rescope to 5–10 Hz spec and/or Orin Nano Super |
| G3 | Memory | Engine + runtime workspace + capture/pre/post pipeline < **2 GB**, leaving headroom in the Nano's 4 GB *unified* (CPU+GPU shared) memory, headless OS assumed | 🟨 provisional pass (R3: engine 8.1 MB @768; CUDA context ~600–800 MB; published Nano YOLO-TRT deployments < 2 GB headless — citation task w/ Thread A) |
| G4 | Accuracy | Deployed artifact (post-export, post-quantization, at deployment imgsz) retains **mAP@0.5 ≥ 0.745** (95% of 0.784) on the noCPLID test split, **and DefDamper AP@0.5 ≥ 0.59** (95% of 0.622) — the hard class may not be sacrificed disproportionately | ✅ **PASSED** (R2/R3: native-768 3-seed 0.769/0.670; deployed FP16 engine 0.768/0.732 seed-0) |

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

### Round 1 (2026-07-07, in progress) — Thread B reported; Thread A pending
**G1 progress:** champion `best.pt` → ONNX **opset 12, imgsz 768, simplified — export success** (10.2 MB, `runs/HROaugnc_v11_s1_s2/weights/best.onnx` on server). Remaining G1 leg: TensorRT engine build + parity val. `onnx onnxslim onnxruntime` installed into `~/atli/env`.

**Thread B (thesis, `opt/thesis-enhancements` @ be81e70) reported.** Headlines: (1) homogeneous/in-domain source pretraining is the thesis's dominant accuracy lever (FASDD→AFSE 79.2 vs COCO-init 64.8 vs 600-ep scratch 69.2, test mAP@0.5); (2) freezing always hurts (up to −15 mAP); (3) cascaded TL never beats single-stage — merge sources instead; (4) lightweight backbone swaps lose 4–14 mAP at small-data scale; (5) OpenVINO export ≈ 2.9× fps free on Pi-CPU (their edge target); pruning/quant/distill unexplored. Ranked proposals: champ_srcTL (in-domain pretrain from `atli_source_dataset`), champ_lowlr (stage-2 lr0 1e-4), edge export, EDP/variance reporting, (deferred) Ghost-neck.

**Verification (orchestrator, primary source = thesis PDF pp. 39–45):** Tables 4.1/4.2 + Fig 4.2 all match B's numbers ✅ (79.2/64.8/69.2; freeze-10 = 79.2→64.2; v8n/11n fine-tune lr0=0.0001, 75 ep). One correction logged: TL's 5-fold mAP std (4.23) beats 300-ep scratch (5.86) but is *worse* than 600-ep scratch (3.85) — claim restated as "TL matches long-scratch generalizability at ~1/4 budget." OpenVINO leg rejected for this project (Intel-CPU toolchain; our target is Jetson GPU/TensorRT) — the correct analog, TensorRT FP16 export, is already gate G1.

**Experiments launched this round:**
- `HR768nc_v11_s{0,1,2}` (orchestrator, GPUs 0–2): champion recipe retrained natively at **imgsz 768** on `ATLI_noCPLID_OS3`, 150+100 ep, scale=0.9, seeds 0–2. Question: does native-768 beat train-1280/infer-768 (0.754/0.593)? Monitor armed.
- Thread B round-2 tasking sent: (1) leakage-gated (pHash+filename vs all noCPLID splits) pretraining source from `atli_source_dataset` + 1-seed champ_srcTL pilot fine-tuned at 768; (2) champ_lowlr pilot (stage-2 lr0=1e-4) at 768. GPUs 6–7 only.

**Thread A (Jetson, `opt/jetson-nano` @ fe6119f) reported** (arrived late in the round). Headlines: (1) original Nano cannot run the 1280 champion in real time (~6 fps inference-only, ~4–5 e2e); (2) **30 fps is unachievable on the Nano at any useful resolution** — realistic ceiling ~24–26 fps inference-only / ~15–20 e2e at 640; (3) INT8 dead end independently confirmed (Maxwell: no DP4A); (4) no published on-drone inspection system runs 1280 on Nano-class hardware; (5) exports at 640/960/1280 functionally verified, static-letterbox cost ≈ −0.012 mAP; (6) RTX 6000 is latency-bound 640→1280 (flat 7.5 ms) so it cannot proxy Nano latency scaling. Ranked actions: Nano-vs-Orin decision memo, on-device trtexec bench, multi-seed re-val of resolution ladder, 640-native student (+CWD distill), 30–50% structured pruning.

**Verification of Thread A (orchestrator):**
- Orin Nano Super YOLO11n TRT FP16 = 4.57 ms @640 ✅ confirmed (docs.ultralytics.com/guides/nvidia-jetson: FP32 7.53 / FP16 4.57 / INT8 3.80 ms; INT8 costs 3.1 mAP50-95 pts there — FP16-terminal even on Orin for nano models).
- fps ceiling ✅ consistent with my independent anchor projection (Round 0).
- **"DD flat under reduced inference resolution" ⚠ corrected: single-seed (seed-0) artifact.** A's numbers match seed 0 exactly; my 3-seed ladder shows seed-averaged DD 0.622@1280 → 0.539@640 (−0.083). Claim *does* hold at ≥960. A's proposed multi-seed re-val was already done (Round 0.5) — deduplicated.
- A's action #3 cancelled (done); #1 (decision memo) and #5 (pruning) tasked for Round 2; #4 (640-native student) partially covered by my HR768nc runs.

**G2 verdict update:** 30 fps batch-1 on the *original* Nano is **infeasible for any G4-passing configuration** (both threads + my projection agree). Path forward is a decision, not an experiment: (a) rescope the fps gate to ~10–15 fps e2e (arguably sufficient for inspection frame-overlap needs — memo requested), or (b) hardware bump to Orin Nano Super ($249; champion @1280 est. 30–55 fps → all gates pass at full accuracy). To be presented at checkpoint; meanwhile the campaign optimizes max-fps-at-G4 (768 envelope + pruning toward ~1.7×).

**Round-2 tasking sent:** Thread A — pruning feasibility for v11n/C2PSA with TensorRT-FP16 latency evidence + Nano/Orin/rescope decision memo + 768 export. Thread B — leakage-gated srcTL pilot + lowlr pilot at 768 (GPUs 6–7). Orchestrator — HR768nc 3-seed (GPUs 0–2, running), TRT FP16 engine parity at 768 (GPU 6, running).

### Round 2 (2026-07-08) — incident postmortem + native-768 result: G4 PASSES

**INCIDENT (postmortem, revised 07-08):** two separate pip mutations of the *shared* `~/atli/env` while jobs were running combined to break numpy: (1) Thread A's Round-1 install of `onnxruntime-gpu 1.18.1` (pins `numpy<2` → downgrade; disclosed by A unprompted), and (2) my `pip install tensorrt` attempt (pulled a broken `tensorrt_cu13_libs`, failed mid-resolution). Main's OBB benchmark runs (GPUs 3–5) crashed on fresh-process numpy imports; long-running trainings (my HR768nc, Thread B's chains) survived because numpy was already in memory. Both actors owned it. TRT parity job never ran (install failed). **Campaign-wide hard rule: `~/atli/env` is FROZEN — never pip install/upgrade/uninstall in it (in either direction: the stray packages already in it stay, removal is also a mutation). All export tooling lives in separate venvs** — orchestrator: `~/atli/env_export` (python 3.10: ultralytics 8.4.90, onnx, onnxslim, onnxruntime, **TensorRT 8.6.1** + pip cuDNN 8 via `LD_LIBRARY_PATH` — note the server is Ubuntu 18.04/glibc 2.27, so TRT ≥ 10 manylinux_2_28 wheels cannot install; TRT 8.6 is also the closer functional proxy for the Nano's TRT 8.2); Thread A: `~/atli/env_jetson` (torch-pruning). The 09:12 GPU 3–5 relaunch (`rerun_obb_champ.sh`) was main relaunching its own numpy-killed OBB benchmark — no unexplained actor.

**Suspect-window re-validation (fresh vals, healthy env, GPU 0 — `results/optimization/reval_round2.csv`):** all 6 runs re-scored on the noCPLID test split @768. Integrity proof: `TH2_champ768_v11_s0` (Thread B's control) re-validated **bit-identical** to `HR768nc_v11_s0` (mAP 0.7773, DD 0.7252, identical per-class) — two independent launches, same config+seed → deterministic match. Suspect-window trainings are trustworthy.

| run | mAP@0.5 | DefDamper AP | verdict |
|---|---|---|---|
| **HR768nc_v11 (champion recipe @768 native, 3 seeds)** | **0.769 ± 0.009** | **0.670 ± 0.049** | **G4 PASS with margin** (floors 0.745 / 0.59) |
| — vs champion @1280 | 0.784 ± 0.011 | 0.622 ± 0.073 | −1.5 mAP, **+4.8 DD** |
| — vs train-1280/infer-768 | 0.754 | 0.593 | native retrain +1.5 mAP, +7.7 DD |
| TH2_lowlr_v11_768_s0 (stage-2 lr0=1e-4) | 0.772 | 0.684 | wash vs control (0.777/0.725), 1 seed |
| TH2_srcTL_v11_768_s0 (gated in-domain pretrain) | 0.702 | 0.524 | **FAILED: −7.5 mAP vs control** — thesis lever did not replicate with `atli_source_dataset` |

**Findings:** (1) **Native-768 is the new deployment candidate**: passes both G4 floors, DD *above* the 1280 champion (small-object DD apparently benefits more from matched train/infer resolution than from raw pixels), at ~18 fps projected on Nano — 3× the champion's speed for −1.5 mAP. (2) **In-domain source pretraining transfers negatively here** — consistent with the earlier universe-damper finding (community-sourced `atli_source_dataset` has domain/label-style mismatch; the thesis's FASDD source was larger and homogeneous-quality). One-seed evidence; Thread B to diagnose before the lever is killed. (3) Thesis low-LR fine-tune: no effect at 1 seed.

**Gate scoreboard after Round 2:** G1 ONNX ✅ / engine parity pending (env_export rebuilding) · G2 at 768: ~18 fps projected — needs ~1.7× from pruning (Thread A critical path) · G3 untested · **G4 ✅ PASSED at 768-native** (0.769/0.670).

### Round 3 (2026-07-08, in progress) — Thread A pruning verdict + decision memo
**Thread A (`opt/jetson-nano` @ 3e8b23e) reported.** Headlines: (1) **structured pruning of the actual champion is mechanically validated** — `prune_v11n.py` (torch-pruning 1.6.0, `ignored_layers=[Detect, Attention]` per the YOLO-Pruning-RKNN recipe + an auto-quarantine loop for two failing C3k2 groups): 4.60→2.63 GMACs @768 = **1.75× FLOPs cut**, params 2.59M→1.17M, survives YOLO() reload/predict/ONNX re-export; only GPU fine-tune + accuracy check remain. Projection: ~30 fps inference-only @768 on Nano. (2) Critical measurement discipline: FLOPs cuts do NOT show up as batch-1 latency on workstation GPUs (LAMP: −35% FLOPs, 126→122 fps; matches A's RTX finding) — acceptance = GMACs + published Nano anchors, never RTX latency. (3) Letterbox cost @768 = −0.0026 mAP (vs −0.012 @1280) — deployable-format accuracy essentially free at 768. (4) `decision_memo.md`: physics says **5–10 Hz is the real detection-rate requirement** (blur-limited ~3.3 m/s flight, 5–15 m standoff, ~5.3 s per-component dwell → ~26 sightings/pass at 5 Hz); 30 fps only needed for closed-loop servoing. Recommendation: buy Orin Nano Super ($249, runs champion @1280 at 30+ fps e2e at full 0.784 accuracy) AND adopt the rescoped 5–10 Hz spec, making the original Nano a viable fallback at pruned-768 (~20 fps e2e).

**Verification (orchestrator):** MCP-YOLO ✅ confirmed (Sensors 25:7049 — 8.65M params, −37.3%, mAP 0.921, 250 fps, Group SLIM). HALP ⚠ caveat: its GPU speedups were measured at **batch 256** on TITAN V (paper checked) — not batch-1; directionally fine for the compute-bound Nano but not a bs=1 guarantee. A's numpy disclosure integrated into the postmortem above.

**Thread B (`opt/thesis-enhancements` @ 3df8b0c) srcTL root-cause: genuine negative transfer, NOT plumbing.** Plumbing verified (fine-tune loaded pretrain weights: "Transferred 499/499" vs COCO-init's 448/499; source pretrain converged, source-val mAP 0.898). Causes: (1) 45.3% of source supervision is transmission_line/tower boxes — classes ATLI deliberately treats as background (orchestrator re-counted on server: 42,097/92,917 ✅); (2) source ≈ one arid-corridor capture campaign (single scene × 11k images) vs ATLI's heterogeneity; (3) catastrophic forgetting of COCO diversity — largest drops exactly in classes absent from source (Birdnest, Broken_Insulator). **T1 (in-domain pretraining) CLOSED for available inventory — external ATLI-domain data is now 0-for-2 mechanisms (co-training, pretrain-init).** A publishable negative result. B's remaining lever: lowlr seeds 1–2 running (GPU 6) + a 4th champ768 replicate (seed 3, GPU 7); thread converging to final synthesis (EDP + variance-as-metric adopted for reporting).

**Round-3 tasking sent:** Thread A — 3-epoch fine-tune smoke test, then the 9-run prune grid {1.5,1.75,2.0}× × 3 seeds **based on the native-768 champion** (`HR768nc_v11_s{0,1,2}_s2`), acceptance mAP ≥ 0.745 & DD ≥ 0.59 @768, GPUs 0–2. Thread B — lowlr 3-seed verdict, then final thread synthesis. Orchestrator — G1 engine parity (building on GPU 0: FP16, 1×3×768×768 static, INT64→INT32 cast warning only).

**G1 RESULT — PASSED.** `HR768nc_v11_s0` best.pt → ONNX (opset 12, 10.2 MB) → **TensorRT 8.6.1 FP16 engine (8.1 MB, 461 s build)** → engine val on noCPLID test @768 bs=1: **mAP@0.5 0.768, DD AP 0.732** vs PyTorch same-weights 0.7773/0.7252 → total export cost −0.009 mAP (FP16 + static letterbox combined; consistent with Thread A's −0.0026 letterbox-only measurement), DD unharmed. Deployed-artifact accuracy still clears both G4 floors. G3 note: 8.1 MB engine; on-Nano process footprint dominated by the ~600–800 MB CUDA context — published Nano YOLO TRT deployments run comfortably under 2 GB headless; G3 marked **provisional pass** pending Thread A's memo citation (no hardware available by mission definition).

### Round 4 (2026-07-08, in progress) — git migration; Thread B complete (T2 closed)
**Git migration:** repo history rewritten twice (user-ordered privacy scrub: noreply author emails; personal names → "the PI"/"a co-PI"/"et al."; server login → `$ATLI_SERVER`; `results/data_outreach.md` purged). Orchestrator worktree reset to new head `b0d8592` (no unpushed work; privacy gate CLEAN; noreply email set). Thread B migrated and pushed `632c042` (independently privacy-scanned by orchestrator: CLEAN). Thread A migration relayed, pending. Standing policy: `GIT.md` on main.

**Thread B final round (`opt/thesis-enhancements` @ 632c042), verified:** lowlr numbers are internally consistent with orchestrator's own revals (seed-0 values match exactly; 3-/4-seed means recompute correctly).
- **T2 (thesis low-LR fine-tune) CLOSED — wash.** lowlr@768 3 seeds: mAP 0.769 ± 0.008, DD 0.659 ± 0.022, DD recall 0.623 vs champion@768 (now **4 seeds** incl. B's s3 replicate): mAP **0.766 ± 0.009**, DD **0.664 ± 0.042**, DD recall 0.667. OneCycle from lr0=0.00334 already reaches the same basin; keep the champion stage-2 LR.
- **Deployment reference band updated (4 seeds): champion@768 = mAP 0.766 ± 0.009, DD 0.664 ± 0.042.**
- `optimization/final_synthesis.md` delivered: T1–T8 lever scoreboard, negative-transfer result framed as a paper contribution, 768 deployment profile, EDP + variance-as-metric adopted. **Thread B complete**; GPUs 6–7 released.

## Ledger of verification verdicts
| Round | Claim | Source of claim | Verdict | Evidence |
|---|---|---|---|---|
| 0 | INT8 gives no speedup on original Nano | orchestrator hypothesis | ✅ confirmed | Qengineering repo measurement; Maxwell sm_53 lacks DP4A |
| 0 | "Jetson Nano" fps numbers in recent posts | web at large | ⚠ mostly Orin Nano | Ultralytics blog/docs benchmark Orin devices only |
| 1 | FASDD→AFSE 79.2 vs COCO 64.8 vs scratch-600 69.2; freeze-10 −15 mAP; v11n FT lr0=1e-4/75ep | Thread B | ✅ confirmed | Thesis PDF Tables 4.1/4.2 (pp. 39–41), read directly |
| 1 | "TL cuts CV variance" | Thread B | ⚠ corrected | Fig 4.2: TL std 4.23 < scratch-300 5.86 but > scratch-600 3.85 |
| 1 | OpenVINO 2.9× free speedup applies to us | Thread B (implicit) | ❌ rejected | OpenVINO = Intel CPU toolchain; Jetson target needs TensorRT — analog already covered by G1 |
| 1 | Orin Nano Super: YOLO11n TRT FP16 4.57 ms @640 | Thread A | ✅ confirmed | docs.ultralytics.com/guides/nvidia-jetson benchmark table |
| 1 | DefDamper AP flat at reduced inference res (0.54 @1280/960/640) | Thread A | ⚠ corrected | Single seed (=champion s0); 3-seed ladder: DD −0.083 at 640, flat only ≥960 (`results/optimization/res_ladder_infer_lo.csv`) |
| 1 | 30 fps unachievable on original Nano at ≥640 | Thread A | ✅ accepted | Matches orchestrator projection from 19-fps anchor; ceiling ~24–26 fps inference-only @640 |
| 2 | Suspect-window trainings valid despite numpy incident | orchestrator | ✅ proven | Independent same-seed replicate re-validated bit-identical (0.7773/0.7252 all classes) |
| 2 | In-domain source pretraining = dominant accuracy lever (thesis) | Thread B / thesis | ❌ did not replicate | srcTL pilot −7.5 mAP vs control on ATLI (`reval_round2.csv`); source-data quality/domain differs from FASDD |
| 2 | Leakage gate counts (11,294→11,188; 104+2 drops) | Thread B | ✅ confirmed | `source_pretrain_gate_report.json` read directly on server |
| 3 | MCP-YOLO: Group SLIM −37.3% params, mAP 0.909→0.921, 161→250 fps | Thread A | ✅ confirmed | Sensors 25:7049 (PMC12656040) |
| 3 | HALP 1.6–1.9× GPU latency from structured pruning | Thread A | ⚠ caveat | Paper measures TITAN V at batch 256, not batch-1; directional for compute-bound Nano only |
| 3 | GPU 3–5 09:12 relaunch actor | (record) | ✅ resolved | Was main relaunching its own OBB benchmark |
| 3 | srcTL fine-tune actually loaded pretrain weights (499/499) | Thread B | ✅ confirmed | Grepped TH2_gpu6.log on server |
| 3 | 45% of source boxes are line/tower (background-conflict) | Thread B | ✅ confirmed | Orchestrator re-count: 42,097/92,917 = 45.3% |
| 3 | srcTL failure = genuine negative transfer (not plumbing) | Thread B | ✅ adopted | Converged source pretrain (0.898) + weight-load proof + class/scene audit |
