# Jetson Nano Deployment Research — Round 1
**Branch:** `opt/jetson-nano` · **Date:** 2026-07-07 · **Author:** Jetson deployment researcher agent

**Subject model ("champion"):** YOLOv11n, 2.58M params, **6.3 GFLOPs @640** (measured, fused), trained at imgsz **1280** with ×3 Defective_Damper oversampling + scale=0.9. Test metrics (120-img clean test, 3 seeds): mAP@0.5 **0.784±0.011**, Defective_Damper AP **0.622±0.073**.

---

## 1. Executive summary / feasibility verdict

- **The original Jetson Nano cannot run the champion at 1280 in real time.** Best published-benchmark-anchored estimate: **~5–7 fps inference-only, ~4–5 fps end-to-end** at 1280 FP16 TensorRT. 30 fps at 1280 is off by ~6×.
- **30 fps is not achievable on the original Nano at any resolution ≥640** with any YOLO variant of useful accuracy. The realistic ceiling is **~20–25 fps inference-only (~15 fps end-to-end) at 640** with a TensorRT FP16 engine, DeepStream pipeline, and pruning. Published anchor: YOLOv8n @640 FP16 = **19 fps** inference-only ([Qengineering](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano)); Ultralytics quotes **6–10 fps** for a full FP16 pipeline on Nano ([Ultralytics DeepStream guide](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson)).
- **INT8 is a dead end on the Nano** — Maxwell predates DP4A (Pascal+) and has no tensor cores; TensorRT INT8 on Nano gives **no speedup and worse mAP** (verified: [NVIDIA forums](https://forums.developer.nvidia.com/t/is-int8-ptq-even-possible-on-jetson-nano/218184), [Qengineering](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano), [NVIDIA DP4A blog](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/)). FP16 is the terminal precision. QAT is therefore *not* on our ladder for this device.
- **Good news from this round's measurements (UNLV server, 1 seed):** the champion's Defective_Damper AP is **flat when inferring at reduced resolution** (0.542 @1280 → 0.546 @960 → 0.540 @640). The −5.0 mAP cost of 640 inference is concentrated in Flashover (−0.14) and Self-Exploded (−0.13) insulators. The "1280-or-nothing" assumption for DD may be a *training*-resolution effect, not an *inference*-resolution one — this is the single most exploitable finding.
- **Strategic recommendation:** confirm with the PI whether the airframe is truly locked to the original Nano (EOL: JetPack 4.6 / Python 3.6 / TensorRT 8.2, discontinued). A **Jetson Orin Nano Super (~$249)** runs YOLO11n @640 FP16 in **4.57 ms** ([Ultralytics benchmarks](https://docs.ultralytics.com/guides/nvidia-jetson/)) and would run the champion at 1280 at an estimated ~30–55 fps — the 30 fps goal at full accuracy becomes trivially feasible. On the original Nano, the goal must be renegotiated to either ~10 fps @960 (mAP 0.759) or ~15–20 fps @640 (mAP 0.726–0.76 after retraining).

---

## 2. Hard constraints of the target device

| Constraint | Value | Implication |
|---|---|---|
| GPU | 128-core Maxwell | No tensor cores, no DP4A → **FP16 only**; INT8 gives no gain |
| Peak compute | **472 GFLOPS FP16** (235 FP32) | ~165 GFLOP/s *effective* on YOLO-class nets (35% utilization, derived §4) |
| Memory | **4 GB LPDDR4, 25.6 GB/s, shared CPU+GPU** | Engine *builds* at 1280 need swap/zram; runtime OK; OS takes ~1.5 GB |
| Power | 5 W / 10 W modes | UAV payload favors 5 W → clocks drop ~30–40% below benchmark numbers (most benchmarks run `nvpmodel -m 0` + `jetson_clocks`) |
| DLA | **None** (Xavier/Orin only) | No offload path; GPU does everything incl. video decode share |
| Software | **EOL**: JetPack ≤4.6.x, Ubuntu 18.04, Python 3.6, TensorRT 8.2.1, CUDA 10.2 | Modern Ultralytics needs Python ≥3.8 → **export ONNX off-device (opset ≤13; we use 12), build engine on-device with `trtexec`**. Ecosystem support is frozen. |

Sources: [NVIDIA forums INT8/Nano](https://forums.developer.nvidia.com/t/is-int8-ptq-even-possible-on-jetson-nano/218184), [Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/) ("only JetPack 4 supported" for original Nano; no benchmark table published for it), [NVIDIA TensorRT/DP4A](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/), [Jetson Nano Python/TRT constraint](https://forums.developer.nvidia.com/t/jetson-nano-python-3-7-version-for-tensorrt/242694).

## 3. What the literature runs on-device (and at what fps)

**Published Nano benchmarks (TensorRT FP16, 640 input, inference-only unless noted):**

| Model | GFLOPs | fps on Nano | Source |
|---|---|---|---|
| YOLOv8n | 8.7 | **19** | [Qengineering TRT C++](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano) (excludes grab/post/viz) |
| YOLOv8s | 28.6 | **9.25** | same |
| YOLOv5n | 7.7 | **~20** (<50 ms) | [Makhalov TRT comparison](https://medium.com/@peter.makhalov/comparing-performance-of-yolo-family-object-detectors-for-tensorrt-implementations-69e7e8e42c69); [Seeed blog ~27 fps](https://www.seeedstudio.com/blog/2022/08/23/faster-inference-with-tensorrt-on-nvidia-jetson-run-yolov5-at-27-fps-on-jetson-nano/) |
| YOLO26/11-class nano, full pipeline | — | **6–10 e2e** | [Ultralytics DeepStream guide](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson) |

Note the ~2× gap between "inference-only" and "full pipeline" numbers — decode, letterbox, NMS and display share the same 4 GB/25.6 GB/s memory and CPU. This gap is the DeepStream/GPU-preprocessing opportunity (§6, rung 5).

**UAV power-line / infrastructure inspection deployments:**

| System | Edge HW | Model | Result | Source |
|---|---|---|---|---|
| Fire-hazard powerline corridor, DJI M300 + Manifold 2-G | **Jetson TX2** (1.3 TFLOPS ≈ 2.8× Nano) | YOLOv8 light, TRT FP16 | **32 fps**, ~95% acc | [Drones 2026, MDPI](https://www.mdpi.com/2504-446X/10/3/183) |
| Distribution-tower defects, autonomous UAV | **Jetson Orin Nano** | YOLOv11n-based, 640 | **28 fps e2e** | [Sensors 2025, MDPI](https://www.mdpi.com/1424-8220/25/20/6445) |
| Insulator defect UAV (MCP-YOLO) | (workstation only; edge deferred to future work) | pruned YOLOv11, 640, 13.8M→8.65M params | 250 fps on dGPU, mAP *improved* 0.909→0.921 after pruning | [Sensors 2025, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656040/) |
| High-voltage line aerial inspection | ground station | YOLOv8 | real-time off-board | [Energies 2024](https://doi.org/10.3390/en17112535) |

**Pattern:** every published *on-device* real-time inspection system runs at **640 or less, FP16 TensorRT, on hardware ≥2.8× the Nano**. Nobody publishes an on-drone 1280-input detector on a Nano-class device. Our 1280 champion as-is is outside the envelope of everything in the literature.

## 4. Feasibility math for the champion

**Anchor-derived effective throughput.** YOLOv8n (8.7 GFLOPs) at 19 fps → 8.7×19 ≈ **165 GFLOP/s effective** FP16 on Nano (~35% of the 472 GFLOPS peak — typical for small convnets that are partly memory-bound; the 25.6 GB/s DRAM is the co-limiter). Cross-check: YOLOv5n 7.7 GFLOPs × 20 fps ≈ 154 GFLOP/s. Use 150–165 GFLOP/s.

**Champion FLOPs vs input size** (conv FLOPs scale with H×W; verified 6.3 GFLOPs @640 fused on server):

| imgsz | GFLOPs | inference-only est. | end-to-end est. (pipeline ≈ 65–75% of inference fps) |
|---|---|---|---|
| 1280 | 25.2 | 150–165/25.2 = **6.0–6.5 fps** (~160 ms) | **~4–5 fps** |
| 960 | 14.2 | **10.6–11.6 fps** (~90 ms) | **~7–9 fps** |
| 640 | 6.3 | **24–26 fps** (~40 ms) | **~15–19 fps** |
| 640 + P2 head | ~11.9 | **12.6–13.9 fps** | ~9–10 fps |

Caveats: (a) these assume 10 W mode + `jetson_clocks`; 5 W mode ≈ −30–40%. (b) YOLOv11n has attention (PSA) blocks absent from v8n; Maxwell+TRT8.2 may run them less efficiently than the FLOPs ratio implies — the on-device `trtexec` benchmark (script §7) is the validation. (c) An RTX 6000 *cannot* validate the scaling: measured this round, the champion is latency-bound on the RTX (inference 7.3→7.8 ms from 640→1280, essentially flat), while on the Nano it is firmly compute-bound and scales ~4× per resolution doubling.

**Verdict: 30 fps on the original Nano is not achievable** at ≥640 for any model with acceptable accuracy (the 19 fps YOLOv8n anchor already exceeds our champion's per-frame budget). Achievable operating points:
- **~5 fps @1280** (full champion accuracy, mAP 0.784) — usable only if the drone flies slow/hovers per structure.
- **~8–10 fps @960** (mAP 0.759 measured, DD flat) — best accuracy-per-watt compromise without retraining.
- **~15–20 fps @640** (mAP 0.726 without retraining; 0.74–0.76 plausible with a 640-native retrain/distill) — closest to "real-time".
- **30 fps** requires either ~416 input (untested, likely kills small-object recall) or a hardware change (Orin Nano Super: YOLO11n @640 FP16 = 4.57 ms ⇒ 1280 ≈ 18 ms ⇒ ~30–55 fps at full champion accuracy).

## 5. Measurements made this round (UNLV RTX 6000 — functional/fidelity only, NOT Nano latency)

Champion weights `~/atli/runs/HROaugnc_v11_s0_s2/weights/best.pt` (single seed; this seed scores mAP 0.7755 / DD 0.542 at 1280 — within the 3-seed spread).

1. **ONNX export works end-to-end**: opset 12 (TRT 8.2-compatible), static batch-1, onnxslim-simplified, at 640/960/1280 → `~/atli/export_jetson/champ_v11n_{640,960,1280}.onnx` (10.3–10.6 MB).
2. **Export fidelity @1280** (120-img test, FP32 ONNX vs PyTorch): mAP@0.5 0.7755 → 0.7635 (**−0.012**); DD 0.542 → 0.565 (+0.023, noise). The delta is the static-square letterbox vs PyTorch rect-val, not numeric error — this is exactly what an on-device engine will see, so **quote 0.763, not 0.776, as the deployable 1280 accuracy for this seed**. FP16 on Nano should add ≲0.002 loss (Ultralytics measured FP32→FP16 as accuracy-neutral on Orin).
3. **Inference-resolution sensitivity of the 1280-trained champion** (PyTorch val, 1 seed):

| eval imgsz | mAP@0.5 | DefDamper | Flashover | Self-Exploded |
|---|---|---|---|---|
| 1280 | 0.776 | 0.542 | 0.683 | 0.850 |
| 960 | 0.759 | **0.546** | 0.612 | 0.846 |
| 640 | 0.726 | **0.540** | 0.546 | 0.718 |

**DefDamper survives 640 inference** (needs multi-seed confirmation given DD's ±0.07 seed noise). The 640 cost is Flashover (−0.14) and Self-Exploded (−0.13) — many-instance classes that a 640-native retrain has the data to recover.

## 6. Optimization ladder (each rung: expected latency gain / accuracy cost / evidence)

| # | Rung | Latency gain | Accuracy cost | Evidence |
|---|---|---|---|---|
| 1 | **TensorRT FP16 engine** (vs PyTorch/ONNX-CPU on device) | **2–3×** | ~0 (mAP50-95 0.479→0.480 measured by Ultralytics on Orin) | [Ultralytics Jetson](https://docs.ultralytics.com/guides/nvidia-jetson/), [Seeed](https://www.seeedstudio.com/blog/2023/03/30/yolov8-performance-benchmarks-on-nvidia-jetson-devices/) |
| 2 | **Resolution 1280→960** | **1.8×** (FLOPs 25.2→14.2) | −0.017 mAP, DD flat (measured, §5) | this round |
| 3 | **Resolution 1280→640** | **4×** | −0.050 mAP as-is; est. −0.02–0.03 after 640-native retrain (the 640-baseline recipe already scores 0.736) | this round + repo baselines |
| 4 | **1280-teacher → 640-student distillation** (CWD/feature-level, exact 2:1 spatial match) | none beyond rung 3 | recovers ~1–3 mAP of rung-3 loss | [2:1 teacher-student resolution KD](https://arxiv.org/pdf/1903.01522); [CWD+pruning framework](https://arxiv.org/abs/2509.12918); [LKD-YOLOv8](https://www.researchgate.net/publication/393217374_LKD-YOLOv8_A_Lightweight_Knowledge_Distillation-Based_Method_for_Infrared_Object_Detection) |
| 5 | **DeepStream/GStreamer pipeline** (NVDEC decode, GPU letterbox via nvstreammux/nvdspreprocess, zero-copy NVMM buffers) | closes the ~2× inference-vs-e2e gap; frees CPU | 0 | [Ultralytics DeepStream](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson), [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo), [nvjpegdec](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvjpegdec.html) |
| 6 | **Structured pruning (LAMP / Group-SLIM) + finetune, 30–50% channels** | est. **1.3–1.6×** on Nano (v11n is already thin; returns diminish) | ≤0.005 mAP at moderate ratios; MCP-YOLO *gained* mAP pruning an insulator detector 37% | [LAMP PCB study](https://arxiv.org/pdf/2507.17176), [MCP-YOLO](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656040/), [pruning+CWD](https://arxiv.org/abs/2509.12918) |
| 7 | **INT8 PTQ/QAT** | **~0 on Nano** (no DP4A/tensor cores); INT8 = −0.03 mAP50-95 even on Orin | strictly dominated by FP16 on this device — **skip** | [NVIDIA forum](https://forums.developer.nvidia.com/t/is-int8-ptq-even-possible-on-jetson-nano/218184), [Qengineering](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano) |
| 8 | **Lighter architecture swap** (v8n 8.7 GFLOPs vs v11n 6.3; edge-specialized nets like EDNet/LEAF-YOLO) | v11n is already the FLOPs-per-mAP winner in our own CV (beats v8n on DD); swaps buy <1.4× | risky re-validation of the whole recipe | repo CV results; [EDNet](https://arxiv.org/pdf/2501.05885) |

## 7. Small-object options at low resolution (the DefDamper problem)

- **P2 head @640:** YOLOv8n 8.9→17.4 GFLOPs ([Ultralytics discussion](https://github.com/orgs/ultralytics/discussions/8227)); a YOLOv11n P2 variant measured 6.3→11.9 GFLOPs, 7.4→12.5 ms, 142→80 fps ([Sci. Reports](https://www.nature.com/articles/s41598-026-35301-2)). **Cost of P2-at-640 ≈ cost of plain 960** (14.2 GFLOPs) — and our own single-split P2 experiments (repo log) did not beat the champion. Prefer plain 960 unless a 640 retrain shows DD collapse.
- **Tiling/SAHI arithmetic:** a 1280 frame as 2×2 overlapping 640 tiles ≈ 5 tiles with 20% overlap → 5 × 40 ms + merge-NMS ≈ **220+ ms ≈ 4.5 fps** — *no cheaper than native 1280* (tiling saves memory, never compute) and SAHI adds repeated pre/post-processing overhead ([SAHI](https://arxiv.org/abs/2202.06934), [ASAHI](https://arxiv.org/html/2604.19233v1) reports only 20–25% overhead reduction). **Tiling is not a Nano speedup path**; it's an accuracy path if we ever need >1280 effective resolution off-line.
- **ROI/corridor cropping:** on an inspection drone the conductor corridor is known from gimbal pose/mission plan; cropping to a 640-tall band around the line before inference processes ~30–50% of the pixels of a full frame at *native* pixel density — e.g. one 1280×512 crop ≈ 10 GFLOPs ≈ **~15 fps** with 1280-level small-object resolution inside the band. Literature analog: drone-view detection via cropped high-res regions ([DroneNet](https://www.mdpi.com/2504-446X/7/7/441), global-local MAV detection [arXiv 2312.11008](https://arxiv.org/pdf/2312.11008)). YOLO supports rectangular input; needs a retune/val at rect shapes. Highest engineering effort, biggest payoff if 960-square is insufficient.
- **Detection-rate reframing:** an inspection drone at ≤5 m/s with a 30 Hz camera does not need 30 Hz *detection* — 5–10 Hz with frame-skipping covers every structure many times over. The 30 fps goal should be re-scoped to "≥5 Hz detection with ≤200 ms latency" unless there is a control-loop reason; the TX2 paper's 32 fps was on 2.8× our compute.

## 8. Ranked recommendations (next actions)

1. **Decision memo to PI: original Nano vs Orin Nano Super** (effort: 0.5 day). The Nano is EOL and caps us at ~5 fps @1280 / ~15–20 fps @640; a $249 Orin Nano Super runs the full 1280 champion at est. 30–55 fps (Ultralytics: 4.57 ms @640). *Validates with:* one on-device trtexec run each. Expected: 30 fps goal met at mAP 0.784 on Orin; unmeetable on Nano.
2. **On-device Nano benchmark of the 3 exported ONNX engines** (effort: 0.5 day with device in hand; scripts ready in `optimization/jetson/`). Build FP16 engines with `build_engine_nano.sh` at 640/960/1280, log trtexec GPU-compute means at 10 W and 5 W. *Validates:* the §4 fps table (±30%), memory headroom at 1280. This is the gating measurement for everything else.
3. **Multi-seed confirmation of the resolution-sensitivity finding + deployable-accuracy table** (effort: 0.5 day on UNLV server, no training). Re-run §5 val at {1280, 960, 640} for the other two champion seeds; report deployable (static-letterbox ONNX) numbers. *Validates:* "960 costs −0.017 mAP, DD flat" as a seed-robust claim. Expected outcome: 960 adopted as the deployment resolution → ~8–10 fps e2e on Nano.
4. **Train a 640-native deployment student** (effort: 2–3 GPU-days). Champion recipe at imgsz 640 (×3 OS, scale=0.9) ± CWD distillation from the 1280 teacher (exact 2:1 feature-map match). Target: mAP ≥0.75 @640, DD ≥0.60 → ~15–20 fps e2e on Nano. *Validates with:* 3 seeds on the 120-img test + fold-0 CV sanity.
5. **Prune the deployment model 30–50% (LAMP or Group-SLIM) + 50-epoch finetune** (effort: 2 days; torch-pruning integrates with Ultralytics). Expected +30–60% fps on Nano at ≤0.005 mAP cost (MCP-YOLO precedent on insulator UAV detection). Only after rungs 2–4 fix the resolution; pruning a 1280 model does not rescue it (1.5× on 6 fps is still 9 fps).

**Explicit non-recommendations:** INT8/QAT on Nano (no hardware support); SAHI/tiling for speed (compute-neutral at best); architecture swap away from v11n (our CV shows v11n already wins DD per FLOP).

## 9. Deliverable scripts (this commit, `optimization/jetson/`)

| Script | Purpose | Status |
|---|---|---|
| `export_champion.py` | ONNX (opset 12, static b1, slimmed) at 640/960/1280; optional local-GPU engine | **Tested on UNLV server** — 3 ONNX files in `~/atli/export_jetson/` |
| `build_engine_nano.sh` | On-Nano trtexec FP16 engine build + 200-iter benchmark (JetPack 4.6 / TRT 8.2 syntax, 4 GB-safe workspace, nvpmodel/jetson_clocks notes) | ready; needs physical device |
| `latency_harness.py` | Batch-1 latency (pre/inf/post split, mean/p95, fps) for .pt/.onnx/.engine; prints device + non-transferability warning | **Tested on RTX 6000** (found champion latency-bound there — flat 7.3–7.8 ms across 640–1280, confirming RTX can't proxy Nano scaling) |
| `accuracy_check.py` | Exported-vs-PyTorch val on the test split (overall + per-class mAP deltas, PASS/FAIL at 0.005 tolerance) + optional per-image parity | **Tested on RTX 6000** (§5.2). Note: onnxruntime-gpu CUDA EP unavailable in `~/atli/env` (cuDNN mismatch) — use `--exported-device cpu` there |

**RTX 6000 ≠ Nano, stated once more:** every latency number produced on the UNLV server in this report is a *functional* or *relative-accuracy* result. The only valid Nano latencies will come from `build_engine_nano.sh` on the physical device; §4's fps table is an estimate anchored to third-party published Nano measurements.

## 10. Key sources

- Qengineering, *YoloV8 TensorRT Jetson Nano* — https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano (Nano FP16 fps anchor; INT8 no-gain note)
- Ultralytics, *NVIDIA Jetson guide* — https://docs.ultralytics.com/guides/nvidia-jetson/ (JetPack-4-only for Nano; Orin Nano Super YOLO11n table: FP16 4.57 ms, INT8 3.80 ms/−0.031 mAP50-95)
- Ultralytics, *DeepStream on Jetson* — https://docs.ultralytics.com/guides/deepstream-nvidia-jetson (Nano 6–10 fps FP16 pipeline)
- NVIDIA forums — INT8 PTQ impossible on Nano: https://forums.developer.nvidia.com/t/is-int8-ptq-even-possible-on-jetson-nano/218184 ; Python/TRT versions: https://forums.developer.nvidia.com/t/jetson-nano-python-3-7-version-for-tensorrt/242694
- NVIDIA, *Fast INT8 with DP4A (Pascal+)* — https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/
- Makhalov, *YOLO-family TensorRT comparison* — https://medium.com/@peter.makhalov/comparing-performance-of-yolo-family-object-detectors-for-tensorrt-implementations-69e7e8e42c69 (YOLOv5n <50 ms on Nano)
- *Towards Autonomous Powerline Inspection…Fire-Related Hazards*, Drones 10(3):183, 2026 — https://www.mdpi.com/2504-446X/10/3/183 (TX2 + TRT FP16 YOLOv8 @32 fps)
- *Fully Autonomous Real-Time Defect Detection for Power Distribution Towers (YOLOv11n)*, Sensors 25(20):6445 — https://www.mdpi.com/1424-8220/25/20/6445 (Orin Nano, 28 fps @640)
- MCP-YOLO, Sensors 25(22):7049 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12656040/ (Group-SLIM pruning 13.79M→8.65M, mAP 0.909→0.921)
- SAHI — https://arxiv.org/abs/2202.06934 ; ASAHI — https://arxiv.org/html/2604.19233v1 (slicing overhead)
- P2-head cost — https://github.com/orgs/ultralytics/discussions/8227 ; https://www.nature.com/articles/s41598-026-35301-2 (v11n P2: 6.3→11.9 GFLOPs, 142→80 fps)
- KD across resolutions — https://arxiv.org/pdf/1903.01522 ; pruning+CWD — https://arxiv.org/abs/2509.12918 ; LAMP — https://arxiv.org/pdf/2507.17176
- Seeed, *YOLOv5 27 fps on Nano* — https://www.seeedstudio.com/blog/2022/08/23/faster-inference-with-tensorrt-on-nvidia-jetson-run-yolov5-at-27-fps-on-jetson-nano/
