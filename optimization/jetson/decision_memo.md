# Decision memo: edge compute for the UAV inspection payload
**To:** PI / project leadership · **From:** Jetson deployment thread · **Date:** 2026-07-08 · **Decision:** which device flies, and what "real-time" must mean

**Model:** champion YOLOv11n (2.6M params), 7-class defect detection, trained @1280. Test accuracy mAP@0.5 **0.784** (3 seeds); at reduced inference resolution: **0.783 @1024 · 0.754 @768 · 0.717 @640**, Defective_Damper holds to ~896–1024 and drops −0.08 at 640 (3-seed ladder, `results/optimization/res_ladder_infer_lo.csv`).

## Option A — Original Jetson Nano (own it already; $0)
- **Ceiling (published-benchmark-anchored, ±30% until on-device bench):** ~6 fps @1280, ~11 @960, **~17–18 @768**, ~25 @640 — inference-only; end-to-end ≈ 65–75% of that. Anchor: YOLOv8n (8.7 GFLOPs) = 19 fps @640 FP16 TensorRT ([Qengineering](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano)); Ultralytics quotes 6–10 fps full FP16 pipeline ([DeepStream guide](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson)).
- With the planned 1.75× structured prune (validated mechanically this round, accuracy TBD): **~30 fps inference-only / ~20–24 e2e @768** at mAP ~0.74–0.75.
- **30 fps at deployment accuracy is not reachable.** INT8 buys nothing (Maxwell: no DP4A/tensor cores — [NVIDIA](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/), [forum](https://forums.developer.nvidia.com/t/is-int8-ptq-even-possible-on-jetson-nano/218184)). Product is **EOL**: JetPack capped at 4.6 (Ubuntu 18.04, Python 3.6, TensorRT 8.2) — every future software step fights the platform.

## Option B — Jetson Orin Nano Super devkit ($249)
- **$249**, 67 INT8 TOPS, 8 GB LPDDR5 @ 102 GB/s, 7/15/25 W modes ([NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/)).
- Measured by Ultralytics: **YOLO11n @640 TensorRT FP16 = 4.57 ms** (≈219 fps inference-only) ([Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson/)). Scaling ×4 for 1280: ≈ 18 ms ⇒ **~55 fps inference-only ⇒ 30+ fps end-to-end at FULL champion accuracy (mAP 0.784 @1280), no pruning, no retraining, no accuracy negotiation.**
- Precedent for the class: published autonomous distribution-tower inspection runs YOLOv11n on an Orin Nano at **28 fps e2e** ([Sensors 25(20):6445](https://www.mdpi.com/1424-8220/25/20/6445)); a TX2 (2.8× original Nano, well below Orin) flies a DJI M300 corridor-hazard YOLOv8 at 32 fps ([Drones 10(3):183](https://www.mdpi.com/2504-446X/10/3/183)).
- Payload: same module form factor class as flown TX2/Manifold systems; can be power-capped to 7/15 W in flight. Current JetPack 6 support (Python 3.10+, TensorRT 10) — no EOL tax.

## Option C — Keep the Nano and rescope "real-time" (the coverage math)
The 30 fps target is inherited from video, not from inspection physics:
- **Flight envelope:** motion-blur analysis caps inspection passes at ~**3.3 m/s** (1/1000 s exposure, ½-pixel-class blur criterion — [US patent 10,564,649, flight planning for tower inspection](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10564649)); Chinese DL/T-standard practice keeps **≥25–30 m** standoff on patrol, **5–15 m** on detailed passes ([Frontiers standardization review](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2021.713634/full), [Remote Sens. 15(19):4841](https://www.mdpi.com/2072-4292/15/19/4841)).
- **Time-in-frame:** with a typical wide inspection camera (~82.9° HFOV, DJI H20 class) at 10 m standoff the frame spans ≈ 2·10·tan(41.5°) ≈ **17.7 m** of scene. A damper passing through at 3.3 m/s stays in frame ≈ **5.3 s**. At a **5 Hz** detection rate that is **~26 independent sightings per component per pass** (at 30 m standoff: >15 s, ~80 sightings). Even 1 Hz yields ~5 sightings.
- Therefore **5–10 Hz with ≤200 ms latency is a defensible detection requirement** for inspection logging/alerting. 30 fps is only genuinely required if detections close a control loop (visual servoing on components, obstacle tracking). Pruned Nano @768 delivers ~20 fps e2e — comfortably 2–4× this requirement.
- Residual costs of C: engineering time already ~2 weeks-equivalent on this thread (pruning grid + DeepStream port + on-device debug against an EOL JetPack still ahead), accuracy at 768 = 0.754 (−3 mAP vs champion), and DD sits at its floor (0.593 vs 0.59) with no margin.

## Recommendation
1. **Buy the Orin Nano Super ($249) — Option B.** It converts a months-long optimization campaign into a solved problem at full accuracy (0.784 @1280, 30+ fps e2e), on a supported software stack, with published precedent for the exact aircraft/mission class. $249 is less than one day of engineering time.
2. **Adopt the rescoped requirement regardless** (5–10 Hz detection, ≤200 ms latency, ≥5 sightings/component/pass): it is the physically grounded spec, makes the original Nano viable as a **fallback/second airframe** (pruned 768 pipeline, Option A+C), and gives the Orin thermal headroom to run at 15 W.
3. Pruning work (already validated mechanically) continues either way — on the Orin it converts to battery life instead of feasibility.
