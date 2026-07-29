# Approximating Jetson Nano FPS from RTX 6000 Benchmarks

## Bottom line
**Don't extrapolate directly from the RTX 6000 latency numbers.** The two devices sit in
different performance regimes, so no simple scaling factor is valid. Anchor to a published
real-Nano benchmark instead and scale by relative FLOPs/params.

## Why direct extrapolation fails
1. **Different bottleneck regime.** RTX 6000 batch-1 numbers (`results/latency_bench.json`)
   are flat 6.3-8.6ms across every model/resolution, and FP16 is *slower* than FP32 — the
   signature of launch-overhead-bound execution, not compute-bound. The 6000 is so
   overpowered for these nano models that wall-clock time reflects framework overhead, not
   FLOPs. Nano is weak enough to be compute- or bandwidth-bound instead — a different point
   on the roofline curve entirely.
2. **No tensor cores on Nano (Maxwell).** FP16 gives only ~2x there, vs. 4-8x tensor-core
   speedup on the RTX 6000 (Turing). Any FP16/FP32 ratio measured on the 6000 doesn't carry
   over.
3. **No INT8/DP4A on this Nano generation.** Confirmed elsewhere in this project — INT8
   quantization paths that help newer edge boards are a dead end here.
4. **Memory architecture differs completely.** Nano: unified LPDDR4 (~25.6 GB/s) shared
   with CPU. RTX 6000: dedicated GDDR6 (~672 GB/s). Detection-head/NMS latency doesn't scale
   with a simple TFLOPS ratio when bandwidth is the limiter.

## Recommended method
1. Find a **published Jetson Nano benchmark** for a comparable model (same family/size
   class, ideally same export path — e.g. TensorRT FP16 engine).
   - This project's existing anchor: Qengineering's ~19 fps YOLOv8n on real Nano hardware.
2. Compute the **relative delta** between your model and that anchor model:
   - GFLOPs ratio (at matched imgsz/precision)
   - Param-count ratio (secondary signal)
3. Scale the anchor's measured fps by the inverse of that ratio to project your model's fps.
4. Treat the result as a **projection, not a measurement** — flag it as such in any writeup.

## Fallback if no anchor exists at all
- Compare **FP16 GFLOPs using compute-bound (large-batch) RTX 6000 throughput** as a rough
  proxy for relative model cost — not batch-1 numbers, which are overhead-bound and useless
  here.
- Expect ~2x error or worse given the architecture-generation gap (Maxwell vs. Turing, no
  tensor cores, no INT8).
- Do not report a hard fps number from this fallback without validating on real Nano
  hardware first.

## Checklist before quoting a Nano fps number
- [ ] Anchored to a real published Nano benchmark (not derived from RTX 6000 alone)?
- [ ] Compared at matched precision (FP16) and matched export path (TensorRT engine)?
- [ ] Used compute-bound throughput, not batch-1/launch-overhead-bound numbers, for any
      GFLOPs-ratio scaling?
- [ ] Labeled the result as a projection with an explicit error margin (~2x)?
