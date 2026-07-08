#!/usr/bin/env python3
"""Batch-1 latency harness for .pt / .onnx / .engine models.

Measures pure model latency (pre/inference/post as reported by Ultralytics,
plus a raw-forward timing loop) at a given imgsz. Works on any CUDA machine
and on a Jetson (JetPack's python + ultralytics).

IMPORTANT: numbers are only meaningful FOR THE MACHINE THEY RUN ON.
A Quadro RTX 6000 (16.3 TFLOPS FP32) measurement is ~35-70x a Jetson Nano
(472 GFLOPS FP16) — use RTX runs only for functional/parity checks and for
*relative* scaling across resolutions, never as Nano fps predictions.

Usage:
  CUDA_VISIBLE_DEVICES=0 python latency_harness.py --model champ_v11n_640.onnx \
      --imgsz 640 --iters 200 --warmup 30
  python latency_harness.py --model best.pt --imgsz 1280 640 --iters 100
"""
import argparse
import platform
import statistics
import time

import numpy as np


def get_device_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return platform.processor() or "cpu"


def bench(model_path, imgsz, iters, warmup, device):
    from ultralytics import YOLO
    model = YOLO(model_path, task="detect")
    img = (np.random.rand(imgsz, imgsz, 3) * 255).astype("uint8")

    # warmup
    for _ in range(warmup):
        model.predict(img, imgsz=imgsz, device=device, verbose=False)

    total, pre, inf, post = [], [], [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, device=device, verbose=False)[0]
        total.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])

    def stats(xs):
        xs = sorted(xs)
        return (statistics.mean(xs), statistics.median(xs),
                xs[int(0.95 * len(xs)) - 1])

    m_t, md_t, p95_t = stats(total)
    m_i, _, _ = stats(inf)
    return dict(mean_total_ms=m_t, median_total_ms=md_t, p95_total_ms=p95_t,
                mean_pre_ms=statistics.mean(pre), mean_inf_ms=m_i,
                mean_post_ms=statistics.mean(post),
                fps_total=1000.0 / m_t, fps_inference_only=1000.0 / m_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt, .onnx, or .engine")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    dev = get_device_name()
    print(f"# device: {dev}")
    print("# NOTE: only valid for this device — a workstation GPU number is NOT a Jetson Nano number.")
    print(f"{'imgsz':>6} {'mean_ms':>8} {'p95_ms':>8} {'pre_ms':>7} {'inf_ms':>7} "
          f"{'post_ms':>8} {'fps_e2e':>8} {'fps_inf':>8}")
    for sz in args.imgsz:
        s = bench(args.model, sz, args.iters, args.warmup, args.device)
        print(f"{sz:>6} {s['mean_total_ms']:>8.2f} {s['p95_total_ms']:>8.2f} "
              f"{s['mean_pre_ms']:>7.2f} {s['mean_inf_ms']:>7.2f} "
              f"{s['mean_post_ms']:>8.2f} {s['fps_total']:>8.1f} "
              f"{s['fps_inference_only']:>8.1f}")


if __name__ == "__main__":
    main()
