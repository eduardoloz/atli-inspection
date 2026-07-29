#!/usr/bin/env python3
"""End-to-end batch-1 FPS benchmark on real Jetson hardware.

Companion to results/jetson_nano_fps_approximation.md, which explicitly warns
against extrapolating Jetson fps from the RTX 6000 numbers (different
bottleneck regime, no tensor cores / INT8 DP4A on the original Maxwell Nano).
A Jetson Orin Nano (Ampere: tensor cores + INT8 DP4A) sidesteps most of those
caveats, so this measures real fps instead of projecting it.

Unlike eval/bench_latency.py (raw fused forward pass only, no pre/post), this
times the FULL predict() pipeline — letterbox, normalize, forward, NMS/OBB
decode — since that's what matters for a deployed fps number.

Run ON the Jetson (needs ultralytics + torch/torchvision matched to the
device's JetPack release, and TensorRT for the "engine" format). Before
running: `sudo nvpmodel -m 0 && sudo jetson_clocks` (or select "Super" mode
if this board supports it) — batch-1 latency is very sensitive to the
default reduced-power profile, and that's a free win the model can't buy
back. Run `tegrastats` alongside to confirm clocks stay pinned (no thermal
throttle skewing the numbers).

    python3 bench_jetson_fps.py --weights best.pt --imgsz 640 768 \
        --formats pt engine --precision fp16 --source sample_images/

INT8 (Orin Nano has DP4A/INT8 tensor cores, unlike the original Maxwell
Nano) needs calibration images — point --int8-data at a dataset yaml (e.g.
a copy of the eduardo CV fold's osall.yaml, train split is used for
calibration):

    python3 bench_jetson_fps.py --weights best.pt --formats engine \
        --precision int8 --int8-data osall.yaml

--conf/--max-det matter here beyond accuracy: fewer candidate boxes means
less NMS work. Worth profiling at both the default conf and the low-conf
recall-boosting operating point used for defect screening (~0.1), since
that op point trades fps for recall as well as precision.

Writes/updates jetson_fps_results.json next to this script (incremental —
safe to re-run with more formats/sizes/precisions later).
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def find_source(source):
    p = Path(source)
    if p.is_dir():
        imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not imgs:
            raise SystemExit(f"no images found in {p}")
        return imgs[0]
    if p.is_file():
        return p
    raise SystemExit(f"--source {source} not found — point it at an image or a folder of images")


def get_model(weights, fmt, imgsz, precision, device, workspace, int8_data):
    """Return a loaded YOLO model for the given export format, exporting first if needed."""
    if fmt == "pt":
        return YOLO(str(weights))
    tag = {"fp32": "", "fp16": "_fp16", "int8": "_int8"}[precision]
    exported = weights.with_name(f"{weights.stem}_{fmt}_{imgsz}{tag}").with_suffix(
        {"onnx": ".onnx", "engine": ".engine"}[fmt])
    if not exported.exists():
        kwargs = dict(format=fmt, imgsz=imgsz, device=device, simplify=True,
                      dynamic=False, workspace=workspace)
        if precision == "fp16":
            kwargs["half"] = True
        elif precision == "int8":
            if fmt != "engine":
                raise SystemExit("--precision int8 requires --formats engine (TensorRT PTQ)")
            if not int8_data:
                raise SystemExit("--precision int8 requires --int8-data <dataset.yaml> for calibration")
            kwargs["int8"] = True
            kwargs["data"] = int8_data
        print(f"[export] {fmt}/{precision} (imgsz={imgsz}) -> {exported.name} ...")
        out = YOLO(str(weights)).export(**kwargs)
        Path(out).rename(exported)
    return YOLO(str(exported))


def bench_one(model, image, imgsz, conf, max_det, warmup, iters):
    kw = dict(source=str(image), imgsz=imgsz, conf=conf, max_det=max_det, verbose=False)
    for _ in range(warmup):
        model.predict(**kw)
    t0 = time.perf_counter()
    for _ in range(iters):
        model.predict(**kw)
    elapsed = time.perf_counter() - t0
    return iters / elapsed, (elapsed / iters) * 1000  # fps, ms/img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path, help="path to a .pt checkpoint (e.g. best.pt)")
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640], help="one or more inference resolutions")
    ap.add_argument("--formats", nargs="+", default=["pt", "engine"], choices=["pt", "onnx", "engine"])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                     help="fp32/fp16 apply to onnx+engine; int8 requires --formats engine + --int8-data")
    ap.add_argument("--int8-data", default=None, help="dataset yaml for TensorRT INT8 PTQ calibration")
    ap.add_argument("--workspace", type=float, default=4, help="TensorRT builder workspace, GiB")
    ap.add_argument("--conf", type=float, default=0.25, help="NMS confidence threshold (try 0.1 for the recall-screening op point)")
    ap.add_argument("--max-det", type=int, default=300, dest="max_det")
    ap.add_argument("--source", default=str(HERE / "sample_images"), help="image or folder of images to run on")
    ap.add_argument("--device", default=0, help="CUDA device index (Jetson has one GPU: 0)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", type=Path, default=HERE / "jetson_fps_results.json")
    args = ap.parse_args()

    image = find_source(args.source)
    print(f"[bench] using sample image: {image}")

    results = json.load(open(args.out)) if args.out.exists() else {}
    key = args.weights.stem
    results.setdefault(key, {})

    for fmt in args.formats:
        for imgsz in args.imgsz:
            precision = "fp32" if fmt == "pt" else args.precision
            entry_key = f"{fmt}_{imgsz}_{precision}_conf{args.conf}"
            print(f"\n=== {key} / {entry_key} ===")
            try:
                model = get_model(args.weights, fmt, imgsz, precision, args.device,
                                   args.workspace, args.int8_data)
                fps, ms = bench_one(model, image, imgsz, args.conf, args.max_det,
                                     args.warmup, args.iters)
            except Exception as e:
                print(f"  FAILED: {e}")
                results[key][entry_key] = {"error": str(e)}
            else:
                print(f"  {fps:.2f} fps  ({ms:.2f} ms/img, end-to-end incl. pre/post)")
                results[key][entry_key] = {"fps": round(fps, 2), "ms_per_img": round(ms, 2)}
            json.dump(results, open(args.out, "w"), indent=1)  # incremental save

    print("\nBENCH_DONE ->", args.out)


if __name__ == "__main__":
    main()
