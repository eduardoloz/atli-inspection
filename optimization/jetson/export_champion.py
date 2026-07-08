#!/usr/bin/env python3
"""Export the champion YOLOv11n to deployment formats for Jetson Nano.

Produces, per requested imgsz:
  - ONNX (opset 12, simplified, static batch=1)  -> portable to the Nano,
    where the FP16 TensorRT engine MUST be built on-device with trtexec
    (TensorRT engines are not portable across GPU architectures — an engine
    built on a Quadro RTX 6000 will not run on the Nano's Maxwell GPU).
  - optionally a TensorRT engine for the *local* GPU (--engine), useful only
    for functional parity testing, NOT for Nano latency numbers.

Usage (UNLV server):
  CUDA_VISIBLE_DEVICES=0 ~/atli/env/bin/python export_champion.py \
      --weights ~/atli/runs/HROaugnc_v11_s0_s2/weights/best.pt \
      --imgsz 640 960 1280 --outdir ~/atli/export_jetson

Notes:
  - opset 12 is chosen for compatibility with TensorRT 8.2.x, the last TRT
    release available on Jetson Nano (JetPack 4.6.x).
  - dynamic=False / batch=1: the Nano deployment is single-stream batch-1;
    static shapes let TRT pick the fastest tactics.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--outdir", default="export_jetson")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--engine", action="store_true",
                    help="also build a TensorRT FP16 engine for the LOCAL gpu "
                         "(functional test only; NOT a Nano artifact)")
    args = ap.parse_args()

    from ultralytics import YOLO

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    for sz in args.imgsz:
        model = YOLO(args.weights)  # reload each time; export mutates model
        print(f"\n=== Exporting ONNX @ imgsz={sz} (opset {args.opset}) ===")
        onnx_path = model.export(
            format="onnx", imgsz=sz, opset=args.opset,
            simplify=True, dynamic=False, batch=1, device=0,
        )
        dst = outdir / f"champ_v11n_{sz}.onnx"
        shutil.copy(onnx_path, dst)
        print(f"saved {dst}")

        if args.engine:
            model = YOLO(args.weights)
            print(f"=== Building local TensorRT FP16 engine @ {sz} (functional test only) ===")
            eng_path = model.export(
                format="engine", imgsz=sz, half=True,
                dynamic=False, batch=1, device=0, workspace=4,
            )
            dst = outdir / f"champ_v11n_{sz}_LOCALGPU.engine"
            shutil.copy(eng_path, dst)
            print(f"saved {dst}  (engine is bound to the local GPU arch)")

    print("\nDone. Copy the .onnx files to the Jetson Nano and build engines "
          "on-device with build_engine_nano.sh")


if __name__ == "__main__":
    main()
