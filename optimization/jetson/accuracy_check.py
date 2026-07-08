#!/usr/bin/env python3
"""Accuracy check: exported model (.onnx/.engine) vs original PyTorch .pt.

Two levels:
  1. `val` — run Ultralytics val on the same data yaml/split with both models
     and compare mAP@0.5 overall + per class (catches precision-loss and
     letterbox/NMS mismatches end-to-end).
  2. `--parity` — feed identical images through both models and compare the
     final detections (boxes/conf/cls) image-by-image.

Usage (UNLV server):
  CUDA_VISIBLE_DEVICES=0 ~/atli/env/bin/python accuracy_check.py \
      --pt ~/atli/runs/HROaugnc_v11_s0_s2/weights/best.pt \
      --exported ~/atli/export_jetson/champ_v11n_1280.onnx \
      --data ~/atli/Merged_Dataset_Stratified/data.yaml --imgsz 1280 --split test
"""
import argparse


def run_val(model_path, data, imgsz, split, device):
    from ultralytics import YOLO
    model = YOLO(model_path, task="detect")
    m = model.val(data=data, imgsz=imgsz, split=split, device=device,
                  batch=1, verbose=False, plots=False)
    per_class = {m.names[i]: float(ap) for i, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return float(m.box.map50), float(m.box.map), per_class


def parity(pt_path, exp_path, data, imgsz, device, n=20, iou_tol=0.9):
    """Compare final detections of pt vs exported model on n val images."""
    from pathlib import Path
    import yaml
    from ultralytics import YOLO

    d = yaml.safe_load(open(Path(data).expanduser()))
    root = Path(d.get("path", Path(data).expanduser().parent))
    test_dir = root / d.get("test", d.get("val"))
    imgs = sorted([p for p in Path(test_dir).rglob("*") if p.suffix.lower()
                   in (".jpg", ".jpeg", ".png")])[:n]

    a, b = YOLO(pt_path), YOLO(exp_path, task="detect")
    mismatches = 0
    for p in imgs:
        ra = a.predict(str(p), imgsz=imgsz, device=device, verbose=False, conf=0.25)[0]
        rb = b.predict(str(p), imgsz=imgsz, device=device, verbose=False, conf=0.25)[0]
        na, nb = len(ra.boxes), len(rb.boxes)
        dconf = 0.0
        if na == nb and na > 0:
            dconf = float((ra.boxes.conf.cpu() - rb.boxes.conf.cpu()).abs().max())
        if na != nb or dconf > 0.02:
            mismatches += 1
            print(f"  DIFF {p.name}: pt={na} boxes, exported={nb} boxes, "
                  f"max|dconf|={dconf:.4f}")
    print(f"parity: {len(imgs) - mismatches}/{len(imgs)} images match "
          f"(same box count, conf within 0.02)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--exported", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=0)
    ap.add_argument("--exported-device", default=None,
                    help="device for the exported model (e.g. cpu if the "
                         "onnxruntime CUDA EP is unavailable); default: --device")
    ap.add_argument("--parity", action="store_true", help="also run per-image parity check")
    args = ap.parse_args()
    exp_dev = args.exported_device if args.exported_device is not None else args.device

    print(f"== val: PyTorch {args.pt} @ {args.imgsz} ==")
    pt50, pt5095, pt_cls = run_val(args.pt, args.data, args.imgsz, args.split, args.device)
    print(f"== val: exported {args.exported} @ {args.imgsz} (device={exp_dev}) ==")
    ex50, ex5095, ex_cls = run_val(args.exported, args.data, args.imgsz, args.split, exp_dev)

    print(f"\n{'':28} {'pytorch':>9} {'exported':>9} {'delta':>8}")
    print(f"{'mAP@0.5':28} {pt50:>9.4f} {ex50:>9.4f} {ex50 - pt50:>+8.4f}")
    print(f"{'mAP@0.5:0.95':28} {pt5095:>9.4f} {ex5095:>9.4f} {ex5095 - pt5095:>+8.4f}")
    for c in sorted(pt_cls):
        e = ex_cls.get(c, float("nan"))
        print(f"{c:28} {pt_cls[c]:>9.4f} {e:>9.4f} {e - pt_cls[c]:>+8.4f}")

    tol = 0.005
    verdict = "PASS" if abs(ex50 - pt50) <= tol else "FAIL"
    print(f"\nexport fidelity ({tol} mAP@0.5 tolerance): {verdict}")

    if args.parity:
        print("\n== per-image parity ==")
        parity(args.pt, args.exported, args.data, args.imgsz, args.device)


if __name__ == "__main__":
    main()
