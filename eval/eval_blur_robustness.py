"""Blur-robustness head-to-head on motion-blurred test images.

For each fold: copy the test split, apply 7px horizontal motion blur to every
image (same perturbation as the 2026-07-08 fragility test), val each condition's
stage-2 best.pt on it. Incremental: conditions already in the output JSON are
skipped, so new conditions can be appended to CONDS and re-run cheaply.
Writes ~/atli/eval_blur_robustness.json.
Run on server: env EVAL_DEV=4 ~/atli/env/bin/python eval_blur_robustness.py
"""
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml as pyyaml
from ultralytics import YOLO

ROOT = Path.home() / "atli"
OBB = ROOT / "CV_eduardo_obb"
WORK = ROOT / "blurtest_eduardo"
DEV = int(os.environ.get("EVAL_DEV", 4))
K = 7  # 7px horizontal motion blur
kernel = np.zeros((K, K), np.float32)
kernel[K // 2, :] = 1.0 / K

# cond -> (run-name prefix, val imgsz)
CONDS = {
    "deg15": ("EDU_deg15_v11", 1280),
    "blurdeg15": ("EDU_blurdeg15_v11", 1280),
    # mixup-realism probe: if mixup's gain were realism-driven it should lift
    # blurred-test mAP like blur-aug did; if it's boundary regularization,
    # the gain over deg15 should stay modest
    "mix15_1280": ("EDU_mix15_1280", 1280),
}

out_path = ROOT / "eval_blur_robustness.json"
out = json.load(open(out_path)) if out_path.exists() else {}
todo = {c: v for c, v in CONDS.items() if c not in out}
results = {c: [] for c in todo}

for f in range(5):
    fold = OBB / f"fold{f}"
    bdir = WORK / f"fold{f}"
    if not (bdir / "test" / "images").exists():
        (bdir / "test").mkdir(parents=True, exist_ok=True)
        shutil.copytree(fold / "test" / "labels", bdir / "test" / "labels",
                        dirs_exist_ok=True)
        (bdir / "test" / "images").mkdir(exist_ok=True)
        for img in (fold / "test" / "images").iterdir():
            im = cv2.imread(str(img))
            cv2.imwrite(str(bdir / "test" / "images" / img.name),
                        cv2.filter2D(im, -1, kernel))
    # val yaml pointing train/val at originals, test at blurred copies
    cfg = pyyaml.safe_load((fold / "osall.yaml").read_text())
    cfg["path"] = str(fold)
    cfg["test"] = str(bdir / "test" / "images")
    ya = bdir / "test_blur.yaml"
    ya.write_text(pyyaml.safe_dump(cfg))
    for cond, (prefix, imz) in todo.items():
        w = ROOT / "runs" / f"{prefix}_f{f}_s2" / "weights" / "best.pt"
        r = YOLO(str(w)).val(data=str(ya), split="test", imgsz=imz, batch=8,
                             device=DEV, verbose=False, plots=False)
        results[cond].append(float(r.box.map50))
        print(f"fold{f} {cond}: blurred-test mAP50 {r.box.map50:.3f}", flush=True)

for c, v in results.items():
    out[c] = {"folds": v, "mean": float(np.mean(v)), "std": float(np.std(v))}
json.dump(out, open(out_path, "w"), indent=1)
for c, s in out.items():
    print(f"{c}: blurred-test mAP50 {s['mean']:.3f} +/- {s['std']:.3f}")
print("ROBUSTNESS_EVAL_DONE")
