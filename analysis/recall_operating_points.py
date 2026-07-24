#!/usr/bin/env python3
"""Recall/precision vs confidence-threshold operating points for defect classes.

One val per fold (conf=0.001), then reads the per-class R-conf / P-conf curves
(metrics.box.r_curve / p_curve, (nc, 1000) over confidence 0..1) to report the
deployable operating points — how much defect recall a lower conf threshold
buys and what precision it costs. Writes ~/atli/recall_operating_points.json.

Run on server:  EVAL_DEV=4 python recall_operating_points.py
"""
import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO

ROOT = Path.home() / "atli"
CLASSES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
           "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
CONDS = [  # key, run prefix, imgsz
    ("mixup640",   "EDU_mixup640",  640),
    ("deg15_1280", "EDU_deg15_v11", 1280),
]
CONFS = [0.05, 0.10, 0.25, 0.50]
DEV = os.environ.get("EVAL_DEV", "cpu")

out = {}
for key, prefix, imz in CONDS:
    folds = []
    for k in range(5):
        w = ROOT / "runs" / f"{prefix}_f{k}_s2" / "weights" / "best.pt"
        m = YOLO(str(w))
        r = m.val(data=str(ROOT / f"CV_eduardo_obb/fold{k}/osall.yaml"), split="test",
                  imgsz=imz, device=DEV, batch=8, verbose=False, plots=False)
        met = r.box
        idx = {r.names[c]: i for i, c in enumerate(met.ap_class_index)}
        d = {}
        for cname in CLASSES:
            if cname not in idx:
                continue
            i = idx[cname]
            for cf in CONFS:
                j = int(cf * 999)
                d[f"{cname}@{cf}"] = {"R": float(met.r_curve[i, j]),
                                      "P": float(met.p_curve[i, j])}
        folds.append(d)
    out[key] = folds
    print(f"=== {key} ===")
    for cname in CLASSES:
        row = []
        for cf in CONFS:
            rs = [f[f"{cname}@{cf}"]["R"] for f in folds if f"{cname}@{cf}" in f]
            ps = [f[f"{cname}@{cf}"]["P"] for f in folds if f"{cname}@{cf}" in f]
            if rs:
                row.append(f"conf{cf}: R {np.mean(rs):.3f}/P {np.mean(ps):.3f}")
        print(f"  {cname:24s} " + "  ".join(row))

json.dump(out, open(ROOT / "recall_operating_points.json", "w"), indent=1)
print("OPPOINTS_DONE")
