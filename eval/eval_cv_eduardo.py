#!/usr/bin/env python3
"""Collect per-class P/R/AP@0.5 for the eduardos CV sweep (4 conditions x 5 folds).
Re-runs `val` on each fold's stage-2 best.pt on the held-out test split, then
aggregates mean+/-std across folds. Writes ~/atli/eval_eduardo_results.json.

Run on server: ~/atli/env/bin/python eval_cv_eduardo.py
"""
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

ROOT = Path.home() / "atli"
CLASSES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
           "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
# key, run-name prefix, task, cv dir, yaml, imgsz
CONDS = [
    ("baseline",   "EDU_base_v11",       "detect", "CV_eduardo_det", "base.yaml",        640),
    ("obbdeg45",   "EDU_obbdeg45_v11",   "obb",    "CV_eduardo_obb", "osall.yaml",       1280),
    ("obbref",     "EDU_obbref_v11",     "obb",    "CV_eduardo_obb", "osall.yaml",       1280),
    ("champosall", "EDU_champosall_v11", "detect", "CV_eduardo_det", "osall.yaml",       1280),
    ("obbdeg20",   "EDU_obbdeg20_v11",   "obb",    "CV_eduardo_obb", "osall.yaml",       1280),
    ("cplid_obb",  "EDU_cplidobb_v11",   "obb",    "CV_eduardo_obb", "osall_cplid.yaml", 1280),
    ("cplid_det",  "EDU_cpliddet_v11",   "detect", "CV_eduardo_det", "osall_cplid.yaml", 1280),
]
DEVICE = 3


def ev(name, task, cvdir, yamlname, imz, fold):
    w = ROOT / "runs" / f"{name}_f{fold}_s2" / "weights" / "best.pt"
    ya = ROOT / cvdir / f"fold{fold}" / yamlname
    m = YOLO(str(w))
    r = m.val(data=str(ya), split="test", imgsz=imz, device=DEVICE,
              verbose=False, save_json=False, plots=False)
    met = r.box   # both DetMetrics and OBBMetrics expose per-class results under .box
    idx = {r.names[c]: i for i, c in enumerate(met.ap_class_index)}
    out = {"mAP50": float(met.map50), "P": float(met.mp), "R": float(met.mr)}
    for c in CLASSES:
        if c in idx:
            out[f"{c}_AP"] = float(met.ap50[idx[c]])
            out[f"{c}_P"] = float(met.p[idx[c]])
            out[f"{c}_R"] = float(met.r[idx[c]])
        else:
            out[f"{c}_AP"] = out[f"{c}_P"] = out[f"{c}_R"] = float("nan")
    return out


def ms(folds, field):
    vals = [f[field] for f in folds if not np.isnan(f.get(field, float("nan")))]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)


results = {}
for key, name, task, cvdir, yamlname, imz in CONDS:
    folds = [ev(name, task, cvdir, yamlname, imz, k) for k in range(5)]
    results[key] = {"task": task, "folds": folds}
    print(f"\n=== {key} ({task}, {imz}) ===")
    mm, ms_ = ms(folds, "mAP50"); pp = ms(folds, "P"); rr = ms(folds, "R")
    print(f"  overall  mAP50 {mm:.3f}+/-{ms_:.3f}   P {pp[0]:.3f}   R {rr[0]:.3f}")
    for c in CLASSES:
        ap = ms(folds, f"{c}_AP"); p = ms(folds, f"{c}_P"); r_ = ms(folds, f"{c}_R")
        print(f"  {c:24s} P {p[0]:.3f}+/-{p[1]:.3f}  R {r_[0]:.3f}+/-{r_[1]:.3f}  AP {ap[0]:.3f}+/-{ap[1]:.3f}")

json.dump(results, open(ROOT / "eval_eduardo_results.json", "w"), indent=1)
print("\nEVAL_DONE ->", ROOT / "eval_eduardo_results.json")
