"""Evaluate all CV conditions: baseline/champ/osall × v5/v8/v11.
Run on server: ~/atli/env/bin/python eval_cv_all.py
"""
from ultralytics import YOLO
from pathlib import Path
import numpy as np

R = Path.home() / "atli"
CV = R / "Merged_CV"
CLASSES = ["Birdnest","Broken_Insulator","Defective_Damper","Flashover_Insulator",
           "Normal_Damper","Normal_Insulators","Self-Exploded_Insulator"]

def ev(run, imz, yaml_path):
    w = R / f"runs/{run}_s2/weights/best.pt"
    if not w.exists():
        return None
    v = YOLO(str(w)).val(data=str(yaml_path), split="test", imgsz=imz,
                         batch=8, device=0, verbose=False, plots=False)
    nm = {v.names[c]: i for i, c in enumerate(v.box.ap_class_index)}
    g = lambda c, a: float(a[nm[c]]) if c in nm else float("nan")
    return {
        "mAP50": float(v.box.map50),
        **{c: g(c, v.box.ap50) for c in CLASSES},
        "DD_recall": g("Defective_Damper", v.box.r),
    }

CONDITIONS = [
    # (prefix, model_tag, imgsz, yaml_name)
    ("CVbase",  "v5",  640,  "base"),
    ("CVbase",  "v8",  640,  "base"),
    ("CVbase",  "v11", 640,  "base"),   # already done, re-eval for consistency
    ("CVchamp", "v5",  1280, "champ"),
    ("CVchamp", "v8",  1280, "champ"),
    ("CVchamp", "v11", 1280, "champ"),  # already done
    ("CVosall", "v5",  1280, "osall"),
    ("CVosall", "v8",  1280, "osall"),
    ("CVosall", "v11", 1280, "osall"),
]

for prefix, mtag, imz, yname in CONDITIONS:
    suffix = f"_{mtag}" if mtag != "v11" or prefix == "CVosall" else ""
    # existing v11 runs: CVbase_f0, CVchamp_f0 (no _v11 suffix)
    if mtag == "v11" and prefix in ("CVbase", "CVchamp"):
        suffix = ""
    results = []
    for k in range(5):
        run = f"{prefix}{suffix}_f{k}"
        yaml_path = CV / f"fold{k}" / f"{yname}.yaml"
        r = ev(run, imz, yaml_path)
        if r:
            results.append(r)
            print(f"  {run}: mAP={r['mAP50']:.3f} DD={r['Defective_Damper']:.3f} "
                  f"BI={r['Broken_Insulator']:.3f} ND={r['Normal_Damper']:.3f}")
    if results:
        keys = ["mAP50"] + CLASSES + ["DD_recall"]
        means = {k: np.nanmean([r[k] for r in results]) for k in keys}
        stds = {k: np.nanstd([r[k] for r in results]) for k in keys}
        print(f"  >>> {prefix} {mtag} ({len(results)} folds): "
              f"mAP={means['mAP50']:.3f} "
              f"DD={means['Defective_Damper']:.3f}±{stds['Defective_Damper']:.3f} "
              f"BI={means['Broken_Insulator']:.3f}±{stds['Broken_Insulator']:.3f} "
              f"FI={means['Flashover_Insulator']:.3f} "
              f"ND={means['Normal_Damper']:.3f}")
        print()
