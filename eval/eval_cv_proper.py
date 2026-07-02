"""Evaluate all proper CV conditions: baseline/champ/osall x v5/v8/v11.
Uses the SEPARATE test split (val != test).
Run on server: ~/atli/env/bin/python eval_cv_proper.py
"""
from ultralytics import YOLO
from pathlib import Path
import numpy as np

R = Path.home() / "atli"
CV = R / "Merged_CV_proper"
CLASSES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
           "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]

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
        "P": float(np.mean(v.box.p)),
        "R": float(np.mean(v.box.r)),
        **{f"{c}_AP": g(c, v.box.ap50) for c in CLASSES},
        **{f"{c}_P": float(v.box.p[nm[c]]) if c in nm else float("nan") for c in CLASSES},
        **{f"{c}_R": float(v.box.r[nm[c]]) if c in nm else float("nan") for c in CLASSES},
    }

CONDITIONS = [
    ("CVP_base",  "v5",  640,  "base"),
    ("CVP_base",  "v8",  640,  "base"),
    ("CVP_base",  "v11", 640,  "base"),
    ("CVP_champ", "v5",  1280, "champ"),
    ("CVP_champ", "v8",  1280, "champ"),
    ("CVP_champ", "v11", 1280, "champ"),
    ("CVP_osall", "v5",  1280, "osall"),
    ("CVP_osall", "v8",  1280, "osall"),
    ("CVP_osall", "v11", 1280, "osall"),
]

for prefix, mtag, imz, yname in CONDITIONS:
    results = []
    for k in range(5):
        run = f"{prefix}_{mtag}_f{k}"
        yaml_path = CV / f"fold{k}" / f"{yname}.yaml"
        r = ev(run, imz, yaml_path)
        if r:
            results.append(r)
            print(f"  {run}: mAP={r['mAP50']:.3f} P={r['P']:.3f} R={r['R']:.3f} "
                  f"DD={r['Defective_Damper_AP']:.3f} BI={r['Broken_Insulator_AP']:.3f} "
                  f"FI={r['Flashover_Insulator_AP']:.3f} ND={r['Normal_Damper_AP']:.3f} "
                  f"NI={r['Normal_Insulators_AP']:.3f}")
    if results:
        def mean_std(key):
            vals = [r[key] for r in results if not np.isnan(r[key])]
            return np.mean(vals), np.std(vals)
        m, s = mean_std("mAP50")
        mp, _ = mean_std("P")
        mr, _ = mean_std("R")
        dd_m, dd_s = mean_std("Defective_Damper_AP")
        bi_m, bi_s = mean_std("Broken_Insulator_AP")
        fi_m, _ = mean_std("Flashover_Insulator_AP")
        nd_m, _ = mean_std("Normal_Damper_AP")
        ni_m, _ = mean_std("Normal_Insulators_AP")
        # Defect recall
        ddr_m, _ = mean_std("Defective_Damper_R")
        bir_m, _ = mean_std("Broken_Insulator_R")
        fir_m, _ = mean_std("Flashover_Insulator_R")
        ser_m, _ = mean_std("Self-Exploded_Insulator_R")
        print(f"  >>> {prefix} {mtag} ({len(results)} folds): "
              f"mAP={m:.3f}+/-{s:.3f} P={mp:.3f} R={mr:.3f} "
              f"DD={dd_m:.3f}+/-{dd_s:.3f} BI={bi_m:.3f}+/-{bi_s:.3f} "
              f"FI={fi_m:.3f} ND={nd_m:.3f} NI={ni_m:.3f} "
              f"DDr={ddr_m:.3f} BIr={bir_m:.3f} FIr={fir_m:.3f} SEr={ser_m:.3f}")
        print()
