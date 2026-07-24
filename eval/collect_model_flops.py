#!/usr/bin/env python3
"""Params + GFLOPs for every model on the accuracy/size frontier (fold-0 best.pt).

GFLOPs reported at 640 and at each condition's native eval imgsz (FLOPs scale
~quadratically with imgsz). Merges into ~/atli/model_flops.json so the FNET pass
can be run separately:
    python collect_model_flops.py                      # everything except fnet
    FNET=1 PYTHONPATH=~/atli/modpatch python collect_model_flops.py   # fnet only
(the FNET sitecustomize rebinds C3Ghost, which corrupts REAL C3Ghost loading —
never mix the passes).
"""
import json
import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path.home() / "atli"
# key, fold-0 run name, native imgsz
CONDS = [
    ("v11_deg15_obb",   "EDU_deg15_v11_f0_s2",      1280),
    ("v8_deg15_obb",    "EDU_v8deg15_f0_s2",        1280),
    ("v11_champ_det",   "EDU_champosall_v11_f0_s2", 1280),
    ("v8_champ_det",    "EDU_v8champ_f0_s2",        1280),
    ("v5_champ_det",    "EDU_v5champ_f0_s2",        1280),
    ("v11_base_det",    "EDU_base_v11_f0_s2",       640),
    ("v11_ghost_obb",   "EDU_ghost_deg15_f0_s2",    1280),
    ("v11_dws_obb",     "EDU_dws_deg15_f0_s2",      1280),
    ("v11_p2_obb",      "EDU_p2_640_f0_s2",         640),
    ("v8_ghost_obb",    "EDU_v8ghost_deg15_f0_s2",  1280),
    ("v8_dws_obb",      "EDU_v8dws_deg15_f0_s2",    1280),
]
if os.environ.get("FNET") == "1":
    CONDS = [
        ("v11_fnet_obb", "EDU_fnet_deg15_f0_s2",   1280),
        ("v8_fnet_obb",  "EDU_v8fnet_deg15_f0_s2", 1280),
    ]

out_path = ROOT / "model_flops.json"
results = json.load(open(out_path)) if out_path.exists() else {}
for key, run, imz in CONDS:
    w = ROOT / "runs" / run / "weights" / "best.pt"
    if not w.exists():
        print(f"{key:16s} SKIP (no weights: {run})")
        continue
    m = YOLO(str(w)).model
    params = get_num_params(m)
    g640 = get_flops(m, imgsz=640)
    gnat = get_flops(m, imgsz=imz)
    results[key] = {"run": run, "params_M": round(params / 1e6, 3),
                    "GFLOPs_640": round(g640, 2),
                    "native_imgsz": imz, "GFLOPs_native": round(gnat, 2)}
    print(f"{key:16s} {params/1e6:5.2f}M  {g640:6.2f} GFLOPs@640  {gnat:7.2f} @{imz}")
    json.dump(results, open(out_path, "w"), indent=1)
print("FLOPS_DONE ->", out_path)
