#!/usr/bin/env python3
"""Batch-1 fused-model GPU latency for the frontier checkpoints at 640..1280.

Raw forward pass (no pre/post-processing), FP32 and FP16, 30 warmup + 100 timed
iters, torch.cuda.synchronize around the timed block. Merges into
~/atli/latency_bench.json (FNET pass separate, as with the other collectors):
    BENCH_DEV=3 python bench_latency.py
    FNET=1 PYTHONPATH=~/atli/modpatch BENCH_DEV=3 python bench_latency.py
Run on an IDLE GPU only — contention invalidates the numbers.
"""
import json
import os
import time
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path.home() / "atli"
MODELS = [
    ("v11_deg15", "EDU_deg15_v11_f0_s2"),
    ("v11_ghost", "EDU_ghost_deg15_f0_s2"),
    ("v11_dws",   "EDU_dws_deg15_f0_s2"),
    ("v8_deg15",  "EDU_v8deg15_f0_s2"),
    ("v8_ghost",  "EDU_v8ghost_deg15_f0_s2"),
    ("v8_dws",    "EDU_v8dws_deg15_f0_s2"),
]
if os.environ.get("FNET") == "1":
    MODELS = [("v11_fnet", "EDU_fnet_deg15_f0_s2"),
              ("v8_fnet",  "EDU_v8fnet_deg15_f0_s2")]
DEV = f"cuda:{os.environ.get('BENCH_DEV', '3')}"
SIZES = [640, 768, 960, 1280]

def bench(model, imz, half):
    x = torch.zeros(1, 3, imz, imz, device=DEV)
    m = model.half() if half else model.float()
    x = x.half() if half else x
    with torch.no_grad():
        for _ in range(30):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            m(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 10  # ms/img

out_path = ROOT / "latency_bench.json"
results = json.load(open(out_path)) if out_path.exists() else {}
for key, run in MODELS:
    w = ROOT / "runs" / run / "weights" / "best.pt"
    if not w.exists():
        print(f"{key:12s} SKIP (no weights)")
        continue
    model = YOLO(str(w)).model.fuse().eval().to(DEV)
    entry = {}
    for imz in SIZES:
        for half in (False, True):
            ms = bench(model, imz, half)
            entry[f"{imz}_{'fp16' if half else 'fp32'}_ms"] = round(ms, 2)
    results[key] = entry
    json.dump(results, open(out_path, "w"), indent=1)
    print(key, entry)
print("BENCH_DONE ->", out_path)
