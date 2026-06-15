#!/usr/bin/env python3
"""Collect the Universe-damper benchmark results.

1. Per-epoch validation mAP curves: concatenates runs/U*_s{1,2}/results.csv into one
   long-format CSV  -> ~/atli/universe_bench/epoch_map.csv
2. Final test-split metrics incl. per-class AP (the number that matters is
   Defective_Damper): runs `model.val(split='test')` on each finished *_s2 best.pt
   -> ~/atli/universe_bench/test_summary.csv

Run on the server:  ~/atli/env/bin/python collect_universe_results.py [--no-test]
Safe to run while the sweep is still going - it collects whatever is finished.
"""
import argparse
import csv
from pathlib import Path

ROOT = Path.home() / "atli"
RUNS = ROOT / "runs"
BENCH = ROOT / "universe_bench"
DATA = {
    "Ufull": ROOT / "Merged_Universe_Stratified" / "merged_universe.yaml",
    "Uto": ROOT / "Merged_Universe_TrainOnly" / "merged_universe_trainonly.yaml",
    "Uw": ROOT / "Merged_Universe_WangboOnly" / "merged_universe_wangbo.yaml",
    "Ucap1": ROOT / "Merged_Universe_Ucap1" / "merged_universe_ucap1.yaml",
    "Ucap2": ROOT / "Merged_Universe_Ucap2" / "merged_universe_ucap2.yaml",
    "B": ROOT / "Merged_Dataset_Stratified" / "merged_stratified.yaml",
}


def data_for(run_name):
    for k in ("Ufull", "Uto", "Uw", "Ucap1", "Ucap2"):
        if run_name.startswith(k):
            return DATA[k]
    return DATA["B"]

ap = argparse.ArgumentParser()
ap.add_argument("--no-test", action="store_true", help="skip test-split evaluation")
args = ap.parse_args()
BENCH.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. epoch curves
rows = []
for rc in sorted(RUNS.glob("[A-Z]*_s[12]/results.csv")):
    run = rc.parent.name                      # e.g. Uto_v8_300p100_s1
    base, stage = run.rsplit("_", 1)
    with open(rc) as f:
        for r in csv.DictReader(f):
            g = lambda *keys: next((r[k].strip() for k in keys if k in r and r[k].strip()), "")
            rows.append({
                "run": base, "stage": stage,
                "epoch": g("epoch", "                 epoch"),
                "mAP50": g("metrics/mAP50(B)"),
                "mAP50_95": g("metrics/mAP50-95(B)"),
                "precision": g("metrics/precision(B)"),
                "recall": g("metrics/recall(B)"),
            })
out1 = BENCH / "epoch_map.csv"
if rows:
    with open(out1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
print(f"[epochs] {len(rows)} rows from {len(set(r['run']+r['stage'] for r in rows))} "
      f"stage-runs -> {out1}")

# ---------------------------------------------------------------- 2. test-split eval
if not args.no_test:
    from ultralytics import YOLO
    summary = []
    for best in sorted(RUNS.glob("[A-Z]*_s2/weights/best.pt")):
        base = best.parent.parent.name.rsplit("_s2", 1)[0]
        data_yaml = data_for(base)
        imgsz = 1280 if base.startswith("HR") else 640
        print(f"[test] {base} (imgsz={imgsz}) ...")
        m = YOLO(str(best))
        v = m.val(data=str(data_yaml), split="test", imgsz=imgsz, batch=16,
                  verbose=False, project=str(BENCH), name=f"testval_{base}", exist_ok=True)
        names = [v.names[i] for i in sorted(v.names)]
        cls_names = [v.names[c] for c in v.box.ap_class_index]
        per_class = dict(zip(cls_names, v.box.ap50))
        per_p = dict(zip(cls_names, v.box.p))
        per_r = dict(zip(cls_names, v.box.r))
        dd = "Defective_Damper"
        summary.append({
            "run": base,
            "test_P": round(float(v.box.mp), 4),
            "test_R": round(float(v.box.mr), 4),
            "test_mAP50": round(float(v.box.map50), 4),
            "test_mAP50_95": round(float(v.box.map), 4),
            f"P_{dd}": round(float(per_p.get(dd, float("nan"))), 4),
            f"R_{dd}": round(float(per_r.get(dd, float("nan"))), 4),
            **{f"AP50_{n}": round(float(per_class.get(n, float('nan'))), 4) for n in names},
        })
    if summary:
        out2 = BENCH / "test_summary.csv"
        with open(out2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader(); w.writerows(summary)
        print(f"[test] {len(summary)} models -> {out2}")
        for s in summary:
            print(f"  {s['run']:22s} mAP50={s['test_mAP50']:.3f}  "
                  f"DefDamper={s.get('AP50_Defective_Damper', 'n/a')}")
    else:
        print("[test] no finished *_s2 models yet")
print("COLLECT_DONE")
