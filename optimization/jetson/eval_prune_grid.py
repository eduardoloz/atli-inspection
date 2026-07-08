#!/usr/bin/env python3
"""Evaluate the prune grid: val every runs_prune/pr*_s*/weights/best.pt on the
test split @768, report per-run and per-condition (3-seed mean/std) mAP@0.5,
Defective_Damper AP, GMACs and params. Writes prune_grid_results.csv.

Usage: CUDA_VISIBLE_DEVICES=<gpu> ~/atli/env_jetson/bin/python \
    eval_prune_grid.py --project ~/atli/runs_prune --data ~/atli/ATLI_noCPLID_OS3/data.yaml
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import torch


def gmacs(weights, imgsz):
    import torch_pruning as tp
    from ultralytics import YOLO
    m = YOLO(weights).model.eval().float()
    macs, params = tp.utils.count_ops_and_params(m, torch.randn(1, 3, imgsz, imgsz))
    return macs / 1e9, params / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="runs_prune")
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", default="prune_grid_results.csv")
    args = ap.parse_args()

    from ultralytics import YOLO

    rows = []
    for best in sorted(Path(args.project).expanduser().glob("pr*_s*/weights/best.pt")):
        name = best.parts[-3]
        cond, seed = name.rsplit("_s", 1)
        g, p = gmacs(str(best), args.imgsz)
        m = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, split="test",
                                batch=8, device=args.device, verbose=False, plots=False)
        per = {m.names[i]: float(a) for i, a in zip(m.box.ap_class_index, m.box.ap50)}
        dd = per.get("Defective_Damper", float("nan"))
        rows.append(dict(run=name, cond=cond, seed=int(seed), map50=float(m.box.map50),
                         map5095=float(m.box.map), dd_ap50=dd, gmacs=round(g, 2),
                         params_m=round(p, 2)))
        print(f"{name}: mAP50 {rows[-1]['map50']:.4f}  DD {dd:.4f}  "
              f"{g:.2f} GMACs  {p:.2f}M params")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("\n== 3-seed condition means ==")
    print(f"{'cond':8} {'n':>2} {'mAP50':>14} {'DD AP50':>14} {'GMACs':>6}")
    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    for cond, rs in sorted(by.items()):
        m50 = [r["map50"] for r in rs]
        dd = [r["dd_ap50"] for r in rs]
        sd = statistics.stdev if len(rs) > 1 else lambda _: 0.0
        print(f"{cond:8} {len(rs):>2} {statistics.mean(m50):.4f} ± {sd(m50):.4f} "
              f"{statistics.mean(dd):.4f} ± {sd(dd):.4f} {rs[0]['gmacs']:>6}")


if __name__ == "__main__":
    main()
