"""Collect SOTA 5-fold CV test results by parsing the final per-class table in each run log."""
import re, json, statistics
from pathlib import Path

ROOT = Path.home() / "atli"
MODELS = ["dino", "rtmdet", "dynamic_rcnn"]
FOLDS = [0, 1, 2, 3, 4]
CLASSES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
           "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]

def parse_log(path):
    txt = path.read_text(errors="ignore")
    # last Epoch(test) marker = the test eval
    test_idx = txt.rfind("Epoch(test)")
    if test_idx == -1:
        return None
    # the per-class table appears just before it; scan backwards for table rows
    head = txt[:test_idx]
    rows = re.findall(r"\|\s*([\w\-]+)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|\s*([\d.]+|nan)\s*\|", head)
    # keep only the last 7 class rows
    cls_rows = [r for r in rows if r[0] in CLASSES][-7:]
    per_class = {r[0]: {"mAP": float(r[1]), "AP50": float(r[2])} for r in cls_rows}
    # last copypaste before/at test region: mAP mAP50 mAP75 s m l
    cp = re.findall(r"bbox_mAP_copypaste:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
    overall = cp[-1] if cp else None
    return {"per_class": per_class,
            "mAP": float(overall[0]) if overall else None,
            "mAP50": float(overall[1]) if overall else None}

out = {}
for m in MODELS:
    out[m] = {}
    for f in FOLDS:
        p = ROOT / f"SOTA_{m}_f{f}.log"
        if not p.exists():
            print(f"MISSING LOG {p}")
            continue
        r = parse_log(p)
        if r is None or len(r["per_class"]) != 7:
            print(f"PARSE PROBLEM {m} f{f}: {r and list(r['per_class'])}")
        out[m][f] = r

print("\n=== 5-fold CV summary (test split) ===")
print(f"{'model':14s} {'mAP@0.5':>12s} {'mAP@.5:.95':>12s} {'DD AP@0.5':>14s}")
summary = {}
for m in MODELS:
    folds = [v for v in out[m].values() if v]
    m50 = [v["mAP50"] for v in folds]
    mall = [v["mAP"] for v in folds]
    dd = [v["per_class"]["Defective_Damper"]["AP50"] for v in folds]
    summary[m] = {
        "n_folds": len(folds),
        "mAP50_mean": statistics.mean(m50), "mAP50_std": statistics.stdev(m50) if len(m50)>1 else 0,
        "mAP_mean": statistics.mean(mall),
        "DD_AP50_mean": statistics.mean(dd), "DD_AP50_std": statistics.stdev(dd) if len(dd)>1 else 0,
        "per_fold": {f: {"mAP50": v["mAP50"], "DD_AP50": v["per_class"]["Defective_Damper"]["AP50"]}
                     for f, v in out[m].items() if v},
        "per_class_AP50_mean": {c: statistics.mean([v["per_class"][c]["AP50"] for v in folds]) for c in CLASSES},
        "per_class_mAP_mean": {c: statistics.mean([v["per_class"][c]["mAP"] for v in folds]) for c in CLASSES},
    }
    s = summary[m]
    print(f"{m:14s} {s['mAP50_mean']:.3f}±{s['mAP50_std']:.3f} {s['mAP_mean']:12.3f} {s['DD_AP50_mean']:.3f}±{s['DD_AP50_std']:.3f}")

print("\n=== per-fold detail ===")
for m in MODELS:
    for f, v in summary[m]["per_fold"].items():
        print(f"{m} f{f}: mAP50={v['mAP50']:.3f} DD_AP50={v['DD_AP50']:.3f}")

print("\n=== per-class AP@0.5 mean ===")
hdr = f"{'class':26s}" + "".join(f"{m:>14s}" for m in MODELS)
print(hdr)
for c in CLASSES:
    print(f"{c:26s}" + "".join(f"{summary[m]['per_class_AP50_mean'][c]:14.3f}" for m in MODELS))

print("\n=== per-class mAP@0.5:0.95 mean ===")
print(hdr)
for c in CLASSES:
    print(f"{c:26s}" + "".join(f"{summary[m]['per_class_mAP_mean'][c]:14.3f}" for m in MODELS))

json.dump(summary, open(ROOT / "sota_cv_summary.json", "w"), indent=1)
print(f"\nSaved {ROOT/'sota_cv_summary.json'}")
