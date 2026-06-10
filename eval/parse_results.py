#!/usr/bin/env python3
"""Parse the test-split metrics for every config out of the sweep lane logs and
emit a markdown summary + per-class table + CSV. Tolerant of yolo's CR progress bars."""
import re, csv
from pathlib import Path

ROOT = Path.home() / "atli"
CONFIGS = ["v5n_sgd_150", "v5n_sgd_300", "v5n_adam_150", "v5n_adam_300",
           "v8n_sgd_150", "v8n_sgd_300"]
CLASSES = ["all", "Birdnest", "Broken_Insulator", "Defective_Damper",
           "Flashover_Insulator", "Normal_Damper", "Normal_Insulators",
           "Self-Exploded_Insulator"]

text = ""
for f in ("laneA.log", "laneB.log"):
    p = ROOT / f
    if p.exists():
        text += p.read_text(errors="ignore")
text = text.replace("\r", "\n")


def section(name):
    m = re.search(rf"--- {name} test ---(.*?)### {name} DONE", text, re.S)
    return m.group(1) if m else ""


rows = {}
for cfg in CONFIGS:
    sec = section(cfg)
    rows[cfg] = {}
    for cls in CLASSES:
        mm = re.findall(
            rf"^\s*{re.escape(cls)}\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
            sec, re.M)
        if mm:
            i, n, P, R, m50, m95 = mm[-1]
            rows[cfg][cls] = (int(i), int(n), float(P), float(R), float(m50), float(m95))

LABEL = {"v5n_sgd_150": "YOLOv5n SGD 150+100", "v5n_sgd_300": "YOLOv5n SGD 300+100",
         "v5n_adam_150": "YOLOv5n Adam 150+100", "v5n_adam_300": "YOLOv5n Adam 300+100",
         "v8n_sgd_150": "YOLOv8n SGD 150+100", "v8n_sgd_300": "YOLOv8n SGD 300+100"}

print("\n## Summary — test split (mAP@0.5)\n")
print("| Config | done | mAP@0.5 | mAP@0.5:0.95 | P | R |")
print("|---|---|---|---|---|---|")
done = []
for cfg in CONFIGS:
    a = rows[cfg].get("all")
    if a:
        done.append((cfg, a[4]))
        print(f"| {LABEL[cfg]} | ✓ | {a[4]:.3f} | {a[5]:.3f} | {a[2]:.3f} | {a[3]:.3f} |")
    else:
        print(f"| {LABEL[cfg]} | … running | – | – | – | – |")

print("\n## Per-class mAP@0.5\n")
hdr = "| Class | " + " | ".join(LABEL[c].replace("YOLO", "") for c in CONFIGS) + " |"
print(hdr); print("|" + "---|" * (len(CONFIGS) + 1))
for cls in CLASSES[1:]:
    cells = []
    for cfg in CONFIGS:
        v = rows[cfg].get(cls)
        cells.append(f"{v[4]:.3f}" if v else "–")
    print(f"| {cls} | " + " | ".join(cells) + " |")

with (ROOT / "results.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config", "class", "images", "instances", "P", "R", "mAP50", "mAP50_95"])
    for cfg in CONFIGS:
        for cls in CLASSES:
            if cls in rows[cfg]:
                w.writerow([cfg, cls, *rows[cfg][cls]])
print(f"\nParsed {len(done)}/6 configs complete. CSV -> {ROOT/'results.csv'}")
