#!/usr/bin/env python3
"""Condition G (revised): Strip Normal_Damper (class 4) annotations from Eduardo images
to reduce the Normal_Damper flood. Keep all images and all other annotations intact.
Only Eduardo images (rf_ prefix) are modified; target images stay untouched.

This brings the DD:ND ratio down without losing any images or defective annotations.
"""
import shutil
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path.home() / "atli"
TPEF = ROOT / "Target_Plus_EduardoFull"
TO = ROOT / "Target_Only_Stratified"
NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]


def instance_counts(label_dir):
    c = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return {NAMES[k]: c.get(k, 0) for k in range(7)}


TPEF_G = ROOT / "Target_Plus_EduardoFull_StripND"
for sub in ("images", "labels"):
    shutil.rmtree(TPEF_G / "train" / sub, ignore_errors=True)
    (TPEF_G / "train" / sub).mkdir(parents=True, exist_ok=True)

stripped_nd = 0
for img in (TPEF / "train" / "images").glob("*"):
    shutil.copy2(img, TPEF_G / "train" / "images" / img.name)
    lbl = TPEF / "train" / "labels" / (img.stem + ".txt")
    if not lbl.exists():
        continue

    is_eduardo = img.name.startswith("rf_")
    if is_eduardo:
        # Strip Normal_Damper (class 4) lines
        new_lines = []
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if int(line.split()[0]) == 4:
                stripped_nd += 1
            else:
                new_lines.append(line)
        (TPEF_G / "train" / "labels" / lbl.name).write_text("\n".join(new_lines))
    else:
        shutil.copy2(lbl, TPEF_G / "train" / "labels" / lbl.name)

(ROOT / "abl_condition_g.yaml").write_text(yaml.safe_dump({
    "path": str(ROOT.resolve()),
    "train": str((TPEF_G / "train" / "images").resolve()),
    "val": str((TO / "val" / "images").resolve()),
    "test": str((TO / "test" / "images").resolve()),
    "nc": 7, "names": NAMES}, sort_keys=False))

g_counts = instance_counts(TPEF_G / "train" / "labels")
c_counts = instance_counts(TPEF / "train" / "labels")
print(f"Stripped {stripped_nd} Normal_Damper annotations from Eduardo images")
print(f"\nC counts: {c_counts}")
print(f"G counts: {g_counts}")
print(f"\nC ratio DD:ND = 1:{c_counts['Normal_Damper']/c_counts['Defective_Damper']:.1f}")
print(f"G ratio DD:ND = 1:{g_counts['Normal_Damper']/g_counts['Defective_Damper']:.1f}")
print("BUILD_G2_DONE")
