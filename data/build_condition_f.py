#!/usr/bin/env python3
"""Condition F: Eduardo Full minus the 4 hard-negative Defective_Damper images
that the model confuses with Normal_Damper.

Same shared val/test as all other conditions.
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

# Only exclude the 4 Eduardo hard negatives (confused Defective_Damper -> Normal_Damper)
EXCLUDE = {
    "moFZlIsDY74112IjjPAx",
    "qlee4twEkzu3cRG405m4",
    "qlreF6XRQomYTZ4TyNNd",
    "wO8pOCBQjbumdG4pE0aL",
}


def counts(label_dir):
    c = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return {NAMES[k]: c.get(k, 0) for k in range(7)}


def should_exclude(filename):
    stem = Path(filename).stem
    for exc_id in EXCLUDE:
        if exc_id in stem:
            return True
    return False


TPEF_F = ROOT / "Target_Plus_EduardoFull_NoHardNeg"
for sub in ("images", "labels"):
    shutil.rmtree(TPEF_F / "train" / sub, ignore_errors=True)
    (TPEF_F / "train" / sub).mkdir(parents=True, exist_ok=True)

copied = 0
excluded = 0
for img in (TPEF / "train" / "images").glob("*"):
    if should_exclude(img.name):
        excluded += 1
        continue
    lbl = TPEF / "train" / "labels" / (img.stem + ".txt")
    shutil.copy2(img, TPEF_F / "train" / "images" / img.name)
    if lbl.exists():
        shutil.copy2(lbl, TPEF_F / "train" / "labels" / lbl.name)
    copied += 1

print(f"Condition F: copied {copied} images, excluded {excluded}")

(ROOT / "abl_condition_f.yaml").write_text(yaml.safe_dump({
    "path": str(ROOT.resolve()),
    "train": str((TPEF_F / "train" / "images").resolve()),
    "val": str((TO / "val" / "images").resolve()),
    "test": str((TO / "test" / "images").resolve()),
    "nc": 7, "names": NAMES}, sort_keys=False))

print("\nC (full Eduardo) train counts:", counts(TPEF / "train" / "labels"))
print("F (no hard neg)  train counts:", counts(TPEF_F / "train" / "labels"))
print("\nDiff C - F (removed):")
c_counts = counts(TPEF / "train" / "labels")
f_counts = counts(TPEF_F / "train" / "labels")
for cls in NAMES:
    diff = c_counts[cls] - f_counts[cls]
    if diff:
        print(f"  {cls}: -{diff}")
print("BUILD_F_DONE")
