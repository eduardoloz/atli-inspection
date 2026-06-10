#!/usr/bin/env python3
"""Condition G: Eduardo Full but undersample Eduardo's Normal_Damper-only images
to bring the Defective_Damper:Normal_Damper ratio closer to 1:5.

Strategy: keep ALL images that have Defective_Damper or Broken_Insulator labels.
Only drop Eduardo images whose ONLY damper-related class is Normal_Damper.
Target images are never touched.

Current ratio in C: Defective_Damper 224 : Normal_Damper 1853 = 1:8.3
Goal: ~1:5 => keep ~1120 Normal_Damper => drop ~733 Normal_Damper
But we only control Eduardo's contribution (~819 Normal_Damper instances from ~246 images).
Drop enough Eduardo Normal_Damper-only images to hit ~1:5.
"""
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml

random.seed(42)
ROOT = Path.home() / "atli"
TPEF = ROOT / "Target_Plus_EduardoFull"
TO = ROOT / "Target_Only_Stratified"
NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]

TARGET_RATIO = 5.0  # Defective_Damper : Normal_Damper = 1 : TARGET_RATIO


def get_classes(label_file):
    classes = set()
    for line in Path(label_file).read_text().splitlines():
        line = line.strip()
        if line:
            classes.add(int(line.split()[0]))
    return classes


def instance_counts(label_dir):
    c = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return {NAMES[k]: c.get(k, 0) for k in range(7)}


# Categorize Eduardo images
eduardo_defective = []  # has Defective_Damper or Broken_Insulator — always keep
eduardo_normal_only = []  # has Normal_Damper but NO defective classes — candidates for dropping
eduardo_other = []  # neither

defective_classes = {1, 2}  # Broken_Insulator, Defective_Damper

for img in sorted((TPEF / "train" / "images").glob("rf_*")):
    lbl = TPEF / "train" / "labels" / (img.stem + ".txt")
    if not lbl.exists():
        eduardo_other.append(img)
        continue
    classes = get_classes(lbl)
    if classes & defective_classes:
        eduardo_defective.append(img)
    elif 4 in classes:  # Normal_Damper
        eduardo_normal_only.append(img)
    else:
        eduardo_other.append(img)

print(f"Eduardo images: {len(eduardo_defective)} defective, {len(eduardo_normal_only)} normal-damper-only, {len(eduardo_other)} other")

# Count current instances
c_counts = instance_counts(TPEF / "train" / "labels")
current_dd = c_counts["Defective_Damper"]
current_nd = c_counts["Normal_Damper"]
print(f"Current: DD={current_dd}, ND={current_nd}, ratio=1:{current_nd/current_dd:.1f}")

# Calculate how many Normal_Damper instances to target
target_nd = int(current_dd * TARGET_RATIO)
to_remove_nd = current_nd - target_nd
print(f"Target: ND={target_nd}, need to remove ~{to_remove_nd} ND instances")

# Count ND instances per Eduardo normal-only image, then drop enough
nd_per_image = []
for img in eduardo_normal_only:
    lbl = TPEF / "train" / "labels" / (img.stem + ".txt")
    nd_count = sum(1 for line in lbl.read_text().splitlines()
                   if line.strip() and int(line.strip().split()[0]) == 4)
    nd_per_image.append((img, nd_count))

# Shuffle and drop until we've removed enough ND instances
random.shuffle(nd_per_image)
drop_set = set()
removed_nd = 0
for img, nd_count in nd_per_image:
    if removed_nd >= to_remove_nd:
        break
    drop_set.add(img.name)
    removed_nd += nd_count

print(f"Dropping {len(drop_set)} Eduardo normal-damper-only images ({removed_nd} ND instances)")

# Build Condition G
TPEF_G = ROOT / "Target_Plus_EduardoFull_Undersample"
for sub in ("images", "labels"):
    shutil.rmtree(TPEF_G / "train" / sub, ignore_errors=True)
    (TPEF_G / "train" / sub).mkdir(parents=True, exist_ok=True)

copied = 0
dropped = 0
for img in (TPEF / "train" / "images").glob("*"):
    if img.name in drop_set:
        dropped += 1
        continue
    lbl = TPEF / "train" / "labels" / (img.stem + ".txt")
    shutil.copy2(img, TPEF_G / "train" / "images" / img.name)
    if lbl.exists():
        shutil.copy2(lbl, TPEF_G / "train" / "labels" / lbl.name)
    copied += 1

(ROOT / "abl_condition_g.yaml").write_text(yaml.safe_dump({
    "path": str(ROOT.resolve()),
    "train": str((TPEF_G / "train" / "images").resolve()),
    "val": str((TO / "val" / "images").resolve()),
    "test": str((TO / "test" / "images").resolve()),
    "nc": 7, "names": NAMES}, sort_keys=False))

g_counts = instance_counts(TPEF_G / "train" / "labels")
print(f"\nCondition G: {copied} images (dropped {dropped})")
print(f"C counts: {instance_counts(TPEF / 'train' / 'labels')}")
print(f"G counts: {g_counts}")
print(f"New ratio DD:ND = 1:{g_counts['Normal_Damper']/g_counts['Defective_Damper']:.1f}")
print("BUILD_G_DONE")
