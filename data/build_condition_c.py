#!/usr/bin/env python3
"""Condition C: target-train + FULL Eduardo, keeping the 152 Defective_Insulators boxes
mapped to Broken_Insulator (target class 1). Same shared val/test as A and B.

Raw Eduardo classes: 0=Defective_Damper 1=Defective_Insulators 2=Normal_Damper 3=Normal_Insulators
Map to target taxonomy: 0->2 (Defective_Damper), 1->1 (Broken_Insulator),
                        2->4 (Normal_Damper),   3->5 (Normal_Insulators)
"""
import shutil
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path.home() / "atli"
EDU_DL = ROOT / "downloads" / "eduardo"
TO = ROOT / "Target_Only_Stratified"          # built by build_ablation.py
NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
MAP = {0: 2, 1: 1, 2: 4, 3: 5}                 # 1->1 keeps defective insulators as Broken_Insulator


def pairs(root):
    out = []
    for lbl in sorted(Path(root).rglob("labels/*.txt")):
        hits = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
        if hits:
            out.append((lbl, hits[0]))
    return out


def counts(label_dir):
    c = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return {NAMES[k]: c.get(k, 0) for k in range(7)}


# ---- process full Eduardo (keep all 4 classes, remap) ----
EF = ROOT / "eduardo_full_all"
for sub in ("images", "labels"):
    shutil.rmtree(EF / sub, ignore_errors=True)
    (EF / sub).mkdir(parents=True, exist_ok=True)
kept = empty = 0
for lbl, img in pairs(EDU_DL):
    new = []
    for line in lbl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        p[0] = str(MAP[int(p[0])])
        new.append(" ".join(p))
    if not new:
        empty += 1
        continue
    kept += 1
    shutil.copy2(img, EF / "images" / img.name)
    (EF / "labels" / (img.stem + ".txt")).write_text("\n".join(new))
ec = counts(EF / "labels")
print(f"eduardo-full kept={kept} empty={empty} counts={ec}")
assert ec["Broken_Insulator"] == 152 and ec["Defective_Damper"] == 151, f"unexpected: {ec}"

# ---- Target_Plus_EduardoFull/train = target-train + eduardo-full ----
TPEF = ROOT / "Target_Plus_EduardoFull"
for sub in ("images", "labels"):
    shutil.rmtree(TPEF / "train" / sub, ignore_errors=True)
    (TPEF / "train" / sub).mkdir(parents=True, exist_ok=True)
for img in (TO / "train" / "images").glob("*"):
    shutil.copy2(img, TPEF / "train" / "images" / img.name)
for lbl in (TO / "train" / "labels").glob("*.txt"):
    shutil.copy2(lbl, TPEF / "train" / "labels" / lbl.name)
for lbl in (EF / "labels").glob("*.txt"):
    img = next((EF / "images").glob(lbl.stem + ".*"))
    shutil.copy2(img, TPEF / "train" / "images" / f"rf_{img.name}")
    shutil.copy2(lbl, TPEF / "train" / "labels" / f"rf_{lbl.name}")

(ROOT / "abl_target_plus_eduardo_full.yaml").write_text(yaml.safe_dump({
    "path": str(ROOT.resolve()),
    "train": str((TPEF / "train" / "images").resolve()),
    "val": str((TO / "val" / "images").resolve()),
    "test": str((TO / "test" / "images").resolve()),
    "nc": 7, "names": NAMES}, sort_keys=False))

print("C +eduardo-full train counts:", counts(TPEF / "train" / "labels"))
print("(Broken_Insulator: A/B train=135  ->  C train should be 135+152=287)")
print("BUILD_C_DONE")
