#!/usr/bin/env python3
"""Build the Eduardo ablation datasets, isolating Eduardo's training contribution.

  Target_Only_Stratified/{train,val,test}   <- 1046 target imgs, stratified seed 42 (70/15/15)
  Target_Plus_Eduardo/train                  <- target-train + all 297 Eduardo (rf_)
  abl_target_only.yaml                       <- A: train/val/test all target-only
  abl_target_plus_eduardo.yaml               <- B: train = target+Eduardo, val/test = SAME target-only

Both conditions share the identical (Eduardo-free) val + test, so the only variable is
whether Eduardo images are in TRAIN.
"""
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = Path.home() / "atli"
TGT_DL = ROOT / "downloads" / "target"
EDU = ROOT / "eduardo_all"
NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]


def pairs(root):
    out = []
    for lbl in sorted(Path(root).rglob("labels/*.txt")):
        hits = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
        if hits:
            out.append((lbl, hits[0]))
    return out


def vec(lbl):
    v = np.zeros(7, dtype=int)
    for line in Path(lbl).read_text().splitlines():
        line = line.strip()
        if line:
            c = int(line.split()[0])
            if 0 <= c < 7:
                v[c] = 1
    return v


def counts(label_dir):
    c = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return {NAMES[k]: c.get(k, 0) for k in range(7)}


# ---- stratified split of target-only (1046) ----
tp = pairs(TGT_DL)
assert len(tp) == 1046, f"target imgs {len(tp)} != 1046"
rows, targets = [], []
for lbl, img in tp:
    v = vec(lbl)
    if v.sum():
        rows.append((img, lbl)); targets.append(v)
targets = np.array(targets); idx = np.arange(len(rows))
tr, tmp = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30,
                                                random_state=42).split(idx, targets))
vl, ts = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50,
                                               random_state=42).split(np.arange(len(tmp)), targets[tmp]))
val_idx, test_idx = tmp[vl], tmp[ts]
print(f"target-only split: train={len(tr)} val={len(val_idx)} test={len(test_idx)}")

TO = ROOT / "Target_Only_Stratified"
for s in ("train", "val", "test"):
    for sub in ("images", "labels"):
        shutil.rmtree(TO / s / sub, ignore_errors=True)
        (TO / s / sub).mkdir(parents=True, exist_ok=True)


def put(indices, name):
    for i in indices:
        img, lbl = rows[i]
        shutil.copy2(img, TO / name / "images" / img.name)
        shutil.copy2(lbl, TO / name / "labels" / lbl.name)


put(tr, "train"); put(val_idx, "val"); put(test_idx, "test")

# ---- Target_Plus_Eduardo/train = target-train + all Eduardo ----
TPE = ROOT / "Target_Plus_Eduardo"
for sub in ("images", "labels"):
    shutil.rmtree(TPE / "train" / sub, ignore_errors=True)
    (TPE / "train" / sub).mkdir(parents=True, exist_ok=True)
for img in (TO / "train" / "images").glob("*"):
    shutil.copy2(img, TPE / "train" / "images" / img.name)
for lbl in (TO / "train" / "labels").glob("*.txt"):
    shutil.copy2(lbl, TPE / "train" / "labels" / lbl.name)
n_edu = 0
for lbl in (EDU / "labels").glob("*.txt"):
    img = next((EDU / "images").glob(lbl.stem + ".*"))
    shutil.copy2(img, TPE / "train" / "images" / f"rf_{img.name}")
    shutil.copy2(lbl, TPE / "train" / "labels" / f"rf_{lbl.name}")
    n_edu += 1
print(f"eduardo added to B-train: {n_edu}")

# ---- yamls (absolute paths override 'path' in both yolov5 & ultralytics) ----
(ROOT / "abl_target_only.yaml").write_text(yaml.safe_dump({
    "path": str(TO.resolve()), "train": "train/images", "val": "val/images",
    "test": "test/images", "nc": 7, "names": NAMES}, sort_keys=False))
(ROOT / "abl_target_plus_eduardo.yaml").write_text(yaml.safe_dump({
    "path": str(ROOT.resolve()),
    "train": str((TPE / "train" / "images").resolve()),
    "val": str((TO / "val" / "images").resolve()),
    "test": str((TO / "test" / "images").resolve()),
    "nc": 7, "names": NAMES}, sort_keys=False))

print("\nA target-only  train counts:", counts(TO / "train" / "labels"))
print("B +eduardo     train counts:", counts(TPE / "train" / "labels"))
print("shared val     counts:", counts(TO / "val" / "labels"))
print("shared TEST    counts:", counts(TO / "test" / "labels"))
print("BUILD_ABLATION_DONE")
