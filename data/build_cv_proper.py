#!/usr/bin/env python3
"""Build 5-fold CV with exact 70/15/15 splits matching the paper.

Each fold uses a different random seed for stratified splitting:
  - 70% train (gradient updates)
  - 15% val (checkpoint selection only)
  - 15% test (final evaluation, never seen during training)

Builds three dataset variants per fold:
  - base.yaml:  train (no oversample)
  - champ.yaml: trainos (DD 3x oversample)
  - osall.yaml: trainosall (all defect 3x oversample)

Run on server: ~/atli/env/bin/python build_cv_proper.py
"""
import shutil, yaml, numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = Path.home() / "atli"
POOL = ROOT / "Merged_Dataset"
OUT = ROOT / "Merged_CV_proper"
NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
DD = 2
DEFECTS = {1, 2, 3, 6}

# Collect all labeled images
rows, tg = [], []
for img in sorted((POOL / "images").glob("*")):
    lbl = POOL / "labels" / (img.stem + ".txt")
    if not lbl.exists():
        continue
    v = np.zeros(7, int)
    for l in lbl.read_text().splitlines():
        if l.strip():
            c = int(l.split()[0])
            if 0 <= c < 7:
                v[c] = 1
    if v.sum():
        rows.append((img, lbl))
        tg.append(v)
tg = np.array(tg)
idx = np.arange(len(rows))

print(f"Total labeled images: {len(rows)}")

# 5 independent 70/15/15 stratified splits with different seeds
SEEDS = [42, 123, 456, 789, 1024]

for k, seed in enumerate(SEEDS):
    f = OUT / f"fold{k}"

    # First split: 70% train vs 30% temp (val+test)
    sp1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sp1.split(idx, tg))

    # Second split: split the 30% into 50/50 = 15% val + 15% test
    temp_local = np.arange(len(temp_idx))
    sp2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_local, test_local = next(sp2.split(temp_local, tg[temp_idx]))
    val_idx = temp_idx[val_local]
    test_idx = temp_idx[test_local]

    # Create directories
    dirs = [
        "test/images", "test/labels",
        "val/images", "val/labels",
        "train/images", "train/labels",
        "trainos/images", "trainos/labels",
        "trainosall/images", "trainosall/labels",
    ]
    for s in dirs:
        shutil.rmtree(f / s, ignore_errors=True)
        (f / s).mkdir(parents=True, exist_ok=True)

    # Copy test (held out completely)
    for i in test_idx:
        im, lb = rows[i]
        shutil.copy2(im, f / "test/images" / im.name)
        shutil.copy2(lb, f / "test/labels" / lb.name)

    # Copy val (checkpoint selection only)
    for i in val_idx:
        im, lb = rows[i]
        shutil.copy2(im, f / "val/images" / im.name)
        shutil.copy2(lb, f / "val/labels" / lb.name)

    # Copy train + build oversample variants
    dd_count = 0
    defect_count = 0
    for i in train_idx:
        im, lb = rows[i]
        shutil.copy2(im, f / "train/images" / im.name)
        shutil.copy2(lb, f / "train/labels" / lb.name)
        shutil.copy2(im, f / "trainos/images" / im.name)
        shutil.copy2(lb, f / "trainos/labels" / lb.name)
        shutil.copy2(im, f / "trainosall/images" / im.name)
        shutil.copy2(lb, f / "trainosall/labels" / lb.name)

        classes = set()
        for line in lb.read_text().splitlines():
            if line.strip():
                classes.add(int(line.split()[0]))

        if DD in classes:
            dd_count += 1
            for j in (1, 2):
                shutil.copy2(im, f / "trainos/images" / f"os{j}_{im.name}")
                shutil.copy2(lb, f / "trainos/labels" / f"os{j}_{lb.name}")

        if classes & DEFECTS:
            defect_count += 1
            for j in (1, 2):
                shutil.copy2(im, f / "trainosall/images" / f"os{j}_{im.name}")
                shutil.copy2(lb, f / "trainosall/labels" / f"os{j}_{lb.name}")

    # Write YAMLs with SEPARATE val and test
    for nm, td in [("base", "train"), ("champ", "trainos"), ("osall", "trainosall")]:
        (f / f"{nm}.yaml").write_text(yaml.safe_dump({
            "path": str(f),
            "train": f"{td}/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 7,
            "names": NAMES,
        }, sort_keys=False))

    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)
    pct = lambda n: f"{100*n/len(rows):.1f}%"

    print(f"fold{k} (seed={seed}): train={n_train} ({pct(n_train)}) "
          f"val={n_val} ({pct(n_val)}) test={n_test} ({pct(n_test)}) "
          f"dd_os={dd_count} defect_os={defect_count}")

print("CV_PROPER_BUILD_DONE")
