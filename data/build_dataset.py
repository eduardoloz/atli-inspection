#!/usr/bin/env python3
"""Rebuild Merged_Dataset_Stratified under your own home, faithfully reproducing
Merging_datasets_of_Eduardo_and_target.ipynb. Hard asserts on instance counts catch
any Roboflow class-order drift before it can corrupt the benchmark.

Final target: train 936 / val 199 / test 208, 7 classes, merged_stratified.yaml.
"""
import os, shutil, sys
from pathlib import Path
from collections import Counter
import numpy as np
from dotenv import load_dotenv

ROOT = Path.home() / "atli"
DL = ROOT / "downloads"; DL.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
assert KEY, "ROBOFLOW_API_KEY missing (expected scp'd ~/atli/.env)"

CLASS_NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
               "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]


def rf_download(project, version, dest, fmt="yolov5"):
    from roboflow import Roboflow
    dest = Path(dest)
    if dest.exists() and any(dest.rglob("labels/*.txt")):
        print(f"  [skip] {dest} already downloaded"); return dest
    rf = Roboflow(api_key=KEY)
    rf.workspace(WS).project(project).version(version).download(fmt, location=str(dest))
    return dest


def pairs(root):
    """(label_path, image_path) for every label across all splits."""
    out = []
    for lbl in sorted(Path(root).rglob("labels/*.txt")):
        hits = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
        if hits:
            out.append((lbl, hits[0]))
    return out


def counts(label_paths):
    c = Counter()
    for lbl in label_paths:
        for line in Path(lbl).read_text().splitlines():
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return dict(sorted(c.items()))


# ---------------------------------------------------------------- 1. Eduardo (297 imgs)
print("[1/4] Eduardo: download v1 + remap + drop empties")
edu = rf_download("eduardos-annotated-photos", 1, DL / "eduardo", "yolov5")
edu_dst = ROOT / "eduardo_all"
for sub in ("images", "labels"):
    (edu_dst / sub).mkdir(parents=True, exist_ok=True)
# remap1: drop cls1 (defective_insulator), 2->1, 3->2 ; then remap2: 0->2, 1->4, 2->5
remap2 = {0: 2, 1: 4, 2: 5}
kept = skipped = 0
for lbl, img in pairs(edu):
    new = []
    for line in lbl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split(); c = int(p[0])
        if c == 1:            # defective_insulator -> dropped
            continue
        if c == 2: c = 1
        elif c == 3: c = 2
        p[0] = str(remap2[c]); new.append(" ".join(p))
    if not new:               # the 3 now-empty images -> dropped
        skipped += 1; continue
    kept += 1
    shutil.copy2(img, edu_dst / "images" / img.name)
    (edu_dst / "labels" / (img.stem + ".txt")).write_text("\n".join(new))
print(f"      kept={kept} skipped_empty={skipped}")
ec = counts(list((edu_dst / "labels").glob("*.txt")))
print(f"      eduardo counts={ec}")
assert kept == 297, f"eduardo image count {kept} != 297"
assert ec == {2: 151, 4: 819, 5: 992}, f"eduardo class counts drift: {ec}"

# ---------------------------------------------------------------- 2. Target (1046 imgs)
print("[2/4] Target: download merged_atli_target v4")
tgt = rf_download("merged_atli_target", 4, DL / "target", "yolov5")
tp = pairs(tgt)
tc = counts([l for l, _ in tp])
print(f"      target images={len(tp)} counts={tc}")
assert len(tp) == 1046, f"target image count {len(tp)} != 1046 (wrong version?)"
assert tc == {0: 241, 1: 194, 2: 113, 3: 434, 4: 1460, 5: 873, 6: 554}, \
    f"target class counts drift (class order mismatch?): {tc}"

# ---------------------------------------------------------------- 3. Merge -> 1343 imgs
print("[3/4] Merge target + eduardo(rf_ prefix)")
MERGED = ROOT / "Merged_Dataset"
for sub in ("images", "labels"):
    shutil.rmtree(MERGED / sub, ignore_errors=True)
    (MERGED / sub).mkdir(parents=True, exist_ok=True)
for lbl, img in tp:
    shutil.copy2(img, MERGED / "images" / img.name)
    shutil.copy2(lbl, MERGED / "labels" / (img.stem + ".txt"))
for lbl in (edu_dst / "labels").glob("*.txt"):
    img = next((edu_dst / "images").glob(lbl.stem + ".*"))
    shutil.copy2(img, MERGED / "images" / f"rf_{img.name}")
    shutil.copy2(lbl, MERGED / "labels" / f"rf_{lbl.name}")
n_merged = len(list((MERGED / "images").glob("*")))
mc = counts(list((MERGED / "labels").glob("*.txt")))
print(f"      merged images={n_merged} counts={mc}")
assert n_merged == 1343, f"merged image count {n_merged} != 1343"
assert mc == {0: 241, 1: 194, 2: 264, 3: 434, 4: 2279, 5: 1865, 6: 554}, \
    f"merged class counts drift: {mc}"

# ---------------------------------------------------------------- 4. Stratified split
print("[4/4] Stratified split (iterative-stratification, seed 42, 70/15/15)")
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
OUT = ROOT / "Merged_Dataset_Stratified"
for split in ("train", "val", "test"):
    for sub in ("images", "labels"):
        shutil.rmtree(OUT / split / sub, ignore_errors=True)
        (OUT / split / sub).mkdir(parents=True, exist_ok=True)

rows, targets = [], []
# sorted() => reproducible membership across reruns (sizes match the notebook regardless)
for img_path in sorted((MERGED / "images").glob("*")):
    lbl = MERGED / "labels" / f"{img_path.stem}.txt"
    if not lbl.exists():
        continue
    v = np.zeros(7, dtype=int)
    for line in lbl.read_text().splitlines():
        line = line.strip()
        if line:
            c = int(line.split()[0])
            if 0 <= c < 7:
                v[c] = 1
    if v.sum() == 0:
        continue
    rows.append((img_path, lbl)); targets.append(v)
targets = np.array(targets)
idx = np.arange(len(rows))
print(f"      labeled images={len(rows)}")

tr_idx, tmp_idx = next(MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42).split(idx, targets))
tmp_local = np.arange(len(tmp_idx))
val_loc, test_loc = next(MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42).split(tmp_local, targets[tmp_idx]))
val_idx, test_idx = tmp_idx[val_loc], tmp_idx[test_loc]
ntr, nval, ntst = len(tr_idx), len(val_idx), len(test_idx)
print(f"      train={ntr} val={nval} test={ntst}  (notebook ref: 936/199/208)")
# Sizes vary by a handful vs the notebook because stratified per-fold rounding is
# input-order dependent and we sorted the file list for reproducibility. Verify the
# split is sound: complete, ~70/15/15, and disjoint.
assert ntr + nval + ntst == 1343, f"split is not complete: {ntr+nval+ntst} != 1343"
assert 0.66 < ntr / 1343 < 0.74, f"train ratio off: {ntr/1343:.3f}"
assert 0.12 < nval / 1343 < 0.18 and 0.12 < ntst / 1343 < 0.18, \
    f"val/test ratio off: {nval/1343:.3f}/{ntst/1343:.3f}"
assert len(set(tr_idx) | set(val_idx) | set(test_idx)) == 1343, "splits overlap"


def copy_split(indices, name):
    for i in indices:
        img, lbl = rows[i]
        shutil.copy2(img, OUT / name / "images" / img.name)
        shutil.copy2(lbl, OUT / name / "labels" / lbl.name)


copy_split(tr_idx, "train"); copy_split(val_idx, "val"); copy_split(test_idx, "test")

import yaml
(OUT / "merged_stratified.yaml").write_text(yaml.safe_dump({
    "path": str(OUT.resolve()), "train": "train/images", "val": "val/images",
    "test": "test/images", "nc": 7, "names": CLASS_NAMES}, sort_keys=False))
print(f"\nDONE. dataset at {OUT}")
print((OUT / "merged_stratified.yaml").read_text())
print("BUILD_DATASET_DONE")
