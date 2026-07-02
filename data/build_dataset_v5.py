#!/usr/bin/env python3
"""Rebuild dataset with merged_atli_target v5 (tightened Normal_Insulators polygons).
Creates Merged_Dataset_v5, Merged_Dataset_Stratified_v5, and Merged_Native_OSall_v5.

Run on server: ~/atli/env/bin/python build_dataset_v5.py
"""
import os, shutil, sys
from pathlib import Path
from collections import Counter
import numpy as np, yaml
from dotenv import load_dotenv

ROOT = Path.home() / "atli"
DL = ROOT / "downloads"; DL.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
assert KEY, "ROBOFLOW_API_KEY missing"

CLASS_NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
               "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
DEFECTS = {1, 2, 3, 6}  # minority classes to oversample


def rf_download(project, version, dest, fmt="yolov5pytorch"):
    from roboflow import Roboflow
    dest = Path(dest)
    if dest.exists() and any(dest.rglob("labels/*.txt")):
        print(f"  [skip] {dest} already downloaded"); return dest
    rf = Roboflow(api_key=KEY)
    rf.workspace(WS).project(project).version(version).download(fmt, location=str(dest))
    return dest


def pairs(root):
    out = []
    for lbl in sorted(Path(root).rglob("labels/*.txt")):
        hits = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
        if hits: out.append((lbl, hits[0]))
    return out


def counts(label_paths):
    c = Counter()
    for lbl in label_paths:
        for line in Path(lbl).read_text().splitlines():
            if line.strip(): c[int(line.split()[0])] += 1
    return dict(sorted(c.items()))


# ---------------------------------------------------------------- 1. Eduardo (same as before)
print("[1/5] Eduardo: reuse existing")
edu_dst = ROOT / "eduardo_all"
assert edu_dst.exists(), "Run build_dataset.py first to create eduardo_all"

# ---------------------------------------------------------------- 2. Target v5 (tightened polygons)
print("[2/5] Target: download merged_atli_target v5 (tightened Normal_Insulators)")
tgt = rf_download("merged_atli_target", 5, DL / "target_v5", "yolov5pytorch")
tp = pairs(tgt)
tc = counts([l for l, _ in tp])
print(f"      target_v5 images={len(tp)} counts={tc}")
assert len(tp) == 1046, f"target image count {len(tp)} != 1046"
# Note: counts may differ from v4 due to tightened polygons
print(f"      v4 had: NI=873, ND=1460 ; v5 has: NI={tc.get(5,'?')}, ND={tc.get(4,'?')}")

# ---------------------------------------------------------------- 3. Merge -> 1343 imgs
print("[3/5] Merge target_v5 + eduardo")
MERGED = ROOT / "Merged_Dataset_v5"
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

# ---------------------------------------------------------------- 4. Stratified split (same seed)
print("[4/5] Stratified split (seed 42, 70/15/15)")
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
OUT = ROOT / "Merged_Dataset_Stratified_v5"
for split in ("train", "val", "test"):
    for sub in ("images", "labels"):
        shutil.rmtree(OUT / split / sub, ignore_errors=True)
        (OUT / split / sub).mkdir(parents=True, exist_ok=True)

rows, targets = [], []
for img_path in sorted((MERGED / "images").glob("*")):
    lbl = MERGED / "labels" / f"{img_path.stem}.txt"
    if not lbl.exists(): continue
    v = np.zeros(7, dtype=int)
    for line in lbl.read_text().splitlines():
        if line.strip():
            c = int(line.split()[0])
            if 0 <= c < 7: v[c] = 1
    if v.sum() == 0: continue
    rows.append((img_path, lbl)); targets.append(v)
targets = np.array(targets)
idx = np.arange(len(rows))

tr_idx, tmp_idx = next(MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42).split(idx, targets))
tmp_local = np.arange(len(tmp_idx))
val_loc, test_loc = next(MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42).split(tmp_local, targets[tmp_idx]))
val_idx, test_idx = tmp_idx[val_loc], tmp_idx[test_loc]
print(f"      train={len(tr_idx)} val={len(val_idx)} test={len(test_idx)}")

for split_name, indices in [("train", tr_idx), ("val", val_idx), ("test", test_idx)]:
    for i in indices:
        img, lbl = rows[i]
        shutil.copy2(img, OUT / split_name / "images" / img.name)
        shutil.copy2(lbl, OUT / split_name / "labels" / lbl.name)

(OUT / "merged_stratified_v5.yaml").write_text(yaml.safe_dump({
    "path": str(OUT), "train": "train/images", "val": "val/images",
    "test": "test/images", "nc": 7, "names": CLASS_NAMES}, sort_keys=False))

# ---------------------------------------------------------------- 5. Build OSall variant
print("[5/5] Build OSall from Stratified_v5")
OSALL = ROOT / "Merged_Native_OSall_v5"
for split in ("train", "val", "test"):
    for sub in ("images", "labels"):
        shutil.rmtree(OSALL / split / sub, ignore_errors=True)
        (OSALL / split / sub).mkdir(parents=True, exist_ok=True)
        for f in (OUT / split / sub).glob("*"):
            shutil.copy2(f, OSALL / split / sub / f.name)

# oversample defect classes 3x in train
defect_count = 0
for lbl in (OUT / "train" / "labels").glob("*.txt"):
    classes = {int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()}
    if classes & DEFECTS:
        defect_count += 1
        img = next((OUT / "train" / "images").glob(lbl.stem + ".*"))
        for j in (1, 2):
            shutil.copy2(img, OSALL / "train" / "images" / f"os{j}_{img.name}")
            shutil.copy2(lbl, OSALL / "train" / "labels" / f"os{j}_{lbl.name}")

(OSALL / "merged_native_osall_v5.yaml").write_text(yaml.safe_dump({
    "path": str(OSALL), "train": "train/images", "val": "val/images",
    "test": "test/images", "nc": 7, "names": CLASS_NAMES}, sort_keys=False))
total = len(list((OSALL / "train" / "images").glob("*")))
print(f"      OSall_v5: {defect_count} defect imgs oversampled 3x, train total={total}")
print("BUILD_DATASET_V5_DONE")
