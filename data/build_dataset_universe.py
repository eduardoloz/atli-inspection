#!/usr/bin/env python3
"""Build the Universe-augmented damper datasets on the GPU server.

Inputs (must already exist on the server):
  ~/atli/Merged_Dataset                 - 1,343-img pool (target v4 + eduardo), from build_dataset.py
  ~/atli/Merged_Dataset_Stratified      - its seed-42 70/15/15 split (for the train-only variant)
  ~/atli/datasets/vetting/u_wangbo      - wangbo/damper-o5wo3 v1 (YOLO, 640x640)
  ~/atli/datasets/vetting/u_yolov11tasks- yolov11-tasks/damper-defect-detection v3
  ~/atli/datasets/vetting/hashes_*.json - pHash caches from vet_universe_dampers.py

Universe filtering (per damper_dataset_vetting.md):
  - drop aug_* files (pre-augmented copies; they'd straddle splits -> intra-dataset leakage)
  - drop any image within Hamming<=8 of ANY pool image (covers the 4 val/test leaks + 29 train dups)
  - dedupe within the combined universe pool (Hamming<=8, keep first in sorted order)
  - class remap by NAME: Broken_damper/defective -> 2 (Defective_Damper);
                         Intact_damper/none_defective -> 4 (Normal_Damper)
  - samiksha-gadhave set EXCLUDED (defect-spot labels, pending taxonomy decision)

Outputs:
  ~/atli/Merged_Universe_Stratified  + merged_universe.yaml
      pool+universe re-stratified seed-42 70/15/15 ("same split as always", literal)
  ~/atli/Merged_Universe_TrainOnly   + merged_universe_trainonly.yaml
      existing split preserved, universe added to TRAIN ONLY (val/test identical to
      Merged_Dataset_Stratified -> directly comparable Defective_Damper test AP)
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path.home() / "atli"
VET = ROOT / "datasets" / "vetting"
POOL = ROOT / "Merged_Dataset"
OLD_SPLIT = ROOT / "Merged_Dataset_Stratified"
CLASS_NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
               "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
NEAR = 8

import imagehash
from PIL import Image


def counts(label_dir):
    c = Counter()
    for lbl in Path(label_dir).rglob("*.txt"):
        for line in lbl.read_text().splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    return dict(sorted(c.items()))


def load_cache(name):
    p = VET / name
    return {k: int(v, 16) for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def hash_pool():
    cache = VET / "hashes_mergedpool.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    for img in sorted((POOL / "images").glob("*")):
        if str(img) not in done:
            done[str(img)] = str(imagehash.phash(Image.open(img)))
    cache.write_text(json.dumps(done))
    return {k: int(v, 16) for k, v in done.items()}


def near_any(h, pool_hashes):
    return any(bin(h ^ v).count("1") <= NEAR for v in pool_hashes.values())


# ---------------------------------------------------------------- sanity on inputs
assert POOL.exists(), f"{POOL} missing - run build_dataset.py first"
mc = counts(POOL / "labels")
assert mc == {0: 241, 1: 194, 2: 264, 3: 434, 4: 2279, 5: 1865, 6: 554}, f"pool drift: {mc}"
assert OLD_SPLIT.exists(), f"{OLD_SPLIT} missing"
for s in ("train", "val", "test"):
    assert (OLD_SPLIT / s / "images").exists(), f"old split incomplete: {s}"

# ---------------------------------------------------------------- 1. filter universe sets
print("[1/4] Filter + remap universe sets")
pool_hashes = hash_pool()
UNI = ROOT / "Universe_Pool"
for sub in ("images", "labels"):
    shutil.rmtree(UNI / sub, ignore_errors=True)
    (UNI / sub).mkdir(parents=True, exist_ok=True)

kept_hashes = {}
stats = Counter()
for slug, prefix in (("u_wangbo", "uw"), ("u_yolov11tasks", "uy")):
    src = VET / slug
    names = None
    for y in src.rglob("data.yaml"):
        n = yaml.safe_load(y.read_text()).get("names")
        names = [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
        break
    assert names, f"no data.yaml in {src}"
    remap = {}
    for i, n in enumerate(names):
        ln = n.lower()
        if ("none" in ln) or ("intact" in ln):
            remap[i] = 4
        elif ("defect" in ln) or ("broken" in ln):
            remap[i] = 2
    print(f"  {slug}: names={names} -> remap={remap}")
    cache = load_cache(f"hashes_{slug}.json")
    for img in sorted(Path(src).rglob("images/*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if img.name.startswith("aug_"):
            stats[f"{slug}_drop_aug"] += 1
            continue
        h = cache.get(str(img))
        if h is None:
            h = int(str(imagehash.phash(Image.open(img))), 16)
        if near_any(h, pool_hashes):
            stats[f"{slug}_drop_pooldup"] += 1
            continue
        if near_any(h, kept_hashes):
            stats[f"{slug}_drop_unidup"] += 1
            continue
        lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lbl.exists():
            stats[f"{slug}_drop_nolabel"] += 1
            continue
        new = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5 and int(p[0]) in remap:
                p[0] = str(remap[int(p[0])])
                new.append(" ".join(p))
        if not new:
            stats[f"{slug}_drop_emptylabel"] += 1
            continue
        out = f"{prefix}_{img.name}"
        shutil.copy2(img, UNI / "images" / out)
        (UNI / "labels" / (Path(out).stem + ".txt")).write_text("\n".join(new))
        kept_hashes[out] = h
        stats[f"{slug}_kept"] += 1

uc = counts(UNI / "labels")
n_uni = len(list((UNI / "images").glob("*")))
print(f"  stats: {dict(stats)}")
print(f"  universe pool: {n_uni} imgs, counts={uc}  "
      f"(Defective_Damper +{uc.get(2,0)}, Normal_Damper +{uc.get(4,0)})")
assert n_uni > 500, "suspiciously few universe images survived filtering"

# ---------------------------------------------------------------- 2. variant A: re-stratified
print("[2/4] Variant A: combined re-stratified split (seed 42, 70/15/15)")
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

OUT_A = ROOT / "Merged_Universe_Stratified"
rows, targets = [], []
for src_dir in (POOL, UNI):
    for img in sorted((src_dir / "images").glob("*")):
        lbl = src_dir / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        v = np.zeros(7, dtype=int)
        for line in lbl.read_text().splitlines():
            if line.strip():
                c = int(line.split()[0])
                if 0 <= c < 7:
                    v[c] = 1
        if v.sum():
            rows.append((img, lbl)); targets.append(v)
targets = np.array(targets)
idx = np.arange(len(rows))
tr, tmp = next(MultilabelStratifiedShuffleSplit(1, test_size=0.30, random_state=42).split(idx, targets))
vl, ts = next(MultilabelStratifiedShuffleSplit(1, test_size=0.50, random_state=42).split(
    np.arange(len(tmp)), targets[tmp]))
splits_A = {"train": tr, "val": tmp[vl], "test": tmp[ts]}
total = len(rows)
assert sum(len(v) for v in splits_A.values()) == total
print(f"  {total} imgs -> " + " ".join(f"{k}={len(v)}" for k, v in splits_A.items()))
for split, indices in splits_A.items():
    for sub in ("images", "labels"):
        shutil.rmtree(OUT_A / split / sub, ignore_errors=True)
        (OUT_A / split / sub).mkdir(parents=True, exist_ok=True)
    for i in indices:
        img, lbl = rows[i]
        shutil.copy2(img, OUT_A / split / "images" / img.name)
        shutil.copy2(lbl, OUT_A / split / "labels" / lbl.name)
yaml_A = OUT_A / "merged_universe.yaml"
yaml_A.write_text(yaml.safe_dump({
    "path": str(OUT_A.resolve()), "train": "train/images", "val": "val/images",
    "test": "test/images", "nc": 7, "names": CLASS_NAMES}, sort_keys=False))

# ---------------------------------------------------------------- 3. variant B: train-only
print("[3/4] Variant B: old split preserved, universe -> train only")
OUT_B = ROOT / "Merged_Universe_TrainOnly"
for split in ("train", "val", "test"):
    for sub in ("images", "labels"):
        shutil.rmtree(OUT_B / split / sub, ignore_errors=True)
        (OUT_B / split / sub).mkdir(parents=True, exist_ok=True)
        for f in (OLD_SPLIT / split / sub).glob("*"):
            shutil.copy2(f, OUT_B / split / sub / f.name)
for img in (UNI / "images").glob("*"):
    shutil.copy2(img, OUT_B / "train" / "images" / img.name)
for lbl in (UNI / "labels").glob("*"):
    shutil.copy2(lbl, OUT_B / "train" / "labels" / lbl.name)
yaml_B = OUT_B / "merged_universe_trainonly.yaml"
yaml_B.write_text(yaml.safe_dump({
    "path": str(OUT_B.resolve()), "train": "train/images", "val": "val/images",
    "test": "test/images", "nc": 7, "names": CLASS_NAMES}, sort_keys=False))

# ---------------------------------------------------------------- 4. report
print("[4/4] Per-split class counts")
for name, out in (("A re-stratified", OUT_A), ("B train-only", OUT_B)):
    for split in ("train", "val", "test"):
        c = counts(out / split / "labels")
        n = len(list((out / split / "images").glob("*")))
        print(f"  {name:16s} {split:5s} imgs={n:5d}  DefDamper={c.get(2,0):4d}  "
              f"NormDamper={c.get(4,0):5d}  all={c}")
print(f"\nyamls:\n  {yaml_A}\n  {yaml_B}")
print("BUILD_UNIVERSE_DONE")
