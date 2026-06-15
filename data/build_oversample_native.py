#!/usr/bin/env python3
"""Targeted oversampling of the native Defective_Damper class (CE-SSD 'augment the
minority past parity' finding, automated). NO new/external data — duplicates native
train images that contain >=1 Defective_Damper (class 2) box; YOLO's online aug
(mosaic/hsv/flip/scale) gives each duplicate a different view every epoch.

Builds Merged_Native_OS3 (x3) and Merged_Native_OS6 (x6). val/test untouched
(identical to Merged_Dataset_Stratified -> comparable to baseline B).
"""
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path.home() / "atli"
SRC = ROOT / "Merged_Dataset_Stratified"
CLASS_NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
               "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
DD = 2


def has_dd(lbl):
    return any(line.split() and int(line.split()[0]) == DD
              for line in lbl.read_text().splitlines() if line.strip())


def counts(label_dir):
    c = Counter()
    for lbl in Path(label_dir).glob("*.txt"):
        for line in lbl.read_text().splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    return c


for factor in (3, 6):
    OUT = ROOT / f"Merged_Native_OS{factor}"
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            shutil.rmtree(OUT / split / sub, ignore_errors=True)
            (OUT / split / sub).mkdir(parents=True, exist_ok=True)
            for f in (SRC / split / sub).glob("*"):
                shutil.copy2(f, OUT / split / sub / f.name)
    # duplicate DD-containing train images (factor-1) extra times
    dd_imgs = dup = 0
    for lbl in (SRC / "train" / "labels").glob("*.txt"):
        if not has_dd(lbl):
            continue
        dd_imgs += 1
        img = next((SRC / "train" / "images").glob(lbl.stem + ".*"))
        for k in range(1, factor):
            shutil.copy2(img, OUT / "train" / "images" / f"os{k}_{img.name}")
            shutil.copy2(lbl, OUT / "train" / "labels" / f"os{k}_{lbl.name}")
            dup += 1
    (OUT / f"merged_native_os{factor}.yaml").write_text(yaml.safe_dump({
        "path": str(OUT), "train": "train/images", "val": "val/images",
        "test": "test/images", "nc": 7, "names": CLASS_NAMES}, sort_keys=False))
    tc = counts(OUT / "train" / "labels")
    print(f"OS{factor}: {dd_imgs} DD-imgs duplicated x{factor} (+{dup} imgs); "
          f"train DefDamper {counts(SRC/'train'/'labels')[DD]} -> {tc[DD]}, "
          f"NormDamper {tc[4]}, total train imgs {len(list((OUT/'train'/'images').glob('*')))}")
print("BUILD_OVERSAMPLE_DONE")
