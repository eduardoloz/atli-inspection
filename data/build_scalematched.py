#!/usr/bin/env python3
"""Staging action #1: SCALE-MATCHED curation. Add ONLY the Universe images whose
defective-damper boxes are at ATLI's tiny scale (zoomed-out shots); drop the 85%
zoomed-in big-damper images that caused the domain/recall mismatch. Native val/test
untouched. Modest dose (no overtraining); co-added dampers are ATLI-scale so ND
shouldn't regress."""
import shutil
from collections import Counter
from pathlib import Path
import yaml

ROOT = Path.home() / "atli"
SRC, UNI = ROOT / "Merged_Dataset_Stratified", ROOT / "Universe_Pool"
NAMES = ["Birdnest","Broken_Insulator","Defective_Damper","Flashover_Insulator",
         "Normal_Damper","Normal_Insulators","Self-Exploded_Insulator"]
DD = 2
THRESH = 0.8   # %img area; keep image only if its LARGEST DD box <= THRESH (ATLI p90=0.52)


def dd_max(lbl):
    m = 0.0
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and int(p[0]) == DD:
            m = max(m, float(p[3]) * float(p[4]) * 100)
    return m


def counts(d):
    c = Counter()
    for lbl in Path(d).glob("*.txt"):
        for line in lbl.read_text().splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    return c


OUT = ROOT / "Merged_USM"
for split in ("train", "val", "test"):
    for sub in ("images", "labels"):
        shutil.rmtree(OUT / split / sub, ignore_errors=True)
        (OUT / split / sub).mkdir(parents=True, exist_ok=True)
        for f in (SRC / split / sub).glob("*"):
            shutil.copy2(f, OUT / split / sub / f.name)

added = 0
for lbl in (UNI / "labels").glob("*.txt"):
    mx = dd_max(lbl)
    if mx == 0 or mx > THRESH:
        continue
    img = next((UNI / "images").glob(lbl.stem + ".*"))
    shutil.copy2(img, OUT / "train" / "images" / img.name)
    shutil.copy2(lbl, OUT / "train" / "labels" / lbl.name)
    added += 1

(OUT / "merged_usm.yaml").write_text(yaml.safe_dump({
    "path": str(OUT), "train": "train/images", "val": "val/images",
    "test": "test/images", "nc": 7, "names": NAMES}, sort_keys=False))
tc = counts(OUT / "train" / "labels")
base = counts(SRC / "train" / "labels")
print(f"scale-matched (DD box<= {THRESH}%): +{added} universe imgs")
print(f"train DefDamper {base[DD]} -> {tc[DD]} | NormDamper {base[4]} -> {tc[4]} | "
      f"total imgs {len(list((OUT/'train'/'images').glob('*')))}")
print("BUILD_USM_DONE")
