#!/usr/bin/env python3
"""Extend the eduardo-CV OBB folds' trainosall (defect images x3) to x6.

For each fold: trainosall6 = full copy of trainosall + os3_/os4_/os5_ copies of
every defect-containing image (classes {1,2,3,6} = BI, DD, Flashover, SE),
continuing build_cv_eduardo.py's os{r}_ prefix convention. Writes osall6.yaml.
Idempotent. Run on server: python build_osall6_eduardo.py
"""
import shutil
from pathlib import Path

ROOT = Path.home() / "atli" / "CV_eduardo_obb"
DEFECTS = {1, 2, 3, 6}

for k in range(5):
    fold = ROOT / f"fold{k}"
    src = fold / "trainosall"
    dst = fold / "trainosall6"
    if (fold / "osall6.yaml").exists() and (dst / "images").exists():
        print(f"fold{k}: SKIP (exists)")
        continue
    for sub in ("images", "labels"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for f in (src / sub).iterdir():
            shutil.copy2(f, dst / sub / f.name)
    extra = 0
    for lab in (src / "labels").iterdir():
        if lab.name.startswith("os"):  # only originals; os1_/os2_ are the x3 dups
            continue
        lines = lab.read_text().split("\n")
        if not any(l.strip() and int(l.split()[0]) in DEFECTS for l in lines):
            continue
        stem = lab.stem
        img = next(p for p in (src / "images").glob(stem + ".*"))
        for r in (3, 4, 5):
            shutil.copy2(img, dst / "images" / f"os{r}_{img.name}")
            shutil.copy2(lab, dst / "labels" / f"os{r}_{lab.name}")
            extra += 1
    base = (fold / "osall.yaml").read_text().replace("trainosall/images",
                                                     "trainosall6/images")
    (fold / "osall6.yaml").write_text(base)
    print(f"fold{k}: +{extra} defect copies -> trainosall6")
print("OSALL6_DONE")
