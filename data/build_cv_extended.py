"""Extend existing CV folds with OSall (all-minority-class 3x oversample) variant.
Adds trainosall/ dir + osall.yaml to each fold without touching existing base/champ.
Run on server: ~/atli/env/bin/python build_cv_extended.py
"""
import shutil, yaml
from pathlib import Path

ROOT = Path.home() / "atli"
CV = ROOT / "Merged_CV"
NAMES = ["Birdnest","Broken_Insulator","Defective_Damper","Flashover_Insulator",
         "Normal_Damper","Normal_Insulators","Self-Exploded_Insulator"]
# All minority/defect classes to oversample 3x
DEFECTS = {1, 2, 3, 6}  # Broken_Ins, Def_Damper, Flashover_Ins, Self-Exploded_Ins

for k in range(5):
    f = CV / f"fold{k}"
    osall = f / "trainosall"
    for s in ("images", "labels"):
        shutil.rmtree(osall / s, ignore_errors=True)
        (osall / s).mkdir(parents=True, exist_ok=True)

    # Copy all train images, then add 2 extra copies of any image with a defect class
    train_imgs = sorted((f / "train" / "images").glob("*"))
    defect_count = 0
    for im in train_imgs:
        lb = f / "train" / "labels" / (im.stem + ".txt")
        shutil.copy2(im, osall / "images" / im.name)
        if lb.exists():
            shutil.copy2(lb, osall / "labels" / lb.name)
            classes = {int(l.split()[0]) for l in lb.read_text().splitlines() if l.strip()}
            if classes & DEFECTS:
                defect_count += 1
                for j in (1, 2):
                    shutil.copy2(im, osall / "images" / f"os{j}_{im.name}")
                    shutil.copy2(lb, osall / "labels" / f"os{j}_{lb.name}")

    # Write osall.yaml
    (f / "osall.yaml").write_text(yaml.safe_dump({
        "path": str(f), "train": "trainosall/images", "val": "test/images",
        "test": "test/images", "nc": 7, "names": NAMES
    }, sort_keys=False))

    total = len(list((osall / "images").glob("*")))
    print(f"fold{k}: train_base={len(train_imgs)} defect_imgs={defect_count} "
          f"trainosall={total} (+{total - len(train_imgs)} oversampled)")

print("CV_OSALL_BUILD_DONE")
