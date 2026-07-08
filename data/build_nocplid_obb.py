#!/usr/bin/env python3
"""Build OBB datasets from the atli_target-minus-the-cplid v2 segmentation export
(contains the user's oriented Normal_Damper polygons, 82 on 27 imgs, plus the
tightened-NI polygons).

Outputs:
  ~/atli/ATLI_noCPLID_OBB/      — full dataset, YOLO-OBB 4-corner labels
  ~/atli/ATLI_noCPLID_OBB_OS3/  — same + Defective_Damper train images x3 (champion)

Polygon lines -> cv2.minAreaRect rotated rect; plain box lines -> axis-aligned
corners (same conversion as data/build_cv_obb.py).

Run on server:  ~/atli/env/bin/python build_nocplid_obb.py
"""
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path.home() / "atli"
SRC = ROOT / "downloads" / "minus_cplid_v2_seg"
OUT = ROOT / "ATLI_noCPLID_OBB"
OUT_OS3 = ROOT / "ATLI_noCPLID_OBB_OS3"
DD = 2  # Defective_Damper index (verified against export data.yaml below)


def seg_line_to_obb(line, w, h):
    p = line.strip().split()
    cls = int(p[0])
    c = [float(v) for v in p[1:]]
    if len(c) == 4:  # cx cy w h box
        cx, cy, bw, bh = c
        pts = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy - bh / 2,
               cx + bw / 2, cy + bh / 2, cx - bw / 2, cy + bh / 2]
        return cls, pts, False
    pts = np.array([[c[i] * w, c[i + 1] * h] for i in range(0, len(c), 2)],
                   dtype=np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    out = []
    for corner in box:
        out += [float(np.clip(corner[0] / w, 0, 1)),
                float(np.clip(corner[1] / h, 0, 1))]
    return cls, out, True


def main():
    names = yaml.safe_load((SRC / "data.yaml").read_text())["names"]
    assert names[DD] == "Defective_Damper", names
    n_poly = Counter()
    for split in ("train", "valid", "test"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in sorted((SRC / split / "images").glob("*")):
            shutil.copy2(img, OUT / split / "images" / img.name)
            lbl = SRC / split / "labels" / (img.stem + ".txt")
            im = cv2.imread(str(img))
            h, w = im.shape[:2]
            lines = []
            for line in (lbl.read_text().splitlines() if lbl.exists() else []):
                if not line.strip():
                    continue
                cls, pts, was_poly = seg_line_to_obb(line, w, h)
                if was_poly:
                    n_poly[names[cls]] += 1
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in pts))
            (OUT / split / "labels" / (img.stem + ".txt")).write_text("\n".join(lines) + "\n")
    (OUT / "data.yaml").write_text(yaml.safe_dump({
        "train": str(OUT / "train" / "images"), "val": str(OUT / "valid" / "images"),
        "test": str(OUT / "test" / "images"), "nc": len(names), "names": names},
        sort_keys=False))
    print("OBB dataset built:", {s: len(list((OUT / s / "images").glob("*")))
                                 for s in ("train", "valid", "test")})
    print("rotated (polygon-derived) instances per class:", dict(n_poly))

    # OS3: duplicate DD-containing train images x3
    if OUT_OS3.exists():
        shutil.rmtree(OUT_OS3)
    shutil.copytree(OUT, OUT_OS3)
    dd_imgs = 0
    for lbl in (OUT / "train" / "labels").glob("*.txt"):
        if not any(l.split() and int(l.split()[0]) == DD
                   for l in lbl.read_text().splitlines()):
            continue
        dd_imgs += 1
        img = next((OUT / "train" / "images").glob(lbl.stem + ".*"))
        for k in (1, 2):
            shutil.copy2(img, OUT_OS3 / "train" / "images" / f"os{k}_{img.name}")
            shutil.copy2(lbl, OUT_OS3 / "train" / "labels" / f"os{k}_{lbl.name}")
    (OUT_OS3 / "data.yaml").write_text(yaml.safe_dump({
        "train": str(OUT_OS3 / "train" / "images"),
        "val": str(OUT_OS3 / "valid" / "images"),
        "test": str(OUT_OS3 / "test" / "images"), "nc": len(names), "names": names},
        sort_keys=False))
    print(f"OS3: {dd_imgs} DD train imgs duplicated x3; train now",
          len(list((OUT_OS3 / "train" / "images").glob("*"))))


if __name__ == "__main__":
    main()
