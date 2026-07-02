#!/usr/bin/env python3
"""Build 5-fold CV with OBB (oriented bounding box) labels.

Downloads polygon (segmentation) annotations from Roboflow, converts each
polygon to a minimum-area rotated rectangle via OpenCV's minAreaRect, and
writes YOLO-OBB format labels (class x1 y1 x2 y2 x3 y3 x4 y4, normalised).

Same 70/15/15 stratified splits and seeds as build_cv_proper.py so results
are directly comparable.

Builds three dataset variants per fold:
  - base.yaml:  train (no oversample)
  - champ.yaml: trainos (DD 3x oversample)
  - osall.yaml: trainosall (all defect 3x oversample)

Run on server: ~/atli/env/bin/python build_cv_obb.py
"""
import os, shutil, yaml, numpy as np, cv2
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = Path.home() / "atli"
DL = ROOT / "downloads"
DL.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
assert KEY, "ROBOFLOW_API_KEY missing"

POOL_SEG = ROOT / "Merged_Dataset_seg"       # segmentation-format pool
POOL_OBB = ROOT / "Merged_Dataset_obb"       # converted OBB labels
OUT = ROOT / "Merged_CV_obb"

NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
DD = 2
DEFECTS = {1, 2, 3, 6}


# ── Step 1: Download segmentation-format annotations from Roboflow ──────────

def rf_download_seg(project, version, dest):
    """Download in yolov5 segmentation format (polygon vertices in labels)."""
    from roboflow import Roboflow
    dest = Path(dest)
    if dest.exists() and any(dest.rglob("labels/*.txt")):
        print(f"  [skip] {dest} already downloaded")
        return dest
    rf = Roboflow(api_key=KEY)
    rf.workspace(WS).project(project).version(version).download(
        "yolov5", location=str(dest))
    return dest


def pairs(root):
    out = []
    for lbl in sorted(Path(root).rglob("labels/*.txt")):
        hits = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
        if hits:
            out.append((lbl, hits[0]))
    return out


# ── Step 2: Convert segmentation polygons → OBB (rotated rectangles) ────────

def seg_line_to_obb(line, img_w, img_h):
    """Convert one YOLO-seg label line to YOLO-OBB format.

    Input:  class_id x1 y1 x2 y2 ... xN yN  (normalised polygon)
    Output: class_id x1 y1 x2 y2 x3 y3 x4 y4  (normalised rotated rect corners)

    If the annotation is already a bounding box (4 values = cx cy w h),
    it's converted to an axis-aligned OBB (4 corners).
    """
    parts = line.strip().split()
    cls = int(parts[0])
    coords = [float(v) for v in parts[1:]]

    if len(coords) == 4:
        # Detection format: cx cy w h → axis-aligned rectangle corners
        cx, cy, w, h = coords
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy - h / 2
        x3, y3 = cx + w / 2, cy + h / 2
        x4, y4 = cx - w / 2, cy + h / 2
        return cls, [x1, y1, x2, y2, x3, y3, x4, y4]

    # Segmentation format: pairs of x, y
    assert len(coords) >= 6 and len(coords) % 2 == 0, \
        f"Bad polygon: {len(coords)} values"

    # Denormalise to pixel coords for minAreaRect
    pts = []
    for i in range(0, len(coords), 2):
        px = coords[i] * img_w
        py = coords[i + 1] * img_h
        pts.append([px, py])
    pts = np.array(pts, dtype=np.float32)

    # Minimum-area rotated rectangle
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)  # 4 corner points

    # Normalise back to [0, 1]
    obb = []
    for corner in box:
        obb.append(float(np.clip(corner[0] / img_w, 0, 1)))
        obb.append(float(np.clip(corner[1] / img_h, 0, 1)))

    return cls, obb


def convert_label_to_obb(seg_lbl, img_path, obb_lbl):
    """Convert a full segmentation label file to OBB format."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_h, img_w = img.shape[:2]

    lines_out = []
    for line in Path(seg_lbl).read_text().splitlines():
        if not line.strip():
            continue
        cls, obb_coords = seg_line_to_obb(line, img_w, img_h)
        coord_str = " ".join(f"{v:.6f}" for v in obb_coords)
        lines_out.append(f"{cls} {coord_str}")

    Path(obb_lbl).write_text("\n".join(lines_out) + "\n" if lines_out else "")


# ── Step 3: Build merged segmentation pool + convert to OBB ─────────────────

print("=" * 60)
print("STEP 1: Download segmentation annotations")
print("=" * 60)

# Target dataset (v5 = tightened NI polygons)
tgt_seg = rf_download_seg("merged_atli_target", 5, DL / "target_v5_seg")
tgt_pairs = pairs(tgt_seg)
print(f"  target_v5_seg: {len(tgt_pairs)} images")

# Eduardo's annotated photos
edu_seg = rf_download_seg("eduardos-annotated-photos", 1, DL / "eduardo_seg")
edu_pairs = pairs(edu_seg)
print(f"  eduardo_seg: {len(edu_pairs)} images")

# ── Merge into a single pool (segmentation format) ──────────────────────────
print("\nSTEP 2: Merge into segmentation pool")
for sub in ("images", "labels"):
    shutil.rmtree(POOL_SEG / sub, ignore_errors=True)
    (POOL_SEG / sub).mkdir(parents=True, exist_ok=True)

for lbl, img in tgt_pairs:
    shutil.copy2(img, POOL_SEG / "images" / img.name)
    shutil.copy2(lbl, POOL_SEG / "labels" / (img.stem + ".txt"))
for lbl, img in edu_pairs:
    shutil.copy2(img, POOL_SEG / "images" / f"rf_{img.name}")
    shutil.copy2(lbl, POOL_SEG / "labels" / f"rf_{lbl.name}")
n_pool = len(list((POOL_SEG / "images").glob("*")))
print(f"  merged pool: {n_pool} images")

# ── Convert segmentation → OBB ──────────────────────────────────────────────
print("\nSTEP 3: Convert segmentation labels → OBB (minAreaRect)")
for sub in ("images", "labels"):
    shutil.rmtree(POOL_OBB / sub, ignore_errors=True)
    (POOL_OBB / sub).mkdir(parents=True, exist_ok=True)

converted, skipped = 0, 0
for img_path in sorted((POOL_SEG / "images").glob("*")):
    seg_lbl = POOL_SEG / "labels" / (img_path.stem + ".txt")
    if not seg_lbl.exists():
        skipped += 1
        continue
    obb_lbl = POOL_OBB / "labels" / (img_path.stem + ".txt")
    shutil.copy2(img_path, POOL_OBB / "images" / img_path.name)
    convert_label_to_obb(seg_lbl, img_path, obb_lbl)
    converted += 1
print(f"  converted: {converted}, skipped (no label): {skipped}")


# ── Step 4: Build 5-fold CV from OBB pool ───────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Build 5-fold 70/15/15 CV (same seeds as proper CV)")
print("=" * 60)

# Collect all labeled images from OBB pool
rows, tg = [], []
for img in sorted((POOL_OBB / "images").glob("*")):
    lbl = POOL_OBB / "labels" / (img.stem + ".txt")
    if not lbl.exists():
        continue
    v = np.zeros(7, int)
    for line in lbl.read_text().splitlines():
        if line.strip():
            c = int(line.split()[0])
            if 0 <= c < 7:
                v[c] = 1
    if v.sum():
        rows.append((img, lbl))
        tg.append(v)
tg = np.array(tg)
idx = np.arange(len(rows))
print(f"Total labeled OBB images: {len(rows)}")

# Same seeds as build_cv_proper.py for direct comparison
SEEDS = [42, 123, 456, 789, 1024]

for k, seed in enumerate(SEEDS):
    f = OUT / f"fold{k}"

    # 70% train vs 30% temp
    sp1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sp1.split(idx, tg))

    # Split 30% into 15% val + 15% test
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

    # Write YAMLs — NOTE: OBB task uses same yaml structure as detect
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

print("CV_OBB_BUILD_DONE")
