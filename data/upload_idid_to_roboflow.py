#!/usr/bin/env python3
"""Upload EPRI IDID dataset to Roboflow after vetting against ATLI.

Steps:
1. Extract IDID zip(s) and inventory images + annotations
2. Parse IDID annotations (CSV format: image_id, Class_ID conf xmin ymin xmax ymax)
3. Convert to YOLO format (class cx cy w h)
4. pHash-check against ATLI merged dataset to flag leakage
5. Upload clean images to a new Roboflow project (dry-run by default)

IDID classes → ATLI mapping:
  0: Broken insulator shell  → Broken_Insulator (class 1)
  1: Flashover damage shell  → Flashover_Insulator (class 3)
  2: Good insulator shell    → Normal_Insulators (class 5)
  3: Insulator string        → (parent object, no direct ATLI equivalent — skip or map)

Usage:
  # Dry run (no upload):
  python upload_idid_to_roboflow.py /path/to/Train_IDID_V1.2

  # Vet + upload:
  python upload_idid_to_roboflow.py /path/to/Train_IDID_V1.2 --yes

  # Run on server:
  ~/atli/env/bin/python upload_idid_to_roboflow.py ~/atli/downloads/idid
"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
ATLI_NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
              "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]

# IDID class → ATLI class index
IDID_TO_ATLI = {
    0: 1,   # Broken insulator shell → Broken_Insulator
    1: 3,   # Flashover damage       → Flashover_Insulator
    2: 5,   # Good insulator shell   → Normal_Insulators
    3: None, # Insulator string (parent) → skip (or keep as-is)
}

IDID_CLASS_NAMES = {
    0: "Broken_insulator_shell",
    1: "Flashover_damage_shell",
    2: "Good_insulator_shell",
    3: "Insulator_string",
}

NEAR_THRESHOLD = 8  # pHash hamming distance for near-duplicate


def find_images(root):
    """Find all image files in directory tree."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts and p.is_file())


def parse_idid_json(json_path):
    """Parse IDID JSON annotation file.

    IDID uses JSON with bounding box annotations per image.
    Returns: {image_filename: [(class_id, xmin, ymin, xmax, ymax), ...]}
    """
    annotations = defaultdict(list)
    with open(json_path) as f:
        data = json.load(f)

    # Handle different possible JSON structures
    if isinstance(data, list):
        for item in data:
            img_id = item.get("image_id") or item.get("file_name") or item.get("filename")
            if img_id is None:
                continue
            # Normalize filename
            img_id = str(img_id)
            if not any(img_id.lower().endswith(e) for e in [".jpg", ".png", ".jpeg"]):
                img_id = img_id + ".jpg"

            boxes = item.get("annotations") or item.get("boxes") or []
            if isinstance(boxes, list):
                for box in boxes:
                    cls = box.get("class_id") or box.get("category_id") or box.get("label", 0)
                    xmin = box.get("xmin") or box.get("x1") or box.get("bbox", [0])[0]
                    ymin = box.get("ymin") or box.get("y1") or box.get("bbox", [0, 0])[1]
                    xmax = box.get("xmax") or box.get("x2") or 0
                    ymax = box.get("ymax") or box.get("y2") or 0
                    if xmax > 0 and ymax > 0:
                        annotations[img_id].append((int(cls), xmin, ymin, xmax, ymax))
    elif isinstance(data, dict):
        # Could be {filename: [annotations]} or {"images": [...], "annotations": [...]}
        if "images" in data and "annotations" in data:
            # COCO-style
            img_map = {img["id"]: img["file_name"] for img in data["images"]}
            img_sizes = {img["id"]: (img.get("width", 1), img.get("height", 1)) for img in data["images"]}
            for ann in data["annotations"]:
                img_id = img_map.get(ann["image_id"], str(ann["image_id"]))
                bbox = ann.get("bbox", [0, 0, 0, 0])
                cls = ann.get("category_id", 0)
                # COCO bbox is [x, y, width, height]
                xmin, ymin, w, h = bbox
                xmax, ymax = xmin + w, ymin + h
                annotations[img_id].append((int(cls), xmin, ymin, xmax, ymax))
        else:
            for img_id, anns in data.items():
                if isinstance(anns, list):
                    for ann in anns:
                        if isinstance(ann, dict):
                            cls = ann.get("class_id", ann.get("category_id", 0))
                            annotations[img_id].append((
                                int(cls),
                                ann.get("xmin", 0), ann.get("ymin", 0),
                                ann.get("xmax", 0), ann.get("ymax", 0),
                            ))

    return dict(annotations)


def parse_idid_csv(csv_path):
    """Parse IDID CSV annotation file.

    Format: image_id, "Class_ID confidence xmin ymin xmax ymax"
    Returns: {image_filename: [(class_id, xmin, ymin, xmax, ymax), ...]}
    """
    annotations = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            img_id = row[0].strip()
            pred_str = row[1].strip()
            # Parse "ClassID conf xmin ymin xmax ymax ClassID conf xmin ymin xmax ymax ..."
            parts = pred_str.split()
            i = 0
            while i + 5 < len(parts):
                cls = int(parts[i])
                # conf = float(parts[i+1])  # skip confidence
                xmin = float(parts[i + 2])
                ymin = float(parts[i + 3])
                xmax = float(parts[i + 4])
                ymax = float(parts[i + 5])
                annotations[img_id].append((cls, xmin, ymin, xmax, ymax))
                i += 6

    return dict(annotations)


def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert absolute xyxy to normalized YOLO cx cy w h."""
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def main():
    parser = argparse.ArgumentParser(description="Upload IDID to Roboflow after vetting")
    parser.add_argument("idid_path", help="Path to extracted IDID directory")
    parser.add_argument("--yes", action="store_true", help="Actually upload (default: dry-run)")
    parser.add_argument("--atli", default=None, help="Path to ATLI Merged_Dataset/images for pHash check")
    args = parser.parse_args()

    idid_root = Path(args.idid_path)
    assert idid_root.exists(), f"IDID path not found: {idid_root}"

    # ── 1. Inventory IDID ───────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Inventory IDID dataset")
    print("=" * 60)
    images = find_images(idid_root)
    print(f"  Found {len(images)} images")

    # Find annotation files
    json_files = sorted(idid_root.rglob("*.json"))
    csv_files = sorted(idid_root.rglob("*.csv"))
    print(f"  JSON files: {[f.name for f in json_files]}")
    print(f"  CSV files:  {[f.name for f in csv_files]}")

    # ── 2. Parse annotations ────────────────────────────────────────────────
    print("\nSTEP 2: Parse annotations")
    annotations = {}
    for jf in json_files:
        print(f"  Parsing JSON: {jf.name}")
        annotations.update(parse_idid_json(jf))
    for cf in csv_files:
        print(f"  Parsing CSV: {cf.name}")
        annotations.update(parse_idid_csv(cf))

    print(f"  Annotated images: {len(annotations)}")
    # Count classes
    cls_count = Counter()
    for img_id, anns in annotations.items():
        for cls, *_ in anns:
            cls_count[cls] += 1
    print(f"  Class distribution:")
    for cls_id, count in sorted(cls_count.items()):
        name = IDID_CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
        atli_cls = IDID_TO_ATLI.get(cls_id)
        atli_name = ATLI_NAMES[atli_cls] if atli_cls is not None else "SKIP"
        print(f"    {cls_id} ({name}): {count} → ATLI class {atli_cls} ({atli_name})")

    # ── 3. Convert to YOLO format ───────────────────────────────────────────
    print("\nSTEP 3: Convert annotations to YOLO format")
    staging = idid_root.parent / "idid_staging"
    for sub in ("images", "labels"):
        shutil.rmtree(staging / sub, ignore_errors=True)
        (staging / sub).mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_no_ann = 0
    skipped_no_atli = 0
    for img_path in images:
        # Match image to annotation
        img_key = img_path.stem
        # Try various key formats
        anns = None
        for key in [img_path.name, img_path.stem, img_path.stem + ".jpg", img_path.stem + ".png"]:
            if key in annotations:
                anns = annotations[key]
                break

        if anns is None:
            skipped_no_ann += 1
            continue

        # Read image dimensions
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception:
            continue

        # Convert to YOLO
        yolo_lines = []
        for cls_id, xmin, ymin, xmax, ymax in anns:
            atli_cls = IDID_TO_ATLI.get(cls_id)
            if atli_cls is None:
                skipped_no_atli += 1
                continue
            cx, cy, w, h = xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            yolo_lines.append(f"{atli_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            shutil.copy2(img_path, staging / "images" / img_path.name)
            (staging / "labels" / (img_path.stem + ".txt")).write_text("\n".join(yolo_lines) + "\n")
            converted += 1

    print(f"  Converted: {converted} images")
    print(f"  Skipped (no annotation): {skipped_no_ann}")
    print(f"  Skipped annotations (no ATLI mapping, class 3 'Insulator string'): {skipped_no_atli}")

    # ── 4. pHash vet against ATLI ───────────────────────────────────────────
    atli_path = args.atli
    if atli_path is None:
        # Try default server path
        default = Path.home() / "atli" / "Merged_Dataset" / "images"
        if default.exists():
            atli_path = str(default)

    if atli_path and Path(atli_path).exists():
        print(f"\nSTEP 4: pHash vet against ATLI ({atli_path})")
        atli_imgs = find_images(atli_path)
        print(f"  Hashing {len(atli_imgs)} ATLI images...")
        atli_hashes = {}
        for p in atli_imgs:
            try:
                atli_hashes[p] = imagehash.phash(Image.open(p))
            except Exception:
                pass

        print(f"  Hashing {converted} IDID images...")
        idid_hashes = {}
        staged_imgs = find_images(staging / "images")
        for p in staged_imgs:
            try:
                idid_hashes[p] = imagehash.phash(Image.open(p))
            except Exception:
                pass

        print("  Finding overlaps...")
        exact, near = 0, 0
        leaked = []
        for pi, hi in idid_hashes.items():
            best_d = 999
            best_match = None
            for pa, ha in atli_hashes.items():
                d = hi - ha
                if d < best_d:
                    best_d = d
                    best_match = pa
            if best_d == 0:
                exact += 1
                leaked.append(pi)
            elif best_d <= NEAR_THRESHOLD:
                near += 1
                leaked.append(pi)

        print(f"  Exact duplicates (d=0): {exact}")
        print(f"  Near duplicates (d<={NEAR_THRESHOLD}): {near}")
        print(f"  Total leaked: {len(leaked)} / {len(idid_hashes)}")

        if leaked:
            print(f"  Removing {len(leaked)} leaked images from staging...")
            for p in leaked:
                p.unlink(missing_ok=True)
                lbl = staging / "labels" / (p.stem + ".txt")
                lbl.unlink(missing_ok=True)
            clean_count = len(list((staging / "images").glob("*")))
            print(f"  Clean images remaining: {clean_count}")
    else:
        print("\nSTEP 4: SKIPPED (no ATLI path found — run on server or pass --atli)")

    # ── 5. Upload to Roboflow ───────────────────────────────────────────────
    final_imgs = list((staging / "images").glob("*"))
    print(f"\nSTEP 5: Upload to Roboflow")
    print(f"  Images to upload: {len(final_imgs)}")

    if not args.yes:
        print("  DRY RUN — pass --yes to actually upload")
        print(f"  Staging directory: {staging}")
        print("DONE (dry run)")
        return

    load_dotenv(Path.home() / "atli" / ".env")
    KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
    assert KEY, "ROBOFLOW_API_KEY missing"

    from roboflow import Roboflow
    rf = Roboflow(api_key=KEY)
    project = rf.workspace(WS).project("epri-idid-insulators")  # create this project first in Roboflow UI

    uploaded = 0
    for img_path in sorted(final_imgs):
        lbl_path = staging / "labels" / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        try:
            project.upload(
                image_path=str(img_path),
                annotation_path=str(lbl_path),
                annotation_format="yolov5",
                tag_names=["epri_idid_v1.2"],
                batch_name="idid_upload",
            )
            uploaded += 1
            if uploaded % 100 == 0:
                print(f"  Uploaded {uploaded}...")
        except Exception as e:
            print(f"  FAILED: {img_path.name}: {e}")

    print(f"\n  Total uploaded: {uploaded}")
    print("DONE")


if __name__ == "__main__":
    main()
