#!/usr/bin/env python3
"""Draw GT vs predicted boxes on hard negative images for visual review.
Green = ground truth, Red = model prediction. Class name shown on each box."""
import cv2
import os
from pathlib import Path

NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]

HERE = Path(__file__).parent
IMG_DIR = HERE / "review" / "images"
GT_DIR = HERE / "review" / "gt_labels"
PRED_DIR = HERE / "review" / "pred_labels"
OUT_DIR = HERE / "review_annotated"
OUT_DIR.mkdir(exist_ok=True)


def parse_yolo(label_file, img_w, img_h):
    boxes = []
    if not label_file.exists():
        return boxes
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


for img_path in sorted(IMG_DIR.glob("*")):
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    # Draw GT boxes (green)
    gt_boxes = parse_yolo(GT_DIR / f"{stem}.txt", w, h)
    for cls, x1, y1, x2, y2 in gt_boxes:
        color = (0, 200, 0)  # green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"GT: {NAMES[cls]}"
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw predicted boxes (red)
    pred_boxes = parse_yolo(PRED_DIR / f"{stem}.txt", w, h)
    for cls, x1, y1, x2, y2 in pred_boxes:
        color = (0, 0, 220)  # red
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"PRED: {NAMES[cls]}"
        cv2.putText(img, label, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = OUT_DIR / f"{stem}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"  {stem}")
    # Show what went wrong
    gt_cls = {NAMES[c] for c, *_ in gt_boxes}
    pred_cls = {NAMES[c] for c, *_ in pred_boxes}
    missing = gt_cls - pred_cls
    added = pred_cls - gt_cls
    if missing:
        print(f"    MISSED: {missing}")
    if added:
        print(f"    FALSE: {added}")

print(f"\nAnnotated images saved to: {OUT_DIR}")
