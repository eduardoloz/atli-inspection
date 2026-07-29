#!/usr/bin/env python3
"""fig_obb_before_after.png — axis-aligned vs OBB *model predictions*, seamless 6x2 mosaic.

Top row: YOLOv11n detection champion (EDU_champosall_v11, det+osall @1280).
Bottom row: YOLOv11n OBB champion (yolov11n_champion_deg15, OBB+osall+deg15 @1280).
Every box is a real prediction labeled "<class> <confidence>". Bare mosaic:
no title, no caption, no row labels (per 2026-07-28 request).

Each column's image sits in the *held-out* split of the fold whose checkpoint
predicts it (all fold2 here), so no model ever trained on the image it is
shown predicting.

Columns (stem, fold): edu_TJB8... (f2 val, Defective_Damper), 0638 (f2 test,
Normal_Insulators close-up), Screenshot-2025-06-26 (f2 val, Normal_Damper row),
1161 (f2 test, Self-Exploded_Insulator), 000068 (f2 test, 3x Birdnest scene),
LTFIGIVP2... (f2 test, 2x Self-Exploded). The old col-4 image (170041,
Flashover) was dropped: its only clean checkpoint (fold4) detects nothing on
it even at conf 0.03.

NOTE on ground-truth geometry: in the CV_eduardo_obb export, Birdnest /
Broken / Flashover / Self-Exploded GT boxes are 100% axis-aligned (annotated
as rectangles in Roboflow, not polygons); only the damper + Normal_Insulators
classes carry real polygon-derived rotations (14-18%). The OBB *predictions*
shown here can still rotate on any class.

Weights (not committed — 6 MB each):
  OBB : gh release download weights-eduardo-cv-2026-07-22 -p 'yolov11n_champion_deg15_fold2.pt'
  det : scp $ATLI_SERVER:~/atli/runs/EDU_champosall_v11_f2_s2/weights/best.pt yolov11n_det_champosall_fold2.pt

Usage:
  python generate_obb_vs_det_predictions.py --weights-dir <dir> [--imgs-dir obb_panel_assets] [--out fig_obb_before_after.png]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

HERE = Path(__file__).parent

COLS = [  # (stem, fold)
    # 2026-07-29 4-panel rework: kept old cols 1/4/5, added a Broken_Insulator
    # panel (100137, fold0 test — det champion finds 0 of 3 BI, OBB finds all;
    # picked from a 336-image det-vs-OBB sweep of every BI holdout image)
    ("edu_TJB8gulZBwN7wzGjaDir_jpg.rf.276eaf5e62a55ce219d32b4907c3ac49", 2),
    ("100137_JPG.rf.823e68591ffe9d0d9c6bc08a580078", 0),
    ("1161_jpg.rf.118cfacf8cadfd4dc1dd4c99f4f611f4", 2),
    ("000068_jpg.rf.11f57b197c4fbab0f0983b30385e352e", 2),
]

COLORS = {  # match the annotation-figure palette: DD green, NI teal, ND purple, defects warm
    "Defective_Damper": (34, 139, 34),
    "Normal_Insulators": (0, 132, 180),
    "Normal_Damper": (128, 0, 160),
    "Flashover_Insulator": (200, 30, 30),
    "Self-Exploded_Insulator": (230, 120, 0),
    "Broken_Insulator": (180, 0, 90),
    "Birdnest": (90, 60, 20),
}

TILE = 840          # each cell rendered at 840x840 -> mosaic 3360 wide like the old figure
CONF = 0.25


def load_font(size):
    for name in ("Arial Bold.ttf", "Helvetica.ttc", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw, x, y, text, color, font, tile_w, placed):
    pad = 4
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    x = min(max(x, 0), tile_w - w)

    def overlaps(rect):
        return any(rect[0] < p[2] and rect[2] > p[0] and rect[1] < p[3] and rect[3] > p[1] for p in placed)

    # anchor above the box corner, else below it, then nudge down until collision-free
    candidates = [y - h] + [y + dy for dy in range(0, tile_w, h + 2)]
    rect = None
    for cy in candidates:
        if cy < 0 or cy + h > tile_w:
            continue
        rect = (x, cy, x + w, cy + h)
        if not overlaps(rect):
            break
    if rect is None:
        rect = (x, max(y - h, 0), x + w, max(y - h, 0) + h)
    placed.append(rect)
    draw.rectangle(rect, fill=color)
    draw.text((rect[0] + pad, rect[1] + pad - t), text, fill=(255, 255, 255), font=font)


def render_tile(img_path, preds, font, lw):
    im = Image.open(img_path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    s = TILE / 640.0
    placed = []
    for cls, conf, kind, geom in sorted(preds, key=lambda p: -p[1]):  # high conf gets first pick of label spots
        color = COLORS.get(cls, (255, 255, 255))
        if kind == "aabb":
            x1, y1, x2, y2 = [v * s for v in geom]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            lx, ly = x1, y1
        else:
            pts = [(geom[i] * s, geom[i + 1] * s) for i in range(0, 8, 2)]
            draw.polygon(pts, outline=color, width=lw)
            lx, ly = min(p[0] for p in pts), min(p[1] for p in pts)
        draw_label(draw, lx, ly, f"{cls.replace('_', ' ')} {conf:.2f}", color, font, TILE, placed)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--imgs-dir", type=Path, default=HERE / "obb_panel_assets")
    ap.add_argument("--out", type=Path, default=HERE / "fig_obb_before_after.png")
    args = ap.parse_args()

    folds = {f for _, f in COLS}
    det = {f: YOLO(str(args.weights_dir / f"yolov11n_det_champosall_fold{f}.pt")) for f in folds}
    obb = {f: YOLO(str(args.weights_dir / f"yolov11n_champion_deg15_fold{f}.pt")) for f in folds}

    font = load_font(26)
    tiles = {"aabb": [], "obb": []}
    for stem, fold in COLS:
        img = args.imgs_dir / f"{stem}.jpg"
        r = det[fold].predict(str(img), imgsz=1280, conf=CONF, device="cpu", verbose=False)[0]
        preds = [(det[fold].names[int(b.cls)], float(b.conf), "aabb", b.xyxy[0].tolist()) for b in r.boxes]
        tiles["aabb"].append(render_tile(img, preds, font, 5))

        r = obb[fold].predict(str(img), imgsz=1280, conf=CONF, device="cpu", verbose=False)[0]
        preds = [(obb[fold].names[int(r.obb.cls[i])], float(r.obb.conf[i]), "obb",
                  r.obb.xyxyxyxy[i].reshape(-1).tolist()) for i in range(len(r.obb))]
        tiles["obb"].append(render_tile(img, preds, font, 5))

    # Figure-1-style layout: white gutters between photos (and rows), white border
    GAP, MARGIN = 28, 12
    n = len(COLS)
    W = MARGIN * 2 + n * TILE + (n - 1) * GAP
    H = MARGIN * 2 + 2 * TILE + GAP
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    for r, row in enumerate(("aabb", "obb")):
        for c, t in enumerate(tiles[row]):
            canvas.paste(t, (MARGIN + c * (TILE + GAP), MARGIN + r * (TILE + GAP)))
    canvas.save(args.out)
    print("wrote", args.out, canvas.size)


if __name__ == "__main__":
    main()
