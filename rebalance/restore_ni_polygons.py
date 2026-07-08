#!/usr/bin/env python3
"""Restore the polygon geometry of the hand-tightened Normal_Insulators
annotations on the four rewritten projects (atli_target-train/-val/-test and
atli_target-minus-the-cplid).

Background: the NI-tightening overwrite pushed Pascal-VOC XML, which flattens
polygons to rectangles. The polygon vertices live in the merged_atli_target v5
segmentation export (~/atli/downloads/target_v5_seg): label lines with >=8
coords are polygon-tool annotations; 4-value lines are plain boxes.

For every image whose v5 seg label contains at least one polygon line, this
script rebuilds the full annotation as COCO JSON (polygons carry a
`segmentation` array, plain boxes just a bbox) and re-posts it with
overwrite=true to each project where the image exists (IDs taken from the
overwrite manifests, so only images we already rewrote are touched).
atli_target-test's classes are literally "0".."6" -> digit category names.

Verifies after each push that the API now reports `points` on the polygon
annotations. Idempotent; re-run safe.

Run on server:  ~/atli/env/bin/python restore_ni_polygons.py
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import requests
import yaml
from PIL import Image
from dotenv import load_dotenv

ROOT = Path.home() / "atli"
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
API = "https://api.roboflow.com"
WS = "tl-target-set-focus"
SEG = ROOT / "downloads" / "target_v5_seg"
DIGIT_CLASS_PROJECTS = {"atli_target-test"}


def name_key(fname):
    return fname.split(".rf.")[0]


def build_coco(lbl_path, img_path, names):
    W, H = Image.open(img_path).size
    anns = []
    n_poly = 0
    for k, line in enumerate(lbl_path.read_text().splitlines()):
        p = line.split()
        if not p:
            continue
        cid = int(float(p[0])) + 1
        vals = list(map(float, p[1:]))
        if len(vals) == 4:  # plain box: cx cy w h
            cx, cy, w, h = vals
            bbox = [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]
            segm = []
        else:  # polygon vertices
            xs = [v * W for v in vals[0::2]]
            ys = [v * H for v in vals[1::2]]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            segm = [[c for xy in zip(xs, ys) for c in xy]]
            n_poly += 1
        anns.append({"id": k + 1, "image_id": 1, "category_id": cid, "bbox": bbox,
                     "segmentation": segm, "area": bbox[2] * bbox[3], "iscrowd": 0})
    coco = {"images": [{"id": 1, "file_name": name_key(img_path.name),
                        "width": W, "height": H}],
            "annotations": anns,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    return coco, n_poly


def main():
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY not found in ~/atli/.env")
    v5_names = yaml.safe_load((SEG / "data.yaml").read_text())["names"]

    # 1) seg labels that contain at least one polygon line
    poly = {}  # name_key -> (label_path, image_path)
    for lbl in SEG.rglob("labels/*.txt"):
        if any(len(l.split()) >= 9 for l in lbl.read_text().splitlines()):
            imgs = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
            if imgs:
                poly[name_key(lbl.name)] = (lbl, imgs[0])
    print(f"[1] {len(poly)} images carry polygon-tool annotations in the v5 seg export")

    # 2) image IDs per project from the overwrite manifests
    targets = defaultdict(dict)  # project -> name_key -> image_id
    for row in csv.DictReader(open(ROOT / "ni_overwrite_manifest.csv")):
        if row["status"].startswith("OVERWRITE"):
            targets[row["project"]][name_key(row["file"])] = row["image_id"]
    for row in csv.DictReader(open(ROOT / "minus_cplid_overwrite_manifest.csv")):
        if row["status"].startswith("OVERWRITE"):
            targets["atli_target-minus-the-cplid"][name_key(row["file"])] = row["image_id"]

    session = requests.Session()
    pushed = verified = failed = 0
    for project, ids in sorted(targets.items()):
        names = ([str(i) for i in range(len(v5_names))]
                 if project in DIGIT_CLASS_PROJECTS else v5_names)
        todo = {k: v for k, v in ids.items() if k in poly}
        print(f"\n[{project}] {len(todo)} polygon images to restore")
        for key_, iid in sorted(todo.items()):
            lbl, img = poly[key_]
            coco, n_poly = build_coco(lbl, img, names)
            r = session.post(f"{API}/dataset/{project}/annotate/{iid}",
                             params={"api_key": KEY, "name": key_ + ".json",
                                     "overwrite": "true"},
                             data=json.dumps(coco),
                             headers={"Content-Type": "text/plain"}, timeout=60)
            if r.status_code != 200 or '"error"' in r.text[:150]:
                failed += 1
                print(f"  FAIL push {key_}: HTTP {r.status_code} {r.text[:100]}")
                continue
            pushed += 1
            g = session.get(f"{API}/{WS}/{project}/images/{iid}",
                            params={"api_key": KEY}, timeout=30)
            boxes = ((g.json().get("image", {}) or {}).get("annotation") or {}).get("boxes", [])
            got = sum(1 for b in boxes if b.get("points"))
            if got >= n_poly:
                verified += 1
            else:
                print(f"  WARN {key_}: pushed {n_poly} polygons, API shows {got}")
    print(f"\nDone: pushed={pushed} verified={verified} failed={failed}")


if __name__ == "__main__":
    main()
