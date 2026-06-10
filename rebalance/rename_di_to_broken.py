#!/usr/bin/env python3
"""Rename the `Defective_Insulators` class -> `Broken_Insulator` on eduardos-annotated-photos.

eduardos-annotated-photos carries a generic `Defective_Insulators` class (152 boxes on
151 images) that has no match in the target taxonomy. The target uses Broken_Insulator /
Flashover_Insulator / Self-Exploded_Insulator. Per user request, map ALL generic boxes to
`Broken_Insulator` so the class names match the target.

How: rewrite the local staged VOC labels (Defective_Insulators -> Broken_Insulator, all
other boxes untouched) and re-upload the 151 images with annotation_overwrite=True. Roboflow
dedups to the existing image and replaces its annotation, so the project's class list loses
`Defective_Insulators` and gains `Broken_Insulator` (no new images; 300 stays 300).

NOTE: this collapses any true Flashover/Self-Exploded instances into Broken_Insulator — a
deliberate simplification requested by the user; subtype refinement can be done later.

Safety: DRY-RUN by default (--yes to write). Tag `eduardos_di_rename_broken_v1`.
UNDO: re-run phase2_reimport_di.py (restores Defective_Insulators) or rename the class back.

Run:  .venv/bin/python rebalance/rename_di_to_broken.py            # dry-run
      .venv/bin/python rebalance/rename_di_to_broken.py --yes      # write
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv("/Users/eddie/Research/Vegas/.env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
API = "https://api.roboflow.com"
DEST = "eduardos-annotated-photos"
STAGE = Path("/Users/eddie/Research/Vegas/datasets/eduardos_staging")
IDFILE = STAGE / "reimport_di_ids.txt"
OLD, NEW = "Defective_Insulators", "Broken_Insulator"


def live_classes(project):
    r = requests.get(f"{API}/{WS}/{project}", params={"api_key": KEY}, timeout=60)
    r.raise_for_status()
    return (r.json().get("project", {}) or {}).get("classes", {}) or {}


def rewrite(xml_path, out_path):
    """Copy a VOC label, swapping OLD->NEW class names. Returns (n_swapped, classes)."""
    root = ET.parse(xml_path).getroot()
    n, cls = 0, []
    for o in root.findall("object"):
        name = o.find("name")
        if name is not None and name.text == OLD:
            name.text = NEW
            n += 1
        cls.append(name.text if name is not None else None)
    ET.ElementTree(root).write(out_path)
    return n, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="eduardos_di_rename_broken_v1")
    ap.add_argument("--yes", action="store_true", help="actually write (default: dry-run)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY missing")
    if not IDFILE.exists():
        sys.exit(f"{IDFILE} missing — run rebalance/verify_di_from_source.py first.")

    ids = [x for x in IDFILE.read_text().split() if x]
    outdir = STAGE / "labels_broken"
    outdir.mkdir(exist_ok=True)

    plan, per_class, missing = [], Counter(), []
    for sid in ids:
        img = STAGE / "images" / f"{sid}.jpg"
        lbl = STAGE / "labels" / f"{sid}.xml"
        if not img.exists() or not lbl.exists():
            missing.append(sid); continue
        out = outdir / f"{sid}.xml"
        n, cls = rewrite(lbl, out)
        if n == 0:
            continue  # no DI box here; skip
        per_class.update(c for c in cls if c)
        plan.append((img, out))
    if missing:
        print(f"  WARNING {len(missing)} ids missing local files: {missing[:5]}")

    before = live_classes(DEST)
    print("live BEFORE:", ", ".join(f"{k}:{v}" for k, v in sorted(before.items())))
    print(f"\n==== PLAN: overwrite {len(plan)} images on {DEST} (split=train, tag '{args.batch}') ====")
    print(f"   rename {OLD} -> {NEW}")
    for n in sorted(per_class):
        print(f"   {n:24s} {per_class[n]}")
    print(f"   => {NEW} boxes after: {per_class.get(NEW,0)}; {OLD} remaining: {per_class.get(OLD,0)}")

    if not args.yes:
        print("\nDRY-RUN. Nothing written. Re-run with --yes to apply.")
        return

    from roboflow import Roboflow
    proj = Roboflow(api_key=KEY).workspace(WS).project(DEST)
    ok = fail = 0
    for img, lbl in plan:
        try:
            proj.single_upload(image_path=str(img), annotation_path=str(lbl),
                               annotation_overwrite=True, split="train",
                               batch_name=args.batch, tag_names=[args.batch],
                               num_retry_uploads=3)
            ok += 1
            if ok % 25 == 0:
                print(f"   {ok}/{len(plan)} ...")
        except Exception as e:
            fail += 1
            print(f"   FAIL {img.stem}: {str(e)[:120]}")
    print(f"\nDONE. overwritten={ok} failed={fail}.")
    after = live_classes(DEST)
    print("live AFTER:", ", ".join(f"{k}:{v}" for k, v in sorted(after.items())))


if __name__ == "__main__":
    main()
