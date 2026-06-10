#!/usr/bin/env python3
"""Re-import the stripped `Defective_Insulators` labels back onto eduardos-annotated-photos.

Context: eduardos-annotated-photos was staged FROM ATLI_source_dataset. 151 of its 300
images carried `Defective_Insulators` boxes in source (verified live, 0 discrepancies by
rebalance/verify_di_from_source.py). Those boxes were later stripped from the Roboflow
project. This restores them.

How: the local staged files in datasets/eduardos_staging/{images,labels} are byte-faithful
copies of the source images + their FULL annotations (DI + co-occurring normals; tower/line
already excluded to match the target taxonomy). We re-upload the 151 DI images with
annotation_overwrite=True, so Roboflow dedups to the EXISTING image and replaces its
annotation with the complete one — restoring the 152 DI boxes, normals unchanged. No new
images are created.

Safety:
  * DRY-RUN by default; pass --yes to actually write.
  * Only touches the 151 images that have a Defective_Insulators box (from reimport_di_ids.txt).
  * Boxes return as GENERIC `Defective_Insulators` — subtyping into Broken_/Flashover_/
    Self-Exploded_Insulator is still a separate manual step before promoting to the target.
  * Tagged/batched `eduardos_di_reimport_v1`. UNDO: in Roboflow, delete the
    `Defective_Insulators` class (it is currently absent, so removing it cleanly reverts).

Run:  .venv/bin/python rebalance/phase2_reimport_di.py            # dry-run plan
      .venv/bin/python rebalance/phase2_reimport_di.py --yes      # write
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
TARGET_CLASS = "Defective_Insulators"


def label_classes(xml_path):
    root = ET.parse(xml_path).getroot()
    return [o.findtext("name") for o in root.findall("object")]


def live_classes(project):
    """Read-only: current class->count on the destination project."""
    r = requests.get(f"{API}/{WS}/{project}", params={"api_key": KEY}, timeout=60)
    r.raise_for_status()
    # /{ws}/{project} returns project.classes as {name: count}
    return (r.json().get("project", {}) or {}).get("classes", {}) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="eduardos_di_reimport_v1")
    ap.add_argument("--yes", action="store_true", help="actually write (default: dry-run)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY missing")
    if not IDFILE.exists():
        sys.exit(f"{IDFILE} missing — run rebalance/verify_di_from_source.py first.")

    ids = [x for x in IDFILE.read_text().split() if x]
    print(f"re-import candidates (from {IDFILE.name}): {len(ids)}")

    # validate local files + tally what will be (re)written
    plan, missing, no_di = [], [], []
    per_class = Counter()
    for sid in ids:
        img = STAGE / "images" / f"{sid}.jpg"
        lbl = STAGE / "labels" / f"{sid}.xml"
        if not img.exists() or not lbl.exists():
            missing.append(sid); continue
        cls = label_classes(lbl)
        if TARGET_CLASS not in cls:
            no_di.append(sid); continue
        per_class.update(cls)
        plan.append((img, lbl))

    if missing:
        print(f"  WARNING {len(missing)} ids have no local image/label: {missing[:5]}")
    if no_di:
        print(f"  WARNING {len(no_di)} ids' local label lacks {TARGET_CLASS}: {no_di[:5]}")

    # before-snapshot from live project (read-only)
    try:
        before = live_classes(DEST)
        print(f"\nlive {DEST} BEFORE: " +
              ", ".join(f"{k}:{v}" for k, v in sorted(before.items())))
        if TARGET_CLASS in before:
            print(f"  NOTE: {TARGET_CLASS} already present ({before[TARGET_CLASS]}) — "
                  "re-import may double boxes; review before --yes.")
    except Exception as e:
        print(f"  (could not read live classes: {e})")

    print(f"\n==== PLAN: overwrite annotations on {len(plan)} images -> {DEST} "
          f"(split=train, tag '{args.batch}') ====")
    for n in sorted(per_class):
        print(f"   {n:24s} {per_class[n]}")
    print(f"   => restores {per_class[TARGET_CLASS]} {TARGET_CLASS} boxes "
          f"on {len(plan)} images")

    if not args.yes:
        print("\nDRY-RUN. Nothing written. Re-run with --yes to apply.")
        print("After applying: subtype each Defective_Insulators box into Broken_/Flashover_/"
              "Self-Exploded_Insulator before promoting to the target.")
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
    print(f"Verify: re-run rebalance/verify_di_from_source.py-style check or explore_roboflow.py "
          f"(expect {TARGET_CLASS} ~{per_class[TARGET_CLASS]} on {DEST}).")
    print(f"UNDO: in Roboflow delete the '{TARGET_CLASS}' class to revert.")


if __name__ == "__main__":
    main()
