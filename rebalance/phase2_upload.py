#!/usr/bin/env python3
"""Phase 2 (THE ONLY STEP THAT WRITES TO ROBOFLOW): upload the relabeled staging set
into merged_ATLI_target's TRAIN split, tagged so it is easy to find / roll back.

Safety:
  * Dry-run by default. It will NOT upload anything unless you pass --yes.
  * Refuses to run while any box is still 'DEFECTIVE_INSULATOR_TODO' (forces you to
    finish subtype labeling first).
  * Every image is uploaded with split="train" and batch/tag "zoomout_defective_v1"
    -> in Roboflow you can filter by that tag and bulk-delete to undo.
  * Converts YOLO labels -> Pascal-VOC XML with the EXACT target class names, so
    Roboflow merges into the existing classes instead of creating new ones.

Run:  python rebalance/phase2_upload.py              # dry-run, prints the plan
      python rebalance/phase2_upload.py --yes        # actually upload
"""
import argparse
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from dotenv import load_dotenv

load_dotenv()
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()

TODO = "DEFECTIVE_INSULATOR_TODO"
# Must match merged_ATLI_target's class spellings EXACTLY (incl. the hyphen / plural):
TARGET_CLASSES = {"Broken_Insulator", "Flashover_Insulator", "Self-Exploded_Insulator",
                  "Normal_Damper", "Normal_Insulators", "Defective_Damper",
                  "Birdnest"}


def load_names(staging):
    return (Path(staging) / "classes.txt").read_text().split()


def yolo_to_voc(img_path, lbl_path, names):
    """Build a Pascal-VOC XML string from a YOLO label (class names as strings)."""
    from PIL import Image
    W, H = Image.open(img_path).size
    objs = []
    for line in Path(lbl_path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        name = names[int(float(p[0]))]
        cx, cy, w, h = map(float, p[1:5])
        xmin, ymin = int((cx - w / 2) * W), int((cy - h / 2) * H)
        xmax, ymax = int((cx + w / 2) * W), int((cy + h / 2) * H)
        objs.append((name, max(0, xmin), max(0, ymin), min(W, xmax), min(H, ymax)))
    parts = [f"<annotation><filename>{escape(img_path.name)}</filename>",
             f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"]
    for name, a, b, c, d in objs:
        parts.append(f"<object><name>{escape(name)}</name><bndbox>"
                     f"<xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax>"
                     f"</bndbox></object>")
    parts.append("</annotation>")
    return "".join(parts), objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="datasets/staging")
    ap.add_argument("--project", default="merged_atli_target")
    ap.add_argument("--batch", default="zoomout_defective_v1")
    ap.add_argument("--yes", action="store_true", help="actually upload (default: dry-run)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY missing from .env")
    names = load_names(args.staging)

    imgs = sorted((Path(args.staging) / "images").glob("*"))
    if not imgs:
        sys.exit(f"No images in {args.staging}/images — run phase1 first.")

    # validate: no unresolved TODO, and every class name exists in the target taxonomy
    bad_todo, bad_name, total_boxes, per_class = [], set(), 0, {}
    voc = {}
    for img in imgs:
        lbl = Path(args.staging) / "labels" / (img.stem + ".txt")
        xml, objs = yolo_to_voc(img, lbl, names)
        voc[img] = xml
        for name, *_ in objs:
            total_boxes += 1
            per_class[name] = per_class.get(name, 0) + 1
            if name == TODO:
                bad_todo.append(img.name)
            elif name not in TARGET_CLASSES:
                bad_name.add(name)
    if bad_todo:
        sys.exit(f"{len(bad_todo)} box(es) still '{TODO}'. Finish subtype labeling first, "
                 f"e.g. {bad_todo[:3]}")
    if bad_name:
        sys.exit(f"Label name(s) not in target taxonomy: {bad_name}. Fix classes.txt.")

    print(f"Plan: upload {len(imgs)} images ({total_boxes} boxes) to "
          f"'{args.project}' split=train, tag='{args.batch}'")
    for n in sorted(per_class):
        print(f"   {n:28s} +{per_class[n]}")

    if not args.yes:
        print("\nDRY-RUN. Re-run with --yes to upload. After uploading you MUST, in the "
              "Roboflow UI, generate a NEW version with 'preserve existing splits' (do NOT "
              "rebalance) and augmentation OFF, then retrain.")
        return

    from roboflow import Roboflow
    proj = Roboflow(api_key=KEY).workspace(WS).project(args.project)
    ok = 0
    for img in imgs:
        xml_path = img.with_suffix(".xml")
        xml_path.write_text(voc[img])
        try:
            proj.upload(image_path=str(img), annotation_path=str(xml_path),
                        split="train", batch_name=args.batch, tag_names=[args.batch],
                        num_retry_uploads=3)
            ok += 1
            print(f"   uploaded {img.name}  ({ok}/{len(imgs)})")
        except Exception as e:
            print(f"   FAILED {img.name}: {e}")
        finally:
            xml_path.unlink(missing_ok=True)
    print(f"\nDone: {ok}/{len(imgs)} uploaded with tag '{args.batch}'.")
    print("Roll back if needed: filter by that tag in Roboflow and delete the batch.")
    print("Next: generate a new version (preserve splits, NO augmentation) and retrain.")


if __name__ == "__main__":
    main()
