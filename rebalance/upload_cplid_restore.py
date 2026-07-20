#!/usr/bin/env python3
"""Upload the 250 CPLID-restore images to the Roboflow project
`cplid-train-restore-set`. Source = the cplid_* files staged in
CV_eduardo_det/fold0/train_cplid (canonical YOLO bbox labels; these are the
merged_atli_target images that are pHash-disjoint from the CV pool). Converts
each to VOC and uploads, tagged for one-click rollback.

Dry-run by default; --yes uploads.  Undo: filter tag 'cplid_restore_v1' in the
project and bulk-delete.

Run on server: ~/atli/env/bin/python upload_cplid_restore.py [--yes]
"""
import argparse, os
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

ROOT = Path.home() / "atli"
load_dotenv(ROOT / ".env")
KEY = os.environ["ROBOFLOW_API_KEY"].strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
DEST = "cplid-train-restore-set"
TAG = "cplid_restore_v1"
SRC = ROOT / "CV_eduardo_det" / "fold0" / "train_cplid"
CANON = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]


def yolo_to_voc(img, lbl):
    W, H = Image.open(img).size
    objs = []
    for ln in Path(lbl).read_text().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(p[0]); cx, cy, w, h = map(float, p[1:5])
        x1 = max(0, min(W, int((cx - w / 2) * W))); x2 = max(0, min(W, int((cx + w / 2) * W)))
        y1 = max(0, min(H, int((cy - h / 2) * H))); y2 = max(0, min(H, int((cy + h / 2) * H)))
        objs.append((CANON[c], x1, y1, x2, y2))
    xml = [f"<annotation><filename>{img.name}</filename>"
           f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"]
    for nm, x1, y1, x2, y2 in objs:
        xml.append(f"<object><name>{nm}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                   f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>")
    xml.append("</annotation>")
    return "".join(xml), len(objs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually upload (default: dry-run)")
    args = ap.parse_args()
    imgs = sorted((SRC / "images").glob("cplid_*"))
    print(f"CPLID images found: {len(imgs)}  ->  {WS}/{DEST}  (split=train, tag='{TAG}')")
    if not imgs:
        raise SystemExit(f"No cplid_* images in {SRC}/images — run build_cplid_cv.py first.")
    if not args.yes:
        print("DRY-RUN. Re-run with --yes to upload. Nothing changed.")
        return
    from roboflow import Roboflow
    proj = Roboflow(api_key=KEY).workspace(WS).project(DEST)
    tmp = ROOT / "cplid_upload_voc"; tmp.mkdir(exist_ok=True)
    ok = fail = 0
    for img in imgs:
        lbl = SRC / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        try:
            xml, n = yolo_to_voc(img, lbl)
            xp = tmp / (img.stem + ".xml"); xp.write_text(xml)
            proj.single_upload(image_path=str(img), annotation_path=str(xp),
                               split="train", batch_name=TAG, tag_names=[TAG])
            ok += 1
            if ok % 25 == 0:
                print(f"  uploaded {ok}/{len(imgs)} ...")
        except Exception as e:
            fail += 1
            print(f"  FAIL {img.stem}: {str(e)[:120]}")
    print(f"DONE. uploaded={ok} failed={fail}.  Undo: filter tag '{TAG}' in {DEST} + bulk-delete.")


if __name__ == "__main__":
    main()
