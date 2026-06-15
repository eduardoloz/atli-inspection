#!/usr/bin/env python3
"""Upload the filtered Universe damper pool to a DEDICATED Roboflow staging project
so the whole-damper-vs-partial box inconsistency can be fixed in the annotation UI.

Writes ONLY to a separate staging project ('universe-damper-staging') — never to
merged_atli_target. Fully reversible: delete the project (or filter by tag) to undo.

  * images + their CURRENT boxes (YOLO -> Pascal-VOC, target class spellings:
    Defective_Damper / Normal_Damper) so you correct boxes instead of redrawing
  * tags: batch 'universe_damper_v1', source ('src_wangbo'/'src_yolov11tasks'),
    and 'has_defective' on images containing >=1 Defective_Damper box
    -> in the UI, filter by 'has_defective' to prioritize review
  * dry-run by default;  --yes uploads;  --limit N for a smoke batch

Runs on the GPU server (expects ~/atli/Universe_Pool from build_dataset_universe.py).
"""
import argparse
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from dotenv import load_dotenv

ROOT = Path.home() / "atli"
load_dotenv(ROOT / ".env")
load_dotenv()
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()

POOL = ROOT / "Universe_Pool"
CLASS_NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
               "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
PROJECT = "universe-damper-staging"
BATCH = "universe_damper_v1"


def yolo_to_voc(img_path, lbl_path):
    from PIL import Image
    W, H = Image.open(img_path).size
    objs = []
    for line in Path(lbl_path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        name = CLASS_NAMES[int(float(p[0]))]
        cx, cy, w, h = map(float, p[1:5])
        objs.append((name, max(0, int((cx - w / 2) * W)), max(0, int((cy - h / 2) * H)),
                     min(W, int((cx + w / 2) * W)), min(H, int((cy + h / 2) * H))))
    parts = [f"<annotation><filename>{escape(img_path.name)}</filename>",
             f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"]
    for name, a, b, c, d in objs:
        parts.append(f"<object><name>{escape(name)}</name><bndbox>"
                     f"<xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax>"
                     f"</bndbox></object>")
    parts.append("</annotation>")
    return "".join(parts), objs


API = "https://api.roboflow.com"


def rest_upload(img_path, xml, tags, session):
    """Direct REST upload (immune to SDK project-lookup cache for new projects)."""
    with open(img_path, "rb") as f:
        r = session.post(f"{API}/dataset/{PROJECT}/upload",
                         params=[("api_key", KEY), ("name", img_path.name),
                                 ("split", "train"), ("batch", BATCH)]
                                + [("tag", t) for t in tags],
                         files={"file": (img_path.name, f, "image/jpeg")}, timeout=120)
    r.raise_for_status()
    j = r.json()
    if not j.get("success", j.get("duplicate")):
        raise RuntimeError(f"upload rejected: {j}")
    img_id = j["id"]
    r = session.post(f"{API}/dataset/{PROJECT}/annotate/{img_id}",
                     params={"api_key": KEY, "name": img_path.stem + ".xml"},
                     data=xml.encode(), headers={"Content-Type": "text/plain"}, timeout=60)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"annotate rejected: {j}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually upload (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="upload only the first N (smoke test)")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N (resume)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY missing")
    imgs = sorted((POOL / "images").glob("*"))
    if not imgs:
        sys.exit(f"no images under {POOL} — run build_dataset_universe.py first")
    if args.skip:
        imgs = imgs[args.skip:]
    if args.limit:
        imgs = imgs[: args.limit]

    plan, n_def = [], 0
    for img in imgs:
        lbl = POOL / "labels" / (img.stem + ".txt")
        xml, objs = yolo_to_voc(img, lbl)
        has_def = any(o[0] == "Defective_Damper" for o in objs)
        n_def += has_def
        src = "src_wangbo" if img.name.startswith("uw_") else "src_yolov11tasks"
        tags = [BATCH, src] + (["has_defective"] if has_def else [])
        plan.append((img, xml, tags))

    print(f"Plan: {len(plan)} images -> project '{PROJECT}' (workspace {WS}), "
          f"batch/tag '{BATCH}'; {n_def} tagged has_defective")
    if not args.yes:
        print("DRY-RUN. Re-run with --yes to upload.")
        return

    import requests
    session = requests.Session()
    print(f"uploading to https://app.roboflow.com/{WS}/{PROJECT}")
    ok = fail = 0
    for i, (img, xml, tags) in enumerate(plan, 1):
        try:
            rest_upload(img, xml, tags, session)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAILED {img.name}: {str(e)[:160]}")
        if i % 100 == 0 or i == len(plan):
            print(f"  {i}/{len(plan)} done (ok={ok} fail={fail})", flush=True)
    print(f"\nDone: {ok} uploaded, {fail} failed, tag '{BATCH}'.")
    print(f"Review at: https://app.roboflow.com/{WS}/{PROJECT}/annotate")
    print("UPLOAD_STAGING_DONE")


if __name__ == "__main__":
    main()
