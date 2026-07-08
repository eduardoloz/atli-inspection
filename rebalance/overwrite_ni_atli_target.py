#!/usr/bin/env python3
"""Overwrite annotations on the atli_target-{train,val,test} Roboflow projects with
the merged_atli_target v5 labels (tightened Normal_Insulators boxes), skipping —
or optionally deleting — the CPLID-duplicate images.

Input: ~/atli/target_join_mapping.json (from build_target_tightni_nocplid.py):
one record per split-wise image {project, splitwise_file, v5_file, v5_label,
phash_dist, is_cplid}. All 1,046 joins are pHash d=0.

Per-project behavior:
  * non-CPLID image  -> overwrite its annotation with the v5 label (YOLO -> VOC XML,
    posted via the REST annotate endpoint with overwrite=true).
  * is_cplid image   -> skipped by default; deleted if --delete-cplid.
  * atli_target-test has classes literally named "0".."6" (positional), so class
    names are translated to digit strings for that project only.

Safety / reversibility:
  * DRY-RUN by default; writes ~/atli/ni_overwrite_manifest.csv and touches nothing.
    --yes applies annotation overwrites; --delete-cplid (with --yes) also deletes.
  * --limit N caps actions per project (pilot run, e.g. --limit 2).
  * Backups: frozen Roboflow versions (train v5 / val v3 / test v3) + their local
    exports at ~/atli/downloads/atli_target_{train_v5,val_v3,test_v3}/ hold the
    pre-overwrite annotations.
  * Candidates are matched by name AND pHash-verified against the local export
    before any action (belt and braces; joins were d=0).

Run on server:  ~/atli/env/bin/python overwrite_ni_atli_target.py [--yes] [--delete-cplid] [--limit N]
"""
import argparse
import csv
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import requests
import yaml
from PIL import Image
import imagehash
from dotenv import load_dotenv

ROOT = Path.home() / "atli"
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
API = "https://api.roboflow.com"
WS = "tl-target-set-focus"

MAPPING = ROOT / "target_join_mapping.json"
V5_YAML = ROOT / "downloads" / "target_v5" / "data.yaml"
MANIFEST = ROOT / "ni_overwrite_manifest.csv"
EXPORT_DIR = {  # local export of each project (pre-overwrite backup + pHash reference)
    "atli_target-train": ROOT / "downloads" / "atli_target_train_v5",
    "atli_target-val": ROOT / "downloads" / "atli_target_val_v3",
    "atli_target-test": ROOT / "downloads" / "atli_target_test_v3",
}
DIGIT_CLASS_PROJECTS = {"atli_target-test"}  # classes on Roboflow are "0".."6"
VERIFY_THRESHOLD = 2


def list_project_images(session, project, expected):
    """Full (id -> name) inventory. The search API's pagination is flaky (pages can
    come back short/empty spuriously), so sweep repeatedly, merging, until we have
    the expected count or 5 sweeps add nothing new."""
    out = {}
    stale = 0
    for attempt in range(15):
        before = len(out)
        for offset in range(0, 2000, 250):
            r = session.post(f"{API}/{WS}/{project}/search", params={"api_key": KEY},
                             json={"fields": ["id", "name"], "limit": 250,
                                   "offset": offset},
                             timeout=60)
            if r.status_code != 200:
                continue
            for rec in r.json().get("results", []):
                out[rec["id"]] = rec.get("name", "")
        if len(out) >= expected:
            break
        stale = stale + 1 if len(out) == before else 0
        if stale >= 5:
            print(f"  WARNING: inventory stuck at {len(out)}/{expected} after "
                  f"{attempt+1} sweeps")
            break
    return out


def find_local(project, fname):
    hits = list(EXPORT_DIR[project].rglob(fname))
    return hits[0] if hits else None


def hosted_phash(session, project, image_id):
    r = session.get(f"{API}/{WS}/{project}/images/{image_id}",
                    params={"api_key": KEY}, timeout=60)
    if r.status_code != 200:
        return None
    url = (r.json().get("image", {}).get("urls") or {}).get("original")
    if not url:
        return None
    d = session.get(url, timeout=120)
    if d.status_code != 200:
        return None
    try:
        return imagehash.phash(Image.open(io.BytesIO(d.content)))
    except Exception:
        return None


def yolo_to_voc(img_path, lbl_path, names, fname):
    W, H = Image.open(img_path).size
    parts = [f"<annotation><filename>{escape(fname)}</filename>",
             f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"]
    n_boxes = 0
    for line in Path(lbl_path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        name = names[int(float(p[0]))]
        cx, cy, w, h = map(float, p[1:5])
        xmin, ymin = int((cx - w / 2) * W), int((cy - h / 2) * H)
        xmax, ymax = int((cx + w / 2) * W), int((cy + h / 2) * H)
        parts.append(f"<object><name>{escape(name)}</name><bndbox>"
                     f"<xmin>{max(0, xmin)}</xmin><ymin>{max(0, ymin)}</ymin>"
                     f"<xmax>{min(W, xmax)}</xmax><ymax>{min(H, ymax)}</ymax>"
                     f"</bndbox></object>")
        n_boxes += 1
    parts.append("</annotation>")
    return "".join(parts), n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="apply (default: dry-run)")
    ap.add_argument("--delete-cplid", action="store_true",
                    help="also DELETE the CPLID-duplicate images (needs --yes)")
    ap.add_argument("--limit", type=int, default=0, help="max actions per project (pilot)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY not found in ~/atli/.env")

    v5_names = yaml.safe_load(V5_YAML.read_text())["names"]
    print(f"v5 class order: {v5_names}")

    records = json.load(open(MAPPING))
    by_project = defaultdict(list)
    for rec in records:
        by_project[rec["project"]].append(rec)

    session = requests.Session()
    rows, plan = [], defaultdict(lambda: defaultdict(int))
    actions = []  # (kind, project, image_id, xml_or_None, name)

    for project, recs in sorted(by_project.items()):
        print(f"\n[{project}] {len(recs)} mapping records; enumerating project images...")
        inv = list_project_images(session, project, expected=len(recs))
        by_name = defaultdict(list)
        for iid, nm in inv.items():
            by_name[nm].append(iid)
        print(f"  {len(inv)} images on Roboflow")
        names_out = ([str(i) for i in range(len(v5_names))]
                     if project in DIGIT_CLASS_PROJECTS else v5_names)
        if project in DIGIT_CLASS_PROJECTS:
            print("  NOTE: translating class names to digit strings '0'..'6' for this project")

        n_actions = 0
        for rec in sorted(recs, key=lambda r: r["splitwise_file"]):
            fname = rec["splitwise_file"]
            name_key = fname.split(".rf.")[0]
            cand = by_name.get(name_key, [])
            local_img = find_local(project, fname)
            status = ""
            image_id = ""
            if not cand:
                status = "UNMATCHED-no-name-hit"
            elif local_img is None:
                status = "ERROR-local-export-missing"
            else:
                ref_hash = imagehash.phash(Image.open(local_img))
                verified = []
                for iid in cand:
                    h = hosted_phash(session, project, iid)
                    if h is not None and (h - ref_hash) <= VERIFY_THRESHOLD:
                        verified.append(iid)
                if len(verified) != 1:
                    status = f"AMBIGUOUS-{len(verified)}-verified-of-{len(cand)}"
                else:
                    image_id = verified[0]
                    if args.limit and n_actions >= args.limit:
                        status = "SKIP-limit"
                    elif rec["is_cplid"]:
                        if args.delete_cplid:
                            status = "DELETE-cplid"
                            actions.append(("delete", project, image_id, None, fname))
                            n_actions += 1
                        else:
                            status = "SKIP-cplid"
                    elif not rec["v5_label"]:
                        status = "ERROR-no-v5-label"
                    else:
                        xml, nb = yolo_to_voc(local_img, rec["v5_label"], names_out, fname)
                        status = f"OVERWRITE-{nb}-boxes"
                        actions.append(("annotate", project, image_id, xml, fname))
                        n_actions += 1
            plan[project][status.split("-")[0]] += 1
            rows.append({"project": project, "file": fname, "image_id": image_id,
                         "status": status})

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["project", "file", "image_id", "status"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nManifest: {MANIFEST}")
    for project in sorted(plan):
        print(f"  {project}: {dict(plan[project])}")
    n_ann = sum(1 for a in actions if a[0] == "annotate")
    n_del = sum(1 for a in actions if a[0] == "delete")
    print(f"\nPlanned: {n_ann} annotation overwrites, {n_del} CPLID deletions")

    if not args.yes:
        print("\nDRY-RUN — nothing changed. Re-run with --yes"
              + (" --delete-cplid" if args.delete_cplid else "")
              + " to apply.")
        return

    print("\nApplying...")
    ok = fail = 0
    failures = []
    annotates = [a for a in actions if a[0] == "annotate"]
    deletes = defaultdict(list)  # project -> [image_id]
    for kind, project, iid, _, fname in actions:
        if kind == "delete":
            deletes[project].append(iid)

    for i, (kind, project, iid, xml, fname) in enumerate(annotates):
        r = session.post(f"{API}/dataset/{project}/annotate/{iid}",
                         params={"api_key": KEY, "name": Path(fname).stem + ".xml",
                                 "overwrite": "true"},
                         data=xml.encode(),
                         headers={"Content-Type": "text/plain"}, timeout=60)
        body = r.text[:150]
        if r.status_code in (200, 204) and '"error"' not in body:
            ok += 1
        else:
            fail += 1
            failures.append((kind, project, fname, r.status_code, body))
        if (i + 1) % 50 == 0:
            print(f"  annotate {i+1}/{len(annotates)} (ok={ok} fail={fail})")

    # batch delete: DELETE /:ws/:project/images with {"images": [...]} -> HTTP 204
    # (per-image DELETE /images/:id does NOT exist; it 200s with an error body)
    for project, ids in sorted(deletes.items()):
        for j in range(0, len(ids), 50):
            chunk = ids[j:j + 50]
            r = session.delete(f"{API}/{WS}/{project}/images",
                               params={"api_key": KEY}, json={"images": chunk},
                               headers={"Content-Type": "application/json"}, timeout=120)
            if r.status_code == 204:
                ok += len(chunk)
            else:
                fail += len(chunk)
                failures.append(("delete-batch", project, f"{len(chunk)} ids",
                                 r.status_code, r.text[:150]))
        print(f"  deleted {len(ids)} CPLID images from {project}")
    print(f"\nDone: ok={ok} fail={fail}")
    for kind, project, fname, code, body in failures[:15]:
        print(f"  FAIL {kind} {project}/{fname}: HTTP {code} {body}")
    print("\nRollback: re-annotate from the local exports in ~/atli/downloads/"
          "atli_target_{train_v5,val_v3,test_v3}/ (same endpoint), or regenerate "
          "from the frozen Roboflow versions.")


if __name__ == "__main__":
    main()
