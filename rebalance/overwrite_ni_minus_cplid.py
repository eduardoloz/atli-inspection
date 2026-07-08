#!/usr/bin/env python3
"""Bring the `atli_target-minus-the-cplid` Roboflow project to spec:
overwrite every image's annotation with its merged_atli_target v5 label (the
hand-tightened Normal_Insulators boxes) and delete the 241 CPLID duplicates
that its earlier curation missed.

Join key: image name -> target_join_mapping.json (name -> v5 label + is_cplid);
every image is additionally pHash-verified against its v5 counterpart by
downloading the hosted original before any action is planned for it.

Safety (this project has ZERO versions, so no Roboflow snapshot exists):
  * Every image's CURRENT annotation JSON is saved to
    ~/atli/backup_minus_cplid_annotations.jsonl during the verification pass
    (dry-run included) — enough to restore via the same annotate endpoint.
  * DRY-RUN by default; --yes applies. Manifest: ~/atli/minus_cplid_overwrite_manifest.csv.
  * Deletes use the batch endpoint (DELETE /:ws/:project/images, HTTP 204);
    the per-image DELETE URL does not exist and 200s with an error body.

Run on server:  ~/atli/env/bin/python overwrite_ni_minus_cplid.py [--yes]
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
WS, PROJECT = "tl-target-set-focus", "atli_target-minus-the-cplid"
BASE = f"{API}/{WS}/{PROJECT}"

MAPPING = ROOT / "target_join_mapping.json"
V5_YAML = ROOT / "downloads" / "target_v5" / "data.yaml"
MANIFEST = ROOT / "minus_cplid_overwrite_manifest.csv"
BACKUP = ROOT / "backup_minus_cplid_annotations.jsonl"
EXPECTED = 1037
VERIFY_THRESHOLD = 2


def enumerate_project(session):
    out = {}
    stale = 0
    for _ in range(15):
        before = len(out)
        for offset in range(0, 2000, 250):
            r = session.post(f"{BASE}/search", params={"api_key": KEY},
                             json={"fields": ["id", "name", "split"], "limit": 250,
                                   "offset": offset}, timeout=60)
            if r.status_code != 200:
                continue
            for rec in r.json().get("results", []):
                out[rec["id"]] = rec
        if len(out) >= EXPECTED:
            break
        stale = stale + 1 if len(out) == before else 0
        if stale >= 5:
            print(f"  WARNING: inventory stuck at {len(out)}/{EXPECTED}")
            break
    return out


def v5_image_path(v5_label):
    """.../labels/x.txt -> the sibling image file."""
    lbl = Path(v5_label)
    img_dir = lbl.parent.parent / "images"
    hits = list(img_dir.glob(lbl.stem + ".*"))
    return hits[0] if hits else None


def yolo_to_voc(img_path, lbl_path, names, fname):
    W, H = Image.open(img_path).size
    parts = [f"<annotation><filename>{escape(fname)}</filename>",
             f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"]
    n = 0
    for line in Path(lbl_path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = names[int(float(p[0]))]
        cx, cy, w, h = map(float, p[1:5])
        xmin, ymin = int((cx - w / 2) * W), int((cy - h / 2) * H)
        xmax, ymax = int((cx + w / 2) * W), int((cy + h / 2) * H)
        parts.append(f"<object><name>{escape(cls)}</name><bndbox>"
                     f"<xmin>{max(0, xmin)}</xmin><ymin>{max(0, ymin)}</ymin>"
                     f"<xmax>{min(W, xmax)}</xmax><ymax>{min(H, ymax)}</ymax>"
                     f"</bndbox></object>")
        n += 1
    parts.append("</annotation>")
    return "".join(parts), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="apply (default: dry-run)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY not found in ~/atli/.env")

    v5_names = yaml.safe_load(V5_YAML.read_text())["names"]
    print(f"v5 class order: {v5_names}")

    # name-key -> candidate mapping records (dedup identical v5 targets)
    by_name = defaultdict(list)
    for m in json.load(open(MAPPING)):
        key = m["splitwise_file"].split(".rf.")[0]
        if not any(e["v5_label"] == m["v5_label"] for e in by_name[key]):
            by_name[key].append(m)

    session = requests.Session()
    print(f"[1] Enumerating {PROJECT}...")
    inv = enumerate_project(session)
    print(f"  {len(inv)} images")

    print("[2] Verifying (pHash vs v5 export), backing up current annotations...")
    rows, annotates, deletes = [], [], []
    backup_f = open(BACKUP, "w")
    done = 0
    for iid, rec in sorted(inv.items(), key=lambda kv: kv[1]["name"]):
        name = rec["name"]
        cands = by_name.get(name, [])
        status = ""
        detail = session.get(f"{BASE}/images/{iid}", params={"api_key": KEY}, timeout=60)
        img_meta = detail.json().get("image", {}) if detail.status_code == 200 else {}
        backup_f.write(json.dumps({"id": iid, "name": name, "split": rec.get("split"),
                                   "annotation": img_meta.get("annotation")}) + "\n")
        if not cands:
            status = "UNMATCHED-no-mapping-record"
        else:
            url = (img_meta.get("urls") or {}).get("original")
            h = None
            if url:
                d = session.get(url, timeout=120)
                if d.status_code == 200:
                    try:
                        h = imagehash.phash(Image.open(io.BytesIO(d.content)))
                    except Exception:
                        pass
            if h is None:
                status = "ERROR-download"
            else:
                best, best_d = None, 999
                for c in cands:
                    vimg = v5_image_path(c["v5_label"]) if c["v5_label"] else None
                    if vimg is None:
                        continue
                    dd = h - imagehash.phash(Image.open(vimg))
                    if dd < best_d:
                        best, best_d, best_img = c, dd, vimg
                if best is None or best_d > VERIFY_THRESHOLD:
                    status = f"AMBIGUOUS-best-d={best_d}"
                elif best["is_cplid"]:
                    status = "DELETE-cplid"
                    deletes.append(iid)
                else:
                    xml, nb = yolo_to_voc(best_img, best["v5_label"], v5_names, name)
                    status = f"OVERWRITE-{nb}-boxes"
                    annotates.append((iid, xml, name))
        rows.append({"file": name, "image_id": iid, "split": rec.get("split"),
                     "status": status})
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(inv)} verified ({len(annotates)} overwrite, "
                  f"{len(deletes)} delete)")
    backup_f.close()

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "image_id", "split", "status"])
        w.writeheader()
        w.writerows(rows)

    kinds = defaultdict(int)
    for r in rows:
        kinds[r["status"].split("-")[0]] += 1
    print(f"\nManifest: {MANIFEST} | backup: {BACKUP}")
    print(f"Plan: {dict(kinds)}")
    print(f"  -> {len(annotates)} annotation overwrites, {len(deletes)} CPLID deletions")

    if not args.yes:
        print("\nDRY-RUN — nothing changed. Re-run with --yes to apply.")
        return

    print("\n[3] Applying annotation overwrites...")
    ok = fail = 0
    failures = []
    for i, (iid, xml, name) in enumerate(annotates):
        r = session.post(f"{API}/dataset/{PROJECT}/annotate/{iid}",
                         params={"api_key": KEY, "name": name + ".xml",
                                 "overwrite": "true"},
                         data=xml.encode(),
                         headers={"Content-Type": "text/plain"}, timeout=60)
        if r.status_code in (200, 204) and '"error"' not in r.text[:150]:
            ok += 1
        else:
            fail += 1
            failures.append((name, r.status_code, r.text[:120]))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(annotates)} (ok={ok} fail={fail})")

    print(f"[4] Batch-deleting {len(deletes)} CPLID images...")
    for j in range(0, len(deletes), 50):
        chunk = deletes[j:j + 50]
        r = session.delete(f"{BASE}/images", params={"api_key": KEY},
                           json={"images": chunk},
                           headers={"Content-Type": "application/json"}, timeout=120)
        if r.status_code == 204:
            ok += len(chunk)
        else:
            fail += len(chunk)
            failures.append((f"delete-batch@{j}", r.status_code, r.text[:120]))

    print(f"\nDone: ok={ok} fail={fail}")
    for name, code, txt in failures[:10]:
        print(f"  FAIL {name}: HTTP {code} {txt}")
    print("Rollback: restore annotations from backup jsonl via the same annotate "
          "endpoint; deleted images recoverable from the v5 export / CPLID GitHub.")


if __name__ == "__main__":
    main()
