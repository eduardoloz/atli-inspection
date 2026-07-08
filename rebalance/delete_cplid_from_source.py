#!/usr/bin/env python3
"""Delete the CPLID-duplicate images from `atli_source_dataset` on Roboflow.

Input: ~/atli/cplid_vs_source_matches.json (from check_cplid_vs_source.py) —
599 CPLID images with a pHash d<=4 match in the source-dataset export at
~/atli/datasets/classvet/source.

Resolution: the Roboflow search API caps at offset 10,000 and has no
name/split filter, so we sweep the plain listing plus several CLIP-prompt
re-orderings and union the results to cover all 11,294 images. Candidates
are matched by name (export filename minus the .rf.<hash> suffix), then
VERIFIED by downloading the hosted original and pHash-comparing it to the
local export file. Only verified matches (d<=2) are marked for deletion.

Safety / reversibility:
  * DRY-RUN by default — writes ~/atli/cplid_delete_manifest.csv and deletes
    nothing. Pass --yes to actually delete.
  * Version v4 of the project is a frozen snapshot and stays downloadable.
  * A full local backup (images + labels) exists at
    ~/atli/datasets/classvet/source, and the originals are in CPLID itself
    (~/atli/downloads/cplid_github).

Run on server:  ~/atli/env/bin/python delete_cplid_from_source.py [--yes]
"""
import argparse
import csv
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image
import imagehash
from dotenv import load_dotenv

ROOT = Path.home() / "atli"
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
API = "https://api.roboflow.com"
WS, PROJECT = "tl-target-set-focus", "atli_source_dataset"
BASE = f"{API}/{WS}/{PROJECT}"

MATCHES_JSON = ROOT / "cplid_vs_source_matches.json"
MANIFEST_CSV = ROOT / "cplid_delete_manifest.csv"
MATCH_THRESHOLD = 4   # d<=4 in the matches JSON = a CPLID duplicate
VERIFY_THRESHOLD = 2  # hosted original vs local export (0 expected; <=2 allows re-encode)

# CLIP prompts to re-order the 10k search window so the union covers all images
SWEEP_PROMPTS = [None, "glass insulator", "vibration damper on power line",
                 "transmission tower", "bird nest"]
PAGE = 250


def sweep_inventory(wanted_names):
    """Union of (id, name, split) over plain + prompt-reordered search sweeps."""
    session = requests.Session()
    seen = {}  # id -> {name, split}
    for prompt in SWEEP_PROMPTS:
        before = len(seen)
        for offset in range(0, 10000, PAGE):
            body = {"fields": ["id", "name", "split"], "limit": PAGE, "offset": offset}
            if prompt:
                body["prompt"] = prompt
            r = session.post(f"{BASE}/search", params={"api_key": KEY}, json=body, timeout=60)
            if r.status_code != 200:
                print(f"    sweep '{prompt}' offset {offset}: HTTP {r.status_code}, stopping this sweep")
                break
            results = r.json().get("results", [])
            if not results:
                break
            for rec in results:
                seen[rec["id"]] = {"name": rec.get("name", ""), "split": rec.get("split", "?")}
        found = sum(1 for v in seen.values() if v["name"] in wanted_names)
        print(f"  sweep prompt={prompt!r}: +{len(seen)-before} new, inventory={len(seen)}, "
              f"wanted-names covered: {found}/{len(wanted_names)}")
        if found >= len(wanted_names):
            break
    return seen


def hosted_phash(session, image_id):
    """Image detail -> download original -> pHash. Returns (phash, split) or (None, err)."""
    r = session.get(f"{BASE}/images/{image_id}", params={"api_key": KEY}, timeout=60)
    if r.status_code != 200:
        return None, f"detail HTTP {r.status_code}"
    img = r.json().get("image", {})
    url = (img.get("urls") or {}).get("original")
    if not url:
        return None, "no original url"
    d = session.get(url, timeout=120)
    if d.status_code != 200:
        return None, f"download HTTP {d.status_code}"
    try:
        return imagehash.phash(Image.open(io.BytesIO(d.content))), img.get("split", "?")
    except Exception as e:
        return None, f"decode: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()

    if not KEY:
        sys.exit("ROBOFLOW_API_KEY not found in ~/atli/.env")

    matches = [m for m in json.load(open(MATCHES_JSON)) if m["d"] <= MATCH_THRESHOLD]
    print(f"[1] {len(matches)} CPLID matches (d<={MATCH_THRESHOLD}) loaded")

    # export file -> search name ('0049_jpg.rf.<hash>.jpg' -> '0049_jpg') + local pHash
    targets = {}  # name -> list of {export_path, phash}
    for m in matches:
        p = Path(m["src"])
        name = p.name.split(".rf.")[0]
        targets.setdefault(name, []).append(
            {"export": p, "phash": imagehash.phash(Image.open(p))})
    print(f"[2] {len(targets)} distinct names to resolve; local export pHashes computed")

    print("[3] Sweeping project inventory via search API...")
    inventory = sweep_inventory(set(targets))
    by_name = defaultdict(list)
    for iid, rec in inventory.items():
        by_name[rec["name"]].append(iid)

    print("[4] Verifying candidates against local export (pHash on hosted originals)...")
    session = requests.Session()
    rows, to_delete = [], []
    unresolved = []
    for name, entries in sorted(targets.items()):
        cand_ids = by_name.get(name, [])
        if not cand_ids:
            unresolved.append(name)
            for e in entries:
                rows.append({"export_file": e["export"].name, "name": name, "image_id": "",
                             "split": "", "phash_dist": "", "action": "UNRESOLVED-not-in-sweep"})
            continue
        for iid in cand_ids:
            h, split_or_err = hosted_phash(session, iid)
            if h is None:
                rows.append({"export_file": "", "name": name, "image_id": iid,
                             "split": "", "phash_dist": "", "action": f"ERROR-{split_or_err}"})
                continue
            best = min(entries, key=lambda e: h - e["phash"])
            d = h - best["phash"]
            if d <= VERIFY_THRESHOLD:
                to_delete.append(iid)
                rows.append({"export_file": best["export"].name, "name": name, "image_id": iid,
                             "split": split_or_err, "phash_dist": d, "action": "DELETE"})
            else:
                rows.append({"export_file": best["export"].name, "name": name, "image_id": iid,
                             "split": split_or_err, "phash_dist": d,
                             "action": "SKIP-name-collision"})
        if len(rows) % 50 < len(cand_ids):
            print(f"    verified {len(rows)} candidates, {len(to_delete)} marked so far...")

    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["export_file", "name", "image_id", "split",
                                          "phash_dist", "action"])
        w.writeheader()
        w.writerows(rows)

    n_split = defaultdict(int)
    for r in rows:
        if r["action"] == "DELETE":
            n_split[r["split"]] += 1
    print(f"\n[5] Manifest written: {MANIFEST_CSV}")
    print(f"    marked DELETE: {len(to_delete)}  (by split: {dict(n_split)})")
    print(f"    name collisions skipped: {sum(1 for r in rows if r['action'].startswith('SKIP'))}")
    print(f"    unresolved (not found in sweeps): {len(unresolved)}"
          + (f" -> {unresolved[:10]}" if unresolved else ""))

    if not args.yes:
        print("\nDRY-RUN — nothing deleted. Re-run with --yes to delete the marked images.")
        return

    print(f"\n[6] DELETING {len(to_delete)} images from {WS}/{PROJECT}...")
    # batch endpoint: DELETE /:ws/:project/images with {"images": [...]} -> HTTP 204
    # (per-image DELETE /images/:id does NOT exist; it 200s with an error body)
    ok = fail = 0
    failures = []
    for j in range(0, len(to_delete), 50):
        chunk = to_delete[j:j + 50]
        r = session.delete(f"{BASE}/images", params={"api_key": KEY},
                           json={"images": chunk},
                           headers={"Content-Type": "application/json"}, timeout=120)
        if r.status_code == 204:
            ok += len(chunk)
        else:
            fail += len(chunk)
            failures.append((f"batch@{j}", r.status_code, r.text[:120]))
        print(f"    {min(j+50, len(to_delete))}/{len(to_delete)} (ok={ok} fail={fail})")
    print(f"    done: deleted={ok} failed={fail}")
    for iid, code, txt in failures[:10]:
        print(f"      FAIL {iid}: HTTP {code} {txt}")
    print("\nNote: version v4 snapshot is unaffected; local backup at "
          "~/atli/datasets/classvet/source; originals in ~/atli/downloads/cplid_github.")


if __name__ == "__main__":
    main()
