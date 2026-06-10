#!/usr/bin/env python3
"""Delete 'too close' and duplicate images from eduardos-annotated-photos on Roboflow.

Dry-run by default. Pass --yes to actually delete.
Uses the SDK's project.delete_images() — the raw REST DELETE endpoint silently fails.

Usage:
    python rebalance/delete_too_close.py          # dry run
    python rebalance/delete_too_close.py --yes     # actually delete
"""
import os, sys, requests
import roboflow
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = "tl-target-set-focus"
PROJECT = "eduardos-annotated-photos"
DRY = "--yes" not in sys.argv

# Images to delete: too close + duplicates
DELETE_IDS = [
    "T0yd1DPGEKTwfzf4gzHE",   # too close
    "CE9yJeQ9qCxdGpZd5v8u",   # too close
    "bObP4BdP0KqzsiwcbSaR",   # too close
    "yICip2evZgrVxQCdta61",    # duplicate
    "LfSGKCgL8FZxNoRgkzap",   # duplicate, questionable
]

BASE = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}"

# Verify which images exist
print(f"{'DRY RUN' if DRY else 'LIVE'} — deleting {len(DELETE_IDS)} images from {PROJECT}")
print()

found = []
for img_id in DELETE_IDS:
    r = requests.get(f"{BASE}/images/{img_id}?api_key={API_KEY}")
    if r.status_code == 200:
        fname = r.json().get("image", {}).get("original_filename", img_id)
        print(f"  FOUND: {fname} ({img_id})")
        found.append(img_id)
    else:
        print(f"  NOT FOUND (already deleted?): {img_id}")

print()
if not found:
    print("Nothing to delete.")
    sys.exit(0)

if DRY:
    print(f"Dry run complete. {len(found)} images would be deleted. Pass --yes to delete for real.")
else:
    rf = roboflow.Roboflow(api_key=API_KEY)
    proj = rf.workspace(WORKSPACE).project(PROJECT)
    proj.delete_images(found)
    # Verify
    gone = 0
    for img_id in found:
        r = requests.get(f"{BASE}/images/{img_id}?api_key={API_KEY}")
        if r.status_code == 404:
            print(f"  DELETED ✓  {img_id}")
            gone += 1
        else:
            print(f"  FAILED ✗  {img_id} (still exists)")
    print(f"\nDone. {gone}/{len(found)} images deleted from {PROJECT}.")
    print("Note: you'll need to generate a new version to reflect these changes in training.")
