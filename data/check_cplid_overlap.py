#!/usr/bin/env python3
"""Check how many images in our merged ATLI dataset overlap with the CPLID dataset.

Downloads CPLID (insulator dataset, 848 images) from its Roboflow source,
computes perceptual hashes (pHash) for both datasets, and reports exact and
near-duplicate matches.

The APET paper states ~200 ATLI images came from CPLID. This script verifies that.

Run on server: ~/atli/env/bin/python check_cplid_overlap.py
Can also run locally if you have the datasets downloaded.
"""
import os
from pathlib import Path
from collections import defaultdict

import imagehash
from PIL import Image
from dotenv import load_dotenv

ROOT = Path.home() / "atli"
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()

# ── Paths ──
MERGED = ROOT / "Merged_Dataset"          # our 1343-image merged pool
CPLID_DIR = ROOT / "downloads" / "cplid"  # will download here

NEAR_THRESHOLD = 8   # hamming distance ≤ 8 = near-duplicate
EXACT_THRESHOLD = 0  # hamming distance = 0 = exact duplicate


def download_cplid():
    """Download CPLID from Roboflow Universe (heitorcfelix/public-insulator-datasets)."""
    if CPLID_DIR.exists() and any(CPLID_DIR.rglob("*.jpg")) or any(CPLID_DIR.rglob("*.png")):
        print(f"  [skip] CPLID already at {CPLID_DIR}")
        return
    # CPLID is available on Roboflow Universe
    # Try the known public source
    from roboflow import Roboflow
    rf = Roboflow(api_key=KEY)
    # The original CPLID (848 insulator images) — try common Roboflow copies
    # heitorcfelix hosts the canonical public-insulator-datasets
    try:
        rf.workspace("heitorcfelix").project("public-insulator-datasets").version(1).download(
            "yolov5", location=str(CPLID_DIR))
    except Exception as e:
        print(f"  Roboflow download failed: {e}")
        print("  Trying alternative: direct CPLID path...")
        # Fallback: check if already on server at known location
        alt = ROOT / "datasets" / "cplid"
        if alt.exists():
            print(f"  Found at {alt}")
            CPLID_DIR.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(alt), str(CPLID_DIR))
        else:
            print("  ERROR: Cannot find CPLID. Download manually from:")
            print("    https://github.com/heitorcfelix/public-insulator-datasets")
            return


def get_image_paths(root):
    """Get all image paths from a directory tree."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            imgs.append(p)
    return imgs


def compute_hashes(image_paths, label=""):
    """Compute pHash for all images."""
    hashes = {}
    errors = 0
    for i, p in enumerate(image_paths):
        try:
            h = imagehash.phash(Image.open(p))
            hashes[p] = h
        except Exception:
            errors += 1
        if (i + 1) % 200 == 0:
            print(f"    {label} hashed {i+1}/{len(image_paths)}...")
    if errors:
        print(f"    {label} {errors} images failed to hash")
    return hashes


def find_overlaps(hashes_a, hashes_b, threshold):
    """Find images in A that have a near-duplicate in B."""
    matches = []
    for path_a, hash_a in hashes_a.items():
        best_dist = 999
        best_match = None
        for path_b, hash_b in hashes_b.items():
            d = hash_a - hash_b
            if d < best_dist:
                best_dist = d
                best_match = path_b
        if best_dist <= threshold:
            matches.append((path_a, best_match, best_dist))
    return matches


def main():
    print("=" * 60)
    print("CPLID ↔ ATLI Merged Dataset Overlap Check")
    print("=" * 60)

    # Download CPLID
    print("\n[1] Downloading CPLID...")
    download_cplid()

    # Get image lists
    print("\n[2] Collecting images...")
    merged_imgs = get_image_paths(MERGED / "images")
    cplid_imgs = get_image_paths(CPLID_DIR)
    print(f"  Merged dataset: {len(merged_imgs)} images")
    print(f"  CPLID:          {len(cplid_imgs)} images")

    if not cplid_imgs:
        print("  ERROR: No CPLID images found. Exiting.")
        return

    # Compute hashes
    print("\n[3] Computing perceptual hashes...")
    merged_hashes = compute_hashes(merged_imgs, "merged")
    cplid_hashes = compute_hashes(cplid_imgs, "cplid")
    print(f"  Hashed: {len(merged_hashes)} merged, {len(cplid_hashes)} CPLID")

    # Find exact duplicates
    print("\n[4] Finding exact duplicates (hamming=0)...")
    exact = find_overlaps(merged_hashes, cplid_hashes, EXACT_THRESHOLD)
    print(f"  Exact duplicates: {len(exact)} merged images match CPLID")

    # Find near-duplicates
    print("\n[5] Finding near-duplicates (hamming≤{})...".format(NEAR_THRESHOLD))
    near = find_overlaps(merged_hashes, cplid_hashes, NEAR_THRESHOLD)
    print(f"  Near-duplicates:  {len(near)} merged images match CPLID")

    # Check which split they fall in
    print("\n[6] Breakdown by filename pattern...")
    # Eduardo images are prefixed with rf_
    eduardo_exact = [m for m in exact if m[0].name.startswith("rf_")]
    atli_exact = [m for m in exact if not m[0].name.startswith("rf_")]
    eduardo_near = [m for m in near if m[0].name.startswith("rf_")]
    atli_near = [m for m in near if not m[0].name.startswith("rf_")]
    print(f"  ATLI target:  {len(atli_exact)} exact, {len(atli_near)} near")
    print(f"  Eduardo:      {len(eduardo_exact)} exact, {len(eduardo_near)} near")

    # Also check if there's a stratified split to see train/val/test breakdown
    strat = ROOT / "Merged_Dataset_Stratified"
    if strat.exists():
        print("\n[7] Checking train/val/test split overlap...")
        for split in ("train", "val", "test"):
            split_imgs = get_image_paths(strat / split / "images")
            split_hashes = compute_hashes(split_imgs, split)
            split_near = find_overlaps(split_hashes, cplid_hashes, NEAR_THRESHOLD)
            print(f"  {split:5s}: {len(split_near)} near-dups with CPLID "
                  f"(out of {len(split_imgs)} images)")

    # Print sample matches
    print("\n[8] Sample matches (first 10):")
    for path_m, path_c, dist in sorted(near, key=lambda x: x[2])[:10]:
        print(f"  d={dist:2d}  {path_m.name}  ↔  {path_c.name}")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(near)} of {len(merged_imgs)} merged images are "
          f"near-duplicates of CPLID (hamming≤{NEAR_THRESHOLD})")
    print(f"  ({len(exact)} are exact duplicates)")
    print(f"  Paper claims ~200 images from CPLID")
    print("=" * 60)


if __name__ == "__main__":
    main()
