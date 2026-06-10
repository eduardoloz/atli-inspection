#!/usr/bin/env python3
"""Condition D: Target + Eduardo Full, MINUS the 'can't tell' images.
Same shared val/test as A/B/C. Tests whether removing ambiguous annotations improves performance.

Excluded images (Eduardo's 'can't tell' annotations):
  EtOrTKYJ10NQvEnvgNvZ, lztSsOub319uAS3ataR0, B2jaYjJtdUbJs4F9aH0k,
  UXfskzargi9XyrJc0lK8, cpoYp8dMuOVCX7YotJqQ, PbWRuppv9pUcxTvpnhPR,
  6q7oplSOyKmpKaWysRM6

Also excludes already-deleted images (too close / duplicates):
  T0yd1DPGEKTwfzf4gzHE, CE9yJeQ9qCxdGpZd5v8u, bObP4BdP0KqzsiwcbSaR,
  yICip2evZgrVxQCdta61, LfSGKCgL8FZxNoRgkzap
"""
import shutil
from pathlib import Path
from collections import Counter
import yaml

ROOT = Path.home() / "atli"
TPEF = ROOT / "Target_Plus_EduardoFull"       # built by build_condition_c.py
TO = ROOT / "Target_Only_Stratified"           # built by build_ablation.py
NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]

# Images to EXCLUDE (can't tell + too close + duplicates)
# These are Eduardo image IDs — in the dataset they have rf_ prefix
EXCLUDE = {
    # can't tell
    "EtOrTKYJ10NQvEnvgNvZ", "lztSsOub319uAS3ataR0", "B2jaYjJtdUbJs4F9aH0k",
    "UXfskzargi9XyrJc0lK8", "cpoYp8dMuOVCX7YotJqQ", "PbWRuppv9pUcxTvpnhPR",
    "6q7oplSOyKmpKaWysRM6",
    # too close / duplicates (already deleted from Roboflow)
    "T0yd1DPGEKTwfzf4gzHE", "CE9yJeQ9qCxdGpZd5v8u", "bObP4BdP0KqzsiwcbSaR",
    "yICip2evZgrVxQCdta61", "LfSGKCgL8FZxNoRgkzap",
}


def counts(label_dir):
    c = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return {NAMES[k]: c.get(k, 0) for k in range(7)}


def should_exclude(filename):
    """Check if this file matches any excluded image ID."""
    stem = Path(filename).stem
    # Eduardo images have rf_ prefix and .rf.<hash> suffix
    # e.g., rf_EtOrTKYJ10NQvEnvgNvZ.rf.abc123.jpg
    for exc_id in EXCLUDE:
        if exc_id in stem:
            return True
    return False


# ---- Build Condition D: copy C's train, minus excluded ----
TPEF_D = ROOT / "Target_Plus_EduardoFull_Clean"
for sub in ("images", "labels"):
    shutil.rmtree(TPEF_D / "train" / sub, ignore_errors=True)
    (TPEF_D / "train" / sub).mkdir(parents=True, exist_ok=True)

copied = 0
excluded = 0
for img in (TPEF / "train" / "images").glob("*"):
    if should_exclude(img.name):
        excluded += 1
        continue
    lbl = TPEF / "train" / "labels" / (img.stem + ".txt")
    shutil.copy2(img, TPEF_D / "train" / "images" / img.name)
    if lbl.exists():
        shutil.copy2(lbl, TPEF_D / "train" / "labels" / lbl.name)
    copied += 1

print(f"Condition D: copied {copied} images, excluded {excluded}")
print(f"  (expected to exclude {len(EXCLUDE)} Eduardo images)")

# ---- Write yaml ----
(ROOT / "abl_target_plus_eduardo_full_clean.yaml").write_text(yaml.safe_dump({
    "path": str(ROOT.resolve()),
    "train": str((TPEF_D / "train" / "images").resolve()),
    "val": str((TO / "val" / "images").resolve()),
    "test": str((TO / "test" / "images").resolve()),
    "nc": 7, "names": NAMES}, sort_keys=False))

print("\nC (full Eduardo) train counts:", counts(TPEF / "train" / "labels"))
print("D (clean Eduardo) train counts:", counts(TPEF_D / "train" / "labels"))
print("\nDiff (C - D = removed instances):")
c_counts = counts(TPEF / "train" / "labels")
d_counts = counts(TPEF_D / "train" / "labels")
for cls in NAMES:
    diff = c_counts[cls] - d_counts[cls]
    if diff:
        print(f"  {cls}: -{diff}")
print("BUILD_D_DONE")
