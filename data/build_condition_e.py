#!/usr/bin/env python3
"""Condition E: Eduardo Full, keep the 'can't tell' defective images, only remove
too-close/duplicates/sloppy/questionable/disconnected.

Compared to Condition D (excluded all 12), Condition E keeps the 7 'can't tell'
images that all have Broken_Insulator labels.
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

# Only exclude non-"can't tell" images (5 total)
EXCLUDE = {
    # too close / duplicates (deleted from Roboflow)
    "T0yd1DPGEKTwfzf4gzHE",  # too close
    "CE9yJeQ9qCxdGpZd5v8u",  # too close
    "bObP4BdP0KqzsiwcbSaR",  # too close
    "yICip2evZgrVxQCdta61",   # duplicate
    "LfSGKCgL8FZxNoRgkzap",  # duplicate, questionable
    # other problematic (also deleted from Roboflow)
    "vYjI25qrvt0Py6rWQfa8",  # line disconnected
    "WbUMZvMXX38vCnSwPgS0",  # sloppy
    "ExPXgikI4S3EvpMyO4Bt",  # questionable
    "NCGeW1xhh6TnRD6tbFtG",  # questionable
}
# KEEPING these 7 "can't tell" images (all have Broken_Insulator labels):
# EtOrTKYJ10NQvEnvgNvZ, lztSsOub319uAS3ataR0, B2jaYjJtdUbJs4F9aH0k,
# UXfskzargi9XyrJc0lK8, cpoYp8dMuOVCX7YotJqQ, PbWRuppv9pUcxTvpnhPR,
# 6q7oplSOyKmpKaWysRM6


def counts(label_dir):
    c = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                c[int(line.split()[0])] += 1
    return {NAMES[k]: c.get(k, 0) for k in range(7)}


def should_exclude(filename):
    stem = Path(filename).stem
    for exc_id in EXCLUDE:
        if exc_id in stem:
            return True
    return False


# ---- Build Condition E ----
TPEF_E = ROOT / "Target_Plus_EduardoFull_KeepCantTell"
for sub in ("images", "labels"):
    shutil.rmtree(TPEF_E / "train" / sub, ignore_errors=True)
    (TPEF_E / "train" / sub).mkdir(parents=True, exist_ok=True)

copied = 0
excluded = 0
for img in (TPEF / "train" / "images").glob("*"):
    if should_exclude(img.name):
        excluded += 1
        continue
    lbl = TPEF / "train" / "labels" / (img.stem + ".txt")
    shutil.copy2(img, TPEF_E / "train" / "images" / img.name)
    if lbl.exists():
        shutil.copy2(lbl, TPEF_E / "train" / "labels" / lbl.name)
    copied += 1

print(f"Condition E: copied {copied} images, excluded {excluded}")
print(f"  (expected to exclude {len(EXCLUDE)} images)")

(ROOT / "abl_target_plus_eduardo_full_keepcanttell.yaml").write_text(yaml.safe_dump({
    "path": str(ROOT.resolve()),
    "train": str((TPEF_E / "train" / "images").resolve()),
    "val": str((TO / "val" / "images").resolve()),
    "test": str((TO / "test" / "images").resolve()),
    "nc": 7, "names": NAMES}, sort_keys=False))

print("\nC (full Eduardo) train counts:", counts(TPEF / "train" / "labels"))
print("D (clean, no cant-tell) counts:", counts((ROOT / "Target_Plus_EduardoFull_Clean" / "train" / "labels")))
print("E (keep cant-tell)      counts:", counts(TPEF_E / "train" / "labels"))
print("\nDiff E vs C (removed):")
c_counts = counts(TPEF / "train" / "labels")
e_counts = counts(TPEF_E / "train" / "labels")
for cls in NAMES:
    diff = c_counts[cls] - e_counts[cls]
    if diff:
        print(f"  {cls}: -{diff}")
print("BUILD_E_DONE")
