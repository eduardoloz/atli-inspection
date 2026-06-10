#!/usr/bin/env python3
"""Phase 1 (READ-ONLY w.r.t. the target): pick ~100 *zoomed-out* defective-insulator
images from ATLI_source_dataset and stage them locally for manual subtype labeling.

What it does:
  1. Downloads a version of the SOURCE  (ATLI_source_dataset) and the TARGET
     (merged_ATLI_target) in YOLO format  ->  datasets/  (git-ignored).
  2. Keeps only source images that contain the generic `Defective_Insulators` class,
     and scores each by how "zoomed out" the insulator is = the normalized bbox area
     (w*h). Small area  ==  small/distant object  ==  zoomed out.
  3. De-duplicates candidates against EVERY split of the target (perceptual hash) so we
     never add an image that is already in the target's train/val/test (leakage guard),
     and de-dups within the candidate set.
  4. Selects N within an area band [floor, ceil] (excludes unannotatable specks and
     close-ups), copies them to datasets/staging/ with NORMALIZED YOLO labels:
        - source Normal_Insulators / Normal_Damper / Defective_Damper boxes are kept
          (they match the target taxonomy -> avoids creating false-negative background),
        - source towers / lines are DROPPED (the target taxonomy excludes them by design),
        - each Defective_Insulators box becomes class `DEFECTIVE_INSULATOR_TODO` for you
          to reassign to Broken / Flashover / Self-Exploded in Phase 1b.
  5. Writes classes.txt + data.yaml (for labelImg / Roboflow) and manifest.csv (review).

This script NEVER writes to Roboflow. The only mutation is local files under datasets/.

Deps:  pip install roboflow imagehash Pillow PyYAML tqdm python-dotenv
Run:   python rebalance/phase1_select.py            # defaults: 100 imgs, source v4, target v4
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()

# --- taxonomy bridge: source class name -> target class name -------------------------
SOURCE_DEFECTIVE = "Defective_Insulators"          # the generic source class we harvest
TODO = "DEFECTIVE_INSULATOR_TODO"                  # placeholder until you assign a subtype
SUBTYPES = ["Broken_Insulator", "Flashover_Insulator", "Self-Exploded_Insulator"]
# co-occurring source boxes that DO map cleanly to the target (kept, names identical):
KEEP_AS_IS = {"Normal_Damper", "Normal_Insulators", "Defective_Damper"}
# source-only classes with no target equivalent (the target excludes towers/conductors):
DROP = {"Transmission_tower", "transmission_line", "defect_transmission_line"}
# staging label order (indices written into the YOLO .txt files):
STAGING_NAMES = SUBTYPES + ["Normal_Damper", "Normal_Insulators", "Defective_Damper", TODO]
NAME_TO_IDX = {n: i for i, n in enumerate(STAGING_NAMES)}
TODO_IDX = NAME_TO_IDX[TODO]


def need(mod):
    try:
        return __import__(mod)
    except ImportError:
        sys.exit(f"Missing dependency '{mod}'. Run:\n"
                 f"  pip install roboflow imagehash Pillow PyYAML tqdm python-dotenv")


def download(project_id, version, dest):
    """Download a Roboflow version in YOLO format (read-only export). Returns dest path."""
    from roboflow import Roboflow
    if Path(dest).exists():
        print(f"  [skip download] {dest} already exists")
        return Path(dest)
    rf = Roboflow(api_key=KEY)
    proj = rf.workspace(WS).project(project_id)
    print(f"  downloading {project_id} v{version} -> {dest}")
    proj.version(version).download("yolov8", location=str(dest))
    return Path(dest)


def load_names(root):
    """Read data.yaml `names` (list or dict) from a downloaded dataset root."""
    yaml = need("yaml")
    for y in Path(root).rglob("data.yaml"):
        names = yaml.safe_load(y.read_text()).get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return names
    sys.exit(f"No data.yaml/names found under {root}")


def iter_label_files(root):
    """Yield (label_txt_path, image_path) for every label in a YOLO download."""
    for lbl in Path(root).rglob("labels/*.txt"):
        img_dir = lbl.parent.parent / "images"
        hits = list(img_dir.glob(lbl.stem + ".*"))
        if hits:
            yield lbl, hits[0]


def read_boxes(lbl_path):
    """Return list of (class_idx, cx, cy, w, h) from a YOLO label file."""
    out = []
    for line in Path(lbl_path).read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), *map(float, p[1:5])))
    return out


def target_phashes(target_root, threshold):
    """Perceptual-hash every target image (all splits) for the leakage/dup guard."""
    import imagehash
    from PIL import Image
    hashes = []
    imgs = [p for p in Path(target_root).rglob("images/*") if p.is_file()]
    print(f"  hashing {len(imgs)} target images (dup/leakage guard, threshold={threshold})")
    for p in imgs:
        try:
            hashes.append(imagehash.phash(Image.open(p).convert("RGB")))
        except Exception:
            pass
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-version", type=int, default=4)
    ap.add_argument("--target-version", type=int, default=4)
    ap.add_argument("-n", "--num", type=int, default=100)
    ap.add_argument("--area-floor", type=float, default=0.0003,
                    help="min normalized bbox area (drop unannotatable specks)")
    ap.add_argument("--area-ceil", type=float, default=0.03,
                    help="max normalized bbox area (drop close-ups; raise to allow bigger)")
    ap.add_argument("--phash-threshold", type=int, default=5,
                    help="Hamming distance under which a candidate counts as a target dup")
    ap.add_argument("--out", default="datasets/staging")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY missing from .env")
    need("roboflow"); need("imagehash"); need("PIL"); need("yaml")
    import imagehash
    from PIL import Image

    print("[1/5] downloading source + target versions (read-only export)")
    src = download("atli_source_dataset", args.source_version, "datasets/atli_source")
    tgt = download("merged_atli_target", args.target_version, "datasets/merged_target")

    src_names = load_names(src)
    if SOURCE_DEFECTIVE not in src_names:
        sys.exit(f"'{SOURCE_DEFECTIVE}' not in source classes: {src_names}")
    def_idx = src_names.index(SOURCE_DEFECTIVE)
    print(f"      source '{SOURCE_DEFECTIVE}' is class index {def_idx}")

    print("[2/5] scoring source images by zoomed-out-ness (bbox area)")
    candidates = []  # (max_def_area, label_path, image_path, boxes)
    for lbl, img in iter_label_files(src):
        boxes = read_boxes(lbl)
        def_areas = [w * h for (c, cx, cy, w, h) in boxes if c == def_idx]
        if not def_areas:
            continue
        candidates.append((max(def_areas), lbl, img, boxes))
    print(f"      {len(candidates)} source images contain a defective insulator")

    print("[3/5] de-duplicating against the target (leakage guard)")
    tgt_hashes = target_phashes(tgt, args.phash_threshold)
    kept, seen = [], []
    in_band = [c for c in candidates if args.area_floor < c[0] < args.area_ceil]
    in_band.sort(key=lambda c: c[0])  # ascending area => most zoomed out first
    dropped_dup = 0
    for area, lbl, img, boxes in in_band:
        try:
            h = imagehash.phash(Image.open(img).convert("RGB"))
        except Exception:
            continue
        if any((h - t) <= args.phash_threshold for t in tgt_hashes) or \
           any((h - s) <= args.phash_threshold for s in seen):
            dropped_dup += 1
            continue
        seen.append(h)
        kept.append((area, lbl, img, boxes))
        if len(kept) >= args.num:
            break
    print(f"      in-band={len(in_band)}  dropped_as_dup={dropped_dup}  selected={len(kept)}")
    if len(kept) < args.num:
        print(f"      NOTE: only {len(kept)} found; widen --area-ceil to get more.")

    print(f"[4/5] staging {len(kept)} images -> {args.out}")
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    manifest = []
    add_counts = {n: 0 for n in STAGING_NAMES}
    for area, lbl, img, boxes in kept:
        new_lines, kinds = [], []
        for (c, cx, cy, w, h) in boxes:
            name = src_names[c]
            if name == SOURCE_DEFECTIVE:
                new_lines.append(f"{TODO_IDX} {cx} {cy} {w} {h}"); kinds.append(TODO)
                add_counts[TODO] += 1
            elif name in KEEP_AS_IS:
                new_lines.append(f"{NAME_TO_IDX[name]} {cx} {cy} {w} {h}"); kinds.append(name)
                add_counts[name] += 1
            # else: source-only (tower/line) -> dropped on purpose
        shutil.copy2(img, out / "images" / img.name)
        (out / "labels" / (Path(img.stem + ".txt"))).write_text("\n".join(new_lines) + "\n")
        manifest.append({"image": img.name, "max_def_area": round(area, 6),
                         "n_todo_boxes": kinds.count(TODO), "kept_boxes": ";".join(kinds)})

    (out / "classes.txt").write_text("\n".join(STAGING_NAMES) + "\n")
    need("yaml")
    import yaml
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"names": STAGING_NAMES, "nc": len(STAGING_NAMES),
         "train": "images", "val": "images"}, sort_keys=False))
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "max_def_area", "n_todo_boxes", "kept_boxes"])
        w.writeheader(); w.writerows(manifest)

    print("[5/5] done. boxes added by class (TODO = to be subtyped by you):")
    for n in STAGING_NAMES:
        if add_counts[n]:
            print(f"      {n:28s} +{add_counts[n]}")
    print(f"\nNext: open {out} in labelImg (uses classes.txt) and reassign every")
    print(f"      '{TODO}' box to one of: {', '.join(SUBTYPES)}.")
    print(f"      Review/curate the selection in {out/'manifest.csv'} (drop any bad ones).")
    print("      Then run:  python rebalance/phase2_upload.py            (dry-run)")


if __name__ == "__main__":
    main()
