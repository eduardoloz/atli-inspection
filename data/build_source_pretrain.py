#!/usr/bin/env python3
"""Build the leakage-gated in-domain PRETRAINING source set (thesis-T1, champ_srcTL).

Steps (read-only wrt Roboflow — download/export only):
  1. Download the LATEST available version of tl-target-set-focus/atli_source_dataset
     (yolov8 detection export) -> SRC_RAW. NOTE: if the newest generated version predates
     the 2026-07-07 CPLID purge, CPLID dups may still be present; they are harmless for
     pretraining (CPLID was removed from the native target entirely) and the gate below
     still governs anything that collides with the native set.
  2. LEAKAGE GATE against the native target ATLI_target_tightNI_noCPLID — ALL splits
     (train/valid/test), per orchestrator directive 2026-07-07:
       a. pHash gate: drop any source image whose perceptual hash is within
          NEAR_DUP hamming distance of ANY native image.
       b. Filename gate: drop any source image whose Roboflow-normalized stem
          (part before "_<ext>.rf.<hash>") matches a native stem.
  3. Emit ATLI_source_pretrain/{train,valid,test}/{images,labels} (symlinks into the
     raw export to save disk) + data.yaml with the SOURCE-native class taxonomy
     (fine-tuning re-initializes the detection head, so taxonomy mismatch is fine).
  4. Write a gate report JSON with per-split drop counts and example collisions.

Usage (on the UNLV server, from ~/atli):  python build_source_pretrain.py
Requires ROBOFLOW_API_KEY in ./.env (or env).
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()  # cwd fallback
except ImportError:
    pass

KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
assert KEY, "ROBOFLOW_API_KEY missing (.env)"

import imagehash
import yaml
from PIL import Image

WS, PROJ = "tl-target-set-focus", "atli_source_dataset"
ROOT = Path(os.environ.get("ATLI_ROOT", Path.home() / "atli"))
NATIVE = ROOT / "ATLI_target_tightNI_noCPLID"
SRC_RAW = ROOT / "ATLI_source_raw"
OUT = ROOT / "ATLI_source_pretrain"
REPORT = ROOT / "source_pretrain_gate_report.json"
NEAR_DUP = 8  # hamming distance (of 64) treated as duplicate — matches vet_universe_dampers.py

RF_STEM = re.compile(r"^(.*?)(?:_(?:jpg|jpeg|png|bmp|webp))?\.rf\.[0-9a-f]{32}$", re.I)


def norm_stem(p: Path) -> str:
    m = RF_STEM.match(p.stem)
    return (m.group(1) if m else p.stem).lower()


def images_of(root):
    return sorted(p for p in Path(root).rglob("*.jpg")) + \
           sorted(p for p in Path(root).rglob("*.jpeg")) + \
           sorted(p for p in Path(root).rglob("*.png"))


def hash_dir(root, cache: Path):
    done = json.loads(cache.read_text()) if cache.exists() else {}
    imgs = [p for p in images_of(root) if "images" in p.parts]
    changed = False
    for i, p in enumerate(imgs):
        k = str(p)
        if k not in done:
            try:
                done[k] = str(imagehash.phash(Image.open(p)))
            except Exception as e:
                print(f"  [warn] unreadable {p.name}: {e}")
                done[k] = None
            changed = True
        if i and i % 2000 == 0:
            print(f"  hashed {i}/{len(imgs)}")
            cache.write_text(json.dumps(done))
    if changed:
        cache.write_text(json.dumps(done))
    return {k: int(v, 16) for k, v in done.items() if v and Path(k).exists()}


def min_hamming(h, pool_items):
    best, who = 65, None
    for k, v in pool_items:
        d = bin(h ^ v).count("1")
        if d < best:
            best, who = d, k
            if best == 0:
                break
    return best, who


def download_source():
    if SRC_RAW.exists() and any(SRC_RAW.rglob("*.jpg")):
        print(f"[skip] source export already at {SRC_RAW}")
        return
    from roboflow import Roboflow
    rf = Roboflow(api_key=KEY)
    proj = rf.workspace(WS).project(PROJ)
    vnums = sorted(int(v.version.split("/")[-1]) for v in proj.versions())
    assert vnums, "no generated versions on atli_source_dataset"
    v = vnums[-1]
    print(f"[source] {PROJ} latest generated version = v{v}; downloading -> {SRC_RAW}")
    proj.version(v).download("yolov8", location=str(SRC_RAW))
    (SRC_RAW / "VERSION.txt").write_text(str(v))


def main():
    assert NATIVE.exists(), f"native target missing: {NATIVE}"
    download_source()

    print("[hash] native (all splits)...")
    nh = hash_dir(NATIVE, ROOT / "hashes_native_tightni.json")
    native_items = list(nh.items())
    native_stems = {norm_stem(Path(k)) for k in nh}
    print(f"  native: {len(nh)} images, {len(native_stems)} unique stems")

    print("[hash] source...")
    sh = hash_dir(SRC_RAW, ROOT / "hashes_source_raw.json")
    print(f"  source: {len(sh)} images")

    dropped = {"phash": [], "filename": []}
    kept = []
    for i, (k, h) in enumerate(sh.items()):
        p = Path(k)
        if norm_stem(p) in native_stems:
            dropped["filename"].append(k)
            continue
        d, who = min_hamming(h, native_items)
        if d <= NEAR_DUP:
            dropped["phash"].append((k, d, who))
            continue
        kept.append(p)
        if i and i % 2000 == 0:
            print(f"  gated {i}/{len(sh)} (dropped so far: "
                  f"{len(dropped['phash'])} phash / {len(dropped['filename'])} name)")

    print(f"[gate] kept {len(kept)} / {len(sh)}  "
          f"(dropped {len(dropped['phash'])} pHash<= {NEAR_DUP}, "
          f"{len(dropped['filename'])} filename)")

    # --- emit gated dataset as symlinks, preserving the export's own splits
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    counts = Counter()
    for img in kept:
        split = next((s for s in ("train", "valid", "test") if s in img.parts), "train")
        for kind, src in (("images", img),
                          ("labels", Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt"))):
            if kind == "labels" and not src.exists():
                continue
            d = OUT / split / kind
            d.mkdir(parents=True, exist_ok=True)
            dst = d / src.name
            if not dst.exists():
                dst.symlink_to(src)
        counts[split] += 1

    names = None
    for y in SRC_RAW.rglob("data.yaml"):
        names = yaml.safe_load(y.read_text()).get("names")
        break
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    (OUT / "data.yaml").write_text(yaml.safe_dump({
        "train": str(OUT / "train" / "images"),
        "val": str(OUT / ("valid" if (OUT / "valid").exists() else "train") / "images"),
        "test": str(OUT / ("test" if (OUT / "test").exists() else "train") / "images"),
        "nc": len(names), "names": list(names)}, sort_keys=False))

    report = {
        "source_version": (SRC_RAW / "VERSION.txt").read_text().strip()
                          if (SRC_RAW / "VERSION.txt").exists() else "?",
        "source_images": len(sh), "native_images": len(nh),
        "kept": len(kept), "kept_per_split": dict(counts),
        "dropped_phash": len(dropped["phash"]),
        "dropped_filename": len(dropped["filename"]),
        "phash_examples": [(Path(a).name, d, Path(b).name) for a, d, b in dropped["phash"][:40]],
        "filename_examples": [Path(k).name for k in dropped["filename"][:40]],
        "near_dup_threshold": NEAR_DUP,
        "classes": names,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if not k.endswith("_examples")}, indent=2))
    print(f"\ndataset -> {OUT}\nreport  -> {REPORT}")


if __name__ == "__main__":
    main()
