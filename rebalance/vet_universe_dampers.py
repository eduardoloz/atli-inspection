#!/usr/bin/env python3
"""READ-ONLY vetting of Roboflow Universe damper-defect datasets as rebalance sources.

For each candidate Universe set:
  1. Download in YOLO format -> datasets/vetting/  (git-ignored; skip if present).
  2. pHash LEAKAGE check vs merged_atli_target (latest version):
       - vs val/test  = CRITICAL (a dup here invalidates evaluation if merged)
       - vs train     = informational (dup adds nothing but doesn't invalidate)
  3. Cross-set duplicate check (community sets are often copies of each other / DVDI).
  4. Zoom-out analysis of the DEFECTIVE class boxes: normalized bbox area distribution,
     counts of distant (UAV-style) vs close-up shots.
  5. Quality heuristics: unlabeled-image fraction, intra-set duplicates, resolution spread.
  6. Stage the N most zoomed-out defective images per set WITH BOXES DRAWN into
     datasets/vetting/review/<set>/ for manual Quick Look review.

Never writes to Roboflow. Output: printed report + datasets/vetting/vetting_report.json
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import os

KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
assert KEY, "ROBOFLOW_API_KEY missing from .env"

import imagehash
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent / "datasets" / "vetting"
ROOT.mkdir(parents=True, exist_ok=True)
REVIEW = ROOT / "review"
N_REVIEW = 40          # most-zoomed-out defective images staged per set
NEAR_DUP = 8           # hamming distance (of 64) considered a near-duplicate
STRONG_DUP = 4

TARGET_WS, TARGET_PROJ = "tl-target-set-focus", "merged_atli_target"
UNIVERSE = [
    # (slug, workspace, project, version)
    ("u_yolov11tasks", "yolov11-tasks", "damper-defect-detection", 3),
    ("u_wangbo", "wangbo", "damper-o5wo3", 1),
    ("u_samiksha", "samiksha-gadhave", "defect-damper", 5),
]
DVDI_DIR = ROOT / "dvdi"

DEFECTIVE_PAT = ("broken", "defect")
NOT_DEFECTIVE_PAT = ("none", "good", "intact", "normal")


def is_defective_name(name):
    n = name.lower()
    if any(p in n for p in NOT_DEFECTIVE_PAT):
        return False
    return any(p in n for p in DEFECTIVE_PAT)


def download_target():
    from roboflow import Roboflow
    dest = ROOT / "target"
    if dest.exists():
        print(f"[skip] target already at {dest}")
        return dest
    rf = Roboflow(api_key=KEY)
    proj = rf.workspace(TARGET_WS).project(TARGET_PROJ)
    vnums = sorted(int(v.version.split("/")[-1]) for v in proj.versions())
    v = vnums[-1]
    print(f"[target] {TARGET_PROJ} latest version = v{v}; downloading...")
    proj.version(v).download("yolov8", location=str(dest))
    (dest / "VERSION.txt").write_text(str(v))
    return dest


def download_universe(slug, ws, proj_id, ver):
    from roboflow import Roboflow
    dest = ROOT / slug
    if dest.exists():
        print(f"[skip] {slug} already at {dest}")
        return dest
    print(f"[universe] downloading {ws}/{proj_id} v{ver} -> {dest}")
    rf = Roboflow(api_key=KEY)
    rf.workspace(ws).project(proj_id).version(ver).download("yolov8", location=str(dest))
    return dest


def download_dvdi():
    if DVDI_DIR.exists():
        print(f"[skip] DVDI already at {DVDI_DIR}")
        return DVDI_DIR
    print("[dvdi] cloning github.com/Emp-8/DVDI (300 test jpgs)...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Emp-8/DVDI",
                    str(DVDI_DIR)], check=True, capture_output=True)
    return DVDI_DIR


def images_of(root):
    return sorted(p for p in Path(root).rglob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "review" not in p.parts)


def hash_dir(root, cache_name):
    """phash every image under root; cache to json (str(path)->hex)."""
    cache = ROOT / cache_name
    done = json.loads(cache.read_text()) if cache.exists() else {}
    imgs = images_of(root)
    changed = False
    for p in imgs:
        k = str(p)
        if k not in done:
            try:
                done[k] = str(imagehash.phash(Image.open(p)))
            except Exception as e:
                print(f"  [warn] unreadable {p.name}: {e}")
            changed = True
    if changed:
        cache.write_text(json.dumps(done))
    return {k: int(v, 16) for k, v in done.items() if Path(k).exists()}


def split_of(path_str):
    parts = Path(path_str).parts
    for s in ("train", "valid", "test"):
        if s in parts:
            return s
    return "?"


def min_hamming(h, pool):
    best, who = 65, None
    for k, v in pool.items():
        d = bin(h ^ v).count("1")
        if d < best:
            best, who = d, k
    return best, who


def yolo_labels(img_path):
    lp = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if not lp.exists():
        return None
    out = []
    for line in lp.read_text().splitlines():
        f = line.split()
        if len(f) >= 5:
            out.append((int(f[0]), *map(float, f[1:5])))
    return out


def names_of(root):
    for y in Path(root).rglob("data.yaml"):
        names = yaml.safe_load(y.read_text()).get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        return list(names)
    return []


def draw_review(img_path, boxes, names, out_path):
    im = Image.open(img_path).convert("RGB")
    d = ImageDraw.Draw(im)
    W, H = im.size
    for cls, x, y, w, h in boxes:
        x0, y0 = (x - w / 2) * W, (y - h / 2) * H
        x1, y1 = (x + w / 2) * W, (y + h / 2) * H
        name = names[cls] if cls < len(names) else str(cls)
        col = "red" if is_defective_name(name) else "lime"
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        d.text((x0 + 2, max(0, y0 - 12)), name, fill=col)
    im.save(out_path, quality=88)


def main():
    report = {}
    target = download_target()
    sets = {}
    for slug, ws, pid, ver in UNIVERSE:
        try:
            sets[slug] = download_universe(slug, ws, pid, ver)
        except Exception as e:
            print(f"[ERROR] {slug} download failed: {e}")
    try:
        download_dvdi()
    except Exception as e:
        print(f"[ERROR] DVDI clone failed: {e}")

    print("\n[hash] target...")
    th = hash_dir(target, "hashes_target.json")
    t_valtest = {k: v for k, v in th.items() if split_of(k) in ("valid", "test")}
    t_train = {k: v for k, v in th.items() if split_of(k) == "train"}
    print(f"  target: {len(t_train)} train / {len(t_valtest)} val+test hashed")

    pools = {}
    for slug in list(sets) + (["dvdi"] if DVDI_DIR.exists() else []):
        print(f"[hash] {slug}...")
        pools[slug] = hash_dir(ROOT / slug, f"hashes_{slug}.json")
        print(f"  {slug}: {len(pools[slug])} images hashed")

    for slug, pool in pools.items():
        r = {"images": len(pool)}
        # -- leakage vs target
        crit, info = [], []
        for k, h in pool.items():
            d, who = min_hamming(h, t_valtest)
            if d <= NEAR_DUP:
                crit.append((Path(k).name, d, Path(who).name))
                continue
            d2, _ = min_hamming(h, t_train)
            if d2 <= NEAR_DUP:
                info.append((Path(k).name, d2))
        r["leak_valtest"] = crit
        r["dup_train"] = len(info)
        # -- intra-set dups (exact phash collisions)
        c = Counter(pool.values())
        r["intra_exact_dups"] = sum(v - 1 for v in c.values() if v > 1)

        if slug != "dvdi":
            names = names_of(ROOT / slug)
            r["classes"] = names
            def_ids = [i for i, n in enumerate(names) if is_defective_name(n)]
            areas, per_img = [], []
            unlabeled = 0
            res = Counter()
            for k in pool:
                im_boxes = yolo_labels(k)
                if im_boxes is None or not im_boxes:
                    unlabeled += 1
                    continue
                try:
                    res[Image.open(k).size] += 1
                except Exception:
                    pass
                da = [w * h for cls, x, y, w, h in im_boxes if cls in def_ids]
                if da:
                    areas += da
                    per_img.append((min(da), k, im_boxes))
            areas.sort()
            q = lambda p: areas[int(p * (len(areas) - 1))] if areas else None
            r.update({
                "unlabeled_images": unlabeled,
                "defective_boxes": len(areas),
                "images_with_defective": len(per_img),
                "def_area_q10_med_q90": [q(.1), q(.5), q(.9)],
                "distant_lt_1pct": sum(a < 0.01 for a in areas),
                "distant_lt_0p2pct": sum(a < 0.002 for a in areas),
                "closeup_gt_5pct": sum(a > 0.05 for a in areas),
                "top_resolutions": res.most_common(3),
            })
            # -- stage most zoomed-out defective images for manual review
            outdir = REVIEW / slug
            outdir.mkdir(parents=True, exist_ok=True)
            per_img.sort(key=lambda t: t[0])
            for rank, (a, k, im_boxes) in enumerate(per_img[:N_REVIEW]):
                try:
                    draw_review(k, im_boxes, names, outdir / f"{rank:02d}_a{a:.4f}_{Path(k).name}")
                except Exception as e:
                    print(f"  [warn] review draw failed {Path(k).name}: {e}")
            r["review_dir"] = str(outdir)
        report[slug] = r

    # -- cross-set duplicate matrix
    slugs = list(pools)
    cross = {}
    for i, a in enumerate(slugs):
        for b in slugs[i + 1:]:
            n = sum(1 for h in pools[a].values()
                    if min_hamming(h, pools[b])[0] <= NEAR_DUP)
            cross[f"{a} ∩ {b}"] = n
    report["cross_set_near_dups"] = cross

    out = ROOT / "vetting_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("\n" + "=" * 70)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out}\nreview images -> {REVIEW}/<set>/")


if __name__ == "__main__":
    main()
