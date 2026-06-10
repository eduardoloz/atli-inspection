#!/usr/bin/env python3
"""READ-ONLY, AUTHORITATIVE overlap report from the downloaded YOLO exports.

Roboflow export filenames are `<stem>.rf.<contenthash>.<ext>`; identical image content
shares the same `.rf.<hash>` across projects. So exact duplicates = shared .rf hash.
We also run a perceptual-hash pass to catch re-encoded near-duplicates.

For each defective source class (Defective_Damper, Defective_Insulators) we report how many
source images are already in the target, broken down by the target split they collide with
(val/test collisions are the leakage risk), and write a CSV of every colliding pair.

Run:  .venv/bin/python rebalance/overlap_from_export.py [--phash] [--phash-thresh 5]
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import yaml

SRC = Path("/Users/eddie/Research/Vegas/datasets/atli_source")
TGT = Path("/Users/eddie/Research/Vegas/datasets/merged_target")
DEFECT = ("Defective_Damper", "Defective_Insulators")
RF_RE = re.compile(r"\.rf\.([0-9a-f]+)\.[^.]+$", re.I)
STEM_RE = re.compile(r"^(.*?)\.rf\.[0-9a-f]+\.[^.]+$", re.I)


def names_for(root):
    y = next(root.rglob("data.yaml"))
    n = yaml.safe_load(y.read_text())["names"]
    return [n[k] for k in sorted(n)] if isinstance(n, dict) else n


def label_classes(lbl, names):
    out = set()
    for line in Path(lbl).read_text().splitlines():
        p = line.split()
        if p:
            out.add(names[int(float(p[0]))])
    return out


def index(root):
    """Map .rf-hash -> dict(split, classes, path); also stem-> same, for each image."""
    names = names_for(root)
    by_hash, by_stem, items = {}, defaultdict(list), []
    for lbl in root.rglob("labels/*.txt"):
        split = lbl.parent.parent.name  # train/valid/test
        imgs = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
        if not imgs:
            continue
        img = imgs[0]
        m, sm = RF_RE.search(img.name), STEM_RE.match(img.name)
        h = m.group(1).lower() if m else None
        stem = sm.group(1).lower() if sm else img.stem.lower()
        rec = {"split": split, "classes": label_classes(lbl, names), "name": img.name,
               "hash": h, "stem": stem, "path": img}
        items.append(rec)
        if h:
            by_hash[h] = rec
        by_stem[stem].append(rec)
    return names, by_hash, by_stem, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phash", action="store_true", help="also run perceptual-hash near-dup pass")
    ap.add_argument("--phash-thresh", type=int, default=5)
    ap.add_argument("--csv", default="/Users/eddie/Research/Vegas/rebalance/overlap_report.csv")
    args = ap.parse_args()
    for r in (SRC, TGT):
        if not r.exists():
            raise SystemExit(f"missing export {r} - run the download first")

    s_names, s_hash, s_stem, s_items = index(SRC)
    t_names, t_hash, t_stem, t_items = index(TGT)
    print(f"source images: {len(s_items)}   target images: {len(t_items)}")
    print(f"source classes: {s_names}")
    print(f"target classes: {t_names}")
    t_split_tot = defaultdict(int)
    for r in t_items:
        t_split_tot[r["split"]] += 1
    print(f"target split sizes: {dict(t_split_tot)}\n")

    tphash = []
    if args.phash:
        import imagehash
        from PIL import Image
        for r in t_items:
            try:
                tphash.append((imagehash.phash(Image.open(r["path"]).convert("RGB")), r))
            except Exception:
                pass

    rows = []
    print("========== OVERLAP (source defective vs target) ==========")
    for cls in DEFECT:
        recs = [r for r in s_items if cls in r["classes"]]
        hash_hit, stem_hit, near_hit = [], [], []
        matched = set()
        for r in recs:
            if r["hash"] and r["hash"] in t_hash:
                hash_hit.append((r, t_hash[r["hash"]])); matched.add(id(r)); continue
            if r["stem"] in t_stem:
                stem_hit.append((r, t_stem[r["stem"]][0])); matched.add(id(r))
        if args.phash:
            import imagehash
            from PIL import Image
            for r in recs:
                if id(r) in matched:
                    continue
                try:
                    h = imagehash.phash(Image.open(r["path"]).convert("RGB"))
                except Exception:
                    continue
                best = min(((h - th, tr) for th, tr in tphash), default=(99, None))
                if best[0] <= args.phash_thresh:
                    near_hit.append((r, best[1]))
        split_ct = defaultdict(int)
        for _, t in hash_hit + stem_hit + near_hit:
            if t:
                split_ct[t["split"]] += 1
        dup = len(hash_hit) + len(stem_hit) + len(near_hit)
        print(f"\n{cls}:")
        print(f"  source images with this class      : {len(recs)}")
        print(f"  already in target  (exact .rf hash) : {len(hash_hit)}")
        print(f"  already in target  (same filename)  : {len(stem_hit)}")
        if args.phash:
            print(f"  near-dup (phash<= {args.phash_thresh})           : {len(near_hit)}")
        print(f"  -> OVERLAP total                    : {dup}")
        print(f"  -> NOT in target (safe to move)     : {len(recs) - dup}")
        if split_ct:
            print(f"  collisions land in target splits    : {dict(split_ct)}  (valid/test = leakage)")
        for tag, hits in (("exact_hash", hash_hit), ("filename", stem_hit), ("phash", near_hit)):
            for s, t in hits:
                rows.append([cls, tag, s["name"], s["split"],
                             t["name"] if t else "", t["split"] if t else "",
                             ";".join(sorted(t["classes"])) if t else ""])

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_class", "match", "source_name", "source_split",
                    "target_name", "target_split", "target_classes"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} colliding pairs -> {args.csv}")


if __name__ == "__main__":
    main()
