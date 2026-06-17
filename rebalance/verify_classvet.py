#!/usr/bin/env python3
"""DOUBLE-CHECK the classvet overlap flags: recompute per-image min-distance to ATLI
target val/test at strict thresholds (==0 exact, <=4, <=8) and render the N closest
candidate<->ATLI pairs side-by-side so a human can confirm they're真 duplicates, not
pHash look-alike false positives. READ-ONLY."""
import json
from collections import Counter
from pathlib import Path

from PIL import Image

OUT = Path.home() / "atli" / "datasets" / "classvet"
PAIRS = OUT / "pairs"; PAIRS.mkdir(exist_ok=True)
CHECK = ["ins_giomartins", "ins_trashdet", "ins_uttam", "nest_nestdet"]  # the flagged ones
KPAIRS = 8


def load(cache):
    p = OUT / cache
    return {k: int(v, 16) for k, v in json.loads(p.read_text()).items() if Path(k).exists()}


def split_of(s):
    for sp in ("valid", "test", "train"):
        if sp in Path(s).parts:
            return sp
    return "?"


tgt = load("hash_target.json")
valtest = {k: v for k, v in tgt.items() if split_of(k) in ("valid", "test")}
print(f"target val/test reference: {len(valtest)} imgs\n")

summary = {}
for slug in CHECK:
    pool = load(f"hash_{slug}.json")
    matches = []  # (dist, candpath, refpath)
    for ck, ch in pool.items():
        best, who = 65, None
        for rk, rh in valtest.items():
            d = bin(ch ^ rh).count("1")
            if d < best:
                best, who = d, rk
        matches.append((best, ck, who))
    matches.sort()
    dist = [m[0] for m in matches]
    buckets = {"exact_0": sum(d == 0 for d in dist), "le_2": sum(d <= 2 for d in dist),
               "le_4": sum(d <= 4 for d in dist), "le_8": sum(d <= 8 for d in dist)}
    summary[slug] = buckets
    print(f"[{slug}] {len(pool)} imgs -> val/test matches: {buckets}")
    # render the KPAIRS closest pairs
    for i, (d, ck, rk) in enumerate(matches[:KPAIRS]):
        try:
            a = Image.open(ck).convert("RGB"); a.thumbnail((300, 300))
            b = Image.open(rk).convert("RGB"); b.thumbnail((300, 300))
            canvas = Image.new("RGB", (a.width + b.width + 12, max(a.height, b.height) + 4),
                               (20, 20, 20))
            canvas.paste(a, (2, 2)); canvas.paste(b, (a.width + 10, 2))
            canvas.save(PAIRS / f"{slug}_{i:02d}_d{d}.png")  # left=candidate right=ATLI
        except Exception as e:
            print(f"  pair render fail {Path(ck).name}: {e}")

(OUT / "verify_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\npairs (left=Universe, right=ATLI val/test) -> {PAIRS}/")
print("Interpretation: d0 pairs that look identical = TRUE leakage; "
      "d6-8 pairs that look like different insulators = pHash false positive.")
print("VERIFY_DONE")
