#!/usr/bin/env python3
"""READ-ONLY vetting of Roboflow Universe insulator-defect & birdnest datasets vs the
ATLI SOURCE (atli_source_dataset) and TARGET (merged_atli_target). Downloads each,
pHash leakage check, class taxonomy, intra/cross-set dups, and a sample-image montage
per set for relevance eyeballing. NEVER merges or writes to Roboflow.

Why: ATLI used ~200 CPLID insulator images, and many Universe insulator-defect sets are
CPLID re-uploads -> high overlap risk (same trap DVDI sprang for dampers). Birdnest sets
exist but power-line provenance is unverified; overlap test + montage settle both.

Output: ~/atli/datasets/classvet/classvet_report.json + review/<slug>.png montages.
"""
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path.home() / "atli"
load_dotenv(ROOT / ".env")
import os

KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
assert KEY, "ROBOFLOW_API_KEY missing"
WS_OURS = "tl-target-set-focus"

import imagehash
import yaml
from PIL import Image

OUT = ROOT / "datasets" / "classvet"
OUT.mkdir(parents=True, exist_ok=True)
REVIEW = OUT / "review"; REVIEW.mkdir(exist_ok=True)
NEAR = 8           # hamming<=8 of 64 => near-duplicate
N_MONTAGE = 24

# (slug, workspace, project, version)
INSULATOR = [
    ("ins_giomartins", "giomartins", "danified-insulators-wyxhs", 2),
    ("ins_trashdet",   "trash-detection-zqfed", "insulator-defect-gcnfq", 1),
    ("ins_hello",      "hello-92xp7", "insulator-defect-detection-yywkb", 3),
    ("ins_uttam",      "uttam", "insulator-defect", 15),
    ("ins_adspq",      "insulator-adspq", "broken-insulator", 3),
]
BIRDNEST = [
    ("nest_nestdet", "nestdet", "nest-9imzt", 3),
    ("nest_cugb",    "cugb", "nest-ezl6r", 1),
    ("nest_gg",      "gg-97eg1", "nest-hgbmq", 1),
]
CANDIDATES = INSULATOR + BIRDNEST
# our reference sets (all splits hashed; target val/test = leakage-critical)
REFS = [("source", WS_OURS, "atli_source_dataset", 4),
        ("target", WS_OURS, "merged_atli_target", 4)]


def download(slug, ws, proj, ver):
    from roboflow import Roboflow
    dest = OUT / slug
    if dest.exists() and any(dest.rglob("*.jpg")) or (dest.exists() and any(dest.rglob("*.png"))):
        print(f"[skip] {slug} present"); return dest
    print(f"[dl] {ws}/{proj} v{ver} -> {slug}")
    Roboflow(api_key=KEY).workspace(ws).project(proj).version(ver).download(
        "yolov8", location=str(dest))
    return dest


def images_of(root):
    return sorted(p for p in Path(root).rglob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "review" not in p.parts)


def hash_dir(root, cache):
    cp = OUT / cache
    done = json.loads(cp.read_text()) if cp.exists() else {}
    for p in images_of(root):
        if str(p) not in done:
            try:
                done[str(p)] = str(imagehash.phash(Image.open(p)))
            except Exception:
                pass
    cp.write_text(json.dumps(done))
    return {k: int(v, 16) for k, v in done.items() if Path(k).exists()}


def split_of(s):
    for sp in ("valid", "test", "train"):
        if sp in Path(s).parts:
            return sp
    return "?"


def near_count(pool, ref):
    """count pool hashes within NEAR of any ref hash; return (count, examples)."""
    hits = []
    for k, h in pool.items():
        for v in ref.values():
            if bin(h ^ v).count("1") <= NEAR:
                hits.append(Path(k).name); break
    return len(hits), hits[:5]


def names_of(root):
    for y in Path(root).rglob("data.yaml"):
        n = yaml.safe_load(y.read_text()).get("names")
        return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    return []


def montage(root, slug):
    imgs = images_of(root)[:N_MONTAGE]
    cols, cell = 6, 200
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * cell), (28, 28, 28))
    for i, p in enumerate(imgs):
        try:
            im = Image.open(p).convert("RGB"); im.thumbnail((cell - 6, cell - 6))
            grid.paste(im, ((i % cols) * cell + 3, (i // cols) * cell + 3))
        except Exception:
            pass
    grid.save(REVIEW / f"{slug}.png")


# -- hash reference sets
print("[refs] hashing source + target ...")
refs = {}
for name, ws, proj, ver in REFS:
    d = download(name, ws, proj, ver)
    refs[name] = hash_dir(d, f"hash_{name}.json")
    print(f"  {name}: {len(refs[name])} imgs")
tgt = refs["target"]
tgt_valtest = {k: v for k, v in tgt.items() if split_of(k) in ("valid", "test")}
tgt_train = {k: v for k, v in tgt.items() if split_of(k) == "train"}
print(f"  target split: {len(tgt_train)} train / {len(tgt_valtest)} val+test")

# -- vet candidates
report, pools = {}, {}
for slug, ws, proj, ver in CANDIDATES:
    try:
        d = download(slug, ws, proj, ver)
    except Exception as e:
        report[slug] = {"error": str(e)[:200]}; print(f"[ERR] {slug}: {e}"); continue
    pools[slug] = hash_dir(d, f"hash_{slug}.json")
    montage(d, slug)
    n_vt, ex_vt = near_count(pools[slug], tgt_valtest)
    n_tr, _ = near_count(pools[slug], tgt_train)
    n_src, _ = near_count(pools[slug], refs["source"])
    dups = sum(c - 1 for c in Counter(pools[slug].values()).values() if c > 1)
    report[slug] = {
        "url": f"https://universe.roboflow.com/{ws}/{proj}",
        "images": len(pools[slug]), "classes": names_of(d),
        "leak_target_valtest": n_vt, "leak_target_valtest_examples": ex_vt,
        "dup_target_train": n_tr, "overlap_source": n_src, "intra_dups": dups,
    }
    print(f"[{slug}] {len(pools[slug])} imgs | val/test leak={n_vt} train={n_tr} "
          f"source={n_src} intra={dups} | classes={names_of(d)}")

# -- cross-candidate near-dup matrix (recycling detection)
slugs = list(pools); cross = {}
for i, a in enumerate(slugs):
    for b in slugs[i + 1:]:
        n, _ = near_count(pools[a], pools[b])
        if n:
            cross[f"{a} ∩ {b}"] = n
report["_cross_candidate_dups"] = cross
(OUT / "classvet_report.json").write_text(json.dumps(report, indent=2))
print("\n=== cross-candidate overlaps (recycling) ===")
print(json.dumps(cross, indent=2) if cross else "  none")
print(f"\nreport -> {OUT/'classvet_report.json'} ; montages -> {REVIEW}/")
print("CLASSVET_DONE")
