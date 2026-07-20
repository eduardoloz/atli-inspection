#!/usr/bin/env python3
"""Add CPLID images to the TRAIN split of the existing eduardos CV folds.

CPLID source = `merged_atli_target` (the pre-purge target that still contains the
CPLID photos). A merged image is a CPLID-restore candidate iff it has NO pHash
match (d<=6) in the CV pool (minus-cplid + eduardos) — i.e. it is one of the
images that was removed during the purge. Because these images are pHash-disjoint
from the whole pool, adding them to train CANNOT leak into any fold's val/test.

For each existing fold in CV_eduardo_det / CV_eduardo_obb this writes:
  train_cplid/         = train + CPLID            -> base_cplid.yaml
  trainosall_cplid/    = trainosall + CPLID       -> osall_cplid.yaml
val/ and test/ are the original (untouched) eval splits.

Run on server: ~/atli/env/bin/python build_cplid_cv.py
"""
import os, io, shutil, yaml
from pathlib import Path
from collections import Counter
import numpy as np, cv2
from PIL import Image
from dotenv import load_dotenv

ROOT = Path.home() / "atli"
DL = ROOT / "downloads"
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
CANON = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
DEFECTS = {1, 2, 3, 6}
DET = ROOT / "CV_eduardo_det"
OBB = ROOT / "CV_eduardo_obb"
POOL_DIRS = [DL / "mcplid_v2", DL / "eduardo_v6"]   # the 974-image CV pool sources


# ---- pHash (same DCT impl as the audit) ----
_D = None
def dct(N):
    global _D
    if _D is None or _D.shape[0] != N:
        n = np.arange(N); _D = np.sqrt(2.0/N)*np.cos(np.pi*(2*n+1)[:, None]*n[None, :]/(2*N)); _D[:, 0] = np.sqrt(1.0/N)
    return _D
def phash(path):
    g = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.LANCZOS), float)
    low = (dct(32).T @ g @ dct(32))[:8, :8].flatten(); m = np.median(low[1:])
    v = 0
    for b in (low > m):
        v = (v << 1) | int(b)
    return v & ~(1 << 63)
def ham(a, b):
    return bin(a ^ b).count("1")


def rf_download(project, version, dest):
    from roboflow import Roboflow
    dest = Path(dest)
    if dest.exists() and (dest / "data.yaml").exists():
        print(f"  [skip] {dest} present"); return dest
    Roboflow(api_key=KEY).workspace(WS).project(project).version(version).download("yolov5", location=str(dest))
    return dest


def canon_lines(lbl, names):
    idmap = {i: (CANON.index(n) if n in CANON else None) for i, n in enumerate(names)}
    out = []
    for ln in Path(lbl).read_text().splitlines():
        p = ln.split()
        if not p:
            continue
        c = idmap.get(int(p[0]))
        if c is not None:
            out.append(f"{c} " + " ".join(p[1:]))
    return out


def to_det(line):
    p = line.split(); c = p[0]; v = [float(x) for x in p[1:]]
    if len(v) == 4:
        return f"{c} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]:.6f}"
    xs, ys = v[0::2], v[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return f"{c} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}"


def to_obb(line, w, h):
    p = line.split(); c = p[0]; v = [float(x) for x in p[1:]]
    if len(v) == 4:
        cx, cy, bw, bh = v
        cs = [cx-bw/2, cy-bh/2, cx+bw/2, cy-bh/2, cx+bw/2, cy+bh/2, cx-bw/2, cy+bh/2]
        return f"{c} " + " ".join(f"{x:.6f}" for x in cs)
    pts = np.array([[v[i]*w, v[i+1]*h] for i in range(0, len(v), 2)], np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    cs = []
    for a, b in box:
        cs += [float(np.clip(a/w, 0, 1)), float(np.clip(b/h, 0, 1))]
    return f"{c} " + " ".join(f"{x:.6f}" for x in cs)


# ---- 1. hash the CV pool ----
print("[1/4] hashing CV pool (minus-cplid + eduardos)")
pool_h = set()
for d in POOL_DIRS:
    for img in Path(d).rglob("images/*"):
        try:
            pool_h.add(phash(img))
        except Exception:
            pass
print(f"      pool hashes: {len(pool_h)}")

# ---- 2. download merged + find CPLID = merged imgs with no pool match ----
print("[2/4] download merged_atli_target v6, isolate CPLID (merged-only)")
merged = rf_download("merged_atli_target", 6, DL / "merged_v6")
names = yaml.safe_load(open(merged / "data.yaml"))["names"]
cplid = []
for lbl in sorted(merged.rglob("labels/*.txt")):
    imgs = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
    if not imgs:
        continue
    try:
        h = phash(imgs[0])
    except Exception:
        continue
    if any(ham(h, ph) <= 6 for ph in pool_h):
        continue                          # in the pool -> not a removed CPLID image
    lines = canon_lines(lbl, names)
    if lines:
        cplid.append({"img": imgs[0], "lines": lines, "stem": "cplid_" + lbl.stem})
cc = Counter(int(l.split()[0]) for it in cplid for l in it["lines"])
print(f"      CPLID-restore images: {len(cplid)}  class instances: "
      f"{ {CANON[k]: cc.get(k, 0) for k in range(7)} }")
assert 200 <= len(cplid) <= 300, f"unexpected CPLID count {len(cplid)}"

DIMS = {}
def dims(img):
    if img not in DIMS:
        im = cv2.imread(str(img)); DIMS[img] = (im.shape[1], im.shape[0])
    return DIMS[img]

# ---- 3. materialize CPLID into each fold's train (det + obb) ----
def add_cplid(fold_root, src_split, dst_split, obb):
    src, dst = fold_root / src_split, fold_root / dst_split
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)                       # start from the existing train
    for it in cplid:
        w, h = dims(it["img"]) if obb else (0, 0)
        lab = "\n".join((to_obb(l, w, h) if obb else to_det(l)) for l in it["lines"])
        shutil.copy2(it["img"], dst / "images" / (it["stem"] + it["img"].suffix))
        (dst / "labels" / (it["stem"] + ".txt")).write_text(lab)

def write_yaml(root, k, train_dir, name):
    (root / f"fold{k}" / f"{name}.yaml").write_text(yaml.safe_dump({
        "path": str(root / f"fold{k}"), "train": f"{train_dir}/images",
        "val": "val/images", "test": "test/images", "nc": 7, "names": CANON}, sort_keys=False))

print("[3/4] adding CPLID to each fold's train (det + obb)")
for k in range(5):
    for root, is_obb in [(DET, False), (OBB, True)]:
        fr = root / f"fold{k}"
        add_cplid(fr, "train", "train_cplid", is_obb)
        add_cplid(fr, "trainosall", "trainosall_cplid", is_obb)
        write_yaml(root, k, "train_cplid", "base_cplid")
        write_yaml(root, k, "trainosall_cplid", "osall_cplid")
    n = len(list((DET / f"fold{k}" / "trainosall_cplid" / "images").glob("*")))
    print(f"      fold{k}: trainosall_cplid = {n} imgs (+{len(cplid)} CPLID)")
print("[4/4] BUILD_CPLID_CV_DONE")
