#!/usr/bin/env python3
"""Group-aware 5-fold CV (detection + OBB) on the ATLI-no-CPLID + eduardos pool.

Pulls fresh from Roboflow:
  * atli_target-minus-the-cplid  v2  (796 imgs, no CPLID)
  * eduardos-annotated-photos    v6  (178 imgs, deduped head)
in `yolov5` format (polygon-preserving: boxes + polygon vertices).

Classes are remapped BY NAME to the canonical 7-class order (robust to export
order; eduardos' subset + any stray 0-instance class handled). eduardos'
21 duplicate clusters (edu_clusters.json) are kept WHOLE within a fold so no
near-duplicate spans train/val/test. Each fold is a 70/15/15 multilabel-
stratified split at the CLUSTER level (5 seeds).

Outputs (same fold membership for det & obb, so the tasks are comparable):
  ~/atli/CV_eduardo_det/fold{k}/{train,trainosall,val,test}/{images,labels}
                               + base.yaml, osall.yaml            (cx cy w h)
  ~/atli/CV_eduardo_obb/fold{k}/{train,trainosall,val,test}/{images,labels}
                               + base.yaml, osall.yaml            (8-corner OBB)

osall = train images containing any defect class {Broken_Insulator,
Defective_Damper, Flashover_Insulator, Self-Exploded_Insulator} duplicated x3
(TRAIN ONLY; val/test never oversampled). Detection labels collapse polygons to
axis-aligned boxes; OBB labels convert every box/polygon via cv2.minAreaRect.

Run on server: ~/atli/env/bin/python build_cv_eduardo.py
"""
import os, re, json, shutil, yaml
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, cv2
from dotenv import load_dotenv
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = Path.home() / "atli"
DL = ROOT / "downloads"; DL.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
assert KEY, "ROBOFLOW_API_KEY missing (~/atli/.env)"

CANON = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
DEFECTS = {1, 2, 3, 6}                      # BI, DD, Flashover, SE
OSFACTOR = 3
SEEDS = [42, 123, 456, 789, 1024]
CLUSTERS = json.load(open(ROOT / "edu_clusters.json"))   # {cluster: [roboflow_id,...]}
DET = ROOT / "CV_eduardo_det"
OBB = ROOT / "CV_eduardo_obb"


def rf_download(project, version, dest):
    from roboflow import Roboflow
    dest = Path(dest)
    if dest.exists() and (dest / "data.yaml").exists():
        print(f"  [skip] {dest} already downloaded"); return dest
    Roboflow(api_key=KEY).workspace(WS).project(project).version(version).download(
        "yolov5", location=str(dest))
    return dest


def load_source(dest, source):
    """Return list of dicts: {img, lines(canonical seg/box lines), cluster}."""
    names = yaml.safe_load(open(Path(dest) / "data.yaml"))["names"]
    idmap = {i: (CANON.index(n) if n in CANON else None) for i, n in enumerate(names)}
    dropped = Counter()
    items = []
    for lbl in sorted(Path(dest).rglob("labels/*.txt")):
        imgs = list((lbl.parent.parent / "images").glob(lbl.stem + ".*"))
        if not imgs:
            continue
        lines = []
        for ln in lbl.read_text().splitlines():
            p = ln.split()
            if not p:
                continue
            c = idmap.get(int(p[0]))
            if c is None:
                dropped[names[int(p[0])]] += 1
                continue
            lines.append(f"{c} " + " ".join(p[1:]))
        if not lines:
            continue
        if source == "edu":
            rid = re.split(r"[._]", lbl.stem)[0]      # roboflow id = leading token
            cid = next((cl for cl, ids in CLUSTERS.items() if rid in ids), f"edu_s_{rid}")
            stem = "edu_" + lbl.stem
        else:
            cid = f"atli_{lbl.stem}"
            stem = lbl.stem
        items.append({"img": imgs[0], "lines": lines, "cluster": cid, "stem": stem})
    if dropped:
        print(f"      [{source}] dropped non-canonical classes: {dict(dropped)}")
    return items


def to_det(line):
    """canonical seg/box line -> 'c cx cy w h' (collapse polygon to AABB)."""
    p = line.split(); c = p[0]; v = [float(x) for x in p[1:]]
    if len(v) == 4:
        return f"{c} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]:.6f}"
    xs = v[0::2]; ys = v[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return f"{c} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}"


def to_obb(line, w, h):
    """canonical seg/box line -> 'c x1 y1 .. x4 y4' via minAreaRect (norm)."""
    p = line.split(); c = p[0]; v = [float(x) for x in p[1:]]
    if len(v) == 4:
        cx, cy, bw, bh = v
        cs = [cx-bw/2, cy-bh/2, cx+bw/2, cy-bh/2, cx+bw/2, cy+bh/2, cx-bw/2, cy+bh/2]
        return f"{c} " + " ".join(f"{x:.6f}" for x in cs)
    pts = np.array([[v[i]*w, v[i+1]*h] for i in range(0, len(v), 2)], dtype=np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    cs = []
    for cx_, cy_ in box:
        cs += [float(np.clip(cx_/w, 0, 1)), float(np.clip(cy_/h, 0, 1))]
    return f"{c} " + " ".join(f"{x:.6f}" for x in cs)


def has_defect(lines):
    return any(int(l.split()[0]) in DEFECTS for l in lines)


def counts(items):
    c = Counter()
    for it in items:
        for l in it["lines"]:
            c[int(l.split()[0])] += 1
    return {CANON[k]: c.get(k, 0) for k in range(7)}


# ---- 1. download + load + remap ------------------------------------------
print("[1/4] download + remap by name")
atli = load_source(rf_download("atli_target-minus-the-cplid", 2, DL / "mcplid_v2"), "atli")
edu = load_source(rf_download("eduardos-annotated-photos", 6, DL / "eduardo_v6"), "edu")
pool = atli + edu
print(f"      atli={len(atli)} imgs {counts(atli)}")
print(f"      edu ={len(edu)} imgs {counts(edu)}")
print(f"      POOL={len(pool)} imgs {counts(pool)}")
assert len(atli) == 796, f"atli imgs {len(atli)} != 796"
assert 170 <= len(edu) <= 185, f"edu imgs {len(edu)} out of range"

# image-dim cache (for OBB)
DIMS = {}
def dims(img):
    if img not in DIMS:
        im = cv2.imread(str(img)); DIMS[img] = (im.shape[1], im.shape[0])
    return DIMS[img]

# ---- 2. cluster-level stratified 5-fold ----------------------------------
print("[2/4] group-aware 5-fold split (cluster-level)")
by_cl = defaultdict(list)
for it in pool:
    by_cl[it["cluster"]].append(it)
cl_ids = list(by_cl)
Y = np.zeros((len(cl_ids), 7), int)
for i, cl in enumerate(cl_ids):
    for it in by_cl[cl]:
        for l in it["lines"]:
            Y[i, int(l.split()[0])] = 1
print(f"      {len(pool)} imgs in {len(cl_ids)} clusters "
      f"({sum(1 for c in cl_ids if len(by_cl[c])>1)} multi-image)")

def materialize(items, root, split, obb, oversample=False):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for it in items:
        reps = OSFACTOR if (oversample and has_defect(it["lines"])) else 1
        w, h = dims(it["img"]) if obb else (0, 0)
        lab = "\n".join((to_obb(l, w, h) if obb else to_det(l)) for l in it["lines"])
        for r in range(reps):
            pre = "" if r == 0 else f"os{r}_"
            shutil.copy2(it["img"], root / split / "images" / (pre + it["stem"] + it["img"].suffix))
            (root / split / "labels" / (pre + it["stem"] + ".txt")).write_text(lab)

def write_yaml(root, k, train_dir, name):
    (root / f"fold{k}" / f"{name}.yaml").write_text(yaml.safe_dump({
        "path": str(root / f"fold{k}"), "train": f"{train_dir}/images",
        "val": "val/images", "test": "test/images",
        "nc": 7, "names": CANON}, sort_keys=False))

for k, seed in enumerate(SEEDS):
    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, tmp = next(s1.split(cl_ids, Y))
    s2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va_rel, te_rel = next(s2.split(tmp, Y[tmp]))
    va, te = tmp[va_rel], tmp[te_rel]
    part = {"train": tr, "val": va, "test": te}
    imgs = {sp: [it for i in ix for it in by_cl[cl_ids[i]]] for sp, ix in part.items()}
    # leakage assert: clusters disjoint across splits
    csets = {sp: {cl_ids[i] for i in ix} for sp, ix in part.items()}
    assert not (csets["train"] & csets["val"]) and not (csets["train"] & csets["test"]) \
        and not (csets["val"] & csets["test"]), f"fold{k}: cluster spans splits!"
    for task_root, is_obb in [(DET, False), (OBB, True)]:
        fr = task_root / f"fold{k}"
        shutil.rmtree(fr, ignore_errors=True)
        materialize(imgs["train"], fr, "train", is_obb)
        materialize(imgs["train"], fr, "trainosall", is_obb, oversample=True)
        materialize(imgs["val"], fr, "val", is_obb)
        materialize(imgs["test"], fr, "test", is_obb)
        write_yaml(task_root, k, "train", "base")
        write_yaml(task_root, k, "trainosall", "osall")
    print(f"      fold{k}: train {len(imgs['train'])} / val {len(imgs['val'])} "
          f"/ test {len(imgs['test'])}  | test defects "
          f"{ {CANON[d]: sum(1 for it in imgs['test'] for l in it['lines'] if int(l.split()[0])==d) for d in sorted(DEFECTS)} }")

# ---- 3/4. summary ---------------------------------------------------------
print("[3/4] per-fold sizes written above; det + obb trees materialized")
print("[4/4] BUILD_CV_EDUARDO_DONE")
