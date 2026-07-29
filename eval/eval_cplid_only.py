#!/usr/bin/env python3
"""CPLID-as-held-out-test experiment (2026-07-29).

Question: how does the current best v11n model generalize to CPLID when the
CPLID data appears ONLY in the testing data (never in training)?

The eduardo-CV pool (974 imgs) is CPLID-free by construction (post-purge
no-CPLID ATLI + eduardos), and the 250 restored CPLID images living in
CV_eduardo_obb/fold*/train_cplid are filename- and pHash-disjoint from every
fold's training split (verified 2026-07-29: 0 overlap vs trainosall for all
5 folds). So any checkpoint NOT trained on a *_cplid.yaml has never seen
CPLID, and the 250 CPLID images form a legitimate held-out test set.

Builds (idempotent, symlinks):
  ~/atli/CPLID_only_obb/            250 CPLID imgs + OBB labels + cplid_only.yaml
  ~/atli/CPLID_plus_test_obb/foldK/ fold-K native test (148) + the 250 CPLID
Evaluates (per fold checkpoint, 5 folds each):
  mix15_1280  @1280  — current best v11n (CV champion, clean 0.804)
  deg15       @1280  — previous 1280 champion, for context
  blurmix_640 @640   — 640 deployment champion, for context
on both test sets -> ~/atli/eval_cplid_only_results.json
"""
import json
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path.home() / "atli"
OBB = ROOT / "CV_eduardo_obb"
CLASSES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
           "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]
DEVICE = int(os.environ.get("EVAL_DEV", 7))
BATCH = int(os.environ.get("EVAL_BATCH", 8))

NAMES_YAML = "nc: 7\nnames:\n" + "".join(f"- {c}\n" for c in CLASSES)


def link(src: Path, dst: Path):
    if not dst.exists() and src.exists():
        dst.symlink_to(src)


def build_sets():
    train = {p.name for p in (OBB / "fold0/train/images").iterdir()}
    cplid = sorted(p for p in (OBB / "fold0/train_cplid/images").iterdir()
                   if p.name not in train)
    assert len(cplid) == 250, f"expected 250 CPLID imgs, got {len(cplid)}"

    only = ROOT / "CPLID_only_obb"
    for sub in ("images", "labels"):
        (only / sub).mkdir(parents=True, exist_ok=True)
    for img in cplid:
        link(img, only / "images" / img.name)
        lab = OBB / "fold0/train_cplid/labels" / (img.stem + ".txt")
        link(lab, only / "labels" / lab.name)
    (only / "cplid_only.yaml").write_text(
        f"path: {only}\ntrain: images\nval: images\ntest: images\n" + NAMES_YAML)

    plus = ROOT / "CPLID_plus_test_obb"
    for k in range(5):
        fd = plus / f"fold{k}"
        for sub in ("images", "labels"):
            (fd / sub).mkdir(parents=True, exist_ok=True)
        for img in (OBB / f"fold{k}/test/images").iterdir():
            link(img, fd / "images" / img.name)
            link(OBB / f"fold{k}/test/labels" / (img.stem + ".txt"),
                 fd / "labels" / (img.stem + ".txt"))
        for img in cplid:
            link(img, fd / "images" / img.name)
            link(OBB / "fold0/train_cplid/labels" / (img.stem + ".txt"),
                 fd / "labels" / (img.stem + ".txt"))
        (fd / "data.yaml").write_text(
            f"path: {fd}\ntrain: images\nval: images\ntest: images\n" + NAMES_YAML)
    n_lab = len(list((only / "labels").iterdir()))
    print(f"[build] CPLID_only_obb: {len(cplid)} imgs / {n_lab} labels; "
          f"plus-test folds built (148+250 imgs each)")


def ev(run, yaml, imz, fold):
    w = ROOT / "runs" / f"{run}_f{fold}_s2" / "weights" / "best.pt"
    r = YOLO(str(w)).val(data=str(yaml), split="test", imgsz=imz, device=DEVICE,
                         batch=BATCH, verbose=False, save_json=False, plots=False)
    met = r.box  # DetMetrics and OBBMetrics both expose per-class results here
    idx = {r.names[c]: i for i, c in enumerate(met.ap_class_index)}
    out = {"mAP50": float(met.map50), "P": float(met.mp), "R": float(met.mr)}
    for c in CLASSES:
        out[f"{c}_AP"] = float(met.ap50[idx[c]]) if c in idx else float("nan")
        out[f"{c}_P"] = float(met.p[idx[c]]) if c in idx else float("nan")
        out[f"{c}_R"] = float(met.r[idx[c]]) if c in idx else float("nan")
    return out


def ms(folds, field):
    vals = [f[field] for f in folds if not np.isnan(f.get(field, float("nan")))]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)


def main():
    build_sets()
    conds = [("mix15_1280", "EDU_mix15_1280", 1280),
             ("deg15", "EDU_deg15_v11", 1280),
             ("blurmix_640", "EDU_blurmix_640", 640)]
    # per user 2026-07-29: the experiment is CPLID ADDED to the native test
    # split (148 native + 250 CPLID = 398 imgs), not a CPLID-only test set.
    variants = [("testpluscplid", lambda k: ROOT / f"CPLID_plus_test_obb/fold{k}/data.yaml")]
    out_path = ROOT / "eval_cplid_only_results.json"
    results = json.load(open(out_path)) if out_path.exists() else {}
    for ckey, run, imz in conds:
        missing = [k for k in range(5)
                   if not (ROOT / "runs" / f"{run}_f{k}_s2/weights/best.pt").exists()]
        if missing:
            print(f"=== {ckey}: SKIP (missing folds {missing}) ===")
            continue
        for vkey, yfn in variants:
            key = f"{ckey}_{vkey}"
            if key in results:
                print(f"=== {key}: SKIP (already evaluated) ===")
                continue
            folds = [ev(run, yfn(k), imz, k) for k in range(5)]
            results[key] = {"task": "obb", "folds": folds}
            json.dump(results, open(out_path, "w"), indent=1)  # incremental
            mm = ms(folds, "mAP50")
            print(f"=== {key}: mAP50 {mm[0]:.3f}+/-{mm[1]:.3f} "
                  f"P {ms(folds,'P')[0]:.3f} R {ms(folds,'R')[0]:.3f} ===")
            for c in CLASSES:
                ap, p, r = ms(folds, f"{c}_AP"), ms(folds, f"{c}_P"), ms(folds, f"{c}_R")
                if not np.isnan(ap[0]):
                    print(f"  {c:24s} P {p[0]:.3f}  R {r[0]:.3f}  AP {ap[0]:.3f}+/-{ap[1]:.3f}")
    print("EVAL_CPLID_ONLY_DONE ->", out_path)


if __name__ == "__main__":
    main()
