#!/usr/bin/env python3
"""Stage ~300 DEFECTIVE source images into `eduardos-annotated-photos` for manual selection.

Pulls images containing Defective_Damper / Defective_Insulators from ATLI_source_dataset
(0 overlap with the target -> all new material, verified), picks a MIX per class of
"zoomed-out" (smallest defective-bbox area) + random, rebuilds the annotations, and uploads
them to `eduardos-annotated-photos` so Eduardo can hand-pick from there.

API-only (no full dataset export). Boxes come from GET /{ws}/{proj}/images/{id} whose
annotation boxes are CENTER x,y + width,height in pixels -> converted to VOC corners.

Reversible: every upload is tagged + batched `eduardos_stage_v1` (filter that tag in the UI
and bulk-delete to undo). DRY-RUN by default; pass --yes to actually upload.

Run:  .venv/bin/python rebalance/stage_to_eduardos.py            # dry-run plan
      .venv/bin/python rebalance/stage_to_eduardos.py --yes      # upload
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.sax.saxutils import escape

import requests
from dotenv import load_dotenv

load_dotenv("/Users/eddie/Research/Vegas/.env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
API = "https://api.roboflow.com"
SOURCE = "atli_source_dataset"
DEST = "eduardos-annotated-photos"
CLASSES = ("Defective_Damper", "Defective_Insulators")
DROP = {"Transmission_tower", "transmission_line", "defect_transmission_line"}  # excluded by target taxonomy
RNG = random.Random(20260602)


def search_class(project, cls, page=250):
    """All image stubs (id,name,url,w,h) for one class via the reliable class_name filter."""
    out, off, total = {}, 0, None
    while True:
        r = requests.post(f"{API}/{WS}/{project}/search", params={"api_key": KEY},
                          json={"fields": ["id", "name", "url", "width", "height"],
                                "limit": page, "offset": off, "class_name": cls}, timeout=120)
        r.raise_for_status()
        d = r.json()
        total = d.get("total") if total is None else total
        res = d.get("results", [])
        if not res:
            break
        for x in res:
            out[x["id"]] = x
        off += len(res)
        if off >= total or len(res) < page:
            break
        time.sleep(0.03)
    return list(out.values()), total


def fetch_boxes(stub):
    """Return stub augmented with 'boxes' (list of dict label,x,y,width,height) or None."""
    try:
        r = requests.get(f"{API}/{WS}/{SOURCE}/images/{stub['id']}", params={"api_key": KEY}, timeout=60)
        r.raise_for_status()
        stub = dict(stub)
        stub["boxes"] = (r.json().get("image", {}).get("annotation", {}) or {}).get("boxes", [])
        return stub
    except Exception:
        return None


def _f(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def def_area(stub, cls):
    """Largest normalized area among this image's boxes of class `cls` (None if missing)."""
    W, H = _f(stub.get("width"), 1) or 1, _f(stub.get("height"), 1) or 1
    areas = [(_f(b["width"]) * _f(b["height"])) / (W * H)
             for b in stub["boxes"] if b.get("label") == cls]
    return max(areas) if areas else None


def voc_xml(stub):
    """Build a Pascal-VOC XML string from CENTER-format boxes, dropping tower/line classes."""
    W, H = int(_f(stub.get("width"))), int(_f(stub.get("height")))
    objs = []
    for b in stub["boxes"]:
        lab = b.get("label")
        if lab in DROP:
            continue
        bx, byc, bw, bh = _f(b["x"]), _f(b["y"]), _f(b["width"]), _f(b["height"])
        xmin = max(0, bx - bw / 2);  ymin = max(0, byc - bh / 2)
        xmax = min(W, bx + bw / 2);  ymax = min(H, byc + bh / 2)
        objs.append(f"<object><name>{escape(lab)}</name><bndbox>"
                    f"<xmin>{int(xmin)}</xmin><ymin>{int(ymin)}</ymin>"
                    f"<xmax>{int(xmax)}</xmax><ymax>{int(ymax)}</ymax></bndbox></object>")
    return (f"<annotation><filename>{escape(stub['fname'])}</filename>"
            f"<size><width>{W}</width><height>{H}</height><depth>3</depth></size>"
            f"{''.join(objs)}</annotation>"), len(objs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150, help="images per defective class (~300 total)")
    ap.add_argument("--zoom-frac", type=float, default=0.5, help="fraction taken as most zoomed-out")
    ap.add_argument("--pool-insulator", type=int, default=900,
                    help="random candidate pool to box-fetch for the huge insulator class")
    ap.add_argument("--area-floor", type=float, default=0.0002, help="drop unannotatable specks")
    ap.add_argument("--area-ceil", type=float, default=0.05, help="drop extreme close-ups")
    ap.add_argument("--tag", default="eduardos_stage_v1")
    ap.add_argument("--out", default="/Users/eddie/Research/Vegas/datasets/eduardos_staging")
    ap.add_argument("--yes", action="store_true", help="actually upload (default: dry-run)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY missing")

    selected, used_ids = [], set()  # global dedup across classes
    for cls in CLASSES:
        stubs, total = search_class(SOURCE, cls)
        print(f"[{cls}] candidates in source: {len(stubs)} (reported {total})")
        pool = stubs if len(stubs) <= args.pool_insulator else RNG.sample(stubs, args.pool_insulator)
        if len(pool) < len(stubs):
            print(f"   box-fetching a random pool of {len(pool)} (class too large to fetch all)")
        boxed = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for f in as_completed([ex.submit(fetch_boxes, s) for s in pool]):
                r = f.result()
                if r is not None:
                    boxed.append(r)
        # score + band-filter
        scored = []
        for s in boxed:
            a = def_area(s, cls)
            if a is not None and args.area_floor < a < args.area_ceil:
                s["_area"] = a; scored.append(s)
        scored.sort(key=lambda s: s["_area"])  # ascending => most zoomed-out first
        n_zoom = int(round(args.per_class * args.zoom_frac))
        n_rand = args.per_class - n_zoom
        picked, seen = [], set()
        for s in scored:                                   # zoomed-out tranche
            if s["id"] in used_ids or s["id"] in seen:
                continue
            picked.append(("zoom", s)); seen.add(s["id"])
            if len(picked) >= n_zoom:
                break
        rest = [s for s in scored if s["id"] not in seen and s["id"] not in used_ids]
        RNG.shuffle(rest)
        for s in rest[:n_rand]:                            # random tranche
            picked.append(("random", s)); seen.add(s["id"])
        for kind, s in picked:
            s["_kind"], s["fname"] = kind, f"{s['id']}.jpg"
            selected.append((cls, s)); used_ids.add(s["id"])
        print(f"   selected {len(picked)}  (zoom={sum(1 for k,_ in picked if k=='zoom')}, "
              f"random={sum(1 for k,_ in picked if k=='random')})  "
              f"area band [{args.area_floor},{args.area_ceil}], in-band={len(scored)}")

    print(f"\n==== PLAN: upload {len(selected)} images -> {DEST}  (tag/batch '{args.tag}', split=train) ====")
    by = {}
    for cls, s in selected:
        by[cls] = by.get(cls, 0) + 1
    for cls, n in by.items():
        print(f"   {cls:24s} {n}")
    if not args.yes:
        print("\nDRY-RUN. Re-run with --yes to download+upload. Nothing changed.")
        # still write a manifest preview
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "plan.csv").write_text("class,kind,id,name,def_area\n" + "".join(
            f"{cls},{s['_kind']},{s['id']},{s['name']},{s['_area']:.5f}\n" for cls, s in selected))
        print(f"wrote preview manifest -> {out/'plan.csv'}")
        return

    # ---- upload path (the only mutation) ----
    from roboflow import Roboflow
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    proj = Roboflow(api_key=KEY).workspace(WS).project(DEST)
    ok = fail = 0
    for cls, s in selected:
        try:
            img_p = out / "images" / s["fname"]
            if not img_p.exists():
                img_p.write_bytes(requests.get(s["url"], timeout=120).content)
            xml, nobj = voc_xml(s)
            lbl_p = out / "labels" / f"{s['id']}.xml"
            lbl_p.write_text(xml)
            proj.single_upload(image_path=str(img_p), annotation_path=str(lbl_p),
                               split="train", batch_name=args.tag, tag_names=[args.tag])
            ok += 1
            if ok % 25 == 0:
                print(f"   uploaded {ok}/{len(selected)} ...")
        except Exception as e:
            fail += 1
            print(f"   FAIL {s['id']}: {str(e)[:120]}")
    print(f"\nDONE. uploaded={ok} failed={fail}.  Undo: filter tag '{args.tag}' in {DEST} and bulk-delete.")


if __name__ == "__main__":
    main()
