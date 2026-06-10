#!/usr/bin/env python3
"""READ-ONLY overlap report: ATLI_source_dataset  vs  merged_atli_target.

Why: before moving defective images source -> (eduardos-annotated-photos) -> target, we must
not re-add images the target already has, or we'd double-count and skew the class balance.

Signals (all metadata-only, NO image download):
  1. shared image id  -> Roboflow dedups uploads per workspace; same id in both projects
                         == the exact same image object. Definitive.
  2. shared filename  -> same original file (re-uploaded / re-encoded). Strong.
  3. (optional) CLIP-embedding cosine >= --cos  -> visual near-duplicate. Use --embeddings.

We focus the source side on images whose annotations contain Defective_Damper /
Defective_Insulators (the two classes you care about), via the per-image `annotations.classes`.

Run:   .venv/bin/python rebalance/check_overlap.py [--embeddings] [--cos 0.97] [--csv out.csv]
Writes nothing to Roboflow.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv

load_dotenv("/Users/eddie/Research/Vegas/.env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
API = "https://api.roboflow.com"
DEFECT_CLASSES = ("Defective_Damper", "Defective_Insulators")


def _post(url, body, tries=5):
    last = ""
    for i in range(tries):
        try:
            r = requests.post(url, params={"api_key": KEY}, json=body, timeout=120)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = str(e)[:200]
        time.sleep(1.5 * (i + 1))
    sys.exit(f"search failed after {tries} tries -> {last}\n  (body offset={body.get('offset')})")


def search_all(project, fields, page=100, max_pages=400):
    url = f"{API}/{WS}/{project}/search"
    offset, seen_ids, total = 0, set(), None
    for _ in range(max_pages):
        data = _post(url, {"fields": fields, "limit": page, "offset": offset})
        if total is None:
            total = data.get("total")
        results = data.get("results", [])
        if not results:
            break
        for rec in results:
            if rec.get("id") in seen_ids:
                continue
            seen_ids.add(rec.get("id"))
            yield rec
        offset += len(results)
        if (total is not None and offset >= total) or len(results) < page:
            break
        time.sleep(0.05)
    if total is not None and len(seen_ids) < total:
        print(f"      [warn] {project}: fetched {len(seen_ids)} unique of {total} reported")


def img_classes(rec):
    return set((rec.get("annotations") or {}).get("classes", {}).keys())


def cosine(a, b):
    s = na = nb = 0.0
    for x, y in zip(a, b):
        s += x * y; na += x * x; nb += y * y
    return s / ((na ** 0.5) * (nb ** 0.5) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="atli_source_dataset")
    ap.add_argument("--target", default="merged_atli_target")
    ap.add_argument("--embeddings", action="store_true",
                    help="also run CLIP-embedding near-dup pass (slower, pulls 768-d vectors)")
    ap.add_argument("--cos", type=float, default=0.97, help="cosine threshold for near-dup")
    ap.add_argument("--csv")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY missing from .env")
    fields = ["id", "name", "filename", "split", "annotations"]
    if args.embeddings:
        fields.append("features")

    print(f"workspace = {WS}")
    print("[1/4] indexing TARGET (all images) ...")
    tgt = list(search_all(args.target, fields))
    t_by_id = {r["id"]: r for r in tgt}
    t_by_name = defaultdict(list)
    for r in tgt:
        t_by_name[(r.get("filename") or r.get("name") or "").lower()].append(r)
    print(f"      target images: {len(tgt)}  (splits: "
          f"{dict((s, sum(1 for r in tgt if r.get('split')==s)) for s in ('train','valid','test'))})")

    print("[2/4] indexing SOURCE (all images), keeping only defective ...")
    src_all = list(search_all(args.source, fields))
    print(f"      source images pulled: {len(src_all)}")
    src_def = {c: [] for c in DEFECT_CLASSES}
    for r in src_all:
        cl = img_classes(r)
        for c in DEFECT_CLASSES:
            if c in cl:
                src_def[c].append(r)

    print("[3/4] matching by id + filename ...")
    rows = []
    summary = {}
    for c in DEFECT_CLASSES:
        recs = src_def[c]
        id_hits, name_hits, dup_ids = [], [], set()
        for r in recs:
            if r["id"] in t_by_id:
                id_hits.append((r, t_by_id[r["id"]])); dup_ids.add(r["id"]); continue
            key = (r.get("filename") or r.get("name") or "").lower()
            if key and key in t_by_name:
                name_hits.append((r, t_by_name[key][0])); dup_ids.add(r["id"])
        dup = len(dup_ids)
        split_ct = defaultdict(int)
        for _, t in id_hits + name_hits:
            split_ct[t.get("split", "?")] += 1
        summary[c] = dict(total=len(recs), dup=dup, by_id=len(id_hits),
                          by_name=len(name_hits), splits=dict(split_ct))
        for s, t in id_hits:
            rows.append([c, "same_id", s.get("name"), s.get("split"), t.get("name"), t.get("split")])
        for s, t in name_hits:
            rows.append([c, "same_filename", s.get("name"), s.get("split"), t.get("name"), t.get("split")])

    if args.embeddings:
        print(f"[3b] CLIP near-dup pass (cos >= {args.cos}) ...")
        tfeat = [(r, r.get("features")) for r in tgt if isinstance(r.get("features"), list)]
        for c in DEFECT_CLASSES:
            near = 0
            already = {r["id"] for r in src_def[c]
                       if r["id"] in t_by_id or
                       (r.get("filename") or r.get("name") or "").lower() in t_by_name}
            for r in src_def[c]:
                if r["id"] in already:
                    continue
                f = r.get("features")
                if not isinstance(f, list):
                    continue
                best = max((cosine(f, tf) for _, tf in tfeat), default=0)
                if best >= args.cos:
                    near += 1
                    rows.append([c, f"near_cos{best:.3f}", r.get("name"), r.get("split"), "", ""])
            summary[c]["near_dup"] = near

    print("\n========== OVERLAP SUMMARY ==========")
    for c in DEFECT_CLASSES:
        s = summary[c]
        extra = f", near-dup(cos>={args.cos}): {s.get('near_dup')}" if args.embeddings else ""
        safe = s["total"] - s["dup"] - (s.get("near_dup", 0) if args.embeddings else 0)
        print(f"{c}:")
        print(f"  source images with this class : {s['total']}")
        print(f"  already in target (same id)   : {s['by_id']}")
        print(f"  already in target (filename)  : {s['by_name']}{extra}")
        print(f"  -> overlap total              : {s['dup'] + (s.get('near_dup',0) if args.embeddings else 0)}")
        print(f"  -> NOT in target (safe to use): {safe}")
        if s["splits"]:
            print(f"  collisions land in target splits: {s['splits']}  (val/test = leakage risk)")
        print()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "match", "source_name", "source_split", "target_name", "target_split"])
            w.writerows(rows)
        print(f"wrote {len(rows)} matched rows -> {args.csv}")


if __name__ == "__main__":
    main()
