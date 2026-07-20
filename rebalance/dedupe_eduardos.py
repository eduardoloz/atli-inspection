#!/usr/bin/env python3
"""Remove the 72 redundant near-duplicate images from `eduardos-annotated-photos`.

Input: rebalance/eduardo_dedupe_plan.json — 39 visually-confirmed duplicate
groups (audit 2026-07-19, contact sheets in results/eduardo_dupes_review/).
Keeper policy: the polygon-annotated "orig" copy first, tie-broken by smallest
mean normalized bbox area. Edit the plan (swap keeper/delete ids, or drop a
group) before running if you disagree with any choice — see plan `_readme`.

Label-conflict merge-gate: 21/39 groups continue the 2026-07-10 REVIEW backlog
(results/eduardos_dedupe_2026-07-10.md) — duplicate copies are separate
annotation passes each labeling a DIFFERENT real defect on the same photo, so
deleting a copy outright erases a defect annotation. Those groups' deletions
stay gated until the keeper's live annotation carries every defect class seen
in the group (merge the boxes onto the keeper in the Roboflow UI, re-run) or
the group is overridden with "accept_keeper_as_is": true in the plan. The
18 clean groups (23 images) delete on the first --yes run.

Safety / reversibility (CPLID-purge pattern):
  * DRY-RUN by default — verifies + backs up + writes the manifest, deletes
    nothing. Pass --yes to actually delete.
  * Aborts unless backup version v5 exists on Roboflow (frozen 201-img
    snapshot, generated 2026-07-19) and the live project still has exactly
    the expected 201 images.
  * Every to-delete image is re-downloaded from source.roboflow.com and
    pHash-verified against the audit hash (d<=2); mismatch => skipped, never
    deleted. The verified bytes + full annotation record are saved to
    rebalance/backups/eduardo_dedupe/ (gitignored) before deletion, and all
    201 annotation records go to annotations_all.jsonl there.
  * API gotcha (results/cplid_purge_roboflow.md): per-image DELETE does not
    exist (returns HTTP 200 + "Endpoint not found"). Correct endpoint is
    DELETE /:ws/:project/images with {"images":[ids]} -> HTTP 204.

Run from the repo root:  python3 rebalance/dedupe_eduardos.py [--yes]
Idempotent: already-deleted images are reported as "gone" and skipped.
"""
import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = Path(__file__).resolve().parent / "eduardo_dedupe_plan.json"
BACKUP_DIR = Path(__file__).resolve().parent / "backups" / "eduardo_dedupe"
MANIFEST_CSV = BACKUP_DIR / "dedupe_manifest.csv"
API = "https://api.roboflow.com"
VERIFY_THRESHOLD = 2   # phash distance hosted-original vs audit (0 expected)
DELETE_CHUNK = 25


def load_env(path):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


# ---- pHash: identical implementation to the 2026-07-19 audit ----
_DCT = None


def dct_matrix(N):
    global _DCT
    if _DCT is None or _DCT.shape[0] != N:
        n = np.arange(N)
        _DCT = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1)[:, None] * n[None, :] / (2 * N))
        _DCT[:, 0] = np.sqrt(1.0 / N)
    return _DCT


def phash_hex(img):
    g32 = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    D = dct_matrix(32)
    low = (D.T @ g32 @ D)[:8, :8]
    vals = low.flatten()
    med = np.median(vals[1:])
    bits = vals > med
    bits[0] = False
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return format(v, "016x")


def ham_hex(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit("ROBOFLOW_API_KEY missing from .env")

    plan = json.load(open(PLAN_PATH))
    ws, project = plan["workspace"], plan["project"]
    base = f"{API}/{ws}/{project}"
    s = requests.Session()

    def get(url, **kw):
        return s.get(url, params={"api_key": key}, timeout=60, **kw)

    # ---- gate 1: backup version exists and holds the full pre-dedupe set ----
    bv = plan["backup_version"]
    r = get(f"{base}/{bv}")
    if r.status_code != 200:
        sys.exit(f"ABORT: backup version v{bv} not readable (HTTP {r.status_code}). "
                 "Do not delete without the frozen snapshot.")
    vinfo = r.json().get("version", {})
    v_imgs = vinfo.get("images")
    if v_imgs != plan["expected_project_images_before"]:
        sys.exit(f"ABORT: backup version v{bv} reports {v_imgs} images, expected "
                 f"{plan['expected_project_images_before']}.")
    print(f"[gate] backup version v{bv} OK ({v_imgs} imgs, name={vinfo.get('name')!r})")

    # ---- gate 2: live project state matches the audit (or a completed re-run) ----
    r = get(f"{API}/{ws}/{project.split('/')[-1]}")
    proj_imgs = (r.json().get("project") or {}).get("images") if r.status_code == 200 else None
    expected_before = plan["expected_project_images_before"]
    expected_after = plan["expected_project_images_after"]
    if proj_imgs == expected_after:
        print(f"[gate] project already at {proj_imgs} images — dedupe appears done.")
    elif proj_imgs == expected_before:
        print(f"[gate] project image count OK ({proj_imgs})")
    elif proj_imgs is not None and expected_after < proj_imgs < expected_before:
        print(f"[gate] project at {proj_imgs} images — resuming a partial run "
              f"({expected_before} -> {expected_after}); already-gone images are skipped.")
    else:
        sys.exit(f"ABORT: project has {proj_imgs} images; audit expected {expected_before} "
                 f"(or {expected_after} if already deduped). State drifted — re-audit first.")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    (BACKUP_DIR / "images").mkdir(exist_ok=True)

    # ---- verify + back up every image in the plan (keepers included) ----
    manifest = []
    verified_deletes = []
    ann_out = open(BACKUP_DIR / "annotations_all.jsonl", "w")
    n_gone = 0
    merge_pending = []
    for g in plan["groups"]:
        # fetch live records for the whole group first (and back them all up)
        live = {}
        for m in g["members"]:
            gr = get(f"{base}/images/{m['id']}")
            rec = (gr.json() if gr.status_code == 200 else {}).get("image") or {}
            live[m["id"]] = rec
            if rec:
                ann_out.write(json.dumps({"group": g["group"],
                                          "action": "keep" if m["id"] == g["keeper"] else "delete",
                                          "record": rec}) + "\n")
        keeper_rec = live.get(g["keeper"])

        # merge-gate for label-conflict groups (2026-07-10 REVIEW pattern):
        # only delete once the keeper's LIVE annotation covers every defect class
        # seen in the group, or the plan explicitly overrides.
        group_active = True
        gate_note = ""
        if not keeper_rec:
            group_active = False
            gate_note = "SKIP GROUP: keeper missing from project — never delete all copies"
        elif "label_conflict" in g.get("flags", []) and not g.get("accept_keeper_as_is"):
            keeper_labels = {b.get("label") for b in
                             ((keeper_rec.get("annotation") or {}).get("boxes") or [])}
            still_missing = [d for d in g.get("defects_missing_from_keeper", [])
                             if d not in keeper_labels]
            if still_missing:
                group_active = False
                gate_note = (f"MERGE-PENDING: keeper lacks {still_missing} — add the "
                             f"box(es) to the keeper in the Roboflow UI, then re-run "
                             f"(or set accept_keeper_as_is in the plan)")
                merge_pending.append((g["group"], g["keeper"], still_missing))

        for m in g["members"]:
            iid = m["id"]
            action = "keep" if iid == g["keeper"] else "delete"
            row = {"group": g["group"], "image_id": iid, "action": action,
                   "phash_dist": "", "status": "", "url": m["url"]}
            if not live[iid]:
                row["status"] = "gone(not in project)"
                n_gone += 1
                manifest.append(row)
                continue
            if not group_active:
                row["status"] = gate_note if action == "delete" else "keeper (group gated)"
                manifest.append(row)
                continue

            try:
                data = s.get(m["url"], timeout=60).content
                d = ham_hex(phash_hex(Image.open(io.BytesIO(data))), m["phash"])
                row["phash_dist"] = d
            except Exception as e:
                row["status"] = f"verify_error({e})"
                manifest.append(row)
                continue

            if d > VERIFY_THRESHOLD:
                row["status"] = "SKIP: phash mismatch — not the audited image"
                manifest.append(row)
                continue

            if action == "delete":
                with open(BACKUP_DIR / "images" / f"{iid}.jpg", "wb") as f:
                    f.write(data)
                verified_deletes.append(iid)
                row["status"] = "verified, backed up"
            else:
                row["status"] = "keeper, verified"
            manifest.append(row)
        time.sleep(0.05)
    ann_out.close()

    print(f"\nverified {len(verified_deletes)}/{plan['n_delete']} deletions "
          f"({n_gone} already gone); backups in {BACKUP_DIR}")
    if merge_pending:
        print(f"\n{len(merge_pending)} label-conflict groups gated (deletions deferred "
              f"until their keeper carries the missing defect boxes):")
        for gnum, keeper, missing in merge_pending:
            print(f"  group {gnum}: keeper {keeper} needs {', '.join(missing)}")
        print(f"  -> merge in the UI: https://app.roboflow.com/{ws}/{project}/annotate")

    # ---- delete (only with --yes) ----
    n_deleted = 0
    if not args.yes:
        print(f"\nDRY-RUN — nothing deleted. Review {MANIFEST_CSV} and the contact "
              f"sheets, then re-run with --yes to delete {len(verified_deletes)} images.")
    elif not verified_deletes:
        print("\nNothing verified to delete.")
    else:
        for i in range(0, len(verified_deletes), DELETE_CHUNK):
            chunk = verified_deletes[i:i + DELETE_CHUNK]
            dr = s.delete(f"{base}/images", params={"api_key": key},
                          json={"images": chunk}, timeout=120)
            # 204 is the ONLY success signal (200 = wrong endpoint, deleted nothing)
            if dr.status_code == 204:
                n_deleted += len(chunk)
                print(f"  deleted {n_deleted}/{len(verified_deletes)}")
            else:
                print(f"  FAILED chunk at offset {i}: HTTP {dr.status_code} "
                      f"{dr.text[:200]}")
                for iid in chunk:
                    for row in manifest:
                        if row["image_id"] == iid:
                            row["status"] = f"DELETE FAILED (HTTP {dr.status_code})"
        for row in manifest:
            if row["status"] == "verified, backed up":
                row["status"] = "DELETED"
        r = get(f"{API}/{ws}/{project.split('/')[-1]}")
        after = (r.json().get("project") or {}).get("images")
        print(f"\ndeleted {n_deleted} images; project now reports {after} "
              f"(expected {expected_after})")

    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"manifest: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
