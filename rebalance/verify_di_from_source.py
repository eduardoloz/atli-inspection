#!/usr/bin/env python3
"""READ-ONLY verification: do the eduardos-annotated-photos images still carry
`Defective_Insulators` labels in the SOURCE dataset (ATLI_source_dataset)?

eduardos-annotated-photos was staged FROM ATLI_source_dataset; each staged image
was named `{source_image_id}.jpg`, so the local label filename stem IS the source id.
We re-query the live source for every staged image and report which ones still have
Defective_Insulators boxes (i.e. are re-importable). Downloads/changes nothing.

Run: .venv/bin/python rebalance/verify_di_from_source.py
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv("/Users/eddie/Research/Vegas/.env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
API = "https://api.roboflow.com"
SOURCE = "atli_source_dataset"
LABELS = Path("/Users/eddie/Research/Vegas/datasets/eduardos_staging/labels")
TARGET_CLASS = "Defective_Insulators"

if not KEY:
    sys.exit("ROBOFLOW_API_KEY missing")


def local_classes(xml_path):
    """Set of class names present in a staged local label (what we uploaded)."""
    try:
        root = ET.parse(xml_path).getroot()
        return [o.findtext("name") for o in root.findall("object")]
    except Exception:
        return []


def source_boxes(src_id):
    """Live source boxes for an image id -> (ok, list_of_labels) or (False, None)."""
    try:
        r = requests.get(f"{API}/{WS}/{SOURCE}/images/{src_id}",
                         params={"api_key": KEY}, timeout=60)
        if r.status_code != 200:
            return src_id, None, r.status_code
        boxes = (r.json().get("image", {}).get("annotation", {}) or {}).get("boxes", [])
        return src_id, [b.get("label") for b in boxes], 200
    except Exception as e:
        return src_id, None, str(e)[:40]


def main():
    xmls = sorted(LABELS.glob("*.xml"))
    print(f"staged images (local labels): {len(xmls)}")
    # source id = label filename stem
    rows = {x.stem: local_classes(x) for x in xmls}
    local_di = {sid for sid, cls in rows.items() if TARGET_CLASS in cls}
    print(f"locally have {TARGET_CLASS}: {len(local_di)} images\n")
    print(f"querying live source ({SOURCE}) for all {len(rows)} images ...")

    results = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(source_boxes, sid) for sid in rows]
        for f in as_completed(futs):
            sid, labels, code = f.result()
            results[sid] = (labels, code)

    missing_in_source = [s for s, (lab, c) in results.items() if lab is None]
    src_has_di = {s for s, (lab, c) in results.items() if lab and TARGET_CLASS in lab}
    src_di_boxes = sum(lab.count(TARGET_CLASS) for s, (lab, c) in results.items() if lab)

    print("\n================ RESULT ================")
    print(f"images not found / errored in source : {len(missing_in_source)}")
    if missing_in_source:
        print("   ", missing_in_source[:10], "..." if len(missing_in_source) > 10 else "")
    print(f"source images WITH {TARGET_CLASS}    : {len(src_has_di)}")
    print(f"source {TARGET_CLASS} boxes (total)  : {src_di_boxes}")
    print(f"\ncross-check vs local labels:")
    print(f"  local-says-DI but source-NO-DI : {len(local_di - src_has_di)}  {sorted(local_di - src_has_di)[:8]}")
    print(f"  source-says-DI but local-NO-DI : {len(src_has_di - local_di)}  {sorted(src_has_di - local_di)[:8]}")
    print(f"  agree (re-importable)          : {len(local_di & src_has_di)}")

    # write the re-importable id list for a potential phase-2 reattach
    out = LABELS.parent / "reimport_di_ids.txt"
    out.write_text("\n".join(sorted(local_di & src_has_di)) + "\n")
    print(f"\nwrote {len(local_di & src_has_di)} re-importable source ids -> {out}")


if __name__ == "__main__":
    main()
