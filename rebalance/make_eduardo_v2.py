#!/usr/bin/env python3
"""Generate a new version of eduardos-annotated-photos named 'eduardo-v2'.

Run AFTER rebalance/phase2_reimport_di.py has restored the Defective_Insulators
boxes. Generates with NO augmentation and auto-orient only (a faithful snapshot,
no reshuffle/inflation). Names the version via the /generate endpoint's `name` field.

Reversible: delete the generated version in the Roboflow UI.

Run:  .venv/bin/python rebalance/make_eduardo_v2.py            # dry-run (prints settings)
      .venv/bin/python rebalance/make_eduardo_v2.py --yes      # generate
"""
import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv("/Users/eddie/Research/Vegas/.env")
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "tl-target-set-focus").strip()
API = "https://api.roboflow.com"
PROJECT = "eduardos-annotated-photos"
VERSION_NAME = "eduardo-v2"

SETTINGS = {
    "name": VERSION_NAME,
    "preprocessing": {"auto-orient": True},   # no resize/static-crop -> native frames kept
    "augmentation": {},                        # NONE
}


def list_versions():
    r = requests.get(f"{API}/{WS}/{PROJECT}", params={"api_key": KEY}, timeout=60)
    r.raise_for_status()
    return r.json().get("versions", []) or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually generate (default: dry-run)")
    args = ap.parse_args()
    if not KEY:
        sys.exit("ROBOFLOW_API_KEY missing")

    before = list_versions()
    print(f"{PROJECT}: {len(before)} existing version(s): "
          + ", ".join(f"v{(v.get('id') or '').split('/')[-1]}='{v.get('name','')}'" for v in before))
    print(f"\nwill generate version named '{VERSION_NAME}' with settings:")
    print(f"   preprocessing = {SETTINGS['preprocessing']}")
    print(f"   augmentation  = {SETTINGS['augmentation']} (none)")

    if not args.yes:
        print("\nDRY-RUN. Re-run with --yes to generate.")
        return

    r = requests.post(f"{API}/{WS}/{PROJECT}/generate", params={"api_key": KEY},
                      json=SETTINGS, timeout=120)
    try:
        j = r.json()
    except Exception:
        sys.exit(f"non-JSON response ({r.status_code}): {r.text[:200]}")
    if r.status_code != 200:
        sys.exit(f"generate failed ({r.status_code}): {j}")
    ver = j.get("version")
    print(f"\nrequested generation: {j.get('message','')}  -> version {ver}")

    # poll until the new version is listed + report its name
    for _ in range(30):
        vs = list_versions()
        match = [v for v in vs if str((v.get('id') or '').split('/')[-1]) == str(ver)]
        if match:
            v = match[0]
            print(f"version v{ver} ready: name='{v.get('name','')}' images={v.get('images','?')}")
            if v.get("name") != VERSION_NAME:
                print(f"  NOTE: name came back as '{v.get('name')}', not '{VERSION_NAME}'. "
                      "Rename in the Roboflow UI if needed.")
            break
        time.sleep(4)
    else:
        print(f"version {ver} requested but not yet listed; check the Roboflow UI shortly.")


if __name__ == "__main__":
    main()
