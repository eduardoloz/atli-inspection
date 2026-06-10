#!/usr/bin/env python3
"""READ-ONLY: figure out which workspace the local ROBOFLOW_API_KEY belongs to and
whether the source/target projects are reachable via the SDK. Downloads nothing."""
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
ENV_WS = os.environ.get("ROBOFLOW_WORKSPACE", "").strip()
assert KEY, "ROBOFLOW_API_KEY missing"

rf = Roboflow(api_key=KEY)
home = rf.workspace()  # the key's default workspace
home_url = getattr(home, "url", None) or getattr(home, "name", None)
print(f".env ROBOFLOW_WORKSPACE = {ENV_WS!r}")
print(f"key's HOME workspace    = {home_url!r}")
print("projects in home workspace:")
try:
    for p in home.project_list:
        print("   ", p.get("id") or p.get("name"))
except Exception as e:
    print("   (could not list:", e, ")")

for proj_slug in ("atli_source_dataset", "merged_atli_target", "eduardos-annotated-photos"):
    for ws in {ENV_WS, "eduardo-lozano-osrja"}:
        try:
            pr = rf.workspace(ws).project(proj_slug)
            vers = [v.version for v in pr.versions()]
            print(f"OK  {ws}/{proj_slug}  versions={vers}")
            break
        except Exception as e:
            print(f"--  {ws}/{proj_slug}: {type(e).__name__}: {str(e)[:80]}")
