#!/usr/bin/env python3
"""Read-only explorer for the Roboflow workspace.

Makes ONLY HTTP GET requests against the Roboflow Platform API
(https://api.roboflow.com). It never uploads, generates versions,
trains, or modifies any data. Safe to run repeatedly.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError

try:
    from dotenv import load_dotenv
except Exception:
    # Minimal fallback loader if python-dotenv is not installed or unresolved by linter.
    def load_dotenv(path='.env'):
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except FileNotFoundError:
            return

API = "https://api.roboflow.com"
load_dotenv()
KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WS = os.environ.get("ROBOFLOW_WORKSPACE", "").strip()

if not KEY:
    sys.exit("ROBOFLOW_API_KEY missing from .env")


def get(path):
    """GET {API}{path}?api_key=... and return parsed JSON (read-only)."""
    sep = "&" if "?" in path else "?"
    url = f"{API}{path}{sep}api_key={urllib.parse.quote(KEY)}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    # 1) GET / confirms auth and reports the key's bound workspace slug (a string).
    #    Then GET /{slug} returns the workspace object with the project list.
    try:
        root = get("/")
        ws_slug = root.get("workspace") or WS
        ws_detail = get(f"/{ws_slug}")
    except (HTTPError, URLError) as e:
        sys.exit(f"Auth/connection failed: {e}")

    ws_obj = ws_detail.get("workspace", {})
    ws_slug = ws_obj.get("url") or ws_slug
    print("=" * 70)
    print(f"WORKSPACE : {ws_obj.get('name', '?')}   (slug: {ws_slug})")
    if WS and ws_slug and WS != ws_slug:
        print(f"  NOTE: .env ROBOFLOW_WORKSPACE='{WS}' but key resolves to '{ws_slug}'")
    print("=" * 70)

    projects = ws_obj.get("projects", [])
    print(f"\nPROJECTS: {len(projects)}\n")
    if not projects:
        print("  (no projects in this workspace)")
        return

    for p in projects:
        pid = (p.get("id") or "").split("/")[-1]
        print("-" * 70)
        print(f"• {p.get('name','?')}   [id: {pid}]")
        print(f"    type     : {p.get('type','?')}")
        print(f"    images   : {p.get('images','?')}")
        print(f"    versions : {p.get('versions','?')}")
        classes = p.get("classes")
        if isinstance(classes, dict):
            cls_str = ", ".join(f"{k}:{v}" for k, v in classes.items())
            print(f"    classes  : {cls_str}")
        elif classes:
            print(f"    classes  : {classes}")

        # 2) Per-project detail -> versions, splits, trained models
        try:
            detail = get(f"/{ws_slug}/{pid}")
        except (HTTPError, URLError) as e:
            print(f"    (could not fetch detail: {e})")
            continue

        proj = detail.get("project", {})
        splits = proj.get("splits")
        if splits:
            print(f"    splits   : {splits}")

        versions = detail.get("versions", []) or []
        if versions:
            print(f"    --- versions ({len(versions)}) ---")
            for v in versions:
                vid = (v.get("id") or "").split("/")[-1]
                line = f"      v{vid}: '{v.get('name','')}'  images={v.get('images','?')}"
                model = v.get("model") or {}
                if model:
                    m = model.get("map") or model.get("mAP")
                    line += f"  model[map={m}, prec={model.get('precision')}, rec={model.get('recall')}]"
                print(line)
        else:
            print("    (no dataset versions yet)")


if __name__ == "__main__":
    main()
