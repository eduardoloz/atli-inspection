#!/usr/bin/env python3
"""Sync paper/draft_sections_2_3 with its Overleaf project (free-plan workaround).

Overleaf's git/GitHub sync is a paid feature, so this drives a browser instead:
  push    upload main.tex, *.bib, and figures/* into the existing project
          (same-named files are replaced; project URL never changes)
  pull    download the project zip and report differences vs local
          (--apply overwrites local files with Overleaf's copies)
  status  same as pull but never writes anything

GitHub is the source of truth: normally edit locally -> commit -> `push`.
Run `pull` first only when a collaborator edited directly on Overleaf.

First run: `python3 scripts/sync_overleaf.py push --headed` and log in to
Overleaf (Google) in the window that opens; the session persists in
~/.overleaf_sync_profile for later headless runs.
"""
import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ID = "6aa0740339f566201feeae3e"
PROJECT_URL = f"https://www.overleaf.com/project/{PROJECT_ID}"
REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper" / "draft_sections_2_3"
PROFILE = Path.home() / ".overleaf_sync_profile"

ROOT_FILES = sorted([PAPER / "main.tex", *PAPER.glob("*.bib")])
FIGURE_FILES = sorted(p for p in (PAPER / "figures").iterdir() if p.is_file())


def launch(pw, headed):
    return pw.chromium.launch_persistent_context(
        str(PROFILE), headless=not headed, viewport={"width": 1440, "height": 900}
    )


def open_editor(page, headed):
    page.goto(PROJECT_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)

    def editor_loaded():
        _, up = find_in_frames(page, lambda f: f.get_by_role("button", name="Upload"))
        return up is not None

    if not editor_loaded():
        if not headed:
            sys.exit("Editor did not load (not logged in, or Cloudflare challenge). "
                     "Rerun with --headed and log in once.")
        print("Log in to Overleaf in the browser window (waiting up to 3 min)...")
        deadline = time.time() + 180
        while time.time() < deadline and not editor_loaded():
            page.wait_for_timeout(3000)
        if not editor_loaded():
            sys.exit("Editor never loaded — gave up.")


def find_in_frames(page, getter):
    for frame in page.frames:
        loc = getter(frame)
        try:
            if loc.count():
                return frame, loc.first
        except Exception:
            pass
    return None, None


def upload_batch(page, paths, into_figures=False):
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    if into_figures:
        _, folder = find_in_frames(page, lambda f: f.get_by_role("treeitem", name="figures"))
        if folder is None:
            sys.exit("figures folder not found in file tree")
        folder.click()
        page.wait_for_timeout(400)
    _, up = find_in_frames(page, lambda f: f.get_by_role("button", name="Upload"))
    if up is None:
        sys.exit("Upload button not found (editor not loaded?)")
    up.click()
    page.wait_for_timeout(1000)
    _, inp = find_in_frames(page, lambda f: f.locator("input[type=file]"))
    if inp is None:
        sys.exit("upload file input not found")
    inp.set_input_files([str(p) for p in paths])
    page.wait_for_timeout(2000 + 1500 * len(paths))
    # accept an overwrite confirmation if one appears
    _, ow = find_in_frames(page, lambda f: f.get_by_role("button", name="Overwrite"))
    if ow is not None:
        ow.click()
        page.wait_for_timeout(2000)
    page.keyboard.press("Escape")


def push(page):
    upload_batch(page, ROOT_FILES)
    print(f"uploaded {len(ROOT_FILES)} root files")
    upload_batch(page, FIGURE_FILES, into_figures=True)
    print(f"uploaded {len(FIGURE_FILES)} figures")
    _, rec = find_in_frames(page, lambda f: f.get_by_role("button", name="Recompile", exact=True))
    if rec is not None:
        rec.click()
        page.wait_for_timeout(20000)
        body = page.frames[-1].locator("body").inner_text()
        bad = "This project has errors" in body or "Compile Error" in body
        print("recompile:", "ERRORS — check Overleaf logs" if bad else "clean")


def fetch_zip(page):
    resp = page.request.get(f"{PROJECT_URL}/download/zip")
    if not resp.ok:
        sys.exit(f"zip download failed: HTTP {resp.status} "
                 "(if 403: rerun with --headed, or log in once via `push --headed`)")
    return zipfile.ZipFile(io.BytesIO(resp.body()))


def pull(page, apply_changes):
    zf = fetch_zip(page)
    differs, missing_local = [], []
    for info in zf.infolist():
        if info.is_dir():
            continue
        local = PAPER / info.filename
        remote = zf.read(info)
        if not local.exists():
            missing_local.append(info.filename)
        elif local.read_bytes() != remote:
            differs.append(info.filename)
        if apply_changes:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(remote)
    if not differs and not missing_local:
        print("in sync: Overleaf matches local")
        return
    for name in differs:
        print(f"DIFFERS: {name}")
    for name in missing_local:
        print(f"ONLY ON OVERLEAF: {name}")
    if apply_changes:
        print(f"applied {len(differs) + len(missing_local)} files locally "
              "— review with `git diff` before committing")
    else:
        print("run `pull --apply` to overwrite local with Overleaf's copies")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["push", "pull", "status"])
    ap.add_argument("--apply", action="store_true", help="pull: overwrite local files")
    ap.add_argument("--headed", action="store_true", help="show the browser (needed for first login)")
    args = ap.parse_args()

    with sync_playwright() as pw:
        ctx = launch(pw, args.headed)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        open_editor(page, args.headed)
        if args.command == "push":
            push(page)
        else:
            pull(page, apply_changes=(args.command == "pull" and args.apply))
        ctx.close()


if __name__ == "__main__":
    main()
