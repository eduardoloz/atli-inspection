---
name: sync-overleaf
description: Sync paper/draft_sections_2_3 with its Overleaf project (free plan — no git bridge). Use when the user says "update overleaf", "push to overleaf", "pull from overleaf", or after committing paper changes.
---

# Sync the paper with Overleaf

Overleaf project: `atli_sections_2_3_overleaf` — https://www.overleaf.com/project/6aa0740339f566201feeae3e
Eddie is on the **free plan**: no git/GitHub sync, so syncing is done by browser
automation. **GitHub is the source of truth** — the normal flow is edit locally
→ build (`latexmk -pdf` with TinyTeX on PATH: `~/Library/TinyTeX/bin/universal-darwin`)
→ commit → push to Overleaf.

## Primary method: the sync script

```bash
python3 scripts/sync_overleaf.py push        # local → Overleaf (replaces same-named files)
python3 scripts/sync_overleaf.py status      # diff Overleaf vs local, read-only
python3 scripts/sync_overleaf.py pull --apply  # Overleaf → local (collaborator edits)
```

- First ever run (or after cookies expire / HTTP 403): `push --headed`, and the
  user logs in to Overleaf (Google) in the window once. Session persists in
  `~/.overleaf_sync_profile`.
- Before `push`: make sure the local build is clean (`latexmk -pdf main.tex` in
  `paper/draft_sections_2_3/`, zero undefined refs in `main.log`).
- If a collaborator (Soum, Giovanny, the PI) may have edited on Overleaf
  directly, run `status` FIRST; if it shows diffs, `pull --apply`, review with
  `git diff`, commit their changes, then `push`. Never push over unreviewed
  Overleaf edits.

## Fallback: drive the Playwright MCP browser directly

If the script's profile is broken, use the mcp__playwright__ tools (that browser
is logged in): navigate to the project URL, then use browser_run_code_unsafe to
click the file-tree Upload button, find `input[type=file]` across frames
(the editor lives in a frame — `text=` selectors on the page will NOT match),
setInputFiles, and Escape. For figures, click the "figures" treeitem first so
uploads land in the folder. Verify by clicking Recompile and checking for
"This project has errors". Working example: session 2026-09-11 (memory:
atli-summer-paper).

## After syncing

Report the project URL and whether Overleaf recompiled clean. If files were
pulled, remind the user to commit (dated message, GIT.md conventions: noreply
email, no advisor names).
