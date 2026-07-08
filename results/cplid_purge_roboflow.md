# CPLID purge + tightened-NI push to Roboflow (2026-07-07)

Follow-up to `results/atli_minus_cplid_build.md`. All mutations verified by GET-404 /
annotation spot-checks. Scripts: `rebalance/delete_cplid_from_source.py`,
`rebalance/overwrite_ni_atli_target.py` (both dry-run by default, pHash-verify every
image before acting).

## What changed on Roboflow (workspace `tl-target-set-focus`)

| project | action | result |
|---|---|---|
| `atli_source_dataset` | deleted the CPLID copies | 596 deleted (586 train / 10 valid per project metadata), 0 failures. 599 CPLID dups found; 3 unresolved rows + 23 same-name lookalikes skipped (see manifest). |
| `atli_target-train` | NI-tightened labels + CPLID removal | 561 overwrites, 171 deleted → **561 imgs** |
| `atli_target-val` | same | 116 overwrites, 41 deleted → **116 imgs** |
| `atli_target-test` | same (class names translated to this project's literal "0"–"6" classes) | 120 overwrites, 37 deleted → **120 imgs** |
| `atli_target-minus-the-cplid` | same (second pass, same day; script `rebalance/overwrite_ni_minus_cplid.py`) | 796 overwrites, 241 deleted (189 train / 22 valid / 30 test) → **796 imgs** (~576/113/107, merged-split layout). Its earlier curation had removed only ~9 CPLID imgs. Had **zero versions** → pre-change annotations backed up to `~/atli/backup_minus_cplid_annotations.jsonl`. |

- "Overwrite" = replace the image's full annotation with its `merged_atli_target` v5
  label (the hand-tightened Normal_Insulators boxes; other classes identical in v5).
  Every image joined to v5 at pHash d=0 before writing.
- `merged_atli_target` itself was NOT touched.

## Manifests / audit trail (server `~/atli/`)
`cplid_delete_manifest.csv` (source), `ni_overwrite_manifest.csv` (target),
`cplid_vs_source_matches.json`, `cplid_vs_target_matches.json`, `target_join_mapping.json`,
logs `cplid_delete_*.log`, `ni_overwrite_apply*.log`.

## Post-change snapshots (2026-07-07, generated after polygon restore; auto-orient only, no aug)
`atli_target-minus-the-cplid` **v1** (796; 576/113/107) — its first version ever ·
`atli_target-train` **v6** (561) · `atli_target-val` **v4** (116) · `atli_target-test` **v4** (120).
These freeze the clean state: CPLID-free + tightened NI + restored polygons.

## Rollback
- Frozen Roboflow versions predate all changes (source v4; target train v5 / val v3 / test v3).
- Full local exports: `~/atli/datasets/classvet/source` (source),
  `~/atli/downloads/atli_target_{train_v5,val_v3,test_v3}` (target, pre-overwrite labels).
- CPLID originals: `~/atli/downloads/cplid_github`.

## ⚠ Eval comparability
The `atli_target-test` project lost 37 CPLID images — numbers computed on the new
test set are NOT comparable to the APET paper or to the 154-run history (all used the
CPLID-contaminated split). The local dataset `~/atli/ATLI_target_tightNI_noCPLID/`
mirrors the new state (797 imgs, original paper split) for training/eval.

## API gotchas (hard-won, keep for future scripts)
- `DELETE /:ws/:project/images/:id` does **not** exist — it returns **HTTP 200** with
  `{"error":"Endpoint not found."}`. Status-code checks alone will report success while
  deleting nothing. Correct: `DELETE /:ws/:project/images` with JSON `{"images":[ids]}`
  → HTTP **204** (SDK `project.delete_images`).
- The search endpoint (`POST /:ws/:project/search`) caps at offset 10,000 (HTTP 500
  past it), ignores `class`/`split` filters, and paginates flakily (short/empty pages at
  random) — sweep repeatedly and merge until the expected count is reached.
- Project metadata (`images`, `splits`) is cached and can stay stale long after
  mutations; verify with per-image GETs, not the project header.
- Annotation overwrite: `POST /dataset/:project/annotate/:imageId?name=<f>.xml&overwrite=true`
  with VOC XML body works and merges by class name (digit-named classes need translation).
- **VOC XML flattens polygons to rectangles.** The first overwrite pass lost the hand-drawn
  NI polygon vertices (box geometry was preserved). Fixed by re-pushing COCO JSON (same
  endpoint, `name=<f>.json`) with `segmentation` arrays built from the v5 *segmentation*
  export — polygon lines there have ≥8 coords, plain boxes 4. 2026-07-07 second pass:
  152 polygon-bearing images restored & verified across all four projects
  (`rebalance/restore_ni_polygons.py`; 303 pushes, 0 failures).
