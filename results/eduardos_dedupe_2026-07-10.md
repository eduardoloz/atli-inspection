# eduardos-annotated-photos intra-project dedupe (2026-07-10)

pHash (d=0) duplicate scan + deletion inside Roboflow project
`tl-target-set-focus/eduardos-annotated-photos`. All mutations verified by per-image
GET (deleted → 404, keepers → present), not project metadata.

## Result

- **264 images scanned** → 45 exact-duplicate groups (pHash distance 0), 98 images involved.
- **35 groups auto-resolved, 53 duplicate images deleted** → project now **211 images**.
- Keeper rule per group: most polygon annotations, then most boxes (most complete
  labeling), then smallest total box area (tightest). Project-wide only 34/264 images
  had any polygon boxes; polygon copies were kept wherever present.
- **10 groups (26 images) NOT deleted — flagged for manual review**: their members are
  pixel-identical but carry *different defect labels* (one copy labels the
  Defective_Damper, the other a Broken_Insulator on the same photo). Deleting either
  copy would erase a real defect annotation; the right fix is merging the defect boxes
  into one copy in the Roboflow UI, then deleting the other. Review list with direct
  URLs: `dedupe_manifest.csv` (rows marked REVIEW) in the backup dir below.
- 117 near-duplicate pairs (0 < pHash d ≤ 6) were detected but deliberately left
  untouched (many are legitimately distinct shots of the same tower); list in
  `phash_groups.json`.

## Rollback / audit trail

`datasets/eduardos_dedupe_backup_2026-07-10/` (local, gitignored):
`images.jsonl` (full pre-deletion annotations for all 264 images),
`eduardos_images/` (all 264 original image files), `dedupe_manifest.csv`
(KEEP/DELETE/REVIEW per image), `phash_groups.json`. A deleted image can be restored
by re-uploading its file and re-pushing its annotation from `images.jsonl`.

## Notes

- Class totals before: Broken_Insulator 124, Defective_Damper 141, Normal_Damper 725,
  Normal_Insulators 894 (project metadata; recount after Roboflow's cache catches up).
- This project is still **excluded from all trained models** (see `models/README.md`);
  the dedupe is prep for the planned leakage-gated train-only merge experiment.
- Same API gotchas as `results/cplid_purge_roboflow.md` apply (batch DELETE endpoint,
  flaky search pagination — the scan swept until 3 consecutive stable rounds).
