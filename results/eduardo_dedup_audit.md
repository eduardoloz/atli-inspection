# eduardos-annotated-photos — duplicate & annotation audit

> **Dedupe plan (2026-07-19)** — ready to execute, nothing deleted yet.
>
> - **Backup snapshot: Roboflow version v5** (201 imgs, auto-orient only, no aug), generated
>   2026-07-19 before any changes. Frozen and downloadable regardless of later deletions
>   (same pattern as the CPLID purge v4 snapshot). Rename it in the UI to
>   `pre-dedupe-backup-2026-07-19` (API default name is a timestamp).
> - **Plan:** `rebalance/eduardo_dedupe_plan.json` — 39 groups, keeper = polygon copy first
>   (tie-break: tightest mean bbox area), 72 deletions total, 201 → 129 imgs. Editable
>   before running (swap keeper/delete ids or drop a group).
> - **⚠ Label-conflict merge-gate — 21/39 groups (49 of the 72 deletions).** These continue
>   the 2026-07-10 REVIEW backlog (`results/eduardos_dedupe_2026-07-10.md`): duplicate
>   copies are separate annotation passes that each label a *different real defect* on the
>   same photo (17 keepers lack the dup's `Defective_Damper`, 4 lack `Broken_Insulator`).
>   Deleting a copy outright would erase that defect box, so the script defers these
>   deletions until the keeper's **live** annotation carries the missing class — merge the
>   defect box onto the keeper in the UI, re-run — or the group is overridden with
>   `"accept_keeper_as_is": true` in the plan (e.g. if the defect is genuinely
>   out-of-frame in the keeper's framing).
> - **Executor:** `python3 rebalance/dedupe_eduardos.py` (dry-run) then `--yes` to delete.
>   Gates: v5 must exist with 201 imgs; live count must match (partial runs resume).
>   Every to-delete image is re-downloaded and pHash-verified (d≤2) before deletion;
>   verified bytes + all annotation records back up to `rebalance/backups/eduardo_dedupe/`
>   (gitignored). Deletes via batch `DELETE /:ws/:project/images` (204), chunks of 25.
>   Never deletes a group whose keeper is missing. Idempotent.
> - **EXECUTED 2026-07-20:** ran `--yes`; **23 clean copies deleted** (all batches HTTP 204),
>   verified gone (spot-checked deleted IDs → 404, keepers present). **Project 201 → 178.**
>   Backups + manifest in `rebalance/backups/eduardo_dedupe/`. The **49 label-conflict
>   deletions remain deferred** pending per-keeper UI merges (then re-run `--yes`; idempotent).
>   _(Roboflow's cached image count lagged at 201 right after the deletes — the search API
>   confirmed 178.)_
> - **Dry-run result (2026-07-19):** gates green; **all 72 dup copies re-verified at pHash
>   distance 0**; with the merge-gate, **23 deletions (18 clean groups) ready now**, 49
>   deferred pending UI merges.
> - **Review sheets:** `results/eduardo_dupes_review/group_01..39.png` (gitignored;
>   re-derivable from the v5 export).

Project `tl-target-set-focus/eduardos-annotated-photos` — 201 images (all in `train`, all tagged `eduardos_stage_v1`). Filenames are Roboflow hash-IDs (no original names preserved), so duplicates were found by perceptual hashing of the pixels (pHash + dHash), clustered by union-find, and **every group was visually confirmed**. "Tight polygon" status comes from the live per-image annotation geometry (`GET /{ws}/{project}/images/{id}` → boxes with a `points` array).

## Summary

| metric | count |
|---|---|
| total images | 201 |
| **duplicate groups** (near-identical drone frames) | **39** |
| images that are part of a dup group | 111 |
| **redundant copies** (extra beyond 1 keeper/group) | **72** |
| unique images (no duplicate) | 90 |
| **polygon-annotated "orig" images** (tight segmentation) | **54** |
| box-only images (looser / no polygon) | 147 |
| polygon origs that sit inside a dup group | 30 |
| polygon origs that are unique | 24 |

Net: if you kept one best copy per group you would drop **72 redundant images** (201 → 129).

## List 1 — Duplicate groups

Members sorted best-first: **polygon copy first (the tight "orig" keeper), then by tightest mean bbox area**. `poly` = # polygon-segmented objects, `bx` = # boxes, `area` = mean normalized bbox area (smaller = tighter). ✅ = suggested keeper.

**Group 1** — 8 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `tR3pJTnPLZEOqi0f3zLj` | 5 | 6 | 0.0028 | Broken_Insulator:1, Normal_Damper:3, Normal_Insulators:2 |
|  | `T0VnV1gF2M4T9OMFSW3F` | 0 | 6 | 0.0047 | Broken_Insulator:1, Normal_Insulators:2, Normal_Damper:3 |
|  | `yBIhY0qTYFPoedch7POC` | 0 | 6 | 0.0047 | Broken_Insulator:1, Normal_Insulators:2, Normal_Damper:3 |
|  | `LzmzfOkFb4H3IcAemfUz` | 0 | 6 | 0.0050 | Broken_Insulator:1, Normal_Damper:3, Normal_Insulators:2 |
|  | `uYx5nqKdk7n3SelPB3cG` | 0 | 7 | 0.0051 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:3 |
|  | `x6e7zOHf75QAyUpgQ6N8` | 0 | 6 | 0.0056 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:2 |
|  | `u8FlLw7jaoEgHiBAszGC` | 0 | 6 | 0.0058 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:2 |
|  | `eDgM8xg6400SrolXFqlU` | 0 | 6 | 0.0071 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:2 |

**Group 2** — 6 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `8qb7BV8q2lLwEK0umzf7` | 4 | 7 | 0.0077 | Broken_Insulator:1, Normal_Insulators:2, Normal_Damper:4 |
|  | `9h5O9p3C8mOw2uMzZ2RP` | 0 | 6 | 0.0061 | Defective_Damper:1, Normal_Insulators:2, Normal_Damper:3 |
|  | `UZ1DgQFN5QrylX7Hypd9` | 0 | 8 | 0.0068 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:4 |
|  | `xkLHIRYcELHTIUEJSGj1` | 0 | 6 | 0.0071 | Defective_Damper:1, Normal_Insulators:2, Normal_Damper:3 |
|  | `DN94ryXq5oVQpIopuYhM` | 0 | 8 | 0.0073 | Defective_Damper:1, Normal_Damper:4, Normal_Insulators:3 |
|  | `TJB8gulZBwN7wzGjaDir` | 0 | 7 | 0.0075 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:3 |

**Group 3** — 5 images  _(⚠ 2 tight copies)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `MDCoNflKrpm6ZSA3XxsM` | 7 | 9 | 0.0036 | Broken_Insulator:1, Normal_Damper:2, Normal_Insulators:6 |
|  | `nxlxdlntplp4e0dvuFJR` | 5 | 10 | 0.0050 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:5 |
|  | `6hj75E3E0S6lTyvHUkzK` | 0 | 10 | 0.0076 | Defective_Damper:1, Normal_Insulators:6, Normal_Damper:3 |
|  | `ks4IadecsLX4HZSn4mTw` | 0 | 10 | 0.0076 | Defective_Damper:1, Normal_Insulators:6, Normal_Damper:3 |
|  | `CjgEPldpk6BmjPQ6xgg6` | 0 | 10 | 0.0076 | Defective_Damper:1, Normal_Insulators:6, Normal_Damper:3 |

**Group 4** — 5 images  _(⚠ 2 tight copies)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `9h9Jmc3U1I2GwkOsDWY7` | 7 | 8 | 0.0024 | Normal_Insulators:3, Normal_Damper:4, Defective_Damper:1 |
|  | `E7vK1tMoCHk0w4UMXdIK` | 6 | 8 | 0.0023 | Normal_Insulators:3, Defective_Damper:1, Normal_Damper:4 |
|  | `JQ6egph0XqgTiGpVTZUh` | 0 | 7 | 0.0025 | Broken_Insulator:1, Normal_Damper:4, Normal_Insulators:2 |
|  | `QdJ7LdOw0D6wmFGuM9gQ` | 0 | 7 | 0.0040 | Defective_Damper:1, Normal_Damper:3, Normal_Insulators:3 |
|  | `xxVmTrFhxzP0yFpQPvAV` | 0 | 7 | 0.0054 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:3 |

**Group 5** — 4 images  _(⚠ 2 tight copies)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `XpL8G7yrqDD4aUjRYHz1` | 3 | 8 | 0.0057 | Normal_Insulators:4, Broken_Insulator:1, Normal_Damper:3 |
|  | `tUtZg1AwJjXWDSDWKZln` | 3 | 9 | 0.0062 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:3 |
|  | `LCg21RHkWaoSp5XFu0fn` | 0 | 10 | 0.0059 | Broken_Insulator:1, Normal_Insulators:5, Defective_Damper:1, Normal_Damper:3 |
|  | `oC4mgSYSJaOcgHlRzfux` | 0 | 9 | 0.0070 | Normal_Insulators:5, Normal_Damper:3, Broken_Insulator:1 |

**Group 6** — 4 images  _(⚠ 2 tight copies)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `V5SejL9WTuLd9T5dV52V` | 8 | 9 | 0.0038 | Broken_Insulator:1, Normal_Damper:3, Normal_Insulators:5 |
|  | `zorqyuCS1kb53IQNy4SZ` | 8 | 9 | 0.0042 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:3 |
|  | `jZ9Nc9hfIVpk4An7PoLS` | 0 | 12 | 0.0058 | Defective_Damper:1, Normal_Insulators:6, Normal_Damper:5 |
|  | `Sqg2ucCf6d5Wp5AFXvHv` | 0 | 7 | 0.0083 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:2 |

**Group 7** — 4 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `87zZTs30jiFCVRMfThbw` | 0 | 9 | 0.0079 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:4 |
|  | `5n30s9tt5KDp2dD4Ie9R` | 0 | 7 | 0.0079 | Defective_Damper:1, Normal_Damper:3, Normal_Insulators:3 |
|  | `9OStwvcpNmcXvSujVHMD` | 0 | 7 | 0.0083 | Defective_Damper:1, Normal_Insulators:2, Normal_Damper:4 |
|  | `Ry3mkwxDihCHTkPl2j75` | 0 | 10 | 0.0086 | Defective_Damper:1, Normal_Damper:3, Normal_Insulators:6 |

**Group 8** — 4 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `jh0PyeaUsdQ2tZSCsG89` | 0 | 9 | 0.0062 | Normal_Damper:3, Normal_Insulators:5, Defective_Damper:1 |
|  | `LuqRI1XWwQOFKm8xvClY` | 0 | 9 | 0.0062 | Defective_Damper:1, Normal_Damper:3, Normal_Insulators:5 |
|  | `MlVGtqbdbfTMenkA6HtA` | 0 | 9 | 0.0062 | Defective_Damper:1, Normal_Damper:3, Normal_Insulators:5 |
|  | `fEmRRbJCyXKCbaznop5s` | 0 | 9 | 0.0064 | Broken_Insulator:1, Normal_Damper:4, Normal_Insulators:4 |

**Group 9** — 4 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `lD6tbFgtGQepPK9dXzCV` | 3 | 8 | 0.0041 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:3 |
|  | `E0jGKn27n2TdbBtcqTX8` | 0 | 8 | 0.0042 | Broken_Insulator:1, Normal_Insulators:3, Normal_Damper:4 |
|  | `b1TOlEG21lgLokOcI32F` | 0 | 8 | 0.0066 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:3 |
|  | `K9033sE4CfZGSVN6yRyQ` | 0 | 8 | 0.0092 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:2 |

**Group 10** — 3 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `yEcW2uQPORF4KlUMxKsR` | 0 | 11 | 0.0034 | Broken_Insulator:1, Normal_Damper:5, Normal_Insulators:5 |
|  | `NbE0mDjQ4gP9hyktqDDK` | 0 | 10 | 0.0038 | Broken_Insulator:1, Normal_Damper:4, Normal_Insulators:5 |
|  | `Xp6fuaXvM3S2oAG9WbMq` | 0 | 9 | 0.0053 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:3 |

**Group 11** — 3 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `CTJow6F2wauWRC5VKTxd` | 6 | 7 | 0.0042 | Broken_Insulator:1, Normal_Insulators:3, Normal_Damper:3 |
|  | `MbgPqG8XIh3v9G6Yehiu` | 0 | 7 | 0.0098 | Defective_Damper:1, Normal_Damper:2, Normal_Insulators:4 |
|  | `XRw0bcKmsdsPP3dq3JR6` | 0 | 7 | 0.0099 | Defective_Damper:1, Normal_Damper:2, Normal_Insulators:4 |

**Group 12** — 3 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `B53HprrW09uSpfl29q4w` | 9 | 11 | 0.0029 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:6 |
|  | `1jpmsMC6ByKj1M2u5cgT` | 0 | 11 | 0.0052 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:6 |
|  | `ph0cRdzJDhoih884hmf4` | 0 | 10 | 0.0081 | Defective_Damper:1, Normal_Damper:5, Normal_Insulators:4 |

**Group 13** — 3 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `fj73EkvQ3Rc5TIDaknFA` | 1 | 6 | 0.0091 | Normal_Insulators:5, Defective_Damper:1 |
|  | `hNC3MpHiaMBirl8gTTK2` | 0 | 7 | 0.0072 | Defective_Damper:1, Normal_Insulators:5, Normal_Damper:1 |
|  | `soYKwJZfv0TO51oy1cUJ` | 0 | 9 | 0.0090 | Normal_Damper:3, Normal_Insulators:5, Broken_Insulator:1 |

**Group 14** — 3 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `MFQ1GRHXlCzbYKcWfeUl` | 3 | 5 | 0.0048 | Normal_Insulators:2, Defective_Damper:1, Normal_Damper:2 |
|  | `EKhjU4T224L8sE12o85p` | 0 | 5 | 0.0061 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:1 |
|  | `Ike92KDilO6xVSEg53XT` | 0 | 4 | 0.0110 | Defective_Damper:1, Normal_Insulators:2, Normal_Damper:1 |

**Group 15** — 3 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `cHtc1LgXqjxO5VxDG0xr` | 4 | 7 | 0.0045 | Normal_Insulators:3, Normal_Damper:3, Defective_Damper:1 |
|  | `GPqM85qAABPozf8RmoTy` | 0 | 8 | 0.0045 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:4 |
|  | `OOtLeGKj1wCOxIoWz7ra` | 0 | 9 | 0.0046 | Broken_Insulator:1, Normal_Insulators:2, Normal_Damper:6 |

**Group 16** — 3 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `drjKSUy37TJTxTjv9T7r` | 0 | 6 | 0.0056 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:2 |
|  | `Rjm092E6RMhgkjgNAn4R` | 0 | 6 | 0.0056 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:2 |
|  | `lKUMAw6LsXl1QSAIfFex` | 0 | 6 | 0.0056 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:2 |

**Group 17** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `lCpNZGNEJIGPzZpYliX3` | 0 | 13 | 0.0047 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:7 |
|  | `WwNNbFAJycgnVKIGJV6T` | 0 | 13 | 0.0049 | Defective_Damper:1, Normal_Damper:6, Normal_Insulators:6 |

**Group 18** — 2 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `lwwEQxN5Raep8k4Z828Z` | 7 | 10 | 0.0059 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:4 |
|  | `jKLt2kPewGzOjJ5n5kIw` | 0 | 10 | 0.0061 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:4 |

**Group 19** — 2 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `wfULgoBtCU5lr6m2zeaw` | 3 | 10 | 0.0065 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:4 |
|  | `EUiphmeofuD1NND6VFUz` | 0 | 9 | 0.0057 | Normal_Damper:3, Broken_Insulator:1, Normal_Insulators:5 |

**Group 20** — 2 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `jftYgjOtiAR2KVAvYi3i` | 3 | 8 | 0.0068 | Broken_Insulator:1, Normal_Insulators:3, Normal_Damper:4 |
|  | `Ll4sPMUpN9n46KRfp95g` | 0 | 8 | 0.0105 | Defective_Damper:1, Normal_Insulators:4, Normal_Damper:3 |

**Group 21** — 2 images  _(⚠ 2 tight copies)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `RdShrorkOfvFpy9Kz6gU` | 7 | 9 | 0.0036 | Normal_Insulators:5, Broken_Insulator:1, Normal_Damper:3 |
|  | `BBjKvLqD0NAWxhjnPWPP` | 1 | 12 | 0.0038 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:6 |

**Group 22** — 2 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `uK4JuvEfNICnCCwc51I8` | 2 | 8 | 0.0080 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:2 |
|  | `SFug90J0ReddRsONO1At` | 0 | 7 | 0.0070 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:2 |

**Group 23** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `CLmGXWQU4cm6sCpGg25i` | 0 | 1 | 0.0014 | Broken_Insulator:1 |
|  | `iRk7ReEMLq3G7NMpKXVC` | 0 | 5 | 0.0033 | Broken_Insulator:1, Normal_Insulators:2, Normal_Damper:2 |

**Group 24** — 2 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `QF9jNqeEfsynxQakWIG8` | 8 | 8 | 0.0036 | Normal_Damper:1, Normal_Insulators:6, Defective_Damper:1 |
|  | `BE031xyZjp7KPtCKrRS7` | 0 | 8 | 0.0071 | Normal_Insulators:5, Broken_Insulator:1, Normal_Damper:2 |

**Group 25** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `qLhNkuEY23pJXSxBe7dY` | 0 | 11 | 0.0028 | Normal_Damper:5, Broken_Insulator:1, Normal_Insulators:5 |
|  | `lrUcH1aEnIlIQrkZ8XUh` | 0 | 8 | 0.0052 | Normal_Damper:2, Broken_Insulator:1, Normal_Insulators:5 |

**Group 26** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `jW6VkKyHWAcIebZft5wM` | 0 | 7 | 0.0070 | Defective_Damper:1, Normal_Damper:4, Normal_Insulators:2 |
|  | `wwrFZwnH185nqfbxZHN2` | 0 | 7 | 0.0070 | Defective_Damper:1, Normal_Damper:4, Normal_Insulators:2 |

**Group 27** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `pniPvESepsbCqa9hCYtF` | 0 | 13 | 0.0029 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:7 |
|  | `HVdBZFbIhHip6cHDBIe7` | 0 | 9 | 0.0036 | Normal_Insulators:5, Normal_Damper:3, Broken_Insulator:1 |

**Group 28** — 2 images  _(⚠ 2 tight copies)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `bfkbCXgL0bAlrhK8BBYp` | 5 | 11 | 0.0050 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:6 |
|  | `apbS1SWZ4k7ojVyXpYWP` | 5 | 7 | 0.0060 | Broken_Insulator:1, Normal_Insulators:3, Normal_Damper:3 |

**Group 29** — 2 images  _(⚠ 2 tight copies)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `8DZSbr3yidooctvywnor` | 7 | 7 | 0.0054 | Normal_Insulators:5, Normal_Damper:2 |
|  | `7W4UGMyoXOSQQFq58Uh7` | 1 | 7 | 0.0081 | Defective_Damper:1, Normal_Insulators:5, Normal_Damper:1 |

**Group 30** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `HfgxFp94Dc3KJd7nht9o` | 0 | 8 | 0.0035 | Broken_Insulator:1, Normal_Insulators:2, Normal_Damper:5 |
|  | `Nw1HGc3nTUhHEB4zAzWp` | 0 | 8 | 0.0047 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:4 |

**Group 31** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `AEW5ROQ6zSzs5cfQU2xU` | 0 | 6 | 0.0057 | Normal_Damper:3, Normal_Insulators:2, Broken_Insulator:1 |
|  | `RrnQgAfLBPiDo781v6iw` | 0 | 7 | 0.0085 | Defective_Damper:1, Normal_Damper:2, Normal_Insulators:4 |

**Group 32** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `sljZWoTsF2ynONwbyIsd` | 0 | 7 | 0.0088 | Broken_Insulator:1, Normal_Insulators:5, Normal_Damper:1 |
|  | `Iif6weucQbQb6U9MblrO` | 0 | 6 | 0.0111 | Normal_Insulators:5, Broken_Insulator:1 |

**Group 33** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `cCGwRTi8xk2MSwmzqPrh` | 0 | 8 | 0.0049 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:4 |
|  | `pVpDPG648ua7u5NZp6Sl` | 0 | 8 | 0.0049 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:4 |

**Group 34** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `sdFgHG5EzPvedsAdcKPQ` | 0 | 6 | 0.0076 | Defective_Damper:1, Normal_Damper:1, Normal_Insulators:4 |
|  | `RGfFfkICOyMSudtTAqyA` | 0 | 6 | 0.0077 | Defective_Damper:1, Normal_Damper:1, Normal_Insulators:4 |

**Group 35** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `BjKPl2BTWYbqYUHIpLFx` | 0 | 7 | 0.0019 | Broken_Insulator:1, Normal_Damper:5, Normal_Insulators:1 |
|  | `Yx5POCiix79l9747QKht` | 0 | 7 | 0.0027 | Defective_Damper:1, Normal_Damper:4, Normal_Insulators:2 |

**Group 36** — 2 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `mLHsn0pTle1Xyia8Zwa7` | 4 | 6 | 0.0070 | Broken_Insulator:1, Normal_Insulators:4, Normal_Damper:1 |
|  | `sCzC3muTALd05xZusmiH` | 0 | 9 | 0.0093 | Defective_Damper:1, Normal_Damper:2, Normal_Insulators:6 |

**Group 37** — 2 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `aKaam0jqs5Tc1p87fGdn` | 5 | 8 | 0.0021 | Broken_Insulator:1, Normal_Insulators:2, Normal_Damper:5 |
|  | `NVIwNMxHyAFVhaGR17t4` | 0 | 9 | 0.0054 | Defective_Damper:1, Normal_Insulators:3, Normal_Damper:5 |

**Group 38** — 2 images

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `hHb6yQiEOQKt8FtYeGkL` | 3 | 6 | 0.0047 | Normal_Insulators:3, Normal_Damper:2, Defective_Damper:1 |
|  | `KbMHHrD26MQk0VqjIPsR` | 0 | 4 | 0.0114 | Defective_Damper:1, Normal_Insulators:2, Normal_Damper:1 |

**Group 39** — 2 images  _(no polygon copy — all box-only)_

| keep | image id | poly | bx | area | classes |
|---|---|---|---|---|---|
| ✅ | `5dQQ2Fb86r8g9BUNdBmp` | 0 | 2 | 0.0021 | Broken_Insulator:1, Normal_Insulators:1 |
|  | `8sqQahpGgORptJACcEJI` | 0 | 5 | 0.0042 | Broken_Insulator:1, Normal_Insulators:3, Normal_Damper:1 |

## List 2 — "Orig" photos (tight polygon segmentation)

The 54 images whose annotations were hand-drawn with the polygon tool (tight segmentation / tight boxes). These are the properly-annotated originals; box-only near-duplicates of them are the loose copies.

### 2a. Unique polygon origs — no duplicate (24)

`1gDeriwBe9EWlpTRcEWm`, `2u4YRbhbPdIUbIBpge36`, `4n0FEHruMZJbyFL64dwJ`, `DDWijklYu3pwpafrD85L`, `DI0KWVtNOEezBC0j45WC`, `DaPPOoXEffKAsk6n3hA1`, `E4cPB7UJz0oEIthVOpko`, `EIQbd9a5LuxK4SBZNI20`, `FoGMLTHTkcfDBAPtUa5z`, `HOli2DaQwICxd2S7Ong6`, `IZTCZ6pIoolLV92npdv1`, `K7OEvq8ZkIDwPcOEPDbZ`, `LKV8BOBbQgUwtsbqTZgH`, `PxqUtKuinsx4hOqTPHm0`, `TFjq60qFN9VIWFCtTIh8`, `Uac5BmtlWZ7N2XTk1ppj`, `VAC7PHbRGNhJ1HEvmtEs`, `VVuUk530AtK5klkMvBX7`, `c8mGKpscMbjEU0vplwRL`, `dgvLA6XQrpWTizet87xq`, `ehmYSlnsqEVKtF6aPX68`, `mT9vFn90JLBMOVx34NCN`, `pPcMYIGVTJSD0LjDoREv`, `xqIqjSoW8Lg4LxZxkc5v`

### 2b. All polygon origs (54)

`1gDeriwBe9EWlpTRcEWm`, `2u4YRbhbPdIUbIBpge36`, `4n0FEHruMZJbyFL64dwJ`, `7W4UGMyoXOSQQFq58Uh7`, `8DZSbr3yidooctvywnor`, `8qb7BV8q2lLwEK0umzf7`, `9h9Jmc3U1I2GwkOsDWY7`, `B53HprrW09uSpfl29q4w`, `BBjKvLqD0NAWxhjnPWPP`, `CTJow6F2wauWRC5VKTxd`, `DDWijklYu3pwpafrD85L`, `DI0KWVtNOEezBC0j45WC`, `DaPPOoXEffKAsk6n3hA1`, `E4cPB7UJz0oEIthVOpko`, `E7vK1tMoCHk0w4UMXdIK`, `EIQbd9a5LuxK4SBZNI20`, `FoGMLTHTkcfDBAPtUa5z`, `HOli2DaQwICxd2S7Ong6`, `IZTCZ6pIoolLV92npdv1`, `K7OEvq8ZkIDwPcOEPDbZ`, `LKV8BOBbQgUwtsbqTZgH`, `MDCoNflKrpm6ZSA3XxsM`, `MFQ1GRHXlCzbYKcWfeUl`, `PxqUtKuinsx4hOqTPHm0`, `QF9jNqeEfsynxQakWIG8`, `RdShrorkOfvFpy9Kz6gU`, `TFjq60qFN9VIWFCtTIh8`, `Uac5BmtlWZ7N2XTk1ppj`, `V5SejL9WTuLd9T5dV52V`, `VAC7PHbRGNhJ1HEvmtEs`, `VVuUk530AtK5klkMvBX7`, `XpL8G7yrqDD4aUjRYHz1`, `aKaam0jqs5Tc1p87fGdn`, `apbS1SWZ4k7ojVyXpYWP`, `bfkbCXgL0bAlrhK8BBYp`, `c8mGKpscMbjEU0vplwRL`, `cHtc1LgXqjxO5VxDG0xr`, `dgvLA6XQrpWTizet87xq`, `ehmYSlnsqEVKtF6aPX68`, `fj73EkvQ3Rc5TIDaknFA`, `hHb6yQiEOQKt8FtYeGkL`, `jftYgjOtiAR2KVAvYi3i`, `lD6tbFgtGQepPK9dXzCV`, `lwwEQxN5Raep8k4Z828Z`, `mLHsn0pTle1Xyia8Zwa7`, `mT9vFn90JLBMOVx34NCN`, `nxlxdlntplp4e0dvuFJR`, `pPcMYIGVTJSD0LjDoREv`, `tR3pJTnPLZEOqi0f3zLj`, `tUtZg1AwJjXWDSDWKZln`, `uK4JuvEfNICnCCwc51I8`, `wfULgoBtCU5lr6m2zeaw`, `xqIqjSoW8Lg4LxZxkc5v`, `zorqyuCS1kb53IQNy4SZ`
