# Universe Vetting — Defective-Insulator & Birdnest Datasets (2026-06-15)

Read-only pHash vetting of Roboflow Universe candidates vs ATLI **source** (`atli_source_dataset` v4)
and **target** (`merged_atli_target` v4, val/test = leakage-critical). **Nothing merged.**
Scripts: `rebalance/vet_universe_class.py` (sweep) + `rebalance/verify_classvet.py` (strict-threshold
double-check + side-by-side pair render). Artifacts: `results/classvet_review/` (report, montages, dup-pairs).

**Double-check method:** every flagged overlap recomputed at exact (d=0) / ≤2 / ≤4 / ≤8 pHash distance,
and the closest pairs rendered side-by-side and eyeballed. The flagged leaks are **exact-0 matches**
(pixel-identical), confirmed visually — not look-alike false positives.

## Defective insulators

| Dataset | imgs | classes | **val/test leak (exact)** | src overlap | intra-dups | verdict |
|---|---|---|---|---|---|---|
| [giomartins/danified-insulators](https://universe.roboflow.com/giomartins/danified-insulators-wyxhs) | 614 | descarga/faltante/quebrado/intacto (PT: flashover/missing/broken/intact) | **92 (15%!)** | 105 | 0 | ❌ heavy ATLI leak |
| [trash-detection/insulator-defect-gcnfq](https://universe.roboflow.com/trash-detection-zqfed/insulator-defect-gcnfq) | 1578 | broken/insulator/pollution-flashover | **74** | 45 | 0 | ❌ leak; ⊂ uttam |
| [uttam/insulator-defect](https://universe.roboflow.com/uttam/insulator-defect) | 4107 | broken/insulator/pollution-flashover | **121** | 75 | 1191 | ❌ leak + 29% self-dups |
| [hello-92xp7/insulator-defect-detection](https://universe.roboflow.com/hello-92xp7/insulator-defect-detection-yywkb) | 5493 | Flash Over / Insulator / Damaged / Good | **0** | 0 | 72 | ✅ clean |
| [insulator-adspq/broken-insulator](https://universe.roboflow.com/insulator-adspq/broken-insulator) | 1028 | broken / dirty / good | **0** (1 train) | 0 | 11 | ✅ clean |

**The "broken/pollution-flashover" trio is one recycled CPLID dataset** (cross-overlap: trashdet ∩ uttam =
1578 = *all* of trashdet; giomartins shares 388–394 with each). Since ATLI seeded ~200 CPLID images, these
re-uploads duplicate ATLI's val/test → **using any of them as-is would invalidate evaluation** (the DVDI trap).
`hello` and `adspq` are clean of ATLI overlap and are the only safe insulator-harvest candidates.

## Birdnest

| Dataset | imgs | classes | **val/test leak (exact)** | src overlap | content (montage) | verdict |
|---|---|---|---|---|---|---|
| [nestdet/nest-9imzt](https://universe.roboflow.com/nestdet/nest-9imzt) | 1583 | nest | **18** | 74 | power-line nests | ⚠️ relevant but leaks ATLI |
| [cugb/nest-ezl6r](https://universe.roboflow.com/cugb/nest-ezl6r) | 5406 | 0/1/2 | 0 | 3 | **natural** nests in trees/reeds | ➖ clean but off-domain |
| [gg-97eg1/nest-hgbmq](https://universe.roboflow.com/gg-97eg1/nest-hgbmq) | 6682 | 0 | **0** | 0 | **power-line infra** against sky | ✅ clean + on-domain |

`nest_gg` is the standout: large, zero ATLI overlap, and the montage is unambiguously transmission-line
imagery. `nestdet` overlaps ATLI's birdnest images (so it *is* power-line, but partly recycled from our own
source/target → must be deduped first). `cugb` is general ornithology imagery (birds + tree nests), off-domain.

## Bottom line (no merges — per instruction)

- **Insulator defects:** the obvious-looking candidates (uttam/trashdet/giomartins) are CPLID recyclings that
  leak into ATLI val/test — reject as-is. **`hello` (5,493; FlashOver/Damaged/Good) and `adspq` (1,028;
  broken/dirty/good) are clean** and worth a closer look as defective-insulator sources, *if* the harvest is
  later routed through the same pHash gate (clean-of-ATLI ≠ proven-not-CPLID; verify taxonomy + style first).
- **Birdnest:** **`gg-97eg1/nest-hgbmq` is the one clean, on-domain candidate** — confirm it actually labels
  nests (class "0") before considering it. `nestdet` is usable only after deduping the 18+ ATLI-overlapping
  images; `cugb` is off-domain.
- Any future use must keep ATLI val/test clean: dedupe candidates vs target val/test, add to **train only**,
  tagged/reversible (the `phase1_select.py` / staging-project workflow).
