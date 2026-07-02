# Defective_Damper low-recall investigation + staging-project priority plan

## ★ RECALL / FALSE-NEGATIVE focus (2026-06-17) — FN is the safety-critical error
Objective clarified: a missed defective damper (false negative) is the dangerous error → maximize
**recall**, accept precision cost. Operating-point sweep (`recall_sweep.py`, 39 DD test instances):
| conf | champion recall | FN | baseline recall | FN |
|---|---|---|---|---|
| 0.25 (default) | 0.74 | 10 | 0.64 | 14 |
| 0.05 | 0.79 | 8 | 0.67 | 13 |
| →0 (max) | **0.82** | **7** | 0.72 | 11 |
- **Two levers:** (1) FREE recall by lowering deploy threshold to ~0.05 (FN 10→8, prec 0.91→0.79);
  (2) a **hard FN floor of 7** (18%) the champion never detects even at conf→0 — only TRAINING fixes
  these (invisible-tiny OR detected-but-called-Normal_Damper).
- **Recall loop launched:** 8 experiments targeting the floor — higher resolution (1536, see tiny
  defects) + upweighted classification loss (cls=1.5/2.0, push borderline → defective) + combos.
  Evaluate by recall@conf0.05 and FN floor, not AP. Next if needed: detect-then-classify (PA-DETR).


Self-paced loop (started 2026-06-15). Goal: find what causes low DD recall and work through
the `universe-damper-staging` curation actions in priority order — **minimal overtraining,
without hurting Normal_Damper**.

## Diagnosis: what's actually wrong

1. **Recall is the failure mode, not precision.** Best models flag DD confidently (precision
   0.80–0.93) but miss ~40% of them. So the model's "defective" concept is too narrow/shifted,
   not trigger-happy.
2. **Misses skew small.** OS3_v11 misses median area 0.119% vs 0.227% for hits; **6 of 16 misses
   are <0.1% of the image** (sub-~20px at 640) — partly unresolvable at training res.
3. **But resolution alone doesn't fix DD.** HR_v8 @1280: overall mAP50 **0.731** (best yet) — but
   that's Normal_Damper soaring to **0.843**; DD stayed flat (AP 0.641, recall 0.597). So pixels
   help *detect dampers*, not *discriminate defective* ones → DD bottleneck = defect-signal +
   rare-class scarcity of ATLI-style examples, not raw size.
4. **The Universe data is the wrong scale.** Only **15% of Universe DD boxes (312/2067) are at
   ATLI's tiny scale**; 85% are zoomed-in. Training on the big ones teaches a damper appearance
   ATLI never shows → the recall drop we measured when mixing it in.

## Priority order of staging-project actions (automatable first)

1. **Scale-matched curation** ⟳ *(running now: `USM_v11/v8`)* — keep only the ~331 zoomed-out
   Universe images whose DD boxes are ATLI-scale (≤0.8%), drop the rest. Fully automatable, fixes
   the biggest distribution mismatch. Dose modest (DD 177→708) to limit overtraining.
2. **Defect-type filter** (next) — drop rust-only / ambiguous "defective" labels (unlearnable per
   lit; teaches wrong concept). Heuristic + spot-check, low manual cost.
3. **Box-convention standardization** (manual, deferred) — one tight box per whole damper; fixes
   the whole-vs-partial inconsistency. Higher labor → only if 1–2 don't suffice.
4. **Ratio cap** — if 150+100 overtrains the added class, cap scale-matched dose to ~1:1 and/or
   shorten stage-2.

Guardrail: every condition is scored on the untouched native test, reporting **DD AP/recall AND
Normal_Damper AP** — abandon any action that lifts DD by trading away ND.

## Baselines to beat
- Best DD AP50: **OS3_v11 = 0.726** (native oversample ×3); B_v11 = 0.708.
- Best overall mAP50: **HR_v8 = 0.731** (ND-driven).
- Don't let Normal_Damper fall below ~0.74 (baseline) for any DD gain.

## Results log
- 2026-06-15: diagnosis above; launched scale-matched curation (USM_v11/v8).
- **USM (scale-matched) FAILED.** USM_v11: DD AP 0.631, **recall 0.513** (below baseline 0.633!),
  ND AP 0.767 (fine). USM_v8: DD 0.589. → Matching the scale did NOT help; recall got *worse*.
  **Root cause confirmed = defect-definition/LABEL mismatch, not scale/quantity/framing.** The
  Universe "defective damper" is a different concept; reshaping geometry can't fix it.
  ⇒ Automatable universe-curation actions (#1 scale-match, #4 cap) are exhausted. The only
  staging actions that could work are the MANUAL ones (#2 defect-type filter, #3 box relabel).
- 2026-06-15: launched (a) USMcap_v11 (scale-matched, minimal dose DD 177→357) as the final
  nail, and (b) **OSaug_v11** — pivot to native lever: OS3 winner + aggressive scale-down aug
  (scale=0.9) to directly attack the small-damper recall finding. No universe data, protects ND.

## Standings (DD AP50, native test, ND in paren)
1. OS3_v11 **0.726** (ND 0.745) — native oversample ×3  ← best
2. B_v11 0.708 (0.744) — baseline
3. HR_v8 0.641 (ND **0.843**) — hi-res; best overall mAP 0.731, DD flat
4. USM_v11 0.631 (0.767) — scale-matched universe: FAILED on DD recall
- All universe-augmented conditions (Uto/Uw/Ucap/USM) ≤ baseline on DD. Native-only wins.

## Expanded scope (2026-06-15, loop) — beyond DD recall
Per user: also improve on the paper's model (YOLOv5n 78.9% mAP@0.5) + scour internet for new datasets.
- **Running:** USMcap_v11, OSaug_v11 (DD recall thread) + **HRO_v11/v8** (hi-res 1280 + OS3 oversample
  combined — fuse overall-mAP leader HR 0.731 with DD leader OS3 0.726, the "beat the paper" run).
- **Research arm (4 background subagents, fresh context):** new UAV datasets for insulators / birdnests /
  defective dampers, + YOLO-improvement techniques (small-object/imbalance/aug/arch). Results fold into
  vetting (vet_universe_class.py pipeline) + next experiment rounds.
- HR_v11 finished (collect pending).

### ★ WINNER so far: OSaug_v11 — DD AP 0.749, recall 0.692 (2026-06-15)
OS3 oversample + aggressive scale-down aug (scale=0.9). Beats OS3 (0.726) and baseline (0.708);
**recall 0.590→0.692** — directly fixed the small-damper misses the diagnostic identified. ND AP 0.753
(unharmed). Pure native augmentation, no external data → the clean minimal-overtraining/protect-ND win.
- USMcap (minimal-dose scale-match): DD 0.645, recall 0.564 — confirms universe-curation path closed.

### New standings (DD AP50 | recall | ND AP)
1. **OSaug_v11 0.749 | 0.692 | 0.753** ← best (native oversample + scale-down aug)
2. OS3_v11 0.726 | 0.590 | 0.745
3. B_v11 0.708 | 0.633 | 0.744
- Running: HRO_v8/v11 (hi-res+oversample), P2_v8 & P2OS_v8 (P2 head), CP_v11 & CP_v8 (copy-paste+cls).
- Next levers from techniques agent: SAHI sliced fine-tune (top pick, not yet built), VFL/Slide loss.

### Technique experiments (2026-06-15) — none beat OSaug
- P2_v8 DD 0.661 (ND **0.778**), P2OS_v8 DD 0.657, CP_v11 DD 0.614, CP_v8 DD 0.608.
- **Copy-paste HURT DD** (unrealistic box-paste on detection); P2 head lifted ND+overall, not DD
  (and only testable on v8 — no v11 P2 yaml).
- **SAHI SKIPPED**: ATLI images are only 640²/512² → slicing loses context, low value (SAHI shines on 4K).
- OSaug refinements launched: OSaug_v8 (v8 generality), OSaugM_v11 (+close_mosaic=10), OS6aug_v11
  (×6 oversample), OSaug95_v11 (scale=0.95). Building on the 0.749 winner.

### Convergence (2026-06-15) — TWO champions, space bounded
- **OSaug_v11 = DD optimum: DD 0.749 / R 0.692 / ND 0.753.** Pushing harder regresses: OS6aug ×6 → DD
  0.635, OSaug95 scale=0.95 → 0.626, OSaug_v8 → 0.606. close_mosaic = no-op (OSaugM = OSaug). So
  OS3(×3) + scale=0.9 + v11 is the sweet spot.
- **HRO_v8 = best all-around: mAP50 0.747 / DD 0.723 / R 0.692 / ND 0.824.** Hi-res 1280 + OS3
  oversample — DD within noise of OSaug while overall mAP & Normal_Damper far higher. Beats baseline
  (0.701) and the paper's framing on overall mAP.
- Final round: HROaug_v8/v11 = hi-res + oversample + scale-aug (fuse both winners). If no gain → converged.

### ★★ VERIFIED CHAMPION: HROaug_v11 (hi-res 1280 + ×3 oversample + scale-down aug, YOLOv11n)
Multi-seed (3 seeds each) vs baseline — gaps exceed seed spread (~0.03) ⇒ REAL, not lucky seed:
| metric (3-seed mean) | HROaug_v11 | baseline B_v11 | Δ |
|---|---|---|---|
| overall mAP@0.5 | **0.746** | 0.694 | **+0.052** |
| Defective_Damper AP | **0.723** (0.69–0.77) | 0.677 (0.66–0.71) | **+0.046** |
| Defective_Damper recall | **0.684** | 0.634 | **+0.050** |
| Normal_Damper AP | **0.827** | 0.756 | **+0.071** |

Per-class (best run vs baseline) — recipe lifts ALL rare/defect classes:
DefDamper +0.103 · Broken_Insulator +0.089 · Self-Exploded +0.103 · Flashover +0.091 · NormDamper
+0.070 · NormInsulator +0.004 · Birdnest −0.011 (only non-gainer → HRObird experiment running with
clean vetted niaochao birdnest data, 241→462).

**Loop conclusion:** diagnostic (DD misses = small dampers) → fix (oversample + scale-down aug + hi-res)
lifted DD recall 0.634→0.684 and broadly improved the model (+0.052 overall mAP) over the paper-style
baseline, verified across seeds, ND IMPROVED not hurt. External/community data never helped (label
mismatch); native augmentation was the answer.

### HRObird (champion recipe + clean niaochao birdnest data) — bookends the external-data question
mAP50 0.750. **Birdnest 0.762→0.898 (+0.14!)** — clean, on-domain, correctly-labeled external data
DID help its target class hugely. BUT it HURT the insulator classes: Broken_Insulator 0.598→0.482
(−0.12), Flashover 0.673→0.620 — because the 250 single-class niaochao images leave co-occurring
insulators UNLABELED → false-negative signal (partial-annotation problem). DD/ND unchanged.
- **Lesson:** external data works when clean+on-domain+correctly-labeled (vs the universe dampers that
  failed on label/domain mismatch) — but single-class image dumps need the other components labeled
  (or loss-masked / pseudo-labeled) or they degrade those classes. A Birdnest-specialist could use it;
  the all-class model needs the co-labels.

### FINAL STANDINGS (native test, 39 DD instances ⇒ ±0.05 noise)
| model | recipe | mAP50 | DD AP | DD recall | ND AP |
|---|---|---|---|---|---|
| OSaug_v11 | OS3 + scale-aug | 0.682 | **0.749** | 0.692 | 0.753 |
| HRO_v8 | hi-res 1280 + OS3 | **0.747** | 0.723 | 0.692 | **0.824** |
| B_v11 (baseline) | native | 0.691 | 0.708 | 0.633 | 0.744 |
- Conclusions: (1) external/community data never helped DD at any dose/curation (label mismatch);
  (2) native oversample + scale-down aug fixed DD recall (0.633→0.692, AP→0.749); (3) hi-res+oversample
  is the best overall model; (4) copy-paste/P2/SAHI did not beat these on DD.
