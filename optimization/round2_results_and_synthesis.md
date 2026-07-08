# Round 2/3 — Pilot Results, srcTL Root-Cause, and Thesis-Thread Synthesis

**Date:** 2026-07-08 · **Branch:** `opt/thesis-enhancements` · **Author:** thesis-analyst thread
**Context docs:** `optimization/thesis_digest.md` (page-referenced thesis digest), `optimization/thesis_enhancements_analysis.md` (T1–T8 transferability + ranked shortlist).

---

## 1. Incident postmortem (numpy corruption, 2026-07-07)

Both GPU-6/7 pilot chains **ran to completion** through the corrupted-env window (long-running trainings had numpy already resident; only fresh process starts crashed). No relaunches were needed:

- `TH2_gpu6.log` → `### GPU6 CHAIN DONE` (Jul 7 21:49): gate build → source pretrain → srcTL fine-tune.
- `TH2_gpu7.log` → `### GPU7 CHAIN DONE` (Jul 7 19:41): lowlr pilot → champ768 control (ran as seed 0; the seed-3 revision arrived after step 2 had started).
- No `Traceback`/numpy errors anywhere in either log.
- Orchestrator re-validated all suspect runs with the healthy env: champ768_s0 re-val **bit-identical** → trainings untainted.

## 2. Pilot results (noCPLID test split @768: 120 imgs / 467 inst; DefDamper = 12 inst — tiny, treat per-class numbers as coarse)

| run | recipe delta vs champion@768 | mAP@0.5 | DD AP | verdict |
|---|---|---|---|---|
| `TH2_champ768_v11_s0` (control) | none (stage-2 lr0 = 0.00334) | 0.777 | 0.725 | reference |
| champion@768 3-seed (orchestrator) | none | **0.769 ± 0.009** | 0.670 ± 0.049 | reference band |
| `TH2_lowlr_v11_768_s0` | stage-2 lr0 = 1e-4 (thesis Table 4.1) | 0.772 | 0.684 | **wash** at 1 seed (inside seed noise) |
| `TH2_srcTL_v11_768_s0` | init from gated in-domain source pretrain | **0.702** | **0.524** | **FAILED: −7.5 mAP, −20 DD AP** |

Source pretrain itself (`TH2_srcpre_v11_s1`, 150 ep @640 on the gated source): source-val mAP@0.5 **0.898** (DefDamper 0.905, generic Defective_Insulators 0.962) — fully converged.
Leakage gate: 11,294 → **11,188 kept** (2 pHash ≤ 8, 104 filename) vs all 797 native images. Gate worked; leakage is not a factor in either direction.

## 3. srcTL root cause: **genuine negative transfer, not plumbing**

**Plumbing verified correct:**
- `runs/TH2_srcTL_v11_768_s0_s1/args.yaml`: `model: .../TH2_srcpre_v11_s1/weights/best.pt` — fine-tune initialized from the source-pretrained weights.
- Log line 901: `Transferred 499/499 items from pretrained weights` — **every** tensor loaded (source nc=7 == native nc=7, so even the detection head transferred; COCO-init runs show 448/499). Note the head transferred with *scrambled class semantics* (source idx 4 = Transmission_tower vs native idx 4 = Normal_Damper), but 150+100 ep at lr0 0.01 relearns a head easily — this is a footnote, not the cause.

**Data-side cause (three converging lines of evidence):**

1. **~45% of source supervision is native background.** Source train label census (8,971 imgs, 10.36 boxes/img): `transmission_line` 34,348 + `Transmission_tower` 6,668 + `defect_transmission_line` 1,081 = 42,097 of ~93k boxes. ATLI deliberately *excludes* lines and towers — they are ubiquitous background in every native image. The pretrain explicitly teaches the model to fire on them; the fine-tune must unlearn it.
2. **The source is essentially one scene.** Visual audit of 20 random source images: a single arid-corridor capture campaign — same terrain, same washed-out lighting, same lattice-tower type, same dark disc-insulator strings, sequential `Transmission-NNNN-*.rf.*` filenames, several near-empty ground frames. 11k images ≈ one domain. Native ATLI is the opposite (internet-sourced: green/snow/water scenes, wooden poles, porcelain and glass insulators, extreme close-ups to wide shots).
3. **Catastrophic forgetting of COCO diversity.** 150 epochs at lr0 0.01 on a narrow single-domain source overwrites COCO's general features (source-val 0.898 shows it specialized hard). The COCO-init control keeps broad features that match ATLI's heterogeneity better. Consistent with the largest srcTL drops being on classes absent from the source: Birdnest 0.994→0.875 (no birdnest class in source), Broken_Insulator 0.665→0.554 (no defect subtypes), Flashover 0.604→0.579.

**Why the thesis's T1 didn't replicate:** Vazquez's FASDD is ~100k *heterogeneous* images matched in task semantics to the target (fire/smoke → fire/smoke). Our `atli_source_dataset` is ~11k images ≈ one corridor, with 45% of boxes in target-background classes and a coarser taxonomy. The lever isn't wrong in general — the precondition (large + diverse + semantically aligned source) simply doesn't exist in our data inventory.

**Combined project finding:** external power-line data is now 0-for-2 mechanisms on ATLI — co-training (F1, universe sweep) *and* pretraining-init (this pilot) both hurt. The only external-data path left open would be assembling a genuinely multi-source, taxonomy-mapped pretraining base (FASDD-scale) — high effort, low prior; **not recommended**.

## 4. Thesis-lever scoreboard (final unless lowlr surprises)

| lever | status for ATLI |
|---|---|
| T1 in-domain source pretraining | **DEAD** with available sources (−7.5 mAP, root-caused above) |
| T2 stage-2 lr0 = 1e-4 | wash at 1 seed; **seeds 1–2 running now** to settle (variance/DD-recall is the remaining question) |
| T3 zero frozen layers | **CONFIRMED** (thesis Table 4.2 + our frz10/frz20) — already champion practice; cite both studies in the paper |
| T4 no cascaded TL | adopted as negative guidance (saved us a sweep) |
| T5 lightweight backbone swaps | **DEAD** for accuracy at our data scale (thesis Table 5.6/5.8) |
| T6 OpenVINO/ONNX export | descoped (Jetson/TensorRT workstream owns deployment); residue: deployment imgsz ≈ 768 |
| T7 EDP + power methodology | **ALIVE — recommend adopting** (~$30 meter + Simpson-rule energy + normalized EDP) for the paper's edge section |
| T8 variance-as-headline-metric | **ALIVE — adopt** (report fold/seed std as a generalizability claim; we already have the CV data) |

## 5. Recommendation

The thesis thread's accuracy levers are exhausted: **the champion recipe stands unmodified.** The thread's durable yield is methodological — (a) EDP/power reporting, (b) variance-as-metric, (c) an independent within-group confirmation that freezing hurts and cascades are useless, and (d) a strong **768 deployment profile**: champion@768 = 0.769 ± 0.009 mAP vs 0.784 ± 0.011 at 1280 — only ~1.5 mAP for ~2.8× fewer pixels, which is the number the Jetson workstream needs. Close the thread after the lowlr seeds report; no further srcTL variants.

## 6. Runs launched this round (GPUs 6–7 only; env untouched)

| run | GPU | purpose | status |
|---|---|---|---|
| `TH2_champ768_v11_s3` | 7 | orchestrator-tasked integrity replicate (seed 3) | launched 09:16, ~22 min/run |
| `TH2_lowlr_v11_768_s1` → `_s2` | 6 | settle T2 (2 extra seeds, chained) | launched 09:32, log `TH2_gpu6_lowlr_s12.log` |

Pilot chosen per Round-3 item 3: **lowlr seeds over a revised srcTL** — the srcTL failure is data-fundamental (source homogeneity + background-class supervision), so a 768-pretrain or merged-source variant has <50% prior; merged co-training is additionally pre-refuted by F1.
