# Prune-grid results — NEGATIVE RESULT (pre-registered outcome 3)
**Round 3 deliverable · branch `opt/jetson-nano` · 2026-07-09 · raw per-run data: `prune_grid_results.csv`**

## Verdict

**Structured global channel pruning of the native-768 YOLOv11n champion cannot be recovered to deployment accuracy on this dataset at ANY tested ratio — including the mild 1.25–1.35× boundary probe.** Acceptance gate (3-seed mean mAP@0.5 ≥ 0.745 AND Defective_Damper AP ≥ 0.59 on the noCPLID test split @768) was missed by every condition. Pruning is **off the Jetson Nano recipe**; per the campaign framing, the unpruned 768 model already satisfies the physically grounded 5–10 Hz detection spec (~17 fps inference-only projected), and the literal 30-fps target on the original Nano is reachable only at unusable accuracy (mAP ≤ 0.48). This is the pre-registered outcome 3: a decision-relevant negative result, not a failed round.

## Setup

21 runs. Base weights: native-768 champion `HR768nc_v11_s{0,1,2}_s2` (reference: **mAP 0.769 ± 0.009, DD 0.670 ± 0.049**), seed-paired (pruned seed *i* fine-tunes from base seed *i*). Prune: torch-pruning GroupNormPruner, GroupMagnitudeImportance(L2), global, round_to=8, ignored = Detect + C2PSA Attention + 3 auto-quarantined convs (see `prune_plan.md` §3); iterative steps until the MACs target. Fine-tune: 100 ep @768 on ATLI_noCPLID_OS3 (633 train imgs), SGD, scale=0.9, batch 16 — two schedules: **baseline** = champion stage-2 (lr0=0.00334, lrf=0.1535); **hlr** = recovery schedule (lr0=0.01, lrf=0.01), triggered per pre-approval after the baseline tier missed. Eval: `eval_prune_grid.py`, test split @768, per-class.

## Results (3-seed mean ± sd; fps = projected Nano inference-only at the 160 GFLOP/s effective anchor, `jetson_nano_research.md` §4 — NOT a measurement)

| condition | FLOPs speedup | GMACs @768 | params | mAP@0.5 | DD AP@0.5 | proj. Nano fps | gate |
|---|---|---|---|---|---|---|---|
| **unpruned reference** | 1.0× | 4.60 | 2.59M | **0.769 ± 0.009** | **0.670 ± 0.049** | ~17 | PASS (ref) |
| pr125_hlr (probe) | 1.25–1.35× | 3.42–3.67 | 2.0–2.2M | 0.695 ± 0.018 | 0.526 ± 0.048 | ~22 | **FAIL** |
| pr150_hlr | 1.53× | 3.00 | 1.5M | 0.639 ± 0.019 | 0.526 ± 0.036 | ~27 | FAIL |
| pr150 (stage-2 LR) | 1.53× | 3.00 | 1.5M | 0.609 ± 0.018 | 0.501 ± 0.076 | ~27 | FAIL |
| pr175_hlr | 1.83× | 2.51 | 1.1M | 0.475 ± 0.040 | 0.501 ± 0.056 | ~32 | FAIL |
| pr175 (stage-2 LR) | 1.83× | 2.51 | 1.1M | 0.458 ± 0.018 | 0.432 ± 0.023 | ~32 | FAIL |
| pr200_hlr | 2.04× | 2.25 | 0.93M | 0.364 ± 0.032 | 0.436 ± 0.095 | ~36 | FAIL |
| pr200 (stage-2 LR) | 2.04× | 2.25 | 0.93M | 0.287 ± 0.014 | 0.388 ± 0.038 | ~36 | FAIL |

Consistency: per-seed spreads are small (mAP sd ≤ 0.04 everywhere); the degradation is monotone in ratio in both schedule arms; hlr > baseline at every ratio (+0.03 to +0.08 mAP). Nothing here looks like a broken run.

## Why the recovery failed (evidence)

1. **Not primarily a schedule problem.** The hlr schedule at the boundary probe (1.25–1.35×) converged to *healthy* loss levels (final ep-100 losses box 1.03 / cls 0.74 / dfl 0.96 — comparable to an intact training) yet still landed at 0.695 mAP, −0.074 below reference and −0.14 on DD. At ≥1.5× the losses additionally never reach healthy levels (pr200_hlr final box 1.78 / cls 2.03), i.e. aggressive tiers are both capacity-starved *and* under-converged within 100 ep.
2. **Capacity floor of an already-nano model.** v11n is 2.6M params / 4.6 GMACs @768; the published wins we cited came from far more redundant starting points (MCP-YOLO pruned a 13.79M model −37%; the LAMP-PCB base solved a near-saturated task at P/R≈0.99 with mAP50 0.992). Removing 23–65% of channels from a model that is already the smallest in its family leaves nothing redundant to remove — consistent with the steep, immediate degradation.
3. **Data floor.** Recovery must re-learn features from 633 train images; a rare class with ~40 train instances-equivalent (Defective_Damper) is hit hardest — DD drops −0.14 even at the probe ratio, roughly 2× the mAP drop. Literature recoveries fine-tune on datasets 5–100× larger.
4. Both LR schedules bracketed the sensible range (0.00334 conservative / 0.01 standard); the ratio→accuracy curve is smooth and monotone across both, so intermediate LRs are very unlikely to change the verdict.

**Booked finding for the paper:** *global structured channel pruning of a nano-scale detector (YOLOv11n, 2.6M params) trained on a small defect dataset (n=633) is unrecoverable at ≥1.25× FLOPs reduction — mAP −0.07 (and rare-class AP −0.14) at 1.3×, degrading monotonically to −0.48 at 2× — despite convergent recovery training. Contrast: the same pipeline's published successes start from ≥5× larger models and ≥5× larger datasets.*

## Decision consequences

- **Nano 30-fps via pruning: dead.** The tiers that clear 30 fps (≥1.83×) sit at mAP ≤ 0.48. The Nano operating point remains the **unpruned 768 model: mAP 0.769 / DD 0.670, projected ~17 fps inference-only (~12–15 e2e)** — which already satisfies the rescoped 5–10 Hz inspection spec from `decision_memo.md`.
- **The Orin Nano Super recommendation strengthens further**: it delivers the literal 30 fps at full accuracy with zero accuracy negotiation (see `decision_memo.md`), and this grid shows there is no software path to 30 fps on the original Nano at deployable accuracy.
- Not pursued (per pre-registration, and unlikely to change the verdict given §Why-1): local per-layer pruning, BN-L1 sparse pre-training, distillation-assisted recovery. If pruning is ever revisited, it should be (a) on the Orin for battery/thermal headroom rather than feasibility, and (b) at ≤1.25× with distillation — but the probe result caps expectations.

## Ops notes

- pr150_s1/pr150_s2 were re-run cleanly after a duplicate-queue launch overwrote their first fine-tunes (stuck chained-ssh launcher fired its legs ~30 min late; all launches are single-purpose ssh calls since). All 21 rows in the CSV are from verified-clean runs.
- All runs from `~/atli/env_jetson` (isolated venv); shared training env untouched. Run dirs: `~/atli/runs_prune/`; pruned pre-finetune checkpoints: `~/atli/export_jetson/pr*.pt`.
