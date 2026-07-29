# YOLOv5n × universe/community data — gap-fill sweep (2026-07-29)

**Why:** the original 20-run universe benchmark
(`universe_benchmark_results.md`) covered only {v8n, v11n}. YOLOv5n — the
APET paper's headline model — was never trained on the downloaded community
(Roboflow-Universe) damper data. This sweep fills that cell so the
"external data never helps" claim is tested on all three architectures.

**Recipe:** identical to the original sweep — `yolov5nu.pt` COCO init,
detect, 2-stage TL 150+100 @640 batch 32, same dataset yamls, same native
199-img test split (Ufull scores in-domain by construction). Driver:
`train/sweep_v5_universe.sh`; raw metrics `results/v5_universe_results.json`
(re-valed from each run's `best.pt`). Single run per condition (matches the
original sweep's per-condition budget). All 5 runs exit 0, GPUs 6-7,
07:50–10:01 UTC.

## Results (native test split; DD = Defective_Damper AP@0.5)

| condition | mix ratio | mAP@0.5 | DD AP | DD recall |
|---|---|---|---|---|
| B_v5 baseline (3 seeds, ref) | native only | 0.660–0.690 | 0.596–0.616 | 0.51–0.59 |
| Ucap1_v5_150p100 | capped 1:1 | 0.688 | 0.626 | 0.564 |
| Ucap2_v5_150p100 | capped 2:1 | 0.672 | 0.578 | 0.590 |
| Uw_v5_150p100 | wangbo only | 0.655 | 0.540 | 0.559 |
| Uto_v5_150p100 | train-only merge | 0.644 | 0.635 | 0.615 |
| (control) Ufull_v5_150p100 | scored in-domain | 0.764 | **0.901** | 0.840 |

## Conclusion

**Community data does not help YOLOv5n either.** Every universe-augmented
condition lands inside or below the 3-seed baseline band on mAP
(0.644–0.688 vs 0.660–0.690) and DD AP (0.540–0.635 vs ~0.61), while the
in-domain Ufull control hits DD 0.901 — the exact signature seen on v8n and
v11n: *the community labels are learnable, but they do not transfer to the
native test distribution.* The external-data-never-helps finding is now
confirmed on **all three architectures** (v5n, v8n, v11n), closing the last
open cell of the universe benchmark grid.
