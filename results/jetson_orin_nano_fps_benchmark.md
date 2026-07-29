# Jetson Orin Nano — real on-device FPS benchmark (2026-07-28)

Real, measured batch-1 end-to-end fps (pre-process + forward + NMS/OBB-decode)
on a **Jetson Orin Nano** (Ampere: tensor cores + INT8 DP4A), superseding the
projection-only methodology in `jetson_nano_fps_approximation.md` (which
targets the *original* Maxwell Jetson Nano — a completely different chip with
no tensor cores). Harness: `hardware_testing/bench_jetson_fps.py`. Board
pinned to its max power profile before every run (`nvpmodel -m 0` — this
board's only two profiles are 15W/7W, no "Super"/MAXN tier — plus
`jetson_clocks`). JetPack 7.2 / L4T R39.2 / CUDA 13.2 / TensorRT 11.1.0 /
PyTorch 2.13.0+cu132 (community prerelease wheels — no official NVIDIA
wheels exist yet for this JetPack release).

**⚠ Hardware note:** these are real Orin Nano numbers, not the "original
Jetson Nano" the poster's placeholder figure (`sample_fps_frontier.png`)
assumed. The two chips are not comparable — Orin Nano is dramatically faster
(tensor cores, newer process node). Any figure/caption using these numbers
must be relabeled to say "Jetson Orin Nano," not left as a generic "Nano"
claim.

## Methodology note
FPS is a function of (architecture, resolution, export format) only — not
training recipe/data. One checkpoint per distinct architecture was
benchmarked (any fold), since different training recipes on the same
architecture have identical inference speed. This covers the full
backbone-graft frontier plus P2-head and v5n variants documented in
`CLAUDE.md`'s experiment log.

## Full architecture × resolution × format matrix
fps, PyTorch fp32 / TensorRT fp16 (`-` = not benchmarked, `ERR` = TensorRT
build failed):

| Architecture | 640px | 768px | 1024px | 1280px |
|---|---|---|---|---|
| YOLOv11n-OBB (champion, `EDU_blurmix_640`) | 22.4 / 34.8 | 20.0 / 23.4 | 14.9 / 11.9 | 10.4 / 6.5 |
| YOLOv8n-OBB (true champion, `EDU_v8deg15`) | 26.3 / 37.9 | 23.0 / 24.2 | 15.4 / 13.0 | 11.4 / 7.7 |
| v11-Ghost graft (OBB) | 22.2 / 45.9 | 19.8 / 32.5 | - / - | 9.6 / 14.4 |
| v8-Ghost graft (OBB) | 23.7 / 53.4 | 20.9 / 38.8 | - / - | 10.7 / 21.2 |
| v11-DWS graft (OBB) | 22.7 / **ERR** | 20.3 / **ERR** | - / - | 11.0 / **ERR** |
| v8-DWS graft (OBB) | 26.4 / **ERR** | 23.1 / **ERR** | - / - | 11.3 / **ERR** |
| v11-FasterNet graft (OBB) | 24.2 / 48.0 | 21.4 / 32.9 | - / - | 10.6 / 12.9 |
| v8-FasterNet graft (OBB) | 26.5 / 23.5 | 23.2 / 18.9 | - / - | 12.2 / **4.5\*** |
| YOLOv5n (detect-only, no OBB variant exists) | 30.2 / 59.9 | 26.3 / 42.9 | - / - | 11.9 / 25.5 |
| YOLOv11n-P2 (extra head, OBB) | 19.6 / **6.7\*** | 15.8 / **5.8\*** | - / - | 7.0 / **5.0\*** |
| YOLOv8n (plain **detect** task — see labeling bug below) | 32.2 / 58.2 | 27.0 / 40.1 | - / - | 12.2 / 23.2 |

**⚠ Labeling bug caught and fixed:** the first pass mislabeled
`EDU_v8champ_f0_s2` as "YOLOv8n-OBB (stock)" — it's actually a **plain-detect
task model** (`task: detect`, 11 output channels; OBB models have 12, the
extra channel is the rotation angle). The real v8n-OBB champion is
`EDU_v8deg15_f0_s2` (`task: obb`, matches CLAUDE.md's "v8n OBB+deg15 @1280 =
0.773"), now benchmarked separately and correctly above as "YOLOv8n-OBB (true
champion)". The mislabeled detect-task entry is kept in the table (last row)
for reference since it was fully measured, but flagged as **not** a
same-task comparison against the OBB architectures.

## Anomalies found (reported honestly, not smoothed over)

1. **DWS graft (v11 and v8) — TensorRT FP16 export hard-fails at every
   resolution.** Root cause confirmed in logs, not a fluke: `IBuilder::
   buildSerializedNetwork: Error Code 10: ... Could not find any
   implementation for node /model.1/conv/Conv` — TensorRT 11.1 on this
   CUDA-13.2 stack has no low-precision kernel for this particular grouped
   -conv configuration (Ultralytics' `DWConv`, `groups=gcd(c1,c2)`). PyTorch
   inference still works fine; only the FP16 TensorRT export path is broken.
   Possible follow-up: retry with `op_block_list` excluding the affected
   Conv layers from FP16 (TensorRT's own suggestion in the warning), or try
   FP32 TensorRT instead of FP16.

2. **P2 head — TensorRT is *slower* than plain PyTorch at all three
   resolutions** (e.g. 640: 6.7 vs 19.6 fps). Measured before any of the
   session's network interruption, under confirmed-pinned clocks, consistent
   across all 3 resolutions — a real, reproducible property of this engine
   build, not noise. Likely related to the extra stride-4 head's much larger
   output tensor (up to 136,000 anchors at 1280) interacting poorly with
   this TensorRT version's tactic selection.

3. **v8-FasterNet @ 1280 and champion @ 1024/1280 — TensorRT slower than
   PyTorch.** Checked against two candidate explanations and ruled both out:
   thermal throttling (re-pinned clocks and re-ran — result barely changed)
   and concurrent-GPU contention from an earlier race condition (re-ran in
   isolation — result reproduced). This looks like genuine TensorRT
   tactic-selection variance for these specific layer shapes on this
   bleeding-edge JetPack 7.2/TensorRT 11.1/CUDA 13.2 combination, not a
   measurement artifact — but it's surprising enough that it deserves a
   skeptical second look (e.g. a persistent timing cache, or `trtexec`
   directly) before being quoted as a stable deployment number.

4. A **race condition** in the benchmark harness: running two
   `bench_jetson_fps.py` processes concurrently against the same `--out`
   json causes a silent read-modify-write clobber (last writer wins, no
   locking). Only run one instance at a time against a shared results file.

## Poster figure data — `sample_fps_frontier.png` / `poster_fig_recipe_ladder`

Real measured numbers to replace the placeholder ("projections for an
original Jetson Nano ... not yet measured on-device") in
`paper/poster/figs_src/gen_poster_figures.py` (~line 406-408). All 5 configs
use the *same* v11n-OBB architecture — only the inference resolution and
which weights happened to be trained differ; fps depends only on
(architecture, resolution, format):

| Config | Res | mAP@0.5 | DD AP | PyTorch fp32 | TensorRT fp16 | Old placeholder |
|---|---|---|---|---|---|---|
| 640 native | 640 | 0.736 | 0.614 | 22.4 | 34.8 | ~25 fps* |
| 768 infer (1280-trained) | 768 | 0.754 | 0.593 | 20.0 | 23.4 | ~18 fps* |
| 768 native retrain | 768 | 0.766 | 0.664 | 20.0 | 23.4 | ~18 fps* |
| 1024 infer (1280-trained) | 1024 | 0.783 | 0.634 | 14.9 | 11.9 | ~10 fps* |
| 1280 native champ | 1280 | 0.784 | 0.622 | 10.4 | 6.5 | ~6 fps* |

**Open decision for the poster:** at 1024/1280 TensorRT comes out *slower*
than PyTorch (anomaly #3 above) — using the TensorRT column would show fps
*and* accuracy moving in the same direction only up to 768, then fps
dropping unusually sharply at 1024/1280. The PyTorch column is monotonic and
arguably the safer number to quote until the TensorRT anomaly is understood
better. Either way, the caption needs to say "measured on Jetson Orin Nano,"
not left as a generic Nano/projection claim.

Raw data: `hardware_testing/jetson_fps_results.json` (pulled from the
Jetson at `/home/eduardo/hardware_testing/jetson_fps_results.json`).
