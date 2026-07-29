# Hardware testing — Jetson Orin Nano FPS benchmark

Real-hardware companion to `results/jetson_nano_fps_approximation.md`, which
projects fps from RTX 6000 numbers because the original Jetson Nano (Maxwell:
no tensor cores, no INT8 DP4A) sits in a completely different performance
regime than the training GPUs. **A Jetson Orin Nano is Ampere** — it has
tensor cores and INT8 support, so most of that doc's caveats don't apply here
and a direct on-device measurement is both possible and worth trusting over
any projection.

## Files
- `bench_jetson_fps.py` — runs **on the Jetson**. Exports a checkpoint to
  ONNX/TensorRT as needed and times the full `predict()` pipeline (pre +
  forward + NMS/OBB-decode, not just the raw forward pass) over N iterations,
  batch size 1. Writes `jetson_fps_results.json` incrementally.
- `transfer_to_jetson.sh` — copies a `best.pt` checkpoint + the benchmark
  script + a handful of sample test images from `$ATLI_SERVER` onto the
  device over SSH/scp.
- `sample_images/` — drop a few representative field images here (gitignored
  contents; the loader just needs one real image to run predict() against).

## One-time setup on the Jetson
1. Flash JetPack (whichever version matches your Orin Nano), which ships the
   NVIDIA-built `torch`/`torchvision`/TensorRT stack — **do not** `pip
   install torch` from PyPI on a Jetson, it won't have CUDA support for the
   Tegra GPU.
2. `pip install ultralytics` (pulls in the rest; torch/torchvision are
   already satisfied by the JetPack wheels).
3. Confirm TensorRT is visible: `python3 -c "import tensorrt; print(tensorrt.__version__)"`.
4. **Before any benchmark run:** `sudo nvpmodel -m 0 && sudo jetson_clocks`
   (max power mode + pinned clocks — the default profile throttles well
   below what the board can do, and this is a free win the model can't buy
   back). Check `nvpmodel -q --verbose` for a "Super" profile too, if this
   board/JetPack supports it — another free clock-ceiling bump. Run
   `tegrastats` in a second terminal during the benchmark to confirm clocks
   stay pinned (thermal throttling on a passively-cooled kit will otherwise
   quietly skew the numbers).
5. There's no DLA on this SKU (unlike Orin NX/AGX Orin) — everything runs on
   the GPU, no DLA-offload path to chase.

## Usage
From this machine, once you have SSH access to the board:
```bash
JETSON_HOST=user@<jetson-ip> ./transfer_to_jetson.sh /path/to/best.pt
ssh $JETSON_HOST
cd ~/hardware_testing
python3 bench_jetson_fps.py --weights best.pt --imgsz 640 768 --formats pt engine --precision fp16
```
- `--formats pt engine` benchmarks the raw PyTorch checkpoint and a TensorRT
  engine (built on first run — TensorRT engines aren't portable across
  devices, so the `.engine` file must be built ON the target board, not
  copied from the server). Add `onnx` if you want that data point too.
- `--imgsz 640 768` matches the two resolutions live in the experiment log's
  640-deployment track (`blurmix_640` / grafts) and the 768 deployment
  candidate from the edge-optimization campaign.
- `--precision fp16` vs `fp32` — on Orin Nano (unlike the original Maxwell
  Nano) FP16 should actually help, since it has tensor cores.
- `--precision int8 --int8-data <fold_osall.yaml>` — Orin Nano has DP4A/INT8
  tensor-core support, another gap the original Nano didn't have. Needs
  calibration images (point at a CV fold's `osall.yaml`, e.g. copy
  `CV_eduardo_obb/fold0/osall.yaml` + its images over). **Check per-class
  P/R/mAP after quantizing, not just fps** — `Defective_Damper` is already
  the lowest-instance, worst-performing class, and INT8 error tends to hit
  minority classes hardest.
- `--conf` / `--max-det` — fewer NMS candidates = less post-processing work.
  Worth profiling at both the default (0.25) and the recall-screening op
  point (~0.1) from the operating-point analysis, since that tradeoff costs
  fps as well as precision.
- **OBB vs. plain-detect fps**, no training needed: benchmark an existing
  `CV_eduardo_det` (axis-aligned) checkpoint against the OBB one at the same
  recipe/resolution. The CV results already found OBB vs. detect is a wash
  on mAP — if OBB's rotated-IoU NMS costs meaningful fps on Jetson's CPU
  (unlike the RTX 6000 where CPU-side NMS is negligible next to the GPU
  work), plain detect could be a free fps win with no accuracy trade.

## Which checkpoint to benchmark
Pull `best.pt` from `$ATLI_SERVER:~/atli/runs/<run_name>_s2/weights/best.pt`.
Current 640-deployment candidates (see `CLAUDE.md` experiment log):
- `EDU_blurmix_640_f*_s2` — current 640 champion (clean 0.757, blur-robust)
- `EDU_ghost_blurmix640_f*_s2` / `EDU_dws_blurmix640_f*_s2` /
  `EDU_fnet_blurmix640_f*_s2` — lighter-backbone grafts on the same recipe
  (phase 18) — these are the ones where a real fps number actually matters,
  since the grafts trade accuracy for speed and RTX 6000 latency showed zero
  wall-clock benefit for them (launch-overhead-bound on a datacenter GPU).
  Any pick a single fold's checkpoint (e.g. `_f0_s2`) — fps shouldn't vary
  meaningfully across folds of the same recipe.

## Interpreting results
Update `results/jetson_nano_fps_approximation.md`'s checklist once real
numbers exist — a measured Orin Nano fps supersedes any projection from it.
Report fps alongside model + imgsz + precision + export format so it's
reproducible.
