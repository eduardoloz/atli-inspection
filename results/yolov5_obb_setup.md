# YOLOv5n-OBB setup — filling the model-family gap

Written 2026-07-28. Status: **sweep launched and running** (not yet complete —
see Status section for how to check progress). This closes the gap noted in
`CLAUDE.md`: "no Ultralytics YOLOv5-OBB variant exists," so YOLOv5n has only
ever been benchmarked in plain-detect mode, never OBB, alongside the
v8n-OBB/v11n-OBB numbers in `cv_obb_results.md`.

## Fork selection

Ultralytics never shipped OBB support for the `yolov5` package (it landed in
v8/v11 only), so any YOLOv5-OBB has to come from a community fork built on
Circular Smooth Label (CSL) rotation detection. Checked before committing:

- **`hukaixuan19970627/yolov5_obb`** (the fork everyone cites) — real OBB
  training support, DOTA-style label format, but last substantive commit is
  from 2022 (repo `pushed_at` 2023-10-13 is a stale merge, not new code),
  191 open issues, unarchived but effectively unmaintained. Its own
  `docs/install.md` only claims tested CUDA 10.0–11.3 / torch ≥1.7 — far
  behind this server's driver (570.207, CUDA 12.8).
- Scanned its forks for anything that modernized the dependency story.
  Most forks are stale mirrors (`pushed_at` identical to upstream). One
  stood out: **`WHFF521/yolov5_obb_compatible`** (pushed 2025-08-05),
  whose README states it minimally patched the deprecated-function breakage
  from newer torch/numpy and confirms `train.py` runs end-to-end on
  `torch/torchvision/torchaudio` from the `cu128` wheel index — i.e.
  already aimed at exactly this server's CUDA 12.8 stack.
- **Chosen: `WHFF521/yolov5_obb_compatible`**, cloned at commit
  `1d9128def0c1dafd9176877fb5d1d8a6d7c41252` (2025-08-05) into
  `~/atli/yolov5_obb/`. Its diff from upstream `hukaixuan19970627/yolov5_obb`
  touches only `train.py`, `models/common.py`, `utils/{datasets,general,loss,
  plots}.py` and the `nms_rotated` extension sources — a small, legible
  compat patch set, not a rewrite. Architecture files (`models/yolov5n.yaml`
  etc.) are untouched stock YOLOv5-OBB (CSL rotation head).

Even with that fork's patches, several more compat issues surfaced under
real use here (see "Additional patches" below) — the upstream code is old
enough that no single fork was going to be drop-in; budget for this when
reusing.

## Environment

- Isolated conda env **`~/atli/env_v5obb`** (Python 3.9), completely separate
  from `~/atli/env` (the project's main Ultralytics env) — never touched.
- `torch==2.8.0+cu128 torchvision torchaudio` from the `cu128` wheel index
  (matches driver 570.207 / CUDA 12.8 confirmed via `nvidia-smi`+`nvcc -V`).
- `numpy<2` (pinned to 1.26.4) + `opencv-python-headless<4.10` (pinned to
  4.9.0) — the server has no `libGL.so.1`, so plain `opencv-python` fails to
  import; headless avoids that. Also `setuptools==68.2.2` (newer setuptools
  dropped bundled `pkg_resources`, which `utils/general.py` still imports).
- Rotated-NMS CUDA extension built via
  `cd utils/nms_rotated && python setup.py build_ext --inplace` — succeeded
  cleanly, auto-detected `sm_75` (Quadro RTX 6000 = Turing, compute 7.5) via
  `TORCH_CUDA_ARCH_LIST` auto-inference. This was the one step expected to be
  fiddly and it wasn't, on this env/driver combination.

## Additional patches applied (on top of the fork's own compat fixes)

Found via an actual smoke train/val run, not by inspection — each of these
silently breaks a fresh clone of this fork under torch 2.8 / numpy 1.26:

1. **`utils/datasets.py::verify_image_label`** — label parsing did
   `np.concatenate((cls_id, label[:8]), axis=None)` where `cls_id` is a
   Python `int` and `label[:8]` a list of **strings** (unconverted split
   tokens). Old numpy silently upcast this to a string array before the
   caller's `np.array(l_, dtype=np.float32)` parsed it back to float; current
   numpy's stricter dtype promotion raises
   `DType <IntegerAbstractDType> could not be promoted by <StrDType>` and the
   fork treats every single image as corrupt. Fixed by converting to floats
   before concatenating: `np.array([cls_id] + [float(v) for v in
   label[:8]], dtype=np.float32)`.
2. **`np.int` / `np.float` / `np.bool`** (removed in modern numpy) — still
   present in `models/common.py`, `utils/general.py`, and several
   `DOTA_devkit/*.py` eval scripts that the fork's own patch pass missed.
   Repo-wide regex fix (`np.int`→`int`, `np.float`→`float`, `np.bool`→`bool`)
   across all `.py` files.
3. **`utils/general.py::check_requirements`** defaulted `install=True`,
   meaning every `train.py`/`val.py` invocation calls `pip install` for any
   requirement it thinks is unmet — and since we install `opencv-python-
   headless` (a different distribution name than the `opencv-python` name in
   `requirements.txt`), it decided opencv was *missing* and reinstalled
   plain `opencv-python` (pulling numpy≥2 with it), silently breaking the
   pinned env on the very first run. Changed the default to `install=False`
   so it only warns instead of mutating the env.
4. **`val.py`** builds `names = {k: v for k, v in enumerate(model.names)}`
   (a dict) and passes it straight into `create_dataloader` →
   `LoadImagesAndLabels` → `verify_image_label`, which calls
   `cls_name_list.index(...)` — `dict` has no `.index`. `train.py`'s own
   internal calls use a plain list and never hit this. Patched `val.py` to
   convert to `list(names.values())` before passing it down.

All of these are patches to **our own clone** in `~/atli/yolov5_obb`, not
upstream — nothing was contributed back (out of scope here), but the diffs
are small and could be upstreamed if useful later.

## Data conversion

The project's OBB CV data lives in `~/atli/CV_eduardo_obb/fold{0..4}/`,
Ultralytics-format OBB labels: `class x1 y1 x2 y2 x3 y3 x4 y4`, normalized
0–1, 4 corners. This fork needs DOTA-style long-edge labels: one `.txt` per
image in a `labelTxt/` dir (sibling to `images/`), each line
`x1 y1 x2 y2 x3 y3 x4 y4 classname difficulty`, **absolute pixel**
coordinates (see its `docs/GetStart.md`).

Conversion script: **`~/atli/yolov5_obb_convert.py`**. For each fold and each
split (`train`, `trainosall`, `val`, `test`):
- Denormalizes each corner by the actual image size (read via PIL — all
  images in this dataset happen to be 640×640 Roboflow exports, but the
  script doesn't assume that).
- Maps the numeric class index to its name via each fold's `base.yaml`
  `names:` list (`Birdnest, Broken_Insulator, Defective_Damper,
  Flashover_Insulator, Normal_Damper, Normal_Insulators,
  Self-Exploded_Insulator`), difficulty hardcoded to `0`.
- Images are **symlinked**, not copied (dataset directory is ~106MB total
  vs. multiple GB of duplicated imagery across 5 folds).
- Emits `base.yaml` (train split) and `osall.yaml` (trainosall split) per
  fold, same `path`/`names`/`nc` structure as the source.

Output: **`~/atli/CV_eduardo_obb_v5fmt/fold{0..4}/`**. Verified by hand
against the source label (`fold0/train/.../0592_...txt`): normalized
`0.484375, 0.384375` on a 640px image → `310.00, 246.00` in the converted
file, matches exactly; class index 5 → `Normal_Insulators`, correct per the
yaml ordering. Per-fold counts (train / trainosall / val / test images):
fold0 683/1647/143/148, fold1 673/1623/155/146, fold2 677/1631/150/147,
fold3 691/1669/142/141, fold4 690/1668/140/144.

## Recipe ladder mapped onto this fork

The fork's `train.py` takes hyperparameters only from a `--hyp` yaml (no CLI
flags for lr0/degrees/etc.), so the project's stage1/stage2 lr schedule and
aug knobs became four hyp files in `~/atli/yolov5_obb/data/hyps/atli/`:

| condition | data yaml | hyp (stage1 / stage2) | degrees | scale | mixup |
|---|---|---|---|---|---|
| **baseline** | `base.yaml` (native train, no oversample) | `hyp.base_s1.yaml` / `hyp.base_s2.yaml` | 0.0 | 0.5 (stock default) | 0.0 |
| **champion-equivalent** | `osall.yaml` (native ×3 DefDamper-oversampled train, already built by the project) | `hyp.champ_s1.yaml` / `hyp.champ_s2.yaml` | 15.0 | 0.9 | 0.15 |

Both stages use lr0=0.01 (stage1, default lrf=0.01) → lr0=0.00334 lrf=0.1535
(stage2), matching `run_config_obb.sh` exactly. Everything else (hsv, flip,
mosaic, momentum, weight decay, warmup) left at Ultralytics-standard values,
identical between conditions, so only the deliberate knobs differ.

**Supported vs. not, honestly:**
- ✅ Rotation (`degrees`), scale-jitter (`scale`), `mixup` — all present in
  this fork's hyp schema and exercised directly.
- ✅ Oversampling — not a fork feature, but didn't need to be: the project's
  existing `osall.yaml`/`trainosall` split (already built by
  `data/build_osall*_eduardo.py`) converts straight through, so the
  champion-equivalent gets real ×3 DefDamper oversampling for free.
- ✅ COCO-init 2-stage transfer learning — `--weights weights/yolov5n.pt`
  (stock Ultralytics **v7.0** release checkpoint, downloaded directly,
  architecturally compatible since this fork's backbone/neck are untouched
  stock YOLOv5n). Verified via smoke test: `Transferred 343/349 items from
  weights/yolov5n.pt` — only the OBB Detect head (extra CSL theta channels)
  is fresh-init, exactly the shape-mismatch-only skip behavior
  `run_config_obb.sh` relies on for v8/v11.
- ❌ No `close_mosaic`, no `multi_scale` equivalent — not in this codebase's
  hyp/CLI surface, not force-fit.
- ❌ No P2 head / architecture grafts — out of scope for this pass, would
  need a custom `models/*.yaml` (feasible later, not attempted).

## Evaluation caveat (read before trusting the numbers)

`val.py`'s built-in metric is **`HBBmAP@.5`** — computed via the standard
Ultralytics `ap_per_class` machinery, but this fork's own axis-aligned
metric, not the rotated-IoU `mAP@0.5` that `yolo obb val` reports for
v8n/v11n in `cv_obb_results.md`. **Baseline vs. champion-equivalent numbers
from this sweep are internally comparable to each other, but not
numerically comparable to the v8n-OBB/v11n-OBB numbers elsewhere in the
project's tables** without an extra step. A true rotated Task1 mAP (poly
IoU, matching Ultralytics OBB) would need the repo's own
`DOTA_devkit/dota_evaluation_task1.py` run directly against the `labelTxt/`
folders (no image-splitting needed, since our images are already single
640px tiles, not giant aerial scenes) — **this has since been built and
automated, see "Rotated Task1-mAP evaluator + auto-eval chain" below; don't
use the HBBmAP@.5 numbers above for any cross-model comparison —
`results/yolov5_obb_results.md` has the comparable number once the chain
finishes.** `--save-json`'s pycocotools path was dropped from `run_v5obb.sh`
since it needs a COCO-format `instances_*.json` this dataset doesn't have
(errors harmlessly if left in, no accuracy signal) — the Task1 evaluator
below regenerates predictions with `--save-json` itself in a separate
eval-only pass, so this doesn't block the rotated-mAP pipeline.

## Launch

- Driver: **`~/atli/run_v5obb.sh`** — same `NAME GPU EP1 EP2 IMGSZ BATCH`
  positional-arg / `DATA`+env-var convention as `run_config_obb.sh`, plus
  `HYP1`/`HYP2` (this fork needs explicit hyp files) and `WEIGHTS` (defaults
  to `weights/yolov5n.pt`). Runs stage1 train → stage2 train (from stage1's
  `best.pt`) → `val.py --task test` in sequence.
- Sweep: **`~/atli/sweep_v5obb.sh`** — 2 conditions × 5 folds = 10 jobs,
  queued 5-at-a-time (`wait` every 5) on **GPUs 1–5** (GPU 0 excluded: another
  user's job, 17GB/~80% util at launch time; GPUs 6–7 left idle as slack on
  this shared box). Same nohup+disown+status-file idiom as
  `sweep_eduardo_p21.sh`: launched via
  `nohup bash ~/atli/sweep_v5obb.sh > ~/atli/logs_v5obb/sweep_v5obb_nohup.log 2>&1 & disown`.
  150+100 epochs, imgsz=1280, batch=16 — same schedule as the project's OBB
  CV sweeps.
- Status file: **`~/atli/sweep_v5obb_status.txt`** (plain-text START/DONE
  lines per job, same format as every other phase's status file). Per-job
  logs: `~/atli/logs_v5obb/v5OBB_{base,champ}_f{0..4}.log`.
- Job naming: `v5OBB_base_f{0..4}` and `v5OBB_champ_f{0..4}`, runs land in
  `~/atli/yolov5_obb/runs/{name}_s1/`, `{name}_s2/`, `{name}_test/`.

## Status as of launch (2026-07-28 19:18 UTC)

Sweep launched and confirmed progressing: all 5 baseline folds (wave 1,
GPUs 1–5) past data loading and into epoch 0/149 with real loss values
(`box/obj/cls/theta` all decreasing) and 70–88% GPU utilization on 4 of 5
GPUs within ~2 minutes of launch (smoke-tested at ~3.8 it/s / 43 it per
epoch on `base.yaml`, ≈13s/epoch → baseline fold ≈1h; `osall.yaml` has
~2.4× the images so champion-equivalent folds are proportionally longer,
roughly ≈2.5h/fold). Wave 2 (champion-equivalent, same 5 GPU slots) starts
automatically once wave 1's `wait` clears — no manual intervention needed.
Check `~/atli/sweep_v5obb_status.txt` for START/DONE lines, or
`tail -f ~/atli/logs_v5obb/v5OBB_*_f*.log` / `nvidia-smi` on GPUs 1–5 for
live progress. Full 10-run sweep (both waves) estimated **≈3.5h** wall
clock from launch, not yet complete as of this writeup.

## Rotated Task1-mAP evaluator + auto-eval chain (added after initial launch)

The `HBBmAP@.5` caveat below was closed properly rather than left as a manual
follow-up:

- **`polyiou` built**: `~/atli/yolov5_obb/DOTA_devkit/` (already vendored in
  the fork) via `python setup.py build_ext --inplace`, compiling the
  pre-generated `polyiou_wrap.cxx` + `polyiou.cpp` directly into
  `_polyiou.cpython-39-*.so`. No working `swig` binary was needed (none is
  installed on this server, and there's no sudo) since the C++ wrapper was
  already checked into the repo — only `swig`'s *output* needed compiling,
  not regenerating from `polyiou.i`. This is a separate, smaller build step
  from the yolov5_obb rotated-NMS CUDA op documented above; the two are easy
  to conflate since both involve "rotated + C extension" but they're
  unrelated code paths (NMS at inference time vs. IoU scoring at eval time).
  **Sanity-checked before trusting it**, against hand-computed geometry:
  identical unit squares → IoU 1.0 exactly; two disjoint unit squares → IoU
  0.0 exactly; two 2×2 squares offset by 1 unit (1×2 intersection / 6 union)
  → IoU 0.33333 exactly. All three matched by-hand expectations bit-for-bit.
- **`~/atli/json2task1.py`**: converts a `val.py --save-json` prediction
  file into DOTA `Task1_<class>.txt` format, parameterized by this dataset's
  7 ATLI classnames (the fork's own `tools/TestJson2VocClassTxt.py` hardcodes
  the 15/16-class DOTA-v1/v1.5 taxonomy, so it was not reusable as-is).
- **`~/atli/eval_v5obb_task1.py`**: full driver — for each of the 10 runs,
  regenerates predictions via `val.py --save-json` (auto-picks a free GPU,
  never GPU 0), converts to Task1 format, scores every class with
  `DOTA_devkit.dota_evaluation_task1.voc_eval` at `ovthresh=0.5` and
  **`use_07_metric=False`** (the modern continuous/all-point AP integration
  — deliberately NOT the legacy VOC07 11-point method the script's own
  `main()` hardcodes, since that would silently not match the AP convention
  used everywhere else in this project), also pulls the fork's native
  `HBBmAP@.5` out of the already-existing training-run log (no rerun needed)
  for reference, aggregates 5-fold mean±std per condition matching
  `eval_cv_eduardo.py`'s convention, and writes both
  `~/atli/eval_v5obb_results.json` (incremental, resumable) and
  **`~/atli/results/yolov5_obb_results.md`** (the human-readable report,
  clearly labeling which number is the comparable rotated Task1 mAP@0.5 vs.
  the fork's own reference-only axis-aligned metric — verified end-to-end on
  an in-progress checkpoint before launch, see below).
- **Verified before arming the chain**, not just inspected: ran the full
  predict → convert → score path by hand against `v5OBB_base_f0`'s
  in-progress stage-1 checkpoint (epoch ~10/149) on the idle GPU 6. Rotated
  Task1 mAP@0.5 came back **0.387**, tracking closely and consistently
  slightly below the same checkpoint's native HBBmAP@.5 of **0.417**
  (rotated IoU is a stricter, non-axis-aligned criterion, so a modest,
  consistent HBB≥rotated gap across all 7 classes is exactly the expected
  relationship, not a red flag) — e.g. Birdnest 0.701→0.687, Normal_Damper
  0.409→0.386, Normal_Insulators 0.634→0.545. This is strong evidence the
  wiring (label formats, class-index mapping, coordinate spaces) is correct,
  on top of the geometric polyiou sanity checks above.
- **Auto-eval chain, armed and running**: `~/atli/run_v5obb_autoeval.sh`
  waits on `sweep_v5obb_status.txt` for the `SWEEP_V5OBB_DONE` marker (same
  `while ! grep -q ... ; do sleep 300; done` idiom used elsewhere in this
  project, e.g. `sweep_eduardo_p21.sh` waiting on `SWEEP_P20_DONE`), then
  runs `eval_v5obb_task1.py` automatically — no manual step required once
  training finishes. Launched via `nohup ... & disown` (confirmed running,
  detached from the SSH session, PID independent of this conversation).
  Status file: **`~/atli/autoeval_v5obb_status.txt`** (`AUTOEVAL_V5OBB_WAITING`
  → `AUTOEVAL_V5OBB_START` → `AUTOEVAL_V5OBB_DONE (exit N)`, each with a
  timestamp). Its own log: `~/atli/logs_v5obb/eval_v5obb_task1.log`. Once it
  finishes, `results/yolov5_obb_results.md` will contain the full 5-fold
  rotated Task1-mAP@0.5 comparison (baseline vs. champion-equivalent) with
  the native HBBmAP@.5 kept alongside for reference only, clearly labeled.

## Known limitations for whoever picks this up next

- This fork is a frozen 2022-era CSL-rotation YOLOv5-OBB codebase with a
  best-effort 2025 compat patch on top (plus this pass's own patches on
  top of that) — expect more friction than the actively-maintained
  Ultralytics v8/v11 OBB path if extended further (e.g. exporting to
  ONNX/TensorRT, adding new hyp knobs, DDP).
- The results-collector (`eval_v5obb_task1.py`) and its auto-eval chain
  (`run_v5obb_autoeval.sh`) are armed and running as of this writeup but
  have not themselves finished a full pass over real (non-smoke-test)
  weights yet — check `~/atli/autoeval_v5obb_status.txt` for
  `AUTOEVAL_V5OBB_DONE` and `results/yolov5_obb_results.md`'s timestamp
  before citing numbers from it.
- `HBBmAP@.5` ≠ the project's usual rotated OBB `mAP@0.5` — see the
  Evaluation caveat and the "Rotated Task1-mAP evaluator" section above;
  use `results/yolov5_obb_results.md`'s Task1 mAP@0.5 column, not this one,
  for any comparison to v8n-OBB/v11n-OBB.
- The rotated Task1-mAP evaluator uses a *different implementation*
  (DOTA_devkit/polyiou + a hand-rolled VOC-style integrator) than
  Ultralytics' internal `OBBMetrics` used for the v8n/v11n-OBB numbers
  elsewhere — same metric family (rotated IoU, 0.5 threshold, all-point AP),
  not guaranteed bit-identical. Also note `dota_evaluation_task1.py`'s HBB
  pre-filter uses the old VOC "+1 pixel" inclusive-width convention before
  falling back to true polygon IoU for the actual score — a minor legacy
  quirk, not patched, shouldn't materially move the numbers.
- No architecture-level P2/graft experiments attempted for this model —
  the params/GFLOPs frontier work done for v8/v11 grafts
  (`results/model_flops.json`) has no v5-OBB counterpart yet.
- The isolated env (`~/atli/env_v5obb`) and repo clone
  (`~/atli/yolov5_obb`) are new, self-contained additions; nothing in
  `~/atli/env` or any existing dataset directory was modified.
