# Backbone-graft provenance

Every yaml here is a hand-edited Ultralytics model-definition file, not a file
pulled from an external repo. "Provenance" below means: which module actually
implements the graft, and which paper the graft's *name* is nominally citing.
Bib keys refer to `paper/refs.bib`.

## Ghost — `yolo11n-ghost-obb.yaml`, `yolov8n-ghost-obb.yaml`

Uses Ultralytics' **native** `GhostConv` / `GhostBottleneck` / `C3Ghost`
modules (`ultralytics/nn/modules/{conv,block}.py`, unmodified) — this repo
only rearranges them into a full backbone (GhostConv as every backbone
downsample, C3Ghost as every stage block; head/neck left stock). Ultralytics'
implementation is a faithful port of the Ghost module/bottleneck from
Han et al., *GhostNet: More Features from Cheap Operations*, CVPR 2020
[`han2020ghostnet`]. The "GhostConv used directly as a stride-2 downsample
layer" pattern mirrors Ultralytics' own official `yolov5s-ghost.yaml`
example, not something we invented.

Application-domain inspiration (the reason we tried this on this dataset):
Lu et al., *A Lightweight Insulator Defect Detection Model Based on Drone
Images*, Drones 2024 [`lu2024iddyolo`] — GhostNet backbone + C3Ghost/GSConv
neck for insulator defect detection. Verified real (code:
github.com/LuYang-2023/Insulator-Defect-Detection-YOLO); previously cited in
code comments only as the shorthand "IDD-YOLO" with no bib entry.

**No known implementation issues** — this graft is architecturally the most
faithful of the three.

## FasterNet — `yolo11n-fnet-obb.yaml`, `yolov8n-fnet-obb.yaml`, `fnet_sitecustomize.py`

Ultralytics has **no native FasterNet/PConv module**, so this one is
hand-implemented in `fnet_sitecustomize.py` (`FasterBlock` / `C3Faster`),
monkey-patched in under the `C3Ghost` name at import time (`FNET=1`). Modeled
on Chen et al., *Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
Networks*, CVPR 2023 [`chen2023fasternet`] (PConv: dense 3x3 conv on 1/4 of
channels via `n_div=4`, left untouched otherwise, + a pointwise MLP over all
channels, residual).

Application-domain inspiration: Chai et al., *FPFS-YOLO: An Insulator Defect
Detection Model Integrating FasterNet and an Attention Mechanism*, Sensors
2025 [`chai2025fpfsyolo`] — a FasterNet-based `C3k2_faster` module on YOLO11n
for insulator defect detection. Verified real; previously cited only as the
shorthand "FPFS-YOLO" with no bib entry.

**Known deviations from the source paper** (flagging per request — not fixed,
since phases 6/14/15/18 already trained and reported results against this
exact block; changing it now would silently invalidate those numbers):
- `mlp_ratio=1.0` here vs. the FasterNet paper's typical expansion ratio of 2
  for the pointwise MLP — ours is a narrower, cheaper block.
- Activation is `SiLU` (matching the rest of the YOLO network) vs. the
  paper's `GELU`.
- Neither deviation is verified against every model size in the source paper
  (its per-scale hyperparameters aren't all reproduced here) — worth a direct
  check against the paper before quoting exact numbers in a methods section.
These were reasonable pragmatic choices for a nano-parameter budget /
YOLO-convention consistency, but they mean this graft is "FasterNet-inspired,"
not a parameter-matched reproduction of the paper's block.

## DWS — `yolo11n-dws-obb.yaml`, `yolov8n-dws-obb.yaml`

**Naming correction:** the yaml header comment calls this a "MobileNet-style
depthwise-separable" graft. That's imprecise. A true depthwise-separable
conv (Howard et al., *MobileNets: Efficient Convolutional Neural Networks for
Mobile Vision Applications*, arXiv 2017 [`howard2017mobilenets`]) is **two**
ops: a depthwise conv (groups = c_in, channel count unchanged) followed by a
separate 1x1 pointwise conv (groups = 1) that does the channel mixing/
expansion. Ultralytics' `DWConv(c1, c2, k, s)` — what this yaml actually
uses — is a **single** grouped conv with `groups = gcd(c1, c2)`
(`ultralytics/nn/modules/conv.py`), which changes channel count and does
spatial filtering in one step. It's a genuine, cheaper grouped convolution
and a reasonable lightweight-backbone lever, but it is not literally
depthwise-separable in the MobileNet sense — there's no standalone pointwise
mixing conv after it anywhere in this backbone.
- The yaml's own parenthetical, "(grouped depthwise)", is the accurate part;
  the "MobileNet-style depthwise-separable" framing in the same comment
  overclaims.
- Not fixing the architecture (it already trained — `dws_deg15` /
  `v8dws_deg15` results in `results/eval_eduardo_results.json` — changing the
  block now would invalidate those numbers without a rerun). The yaml header
  comments have been corrected to describe what the block actually is.
- If this is written up formally, describe it as a "grouped-convolution
  backbone graft (approximating MobileNet-style depthwise-separable convs)"
  rather than claiming full depthwise-separable design, and cite
  [`howard2017mobilenets`] only as the *inspiration*, not as an exact
  architectural match.

## P2 head — `yolo11n-p2-obb.yaml`

Not from a paper — the header comment says it "follows Ultralytics' own
`yolov8-p2.yaml`" example config (a real file Ultralytics ships for
small-object detection), adapted to v11 + OBB. No bib entry needed.
