# Structured Pruning Plan — YOLOv11n champion → ~1.7× on Jetson Nano
**Round 2 deliverable · branch `opt/jetson-nano` · 2026-07-08**

> **STATUS (2026-07-09): EXECUTED — NEGATIVE RESULT.** The 21-run grid
> ({1.25, 1.5, 1.75, 2.0}× × 3 seeds × 2 LR schedules) missed the acceptance
> gate at every ratio; pruning is off the Nano recipe. See
> `prune_grid_results.md` for the table, evidence, and the booked finding.
> The mechanics below (v11n + torch-pruning recipe, quarantine fix, runbook)
> remain valid and reusable.

## Verdict (headline)

**Feasible, and the pipeline is already functionally validated.** `optimization/jetson/prune_v11n.py` pruned the champion to the exact target this round (CPU-only, no training): **4.60 → 2.63 GMACs @768 (57.3% kept = 1.75× FLOPs speedup), 2.59M → 1.17M params (−55%)**, survived reload through `YOLO()`, ran inference, and exported to a clean ONNX (4.8 MB, opset 12: `~/atli/export_jetson/pruned_v11n_59pct.{pt,onnx}` on the UNLV server). What remains is the fine-tune (GPU, next round) and accuracy verification.

Arithmetic to the goal: champion @768 = 9.2 GFLOPs → ~17–18 fps inference-only on Nano (at the ~160 GFLOP/s effective rate derived in `optimization/jetson_nano_research.md` §4). Pruned to 5.3 GFLOPs → **~30 fps inference-only, ~20–24 fps end-to-end**; the remaining e2e gap is the DeepStream/pipeline work, not the model.

## 1. Does structured pruning give real TensorRT FP16 wins on Maxwell? Yes — with one big caveat

- **Unstructured sparsity is worthless here, categorically.** No GPU executes unstructured sparse conv kernels usefully; NVIDIA's sparsity acceleration (2:4 structured sparsity) starts with **Ampere** sparse tensor cores. Maxwell has no tensor cores at all and no DP4A ([NVIDIA DP4A blog](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/)). Anything that leaves weights zeroed in place changes nothing about latency.
- **Channel (structured/dense) pruning shrinks the dense GEMMs TensorRT actually launches** — "CONV layers after structured pruning transform to a full matrix multiplication with reduced matrix size" ([survey](https://arxiv.org/pdf/1907.02124)). Measured GPU-latency evidence: NVIDIA's HALP achieves **1.6–1.9× measured GPU latency reduction** at preserved accuracy with latency-aware structured pruning ([HALP, arXiv:2110.10811](https://arxiv.org/pdf/2110.10811)); the YOLOv8m aerial-detection compression framework measured **26 → 45 fps (+73%) after 73.5% param pruning, 68 fps after TensorRT** on VisDrone ([arXiv:2509.12918](https://arxiv.org/abs/2509.12918)).
- **The caveat — where the speedup does and does not appear:** FLOP cuts convert to batch-1 latency only on **compute-bound** devices. The LAMP PCB study is the cleanest warning: cutting FLOPs 35% left bs1 fps essentially unchanged on a workstation GPU (126.2 → 122.3) while bs32 throughput rose 18% (1032 → 1219) ([arXiv:2507.17176](https://arxiv.org/pdf/2507.17176), Table 1). Our own round-1 measurement showed the same effect (champion latency flat 640→1280 on the RTX 6000). The Jetson Nano at 768 input is firmly compute-bound (~57 ms of pure conv work vs ~0.5 ms launch overhead), so the FLOPs ratio should translate ≈ proportionally there. **Protocol consequence: never judge this pruning by RTX 6000 latency — judge by GMACs (proxy) and the on-Nano trtexec benchmark (truth).**
- Alignment note: pruned channel counts are forced to multiples of 8 (`round_to=8`). On tensor-core GPUs multiples of 8/16 matter a lot; Maxwell FP16 is less picky, but rounding costs almost nothing and keeps the model fast on any future hardware (Orin).

## 2. Expected speedup-vs-mAP curve (published evidence)

| Source | Base | Ratio | Params | FLOPs | Accuracy after finetune | bs1 speed |
|---|---|---|---|---|---|---|
| [LAMP PCB, Tbl 1](https://arxiv.org/pdf/2507.17176) | 1.38M / 3.7G lightweight YOLOv8 derivative (same scale as v11n) | **1.5×** | −51% | 3.7→2.4G | mAP50 **+0.001** (0.9920→0.9932), mAP50-90 +0.088 | 126→122 fps (latency-bound GPU) |
| same | | **2×** | −71% | →1.8G | mAP50 **−0.022** (→0.9701) | 126 fps |
| same | | **2.5×** | −80% | →1.4G | mAP50 −0.031 | 130 fps |
| same | | **3×** | −85% | →1.2G | mAP50 **−0.043**, recall −0.058 (over-pruned) | 133 fps |
| [MCP-YOLO (UAV insulators)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656040/) | 13.79M YOLOv11-based | speed-up param **1.55** (Group SLIM, 500 sparse rounds, reg 5e-4) | −37.3% | n/s | mAP50 **+0.012** (0.909→0.921) | 161→250 fps |
| [YOLOv8m + CWD distill](https://arxiv.org/abs/2509.12918) | 25.85M / 49.6G | **−73.5% params** | →6.85M | →13.3G | AP50 **−2.7 pts** (VisDrone; recovered via channel-wise distillation) | 26→45 fps (+TRT: 68) |

**Forecast for our 1.7–1.75× target (interpolating LAMP 1.5×→2×):** mAP50 delta between **+0.00 and −0.015** after a proper fine-tune. Honest caveats: (a) both "pruning improved mAP" results are on easier/over-parameterized setups (PCB P/R≈0.99; 13.8M-param model); v11n at 2.6M is already lean, so we sit closer to the loss side of the band; (b) our fragile class is Defective_Damper (±0.07 seed noise) — a mean-mAP-neutral prune can still hide a DD regression, so acceptance must be per-class and multi-seed. **Acceptance gate: 3-seed mean mAP50 ≥ 0.745 and DD AP ≥ 0.59 at 768 (the orchestrator's accuracy floors), else fall back to the 1.5× ratio.**

## 3. Why v11n pruning is non-trivial, and the working recipe

- **Torch-Pruning has no YOLO11 example** (examples dir: yolov5/v7/v8 only) and an **open issue of YOLO11 hanging during pruner construction** ([VainF/Torch-Pruning#454](https://github.com/VainF/Torch-Pruning/issues/454)). The v8 example itself needed a C2f forward rewrite to survive the `chunk()` op ([#147](https://github.com/VainF/Torch-Pruning/issues/147)).
- **Working precedent:** [heyongxin233/YOLO-Pruning-RKNN](https://github.com/heyongxin233/YOLO-Pruning-RKNN) prunes YOLOv3→v12 with torch-pruning inside a patched ultralytics trainer. Verified from its source (`ultralytics/engine/trainer.py`): the entire v11 fix is `ignored_layers = [m for m in model.modules() if isinstance(m, (Detect, Attention))]` with a `MagnitudePruner` — i.e. **don't touch the detect head or the C2PSA attention internals**; everything around them is pruned via dependency propagation, and the attention head-dim constraint never triggers.
- **Our addition (needed on ultralytics 8.4.60 + torch-pruning 1.6.0):** two C3k2 groups still mis-trace (`IndexError: index 384 out of bounds for 256` — TP fuses two different split ops in the graph). `prune_v11n.py` handles this generically with an **auto-quarantine loop**: score every dependency group once; any group that raises gets its root conv added to `ignored_layers`; rebuild and rescan until clean. Cost on v11n: 3 quarantined convs (a 16→32 and two 32→64 3×3s — early layers, tiny share of FLOPs), after which the pruner ran 11 clean global iterative steps to target.
- Importance criterion: `GroupMagnitudeImportance(p=2)` default (the precedent's choice); `--importance lamp` switches to `LAMPImportance` (the criterion with the best published nano-scale results above) for the ratio ablation.

## 4. Runbook (next round, GPU)

```bash
# Phase 1 — prune (CPU, seconds; already validated this round)
~/atli/env_jetson/bin/python prune_v11n.py \
  --weights ~/atli/runs/HROaugnc_v11_s0_s2/weights/best.pt \
  --target-flops-ratio 0.59 --imgsz 768 \
  --out ~/atli/export_jetson/pruned_v11n_59pct.pt

# Phase 2 — fine-tune (1 GPU, ~6-8 h at 768) — champion stage-2 recipe
# (SGD lr0=0.00334 lrf=0.1535 scale=0.9, 100 ep). Driver included:
... prune_v11n.py ... --finetune --data ~/atli/ATLI_noCPLID_OS3/data.yaml \
  --epochs 100 --batch 16 --device <free-gpu>

# Phase 3 — eval + export
python accuracy_check.py --pt <champ> --exported <pruned_ft.pt> --imgsz 768 ...
python export_champion.py --weights <pruned_ft.pt> --imgsz 768 ...
# Phase 4 — on-Nano trtexec benchmark (build_engine_nano.sh) = the real number
```

Grid (9 runs, ~2–3 GPU-days): {1.5×, 1.75×, 2×} ratios × 3 seeds at 768, all through `accuracy_check.py` per-class. Optional arm if recovery is poor: 25-epoch sparse pre-training with BN-scale L1 (Group-SLIM style, `reg=5e-4` per MCP-YOLO) before pruning — supported by `BNScalePruner`, one-flag change.

Environment note: all export/pruning tooling lives in the isolated venv `~/atli/env_jetson` (system-site-packages over the training env, torch-pruning added there). **Nothing gets pip-installed into `~/atli/env`.** The pruning fine-tune itself uses stock ultralytics from the shared env via env_jetson's interpreter.

## 5. Risks

| Risk | Mitigation |
|---|---|
| DD AP regression hidden by healthy mean mAP | per-class 3-seed acceptance gate (mAP≥0.745, DD≥0.59) |
| global_pruning over-thins one stage | fallback: per-layer uniform (`global_pruning=False`), matches RKNN precedent |
| C2PSA left unpruned caps max ratio (attention ≈ fixed cost) | fine below 2×; at ≥2.5× revisit with attention-dim-aware pruning |
| finetune driver (in-process DetectionTrainer with pre-built nn.Module) untested on GPU | 30-min smoke run (3 epochs) before launching the grid |
| pruned+finetuned model must re-export | already validated for the pruned-unfinetuned model; re-check post-finetune |
