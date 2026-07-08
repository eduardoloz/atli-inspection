#!/usr/bin/env python3
"""Structured channel pruning for the champion YOLOv11n with Torch-Pruning.

Approach (working precedent: heyongxin233/YOLO-Pruning-RKNN, which prunes
YOLOv3..v12 with torch-pruning by ignoring Detect + Attention modules;
Torch-Pruning itself ships only a yolov8 example and has an open issue #454
about YOLO11 hangs — the ignore-list below is what makes v11 work):

  - Build a DependencyGraph over the fused-free training model.
  - ignored_layers = every `Detect` (head + DFL) and `Attention` (inside
    C2PSA/PSA blocks) module. Their *surrounding* convs still get pruned via
    dependency propagation; the attention qkv/proj and head convs keep their
    channel counts, which sidesteps the C2PSA head-dim constraint.
  - GroupNormImportance (L2), global ranking, iterative steps until the
    MACs target is met, round_to=8 for kernel-friendly channel counts.

Latency reality check: this is *structured/dense* pruning — the only kind
that speeds up GPUs without sparsity hardware. Jetson Nano (Maxwell) has no
sparse tensor cores and no DP4A, so unstructured/2:4 sparsity is worthless
there; channel pruning shrinks dense GEMMs and translates ~proportionally to
latency ONLY on compute-bound devices (the Nano). On big GPUs at batch 1 it
does nothing (measured: LAMP-1.5x cut 35% FLOPs, bs1 fps 126->122 on a
workstation GPU) — do NOT judge the speedup on the RTX 6000; judge it by
GFLOPs and the on-Nano trtexec benchmark.

Two-phase usage:
  # phase 1: prune only (CPU is fine), inspect MACs/params, save pruned ckpt
  python prune_v11n.py --weights best.pt --target-flops-ratio 0.59 \
      --imgsz 768 --out pruned_v11n.pt

  # phase 2: finetune the pruned model (GPU)
  python prune_v11n.py --weights best.pt --target-flops-ratio 0.59 \
      --imgsz 768 --out pruned_v11n.pt \
      --finetune --data ATLI_noCPLID_OS3/data.yaml --epochs 100 --device 0

--target-flops-ratio 0.59 means "keep 59% of MACs" (= 1.7x speedup target).
Finetune uses the champion stage-2 recipe (SGD lr0=0.00334, lrf=0.1535).
"""
import argparse
import copy
from pathlib import Path

import torch


def get_ignored_layers(model):
    """Detect heads + C2PSA attention blocks must not be pruned (v11 recipe)."""
    ignored = []
    for m in model.modules():
        name = type(m).__name__
        if name in ("Detect", "Attention", "AAttn", "A2C2f"):
            ignored.append(m)
    return ignored


def prune_model(yolo, target_ratio, imgsz, max_steps=20, importance="l2"):
    import torch_pruning as tp

    model = yolo.model  # DetectionModel (unfused)
    model.eval().float()
    for p in model.parameters():
        p.requires_grad_(True)  # tp needs grads enabled to trace some ops

    example = torch.randn(1, 3, imgsz, imgsz)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)

    # torch-pruning >=1.6 API names
    imp = (tp.importance.LAMPImportance() if importance == "lamp"
           else tp.importance.GroupMagnitudeImportance(p=2))

    def build_pruner(ignored):
        return tp.pruner.GroupNormPruner(
            model, example, importance=imp,
            iterative_steps=max_steps,
            pruning_ratio=1.0,      # upper bound; we stop early on MACs target
            global_pruning=True,
            ignored_layers=ignored, round_to=8,
        )

    # Auto-quarantine: torch-pruning 1.6.0 mis-traces the chunk/split in a
    # couple of C3k2 blocks on recent ultralytics (index-out-of-bounds when
    # scoring the group). Scan all groups; add the root module of any failing
    # group to ignored_layers and rebuild, until every group scores cleanly.
    # This sacrifices a tiny amount of prunable surface (2 groups on v11n)
    # instead of forking ultralytics.
    ignored = list(get_ignored_layers(model))
    for _ in range(10):
        pruner = build_pruner(ignored)
        bad_roots = []
        for group in pruner.DG.get_all_groups(
                ignored_layers=pruner.ignored_layers,
                root_module_types=pruner.root_module_types):
            try:
                imp(group)
            except IndexError:
                bad_roots.append(group[0].dep.target.module)
        if not bad_roots:
            break
        for m in bad_roots:
            print(f"quarantining incompatible group root: {m}")
        ignored.extend(bad_roots)
    else:
        raise RuntimeError("could not stabilize dependency groups")

    macs, params = base_macs, base_params
    for step in range(max_steps):
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example)
        print(f"step {step + 1}: MACs {macs / 1e9:.2f}G ({macs / base_macs:.1%}), "
              f"params {params / 1e6:.2f}M ({params / base_params:.1%})")
        if macs / base_macs <= target_ratio:
            break

    # sanity forward
    with torch.no_grad():
        model(example)
    print(f"\npruned: {base_macs / 1e9:.2f}G -> {macs / 1e9:.2f}G MACs "
          f"({macs / base_macs:.1%} kept, {base_macs / macs:.2f}x FLOPs speedup), "
          f"{base_params / 1e6:.2f}M -> {params / 1e6:.2f}M params")
    return model


def save_pruned(yolo, out):
    """Save a YOLO-loadable checkpoint with the full (pruned) module pickled."""
    m = copy.deepcopy(yolo.model).half()
    ckpt = {"model": m, "train_args": {}, "date": None, "version": None}
    torch.save(ckpt, out)
    print(f"saved pruned checkpoint: {out}")


def finetune(pruned_model, args):
    """Fine-tune in-process: BaseTrainer.setup_model() returns early when
    self.model is already an nn.Module, so we drive DetectionTrainer directly
    (same pattern as Torch-Pruning's yolov8 example train_v2)."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    overrides = dict(
        model="yolo11n.yaml",  # placeholder; replaced by the pruned module below
        data=args.data, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device, optimizer="SGD",
        lr0=0.00334, lrf=0.1535, scale=0.9, seed=args.seed,
        project=args.project, name=args.name or Path(args.out).stem + "_ft",
        val=True, amp=True, exist_ok=True,
    )
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = pruned_model.float()
    trainer.train()
    print(f"finetune done, best weights: {trainer.best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--target-flops-ratio", type=float, default=0.59,
                    help="fraction of MACs to KEEP (0.59 = 1.7x speedup)")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--importance", choices=["l2", "lamp"], default="l2",
                    help="channel importance: group L2 (precedent default) or LAMP")
    ap.add_argument("--out", default="pruned_v11n.pt")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs_prune")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    yolo = YOLO(args.weights)
    model = prune_model(yolo, args.target_flops_ratio, args.imgsz,
                        importance=args.importance)
    save_pruned(yolo, args.out)

    # reload sanity check: the pickled pruned model must load through YOLO().
    # NB: ultralytics select_device('cpu') clobbers CUDA_VISIBLE_DEVICES for
    # the whole process, which would blind the later GPU finetune — snapshot
    # and restore it around the CPU predict.
    import os
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    check = YOLO(args.out)
    import numpy as np
    check.predict((np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype("uint8"),
                  imgsz=args.imgsz, device="cpu", verbose=False)
    if cvd is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
    print("reload + predict sanity check: OK")

    if args.finetune:
        assert args.data, "--finetune requires --data"
        finetune(model, args)


if __name__ == "__main__":
    main()
