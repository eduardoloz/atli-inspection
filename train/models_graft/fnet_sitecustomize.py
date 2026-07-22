"""FasterNet graft hook — deploy as ~/atli/modpatch/sitecustomize.py.

When FNET=1, rebinds ultralytics.nn.tasks.C3Ghost to a FasterNet-style block
(PConv + MLP residual, Chen et al. "Run, Don't Walk", CVPR 2023 / FPFS-YOLO).
Rebinding an EXISTING global is required: parse_model recognizes modules by
class identity inside literal sets, so a brand-new class name would not get
the standard (c1, c2, n) channel/depth treatment. Only yolo11n-fnet-obb.yaml
runs may set FNET=1 — a real Ghost yaml under FNET=1 would silently become
FasterNet.
"""
import os
import sys

if os.environ.get("FNET") == "1":
    try:
        import torch
        import torch.nn as nn
        from ultralytics.nn.modules import Conv
        import ultralytics.nn.tasks as T
        import ultralytics.nn.modules as M
        import ultralytics.nn.modules.block as B

        class FasterBlock(nn.Module):
            """PConv (3x3 on 1/4 of channels) + inverted-bottleneck MLP, residual."""

            def __init__(self, dim, n_div=4, mlp_ratio=1.0):
                super().__init__()
                self.dc = dim // n_div
                self.pconv = nn.Conv2d(self.dc, self.dc, 3, 1, 1, bias=False)
                h = int(dim * mlp_ratio)
                self.mlp = nn.Sequential(
                    nn.Conv2d(dim, h, 1, bias=False), nn.BatchNorm2d(h), nn.SiLU(),
                    nn.Conv2d(h, dim, 1, bias=False))

            def forward(self, x):
                y = torch.cat((self.pconv(x[:, :self.dc]), x[:, self.dc:]), 1)
                return x + self.mlp(y)

        class C3Faster(nn.Module):
            """Stage: optional 1x1 channel projection, then n FasterBlocks."""

            def __init__(self, c1, c2, n=1, *args):
                super().__init__()
                self.cv1 = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
                self.m = nn.Sequential(*(FasterBlock(c2) for _ in range(max(n, 1))))

            def forward(self, x):
                return self.m(self.cv1(x))

        for mod in (T, M, B):
            setattr(mod, "C3Ghost", C3Faster)
        # stderr ONLY: `conda activate` evals python stdout, a stdout print breaks it
        print("[fnetpatch] active: C3Ghost -> C3Faster (FasterNet PConv blocks)", file=sys.stderr)
    except Exception as e:  # never break the interpreter
        print("[fnetpatch] FAILED:", e, file=sys.stderr)
