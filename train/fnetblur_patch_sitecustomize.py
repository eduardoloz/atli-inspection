"""Combined FasterNet-graft + blur-aug hook — deploy as
~/atli/fnetblur_patch/sitecustomize.py.

Python only imports ONE `sitecustomize` module (first hit on PYTHONPATH), so
the fnet condition can't just concatenate ~/atli/modpatch and ~/atli/blurpatch
on PYTHONPATH the way ghost/dws do — both patches have to live in one file.
This is the union of modpatch/sitecustomize.py (FNET=1: rebind
ultralytics.nn.tasks.C3Ghost -> FasterNet PConv block) and
blurpatch/sitecustomize.py (BLUR_AUG=1: MotionBlur/GaussianBlur into the
Albumentations pipeline), each independently gated by its own env var so this
file is safe to use for fnet_blurmix640 (both set) without affecting any
other condition.
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
        print("[fnetblur_patch] fnet active: C3Ghost -> C3Faster (FasterNet PConv blocks)", file=sys.stderr)
    except Exception as e:  # never break the interpreter
        print("[fnetblur_patch] fnet FAILED:", e, file=sys.stderr)

if os.environ.get("BLUR_AUG") == "1":
    try:
        import albumentations as A
        from ultralytics.data import augment as _aug
        _orig = _aug.Albumentations.__init__

        def _patched(self, p=1.0, transforms=None):
            T = [A.MotionBlur(blur_limit=(3, 9), p=0.3),
                 A.GaussianBlur(blur_limit=(3, 7), p=0.2)]
            _orig(self, p=p, transforms=T)

        _aug.Albumentations.__init__ = _patched
        print("[fnetblur_patch] blur active: MotionBlur(3-9,p=0.3) + GaussianBlur(3-7,p=0.2)", file=sys.stderr)
    except Exception as e:
        print("[fnetblur_patch] blur FAILED:", e, file=sys.stderr)
