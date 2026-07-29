#!/usr/bin/env python3
"""Edge-deployment flowchart (poster companion to pipeline_flowchart_compact).

Same visual language as Figure 2 (grey rounded boxes, dashed group border):
trained checkpoint -> FP32/FP16 paths -> Jetson Orin Nano batch-1 bench ->
fps x mAP at four inference resolutions. Backbone-graft branch shown crossed
out (benchmarked, no frontier gain).

Three variants:
  edge_deploy_flowchart.{svg,png,pdf}          wide horizontal strip
  edge_deploy_flowchart_compact.{svg,png,pdf}  squished horizontal, note under graft box
  edge_deploy_flowchart_vertical.{svg,png,pdf} vertical (Figure-2-shaped)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

BOX_FC, BOX_EC = "#f4f3f1", "#8a8a8a"
ARROW = "#555555"
SUB = "#555555"
FAIL = "#b0413e"
FS_TITLE, FS_SUB, FS_LABEL = 10.5, 9.0, 8.5

TRT_SUBS = ["ONNX $\\rightarrow$ engine,", "built on-device"]
FAIL_NOTE = "$\\times$ no frontier gain (Fig. 10)"


class Canvas:
    def __init__(self, figsize, xspan, yspan):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, xspan)
        self.ax.set_ylim(0, yspan)
        self.ax.axis("off")
        # keep rounded corners visually circular regardless of canvas shape
        self.aspect = (xspan / yspan) * (figsize[1] / figsize[0])

    def box(self, x0, y0, x1, y1, title, subs, ls="-",
            line_h=2.6, title_gap=3.0):
        self.ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            fc=BOX_FC, ec=BOX_EC, lw=1.1, ls=ls,
            mutation_aspect=self.aspect))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        n = len(subs)
        ty = cy if n == 0 else \
            cy + (1.6 if n == 1 else 2.4 if n == 2 else 2.9) * line_h / 2.6
        self.ax.text(cx, ty, title, ha="center", va="center",
                     fontsize=FS_TITLE, fontweight="bold", color="#222")
        for i, s in enumerate(subs):
            self.ax.text(cx, ty - title_gap - line_h * i, s, ha="center",
                         va="center", fontsize=FS_SUB, color=SUB)
        return (x0, y0, x1, y1)

    def arrow(self, p0, p1, ls="-"):
        self.ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=13,
            color=ARROW, lw=1.3, ls=ls, shrinkA=0, shrinkB=0))

    def note(self, x, y, text, color=SUB, ha="center", bold=False):
        self.ax.text(x, y, text, ha=ha, va="center", fontsize=FS_LABEL,
                     color=color, style="italic" if color == SUB else "normal",
                     fontweight="bold" if bold else "normal")

    def save(self, stem):
        self.fig.tight_layout(pad=0.3)
        out = Path(__file__).resolve().parent
        for ext in ("svg", "png", "pdf"):
            self.fig.savefig(out / f"{stem}.{ext}",
                             dpi=300 if ext == "png" else None,
                             facecolor="white", bbox_inches="tight",
                             pad_inches=0.12)
        plt.close(self.fig)
        print("wrote", out / f"{stem}.{{svg,png,pdf}}")


def horizontal(stem, squish):
    if squish:
        c = Canvas((10.6, 3.4), 82, 28)
        ax_, bx0, bx1, cx0, cx1, dx0, dx1 = 1.5, 19.5, 34.5, 39.5, 55.5, 61.5, 77.5
    else:
        c = Canvas((12.6, 3.1), 100, 26)
        ax_, bx0, bx1, cx0, cx1, dx0, dx1 = 1.5, 21, 36, 42.5, 58.5, 64.5, 80.5
    ytop = 28 if squish else 26
    ymid = ytop / 2 + (2.5 if squish else 0.25)  # main row center
    A = c.box(ax_, ymid - 4.25, ax_ + 13, ymid + 4.25, "Trained model",
              ["best checkpoint", "(Fig. 2 pipeline)"])
    if squish:
        B1 = c.box(bx0, ymid + 4.5, bx1, ymid + 10, "TensorRT FP16", [])
        B2 = c.box(bx0, ymid - 10, bx1, ymid - 4.5, "PyTorch FP32", [])
    else:
        B1 = c.box(bx0, ymid + 3.25, bx1, ymid + 11.25, "TensorRT FP16",
                   TRT_SUBS)
        B2 = c.box(bx0, ymid - 11.75, bx1, ymid - 3.75, "PyTorch FP32",
                   ["no export"])
    C = c.box(cx0, ymid - 4.25, cx1, ymid + 4.25, "Jetson Orin Nano",
              ["batch-1 end-to-end", "(pre + forward + NMS)"])
    D = c.box(dx0, ymid - 4.25, dx1, ymid + 4.25, "Measure fps × mAP",
              ["infer @ 640 / 768 / 1024 / 1280", "one model, no retraining"])
    c.arrow((A[2], ymid + 2), (B1[0], ymid + 7))
    c.arrow((A[2], ymid - 2), (B2[0], ymid - 7))
    c.arrow((B1[2], ymid + 7), (C[0], ymid + 2))
    c.arrow((B2[2], ymid - 7), (C[0], ymid - 2))
    c.arrow((C[2], ymid), (D[0], ymid))
    c.note((C[0] + C[2]) / 2, C[1] - 1.6, "max power, pinned clocks")
    gx0, gx1 = dx0, dx1
    gy1 = ymid - 7.5
    G = c.box(gx0, gy1 - 5.5, gx1, gy1, "Backbone grafts",
              ["Ghost / DWS / FasterNet"], ls=(0, (4, 3)))
    c.arrow(((G[0] + G[2]) / 2, G[3]), ((D[0] + D[2]) / 2, D[1]),
            ls=(0, (4, 3)))
    if squish:
        c.note((G[0] + G[2]) / 2, G[1] - 1.8, FAIL_NOTE, color=FAIL, bold=True)
    else:
        c.note(G[2] + 1.2, (G[1] + G[3]) / 2 + 0.2, FAIL_NOTE, color=FAIL,
               ha="left", bold=True)
    c.save(stem)


def vertical(stem):
    c = Canvas((5.9, 9.0), 56, 100)
    kw = dict(line_h=3.6, title_gap=4.2)
    A = c.box(16, 86, 40, 97, "Trained model",
              ["best checkpoint", "(Fig. 2 pipeline)"], **kw)
    B1 = c.box(1.5, 63, 26.5, 77, "TensorRT FP16", TRT_SUBS, **kw)
    B2 = c.box(29.5, 63, 54.5, 77, "PyTorch FP32", ["no export"], **kw)
    C = c.box(16, 40, 40, 54, "Jetson Orin Nano",
              ["batch-1 end-to-end", "(pre + forward + NMS)"], **kw)
    D = c.box(14.5, 14, 41.5, 31, "Measure fps × mAP",
              ["infer @ 640 / 768 /", "1024 / 1280", "one model, no retraining"],
              **kw)
    G = c.box(16, 0.5, 40, 9.5, "Backbone grafts",
              ["Ghost / DWS / FasterNet"], ls=(0, (4, 3)), **kw)
    c.arrow((22, A[1]), ((B1[0] + B1[2]) / 2, B1[3]))
    c.arrow((34, A[1]), ((B2[0] + B2[2]) / 2, B2[3]))
    c.arrow(((B1[0] + B1[2]) / 2, B1[1]), (22, C[3]))
    c.arrow(((B2[0] + B2[2]) / 2, B2[1]), (34, C[3]))
    c.arrow((28, C[1]), (28, D[3]))
    c.note(29.5, (C[1] + D[3]) / 2, "max power, pinned clocks", ha="left")
    c.arrow((28, G[3]), (28, D[1]), ls=(0, (4, 3)))
    c.note(29.5, (G[3] + D[1]) / 2, FAIL_NOTE, color=FAIL, ha="left",
           bold=True)
    c.save(stem)


horizontal("edge_deploy_flowchart", squish=False)
horizontal("edge_deploy_flowchart_compact", squish=True)
vertical("edge_deploy_flowchart_vertical")
