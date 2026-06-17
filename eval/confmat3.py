from ultralytics import YOLO
from pathlib import Path
import numpy as np
R=Path.home()/"atli"; YAML=str(R/"Merged_Dataset_Stratified/merged_stratified.yaml")
N=["Birdnest","Broken_Ins","Def_Damper","Flashover","Norm_Damper","Norm_Ins","SelfExpl","bg"]
def cm(run,imz):
    m=YOLO(str(R/f"runs/{run}_s2/weights/best.pt"))
    m.val(data=YAML,split="test",imgsz=imz,batch=8,device="cpu",verbose=False,plots=False)
    return np.array(m.validator.confusion_matrix.matrix).astype(int)  # [pred, true]
DEF_INS=[1,3,6]; NI=5; DD=2; ND=4; BG=7
out=[]
Ms={}
for run,imz,tag in [("B_v11_150p100",640,"BEFORE"),("HROaug_v11",1280,"NOW")]:
    M=cm(run,imz); Ms[tag]=M
    out.append(f"\n== {tag} {run} ==  rows=pred cols=true")
    out.append("          "+" ".join(f"{n[:9]:>9}" for n in N))
    for i,row in enumerate(M): out.append(f"{N[i][:9]:>9} "+" ".join(f"{x:>9d}" for x in row))
b,n=Ms["BEFORE"],Ms["NOW"]
def line(name,bv,nv): out.append(f"  {name:42s} before={bv:3d}  now={nv:3d}  delta={nv-bv:+d}")
out.append("\n== KEY CONFUSIONS (before -> now) ==")
line("DefDamper(true) called Normal_Damper", b[ND,DD], n[ND,DD])
line("DefDamper(true) MISSED (->background)", b[BG,DD], n[BG,DD])
line("Normal_Damper(true) called DefDamper", b[DD,ND], n[DD,ND])
line("DefInsulators(true) called Normal_Ins", sum(b[NI,c] for c in DEF_INS), sum(n[NI,c] for c in DEF_INS))
line("Normal_Ins(true) called DefInsulator", sum(b[c,NI] for c in DEF_INS), sum(n[c,NI] for c in DEF_INS))
line("Normal_Ins(true) MISSED (->background)", b[BG,NI], n[BG,NI])
txt="\n".join(out); open(R/"universe_bench/confmat_full.txt","w").write(txt); print(txt)
