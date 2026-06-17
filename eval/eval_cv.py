from ultralytics import YOLO
from pathlib import Path
import numpy as np, sys
R=Path.home()/"atli"
def ev(run,imz,yaml):
    w=R/f"runs/{run}_s2/weights/best.pt"
    if not w.exists(): return None
    v=YOLO(str(w)).val(data=yaml,split="test",imgsz=imz,batch=8,device=0,verbose=False,plots=False)
    nm={v.names[c]:i for i,c in enumerate(v.box.ap_class_index)}
    g=lambda c,a: float(a[nm[c]]) if c in nm else float("nan")
    return (float(v.box.map50),g("Defective_Damper",v.box.ap50),g("Defective_Damper",v.box.r),g("Normal_Damper",v.box.ap50))
for cond,imz,yk in [("base",640,"base"),("champ",1280,"champ")]:
    res=[]
    for k in range(5):
        y=str(R/f"Merged_CV/fold{k}/{yk}.yaml")
        r=ev(f"CV{cond}_f{k}",imz,y)
        if r: res.append(r); print(f"CV{cond}_f{k}: mAP={r[0]:.3f} DD={r[1]:.3f} DDr={r[2]:.3f} ND={r[3]:.3f}")
    if res:
        a=np.array(res)
        print(f"  >>> {cond} CV mean ({len(res)} folds): mAP={a[:,0].mean():.3f} DD={a[:,1].mean():.3f}±{a[:,1].std():.3f} DDr={a[:,2].mean():.3f} ND={a[:,3].mean():.3f}\n")
