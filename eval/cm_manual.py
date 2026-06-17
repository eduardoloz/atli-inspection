from ultralytics import YOLO
from pathlib import Path
import numpy as np
R=Path.home()/"atli"; TEST=R/"Merged_Dataset_Stratified"/"test"
def iou(a,b):
    ax0,ay0,ax1,ay1=a[0]-a[2]/2,a[1]-a[3]/2,a[0]+a[2]/2,a[1]+a[3]/2
    bx0,by0,bx1,by1=b[0]-b[2]/2,b[1]-b[3]/2,b[0]+b[2]/2,b[1]+b[3]/2
    iw=max(0,min(ax1,bx1)-max(ax0,bx0)); ih=max(0,min(ay1,by1)-max(ay0,by0)); inter=iw*ih
    ua=a[2]*a[3]+b[2]*b[3]-inter; return inter/ua if ua>0 else 0
def confmat(run,imz):
    m=YOLO(str(R/f"runs/{run}_s2/weights/best.pt"))
    M=np.zeros((8,8),int)  # M[true, pred]; index 7 = background
    for img in sorted((TEST/"images").glob("*")):
        lbl=TEST/"labels"/(img.stem+".txt")
        gts=[(int(p[0]),tuple(map(float,p[1:5]))) for p in (l.split() for l in lbl.read_text().splitlines()) if len(p)>=5] if lbl.exists() else []
        r=m.predict(str(img),conf=0.25,iou=0.45,imgsz=imz,device=0,verbose=False)[0]
        preds=[(int(c),tuple(b)) for b,c in zip(r.boxes.xywhn.tolist(),r.boxes.cls.tolist())]
        used=set()
        for gc,gb in gts:
            best=0.5; bi=-1
            for i,(pc,pb) in enumerate(preds):
                if i in used: continue
                v=iou(gb,pb)
                if v>=best: best=v; bi=i
            if bi>=0: used.add(bi); M[gc,preds[bi][0]]+=1
            else: M[gc,7]+=1
        for i,(pc,pb) in enumerate(preds):
            if i not in used: M[7,pc]+=1
    return M
DI=[1,3,6]; NI=5; DD=2; ND=4; BG=7
b=confmat("B_v11_150p100",640); n=confmat("HROaug_v11",1280)
def L(name,bv,nv): print(f"  {name:46s} before={bv:3d}  now={nv:3d}  delta={nv-bv:+d}")
print("KEYCONF")
L("DefDamper(true) -> called Normal_Damper", b[DD,ND], n[DD,ND])
L("DefDamper(true) -> MISSED (background)",  b[DD,BG], n[DD,BG])
L("Normal_Damper(true) -> called DefDamper", b[ND,DD], n[ND,DD])
L("DefInsul(true) -> called Normal_Insul",   sum(b[c,NI] for c in DI), sum(n[c,NI] for c in DI))
L("Normal_Insul(true) -> called DefInsul",   sum(b[NI,c] for c in DI), sum(n[NI,c] for c in DI))
L("Normal_Insul(true) -> MISSED (background)",b[NI,BG], n[NI,BG])
