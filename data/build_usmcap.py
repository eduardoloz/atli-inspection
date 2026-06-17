# Scale-matched BUT ratio-capped to ~1:1 (minimal dose) — final test of "minimal overtraining"
import shutil, yaml
from collections import Counter
from pathlib import Path
ROOT=Path.home()/"atli"; SRC,UNI=ROOT/"Merged_Dataset_Stratified",ROOT/"Universe_Pool"
NAMES=["Birdnest","Broken_Insulator","Defective_Damper","Flashover_Insulator","Normal_Damper","Normal_Insulators","Self-Exploded_Insulator"]
DD=2; THRESH=0.8; CAP=177
def dd_boxes(lbl):
    return [float(p[3])*float(p[4])*100 for p in (l.split() for l in lbl.read_text().splitlines()) if len(p)>=5 and int(p[0])==DD]
OUT=ROOT/"Merged_USMcap"
for sp in ("train","val","test"):
    for sub in ("images","labels"):
        shutil.rmtree(OUT/sp/sub,ignore_errors=True); (OUT/sp/sub).mkdir(parents=True,exist_ok=True)
        for f in (SRC/sp/sub).glob("*"): shutil.copy2(f,OUT/sp/sub/f.name)
# candidates = scale-matched images, sorted smallest-DD-first (most ATLI-like)
cands=[]
for lbl in (UNI/"labels").glob("*.txt"):
    b=dd_boxes(lbl)
    if b and max(b)<=THRESH: cands.append((min(b),lbl))
cands.sort()
added_dd=added=0
for _,lbl in cands:
    if added_dd>=CAP: break
    img=next((UNI/"images").glob(lbl.stem+".*"))
    shutil.copy2(img,OUT/"train"/"images"/img.name); shutil.copy2(lbl,OUT/"train"/"labels"/lbl.name)
    added_dd+=len(dd_boxes(lbl)); added+=1
(OUT/"merged_usmcap.yaml").write_text(yaml.safe_dump({"path":str(OUT),"train":"train/images","val":"val/images","test":"test/images","nc":7,"names":NAMES},sort_keys=False))
c=Counter()
for lbl in (OUT/"train"/"labels").glob("*.txt"):
    for l in lbl.read_text().splitlines():
        if l.strip(): c[int(l.split()[0])]+=1
print(f"USMcap: +{added} imgs, +{added_dd} DD -> train DefDamper {c[DD]}, NormDamper {c[4]}")
print("BUILD_USMCAP_DONE")
