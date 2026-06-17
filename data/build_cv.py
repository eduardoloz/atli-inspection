import shutil, yaml, numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
ROOT=Path.home()/"atli"; POOL=ROOT/"Merged_Dataset"; OUT=ROOT/"Merged_CV"
NAMES=["Birdnest","Broken_Insulator","Defective_Damper","Flashover_Insulator","Normal_Damper","Normal_Insulators","Self-Exploded_Insulator"]; DD=2
rows=[]; tg=[]
for img in sorted((POOL/"images").glob("*")):
    lbl=POOL/"labels"/(img.stem+".txt")
    if not lbl.exists(): continue
    v=np.zeros(7,int)
    for l in lbl.read_text().splitlines():
        if l.strip():
            c=int(l.split()[0])
            if 0<=c<7: v[c]=1
    if v.sum(): rows.append((img,lbl)); tg.append(v)
tg=np.array(tg); idx=np.arange(len(rows))
mskf=MultilabelStratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for k,(tr,te) in enumerate(mskf.split(idx,tg)):
    f=OUT/f"fold{k}"
    for s in ("test/images","test/labels","train/images","train/labels","trainos/images","trainos/labels"):
        shutil.rmtree(f/s,ignore_errors=True); (f/s).mkdir(parents=True,exist_ok=True)
    for i in te:
        im,lb=rows[i]; shutil.copy2(im,f/"test/images"/im.name); shutil.copy2(lb,f/"test/labels"/lb.name)
    ddc=0
    for i in tr:
        im,lb=rows[i]
        for d in ("train","trainos"): shutil.copy2(im,f/d/"images"/im.name); shutil.copy2(lb,f/d/"labels"/lb.name)
        if any(int(l.split()[0])==DD for l in lb.read_text().splitlines() if l.strip()):
            ddc+=1
            for j in (1,2):
                shutil.copy2(im,f/"trainos/images"/f"os{j}_{im.name}"); shutil.copy2(lb,f/"trainos/labels"/f"os{j}_{lb.name}")
    for nm,td in (("base","train"),("champ","trainos")):
        (f/f"{nm}.yaml").write_text(yaml.safe_dump({"path":str(f),"train":f"{td}/images","val":"test/images","test":"test/images","nc":7,"names":NAMES},sort_keys=False))
    print(f"fold{k}: train={len(tr)} test={len(te)} dd_imgs_oversampled={ddc}")
print("CV_BUILD_DONE")
