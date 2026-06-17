from ultralytics import YOLO
from pathlib import Path
import numpy as np, yaml, csv, re
from collections import defaultdict
R=Path.home()/"atli"; RUNS=R/"runs"; Y=str(R/"Merged_Dataset_Stratified/merged_stratified.yaml")
OUT=R/"universe_bench/agg_results.csv"
cache={}
if OUT.exists():
    for r in csv.DictReader(open(OUT)): cache[r["run"]]=r
rows=[]
pat=re.compile(r"^(HRO|Bv11|B_v11|OSaug|OS6aug|HROfrz|HRObird|USM)")
for d in sorted(RUNS.glob("*_s2")):
    run=d.name[:-3]
    if not pat.match(run) or not (d/"weights/best.pt").exists(): continue
    if run in cache: rows.append(cache[run]); continue
    imz=640
    if (d/"args.yaml").exists(): imz=int(yaml.safe_load(open(d/"args.yaml")).get("imgsz",640))
    try:
        v=YOLO(str(d/"weights/best.pt")).val(data=Y,split="test",imgsz=imz,batch=8,device=0,verbose=False,plots=False)
        nm={v.names[c]:i for i,c in enumerate(v.box.ap_class_index)}
        gg=lambda c,a: round(float(a[nm[c]]),4) if c in nm else float("nan")
        rows.append({"run":run,"imgsz":imz,"mAP":round(float(v.box.map50),4),"DD":gg("Defective_Damper",v.box.ap50),"DDr":gg("Defective_Damper",v.box.r),"ND":gg("Normal_Damper",v.box.ap50)})
    except Exception as e: print("skip",run,str(e)[:50])
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["run","imgsz","mAP","DD","DDr","ND"]); w.writeheader()
    for r in rows: w.writerow({k:r[k] for k in ["run","imgsz","mAP","DD","DDr","ND"]})
def cfg(r):
    r=re.sub(r"_(s\d+seed|sb\d+|a\d+|s\d+)$","",r)
    return "baseline" if r in ("B_v11_150p100","Bv11") else r
g=defaultdict(list)
for r in rows: g[cfg(r["run"])].append(r)
print("\nCONFIG_TABLE")
for c,rs in sorted(g.items(),key=lambda x:-np.mean([float(r["DD"]) for r in x[1]])):
    dd=[float(r["DD"]) for r in rs]; mp=[float(r["mAP"]) for r in rs]; nd=[float(r["ND"]) for r in rs]
    print(f"{c:22s} n={len(rs):2d}  DD {np.mean(dd):.3f}±{np.std(dd):.3f}  mAP {np.mean(mp):.3f}  ND {np.mean(nd):.3f}")
