#!/usr/bin/env bash
# v5n x universe/community-data sweep — fills the gap found 2026-07-29: the
# original 20-run universe benchmark (results/universe_benchmark_results.md)
# only ran {v8n, v11n}; YOLOv5n was never trained on the community
# (Roboflow-Universe) damper data. Same recipe as the original sweep:
# yolov5nu.pt COCO init, detect, 2-stage TL 150+100 @640 batch32, identical
# dataset yamls (native 199-img test split, except Ufull which is scored
# in-domain by construction). Baseline B_v5 already exists (3 seeds).
# 5 runs on GPUs 6-7 (0-5 belong to the concurrent phase-22 graft sweep).
set -uo pipefail
ROOT="$HOME/atli"
EXTR="$ROOT/run_config_ext.sh"; PROJ="$ROOT/runs"
ST="$ROOT/sweep_v5_universe_status.txt"; LOGS="$ROOT/logs_eduardo"; mkdir -p "$LOGS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /opt/miniconda3/etc/profile.d/conda.sh; conda activate "$ROOT/env"
echo "SWEEP_V5U_START $(date)" >> "$ST"

run_one(){ local name=$1 yaml=$2 gpu=$3
  [ -d "$PROJ/${name}_test" ] && { echo "[$(date)] SKIP $name" >>"$ST"; return 0; }
  echo "[$(date)] START $name gpu=$gpu data=$(basename "$yaml")" >>"$ST"
  MODEL="yolov5nu.pt" DATA="$yaml" \
    bash "$EXTR" "$name" "$gpu" 150 100 640 32 >"$LOGS/${name}.log" 2>&1
  echo "[$(date)] DONE $name (exit $?)" >>"$ST"
  [ -f "$PROJ/${name}_s2/weights/best.pt" ] || echo "[$(date)] WARN $name missing s2 best.pt" >>"$ST"; }

# GPU 6 gets the big Ufull first; GPU 7 the two capped variants.
worker6(){
  run_one "Ufull_v5_150p100" "$ROOT/Merged_Universe_Stratified/merged_universe.yaml" 6
  run_one "Ucap1_v5_150p100" "$ROOT/Merged_Universe_Ucap1/merged_universe_ucap1.yaml" 6
  run_one "Uw_v5_150p100"    "$ROOT/Merged_Universe_WangboOnly/merged_universe_wangbo.yaml" 6
}
worker7(){
  run_one "Ucap2_v5_150p100" "$ROOT/Merged_Universe_Ucap2/merged_universe_ucap2.yaml" 7
  run_one "Uto_v5_150p100"   "$ROOT/Merged_Universe_TrainOnly/merged_universe_trainonly.yaml" 7
}
worker6 & worker7 & wait

# Collect: re-val every best.pt on ITS OWN yaml's test split (matches the
# original benchmark's scoring) -> ~/atli/v5_universe_results.json
echo "V5U_COLLECT $(date)" >>"$ST"
python - <<'EOF' >"$LOGS/v5_universe_collect.log" 2>&1
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
ROOT = Path.home() / "atli"
CLASSES = ["Birdnest","Broken_Insulator","Defective_Damper","Flashover_Insulator",
           "Normal_Damper","Normal_Insulators","Self-Exploded_Insulator"]
CONDS = [
    ("Ufull_v5_150p100", ROOT/"Merged_Universe_Stratified/merged_universe.yaml"),
    ("Ucap1_v5_150p100", ROOT/"Merged_Universe_Ucap1/merged_universe_ucap1.yaml"),
    ("Ucap2_v5_150p100", ROOT/"Merged_Universe_Ucap2/merged_universe_ucap2.yaml"),
    ("Uw_v5_150p100",    ROOT/"Merged_Universe_WangboOnly/merged_universe_wangbo.yaml"),
    ("Uto_v5_150p100",   ROOT/"Merged_Universe_TrainOnly/merged_universe_trainonly.yaml"),
]
out_path = ROOT / "v5_universe_results.json"
results = json.load(open(out_path)) if out_path.exists() else {}
for name, yaml in CONDS:
    w = ROOT/"runs"/f"{name}_s2"/"weights"/"best.pt"
    if name in results or not w.exists():
        print(f"skip {name} (done={name in results}, weights={w.exists()})"); continue
    r = YOLO(str(w)).val(data=str(yaml), split="test", imgsz=640, device=6,
                         batch=16, verbose=False, plots=False)
    met = r.box
    idx = {r.names[c]: i for i, c in enumerate(met.ap_class_index)}
    o = {"mAP50": float(met.map50), "P": float(met.mp), "R": float(met.mr)}
    for c in CLASSES:
        o[f"{c}_AP"] = float(met.ap50[idx[c]]) if c in idx else float("nan")
        o[f"{c}_P"]  = float(met.p[idx[c]])   if c in idx else float("nan")
        o[f"{c}_R"]  = float(met.r[idx[c]])   if c in idx else float("nan")
    results[name] = o
    print(name, "mAP50 %.3f  DD AP %.3f" % (o["mAP50"], o["Defective_Damper_AP"]))
json.dump(results, open(out_path, "w"), indent=1)
print("V5U_COLLECT_DONE ->", out_path)
EOF
echo "SWEEP_V5U_DONE $(date)" >> "$ST"
