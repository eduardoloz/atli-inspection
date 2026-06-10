#!/usr/bin/env bash
# Hard negative mining: find images where v11n Condition C confuses
# Defective_Damper <-> Normal_Damper or Broken_Insulator <-> Normal_Insulators.
#
# Runs the best model (v11_tpef s2) on the TRAINING set and saves predictions.
# Then a Python script compares predictions vs ground truth to find mismatches.
set -euo pipefail
ROOT=$HOME/atli
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"

W="$ROOT/runs/v11_tpef_s2/weights/best.pt"
TRAIN_IMGS="$ROOT/Target_Plus_EduardoFull/train/images"
TRAIN_LBLS="$ROOT/Target_Plus_EduardoFull/train/labels"
OUT="$ROOT/hard_negatives"
mkdir -p "$OUT"

echo "Running inference on training set with v11n Condition C..."
yolo detect predict model="$W" source="$TRAIN_IMGS" imgsz=640 \
  conf=0.25 save_txt=True device=0 \
  project="$OUT" name=preds exist_ok=True

echo "Analyzing predictions vs ground truth..."
python3 - <<'PYEOF'
import os
from pathlib import Path
from collections import defaultdict

NAMES = ["Birdnest", "Broken_Insulator", "Defective_Damper", "Flashover_Insulator",
         "Normal_Damper", "Normal_Insulators", "Self-Exploded_Insulator"]

gt_dir = Path(os.path.expanduser("~/atli/Target_Plus_EduardoFull/train/labels"))
pred_dir = Path(os.path.expanduser("~/atli/hard_negatives/preds/labels"))
out_dir = Path(os.path.expanduser("~/atli/hard_negatives"))

# Confusion pairs we care about
PAIRS = [
    (2, 4, "Defective_Damper predicted as Normal_Damper"),
    (4, 2, "Normal_Damper predicted as Defective_Damper"),
    (1, 5, "Broken_Insulator predicted as Normal_Insulators"),
    (5, 1, "Normal_Insulators predicted as Broken_Insulator"),
]

def get_classes(label_file):
    classes = set()
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if line:
                classes.add(int(line.split()[0]))
    return classes

# Find mismatches
results = defaultdict(list)
gt_files = sorted(gt_dir.glob("*.txt"))

for gt_file in gt_files:
    stem = gt_file.stem
    pred_file = pred_dir / f"{stem}.txt"

    gt_classes = get_classes(gt_file)
    pred_classes = get_classes(pred_file)

    for gt_cls, pred_cls, desc in PAIRS:
        # GT has the class but prediction doesn't, AND prediction has the confused class
        if gt_cls in gt_classes and gt_cls not in pred_classes and pred_cls in pred_classes:
            results[desc].append(stem)
        # GT has the class but prediction also adds the wrong class
        if gt_cls in gt_classes and pred_cls in pred_classes and gt_cls not in gt_classes:
            results[f"{desc} (false positive)"].append(stem)

    # Also find: GT has Defective_Damper but model completely misses it
    if 2 in gt_classes and 2 not in pred_classes:
        results["Defective_Damper MISSED entirely"].append(stem)
    if 1 in gt_classes and 1 not in pred_classes:
        results["Broken_Insulator MISSED entirely"].append(stem)

# Report
print("\n" + "="*70)
print("HARD NEGATIVE MINING RESULTS")
print("="*70)
for desc, files in sorted(results.items()):
    print(f"\n{desc}: {len(files)} images")
    for f in files[:20]:  # show first 20
        print(f"  {f}")
    if len(files) > 20:
        print(f"  ... and {len(files)-20} more")

# Save full lists
for desc, files in results.items():
    safe_name = desc.replace(" ", "_").replace("/", "_").lower()
    (out_dir / f"{safe_name}.txt").write_text("\n".join(files))

print(f"\nFull lists saved to {out_dir}/")
PYEOF
echo "HARD_NEGATIVES_DONE"
