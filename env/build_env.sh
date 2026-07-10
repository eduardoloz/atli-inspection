#!/usr/bin/env bash
# Rebuild the torch-2.6 / YOLOv5-v7 / Ultralytics-8.4.9 environment that
# Untitled2.ipynb ran in, under your own home (the original account's home is private).
set -euo pipefail
ROOT="$HOME/atli"
mkdir -p "$ROOT"
cd "$ROOT"
source /opt/miniconda3/etc/profile.d/conda.sh

echo "=== [1/5] conda env (python 3.10) ==="
if [ ! -d "$ROOT/env" ]; then
  conda create -y -p "$ROOT/env" python=3.10
fi
conda activate "$ROOT/env"
python --version

echo "=== [2/5] torch 2.6.0 + cu118 ==="
pip install --upgrade pip wheel >/dev/null
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

echo "=== [3/5] clone yolov5 ==="
if [ ! -d "$ROOT/yolov5" ]; then
  git clone --depth 1 https://github.com/ultralytics/yolov5.git "$ROOT/yolov5"
fi
cd "$ROOT/yolov5"
pip install -r requirements.txt

echo "=== [4/5] ultralytics (yolov8) + dataset tools ==="
pip install "ultralytics==8.4.9" roboflow iterative-stratification

echo "=== [5/5] sanity ==="
python - <<'PY'
import torch, ultralytics
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
print("ultralytics", ultralytics.__version__)
PY
echo "BUILD_ENV_DONE"
