#!/bin/bash
# Build + benchmark a TensorRT FP16 engine ON the Jetson Nano (JetPack 4.6.x,
# TensorRT 8.2.x). Run this ON THE NANO, not on a workstation — TRT engines
# are architecture-specific.
#
# Usage:  ./build_engine_nano.sh champ_v11n_640.onnx [workspace_MB]
#
# Prereqs on the Nano:
#   - JetPack 4.6.x (last release supporting the original Nano)
#   - trtexec lives at /usr/src/tensorrt/bin/trtexec
#   - Max out clocks first:  sudo nvpmodel -m 0 && sudo jetson_clocks
#   - 4 GB RAM is shared CPU+GPU; add zram/swap before building 960/1280
#     engines:  sudo systemctl enable nvzramconfig  (or a 4G swapfile)
set -euo pipefail

ONNX=${1:?usage: $0 model.onnx [workspace_MB]}
WS=${2:-2048}   # MB; keep <=2048 on the 4GB Nano to avoid OOM during build
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ENGINE="${ONNX%.onnx}_fp16.engine"

echo "== Building FP16 engine: $ONNX -> $ENGINE (workspace ${WS}MB) =="
# TRT 8.2 syntax (--workspace in MB). Build can take 10-30 min on the Nano.
$TRTEXEC \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --workspace="$WS" \
  --buildOnly

echo "== Benchmarking (batch 1, 200 iterations, includes H2D/D2H copies) =="
$TRTEXEC \
  --loadEngine="$ENGINE" \
  --iterations=200 \
  --avgRuns=50 \
  --useSpinWait \
  --noDataTransfers=false 2>&1 | tee "${ENGINE%.engine}_bench.log" \
  | grep -E "mean|median|percentile|throughput|GPU Compute"

echo
echo "Report the 'GPU Compute Time mean' as inference latency and 1000/mean"
echo "as inference-only fps. End-to-end fps (camera->decode->preproc->infer->"
echo "NMS) will be lower; measure with the DeepStream pipeline for the real number."
