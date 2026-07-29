#!/usr/bin/env bash
# Copy a trained checkpoint + this benchmark harness + a few sample test
# images onto the Jetson over SSH/rsync.
#
# Usage:
#   JETSON_HOST=user@jetson-ip ./transfer_to_jetson.sh /path/to/best.pt
#
# JETSON_HOST can also be a ~/.ssh/config alias. Sample images are pulled
# from the eduardo CV test split on $ATLI_SERVER (fold0) if SAMPLE_SRC isn't
# overridden — swap in any small folder of representative field images.
set -euo pipefail
WEIGHTS="${1:?usage: JETSON_HOST=user@host ./transfer_to_jetson.sh /path/to/best.pt}"
: "${JETSON_HOST:?set JETSON_HOST=user@jetson-ip (or an ssh config alias)}"
REMOTE_DIR="${JETSON_REMOTE_DIR:-~/hardware_testing}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ssh "$JETSON_HOST" "mkdir -p $REMOTE_DIR/sample_images"
scp "$WEIGHTS" "$JETSON_HOST:$REMOTE_DIR/best.pt"
scp "$HERE/bench_jetson_fps.py" "$JETSON_HOST:$REMOTE_DIR/"

if [ -n "${SAMPLE_SRC:-}" ]; then
  scp "$SAMPLE_SRC"/*.jpg "$JETSON_HOST:$REMOTE_DIR/sample_images/" 2>/dev/null || true
elif [ -n "${ATLI_SERVER:-}" ]; then
  echo "[info] pulling a few sample images via $ATLI_SERVER (fold0 test split)"
  scp "$ATLI_SERVER:~/atli/CV_eduardo_obb/fold0/test/images/*.jpg" /tmp/jetson_samples_tmp/ 2>/dev/null || true
  mkdir -p /tmp/jetson_samples_tmp
  ssh "$ATLI_SERVER" "ls ~/atli/CV_eduardo_obb/fold0/test/images/*.jpg | head -5" | \
    xargs -I{} scp "$ATLI_SERVER:{}" /tmp/jetson_samples_tmp/ 2>/dev/null || true
  scp /tmp/jetson_samples_tmp/*.jpg "$JETSON_HOST:$REMOTE_DIR/sample_images/" 2>/dev/null || \
    echo "[warn] no sample images copied — point --source at your own images on the device"
else
  echo "[warn] no SAMPLE_SRC and no ATLI_SERVER — point --source at your own images on the device"
fi

echo "[done] weights + harness on $JETSON_HOST:$REMOTE_DIR"
echo "next: ssh $JETSON_HOST"
echo "      cd $REMOTE_DIR && python3 bench_jetson_fps.py --weights best.pt --imgsz 640 768 --formats pt engine --half"
