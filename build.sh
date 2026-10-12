#!/usr/bin/env bash
# Build UC-SLAM in one step.
#   ./build.sh            # Pangolin v0.6 (submodule), the ORB vocabulary, then UC-SLAM (cpp/ucslam)
#   JOBS=8 ./build.sh     # fewer parallel jobs
#   ./build.sh --clean    # rebuild UC-SLAM from scratch
# The submodule third_party/ORB_SLAM3 (upstream commit 4452a3c) is not built. UC-SLAM takes its ORB vocabulary,
# and the runner takes its EuRoC and TUM-VI timestamp lists and TUM-VI IMU files.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
JOBS=${JOBS:-$(nproc)}
log() { echo "[build $(date +%H:%M:%S)] $*"; }

log "submodules"
git -C "$ROOT" submodule update --init --recursive

PANGO="$ROOT/third_party/Pangolin"
PANGO_INSTALL="$PANGO/install"
if [ ! -f "$PANGO_INSTALL/lib/cmake/Pangolin/PangolinConfig.cmake" ]; then
  log "building Pangolin (core library only)"
  # Pangolin v0.6 uses std::numeric_limits in some files without including <limits>, and g++ 11 (Ubuntu 22.04,
  # JetPack 6) then fails. "-include limits" adds it to every C++ file.
  cmake -S "$PANGO" -B "$PANGO/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PANGO_INSTALL" \
    -DCMAKE_CXX_FLAGS="-include limits" \
    -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF \
    -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBREALSENSE=OFF -DBUILD_PANGOLIN_LIBREALSENSE2=OFF \
    -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF -DBUILD_PANGOLIN_LIBUVC=OFF \
    -DBUILD_PANGOLIN_DEPTHSENSE=OFF -DBUILD_PANGOLIN_TELICAM=OFF -DBUILD_PANGOLIN_PLEORA=OFF \
    -DBUILD_PANGOLIN_V4L=OFF -DBUILD_PANGOLIN_LIBDC1394=OFF -DBUILD_PANGOLIN_LIBPNG=ON \
    -DBUILD_PANGOLIN_LIBJPEG=ON -DBUILD_PANGOLIN_LIBTIFF=OFF -DBUILD_PANGOLIN_LIBOPENEXR=OFF \
    -DBUILD_PANGOLIN_ZSTD=OFF -DBUILD_PANGOLIN_LZ4=OFF -DBUILD_PANGOLIN_TOON=OFF
  make -C "$PANGO/build" -j"$JOBS"
  make -C "$PANGO/build" install
fi

VOC="$ROOT/third_party/ORB_SLAM3/Vocabulary"
if [ ! -f "$VOC/ORBvoc.txt" ]; then
  log "unpacking the ORB vocabulary"
  tar -xzf "$VOC/ORBvoc.txt.tar.gz" -C "$VOC"
fi

log "building UC-SLAM"
JOBS=$JOBS bash "$ROOT/scripts/build_ucslam.sh" "$@"
