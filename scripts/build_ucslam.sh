#!/usr/bin/env bash
# Build the UC-SLAM fork in cpp/ucslam (upstream ORB_SLAM3 4452a3c plus our changes).
#
# Called by ./build.sh, which builds Pangolin from the submodule third_party/Pangolin
# and unpacks the vocabulary of the submodule third_party/ORB_SLAM3 first.
#
#   scripts/build_ucslam.sh            # full build, Release, -j12
#   JOBS=8 scripts/build_ucslam.sh     # fewer cores
#   scripts/build_ucslam.sh --clean    # remove build dirs first
#   FORK=/path/to/mirror/ucslam scripts/build_ucslam.sh   # build a copy of the fork instead
#
# FORK is the fork tree to build and STOCK the stock tree whose vocabulary is
# linked. Both default to the paths in this repository, and PANGO_INSTALL to the
# Pangolin that ./build.sh installs (the ROS 2 container sets its own). A mirror is an rsync
# of cpp/ucslam without build, lib and Vocabulary. Building a mirror leaves
# the binaries of cpp/ucslam untouched while a benchmark uses them.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
FORK="${FORK:-$REPO_ROOT/cpp/ucslam}"
STOCK="${STOCK:-$REPO_ROOT/third_party/ORB_SLAM3}"
PANGO_INSTALL="${PANGO_INSTALL:-$REPO_ROOT/third_party/Pangolin/install}"
JOBS=${JOBS:-12}
NICE=${NICE:-10}
log() { echo "[build_ucslam $(date +%H:%M:%S)] $*"; }

if [ "${1:-}" = "--clean" ]; then
  rm -rf "$FORK/build" "$FORK/lib" "$FORK"/Thirdparty/*/build "$FORK"/Thirdparty/*/lib
fi
if [ ! -f "$PANGO_INSTALL/lib/cmake/Pangolin/PangolinConfig.cmake" ]; then
  echo "Pangolin not found at $PANGO_INSTALL. Run ./build.sh first." >&2
  exit 1
fi
mkdir -p "$FORK/Vocabulary"
if [ ! -f "$FORK/Vocabulary/ORBvoc.txt" ]; then
  if [ -f "$STOCK/Vocabulary/ORBvoc.txt" ]; then
    log "linking vocabulary from the stock tree"
    ln -sf "$STOCK/Vocabulary/ORBvoc.txt" "$FORK/Vocabulary/ORBvoc.txt"
  else
    echo "no ORBvoc.txt in $STOCK/Vocabulary. Run ./build.sh first." >&2
    exit 1
  fi
fi
build_dir() {  # <source dir> [extra cmake args]
  local src=$1; shift
  nice -n "$NICE" cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release "$@"
  nice -n "$NICE" make -C "$src/build" -j"$JOBS"
}
for tp in DBoW2 g2o; do
  log "building Thirdparty/$tp"
  build_dir "$FORK/Thirdparty/$tp"
done
# Sophus is header-only for UC-SLAM. Its own tests and examples build with -Werror and are skipped.
log "configuring Thirdparty/Sophus"
build_dir "$FORK/Thirdparty/Sophus" -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
log "building UC-SLAM"
# UCSLAM_CMAKE_ARGS: extra CMake arguments, for example -DCMAKE_DISABLE_FIND_PACKAGE_realsense2=TRUE
build_dir "$FORK" -DPangolin_DIR="$PANGO_INSTALL/lib/cmake/Pangolin" ${UCSLAM_CMAKE_ARGS:-}
log "done. binaries:"
ls -1 "$FORK"/Examples/Monocular-Inertial/mono_inertial_euroc "$FORK"/Examples/Stereo-Inertial/stereo_inertial_euroc "$FORK"/Examples/Monocular-Inertial/mono_inertial_tum_vi 2>/dev/null || true
