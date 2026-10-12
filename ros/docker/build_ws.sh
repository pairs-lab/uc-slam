#!/usr/bin/env bash
# Builds UC-SLAM and the ROS 2 packages inside the container of ros/docker/Dockerfile:
#   /ws/ucslam/ros/docker/build_ws.sh            # JOBS=8 for fewer cores
# The repository at /ws/ucslam stays read only. UC-SLAM writes its binaries into its source tree, so it is built
# in a copy under /ws/build/ucslam (the mirror of scripts/build_ucslam.sh). A host build in cpp/ucslam is never
# touched. Then colcon builds the copy of ros/ros2 into /ws/build/{build,install,log}. Run it again after a
# source change. The depth node reads ucslam/ and configs/ from /ws/ucslam at run time.
set -euo pipefail
REPO=${UCSLAM_REPO:-/ws/ucslam}
OUT=${UCSLAM_BUILD:-/ws/build}
JOBS=${JOBS:-$(nproc)}
log() { echo "[build_ws $(date +%H:%M:%S)] $*"; }

MIRROR="$OUT/ucslam"
log "copying cpp/ucslam and ros/ to $MIRROR (sources only)"
mkdir -p "$MIRROR/cpp" "$MIRROR/scripts"
# No build products and no IMU or timestamp lists of a host checkout. Files without an extension are binaries.
rsync -a --delete \
      --exclude '/build/' --exclude '/lib/' --exclude '/Vocabulary/' \
      --exclude '/Thirdparty/*/build/' --exclude '/Thirdparty/*/lib/' \
      --exclude 'TUM_IMU/' --exclude 'EuRoC_IMU/' --exclude 'TUM_TimeStamps/' --exclude 'EuRoC_TimeStamps/' \
      --include '*/' --include '*.*' --include 'CMakeLists.txt' --exclude '*' \
      "$REPO/cpp/ucslam/" "$MIRROR/cpp/ucslam/"
rsync -a --delete "$REPO/ros/" "$MIRROR/ros/"
cp "$REPO/scripts/build_ucslam.sh" "$MIRROR/scripts/"

VOC="$REPO/third_party/ORB_SLAM3/Vocabulary"
STOCK="$OUT/stock"
mkdir -p "$STOCK/Vocabulary"
if [ ! -f "$STOCK/Vocabulary/ORBvoc.txt" ]; then
  if [ -f "$VOC/ORBvoc.txt" ]; then
    cp "$VOC/ORBvoc.txt" "$STOCK/Vocabulary/"
  else
    log "unpacking the ORB vocabulary"
    tar -xzf "$VOC/ORBvoc.txt.tar.gz" -C "$STOCK/Vocabulary"
  fi
fi

log "building UC-SLAM (Pangolin of the image at /opt/pangolin)"
# librealsense2 of the ROS packages would add the RealSense examples of the fork. The host build has none of them.
FORK="$MIRROR/cpp/ucslam" STOCK="$STOCK" PANGO_INSTALL=/opt/pangolin JOBS="$JOBS" NICE=0 \
  UCSLAM_CMAKE_ARGS="-DCMAKE_DISABLE_FIND_PACKAGE_realsense2=TRUE" bash "$MIRROR/scripts/build_ucslam.sh"

log "building the ROS 2 packages"
set +u   # the ROS setup scripts read unset variables
source /opt/ros/humble/setup.bash
set -u
cd "$OUT"
colcon build --base-paths "$MIRROR/ros/ros2" --build-base "$OUT/build" --install-base "$OUT/install" \
  --parallel-workers "$JOBS" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DUCSLAM_REPO="$MIRROR" -DUCSLAM_ROOT="$MIRROR/cpp/ucslam" \
               -DPANGOLIN_ROOT=/opt/pangolin
log "done. Run: source $OUT/install/setup.bash"
log "vocabulary: $STOCK/Vocabulary/ORBvoc.txt"
