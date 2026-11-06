#!/usr/bin/env bash
# Set up UC-SLAM on a Jetson Orin NX (JetPack 6.x, Ubuntu 22.04, aarch64, ROS 2 Humble). Guide: runJetson.md.
#
#   scripts/jetson/setup_jetson.sh                       # every step, in order
#   scripts/jetson/setup_jetson.sh --steps check         # read-only system check
#   scripts/jetson/setup_jetson.sh --steps build,ws,verify --jobs 2
#   scripts/jetson/setup_jetson.sh --dry-run             # print what would run, change nothing
#
# Steps (in this order, each one can run again without harm):
#   check   read-only: JetPack, CUDA, TensorRT, OpenCV, Python, ROS, power mode, memory, swap, disk, socket buffer,
#           serial ports, camera, repository state
#   deps    sudo: ROS 2 apt source (packages.ros.org), ROS and build packages, MAVROS, GeographicLib datasets,
#           RealSense driver from apt (skip with --no-realsense-apt), dialout group. No 'apt upgrade' (it can move
#           the L4T packages): run it yourself if you want it (runJetson.md section 3.1)
#   sysctl  sudo: /etc/sysctl.d/10-cyclone-max.conf (net.core.rmem_max), and ~/.ros/cyclonedds.xml
#   build   UC-SLAM library with ./build.sh (Pangolin, vocabulary, cpp/ucslam)
#   ws      colcon workspace of ros/ros2, the one-OpenCV check, and <ws>/ucslam_env.sh. When
#           ~/rs_ws/install/setup.bash exists (RealSense built from source), the workspace and the env file chain it
#   venv    Python venv of the depth node (system site packages, Jetson torch, transformers)
#   model   the pinned depth model weights in the Hugging Face cache
#   verify  ldd, packages, launch files, Python imports, processor, a short network run
#
# Run it as your user, not with sudo: the deps and sysctl steps call sudo themselves.
# The script calls apt-get install with --no-remove (apt stops instead of removing a package, OpenCV included),
# never purges a package, never reboots, and never changes the power mode (nvpmodel) or the clocks
# (jetson_clocks). It prints those commands for you to run.
set -Eeuo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WS="$HOME/ucslam_ws"
VENV="$HOME/ucslam-venv"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
LOG_DIR="$HOME/ucslam_logs/setup"
JOBS=""
STEPS="check,deps,sysctl,build,ws,venv,model,verify"
ALL_STEPS="check deps sysctl build ws venv model verify"
DRY=0
ENGINE=""
BACKEND=""
NO_RS_APT=0
NET_TEST=1

# Versions of ros/docker/Dockerfile (the environment that made the depth cache). The Jetson wheels of torch and
# torchvision come from the Jetson AI Lab index (community run, CUDA 12.6 builds for JetPack 6).
TORCH_VER=2.11.0
TORCHVISION_VER=0.26.0
TRANSFORMERS_VER=5.17.0
JAI_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
PYPI_INDEX=https://pypi.org/simple
RMEM_MAX=2147483647          # ROS 2 DDS tuning guide, Cyclone DDS section
SYSCTL_FILE=/etc/sysctl.d/10-cyclone-max.conf
CYCLONE_XML="$HOME/.ros/cyclonedds.xml"
RS_WS_SETUP="$HOME/rs_ws/install/setup.bash"   # RealSense built from source (runJetson.md section 4.3)
ONE_IMAGE_BYTES=1221120      # one 848x480 rgb8 image

usage() {
  cat <<EOF
usage: setup_jetson.sh [--steps a,b,...] [--dry-run] [--jobs N] [--ws DIR] [--venv DIR] [--repo DIR]
                       [--hf-home DIR] [--log-dir DIR] [--engine FILE] [--backend B] [--no-realsense-apt]
                       [--no-net-test]
  --steps     comma list of: $ALL_STEPS (default: all, in this order).
              deps and sysctl use sudo. The others run as you.
  --dry-run   print the commands that change something, run only the read-only checks
  --jobs      parallel compile jobs (default: RAM in GiB / 3, at least 1, at most the CPU count)
  --ws        colcon workspace (default: $WS)
  --venv      Python venv of the depth node (default: $VENV)
  --repo      UC-SLAM repository (default: the repository of this script, $REPO)
  --hf-home   Hugging Face home with hub/ (default: \$HF_HOME or ~/.cache/huggingface, now $HF_DIR)
  --log-dir   where the log of this run goes (default: $LOG_DIR)
  --engine    verify: also run this TensorRT engine once (backend from --backend, default trt-fp16)
  --backend   verify: backend of --engine, trt-fp16 or trt-fp32
  --no-realsense-apt  deps: do not install ros-humble-realsense2-camera (you build the driver from source)
  --no-net-test       verify: skip the short depth network run
Run it as your user, not with sudo. Exit status: 0 when no check failed, 1 otherwise, 2 for a usage error.
Guide: runJetson.md at the repository root.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS=${2:?--steps needs a value}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --jobs) JOBS=${2:?--jobs needs a value}; shift 2 ;;
    --ws) WS=${2:?--ws needs a value}; shift 2 ;;
    --venv) VENV=${2:?--venv needs a value}; shift 2 ;;
    --repo) REPO=${2:?--repo needs a value}; shift 2 ;;
    --hf-home) HF_DIR=${2:?--hf-home needs a value}; shift 2 ;;
    --log-dir) LOG_DIR=${2:?--log-dir needs a value}; shift 2 ;;
    --engine) ENGINE=${2:?--engine needs a value}; shift 2 ;;
    --backend) BACKEND=${2:?--backend needs a value}; shift 2 ;;
    --no-realsense-apt) NO_RS_APT=1; shift ;;
    --no-net-test) NET_TEST=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Under sudo, HOME is /root: the workspace, venv, cache and logs would go there and the build folders would
# become owned by root.
if [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ]; then
  echo "run as your user, not with sudo (the deps and sysctl steps call sudo themselves)" >&2
  exit 2
fi

for s in ${STEPS//,/ }; do
  case " $ALL_STEPS " in *" $s "*) ;; *) echo "unknown step: $s (steps: $ALL_STEPS)" >&2; exit 2 ;; esac
done
if [ -n "$JOBS" ] && ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then echo "--jobs must be a positive integer" >&2; exit 2; fi
if [ -n "$BACKEND" ] && [ "$BACKEND" != trt-fp16 ] && [ "$BACKEND" != trt-fp32 ]; then
  echo "--backend must be trt-fp16 or trt-fp32" >&2; exit 2
fi
[ -n "$ENGINE" ] && [ -z "$BACKEND" ] && BACKEND=trt-fp16
REPO=$(cd "$REPO" && pwd)
case "$WS" in /*) ;; *) WS="$PWD/$WS" ;; esac
case "$VENV" in /*) ;; *) VENV="$PWD/$VENV" ;; esac
case "$HF_DIR" in /*) ;; *) HF_DIR="$PWD/$HF_DIR" ;; esac

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
TEE_PID=$!
# Let tee write everything before the script ends.
finish() { exec 1>&- 2>&-; wait "$TEE_PID" 2>/dev/null || true; }
trap finish EXIT
# Any unexpected failure stops the run with the line that failed.
trap 'echo "[STOP] command failed (line $LINENO): $BASH_COMMAND"; echo "log: $LOG"' ERR
ME=$(id -un 2>/dev/null) || ME=$(id -u)

NFAIL=0
NWARN=0
ok()   { echo "[ OK ] $*"; }
info() { echo "[info] $*"; }
warn() { echo "[WARN] $*"; NWARN=$((NWARN + 1)); }
fail() { echo "[FAIL] $*"; NFAIL=$((NFAIL + 1)); }
die()  { echo "[STOP] $*"; echo "log: $LOG"; exit 1; }
step() { echo; echo "==== $* ($(date +%H:%M:%S))"; }
run()  { echo "+ $*"; if [ "$DRY" = 1 ]; then return 0; fi; "$@"; }
has()  { command -v "$1" >/dev/null 2>&1; }
want() { case ",$STEPS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
is_jetson() { [ -f /etc/nv_tegra_release ]; }
ros_source() {  # the ROS setup scripts read unset variables
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  if [ -f "$RS_WS_SETUP" ]; then
    # shellcheck disable=SC1090
    source "$RS_WS_SETUP"
  fi
  if [ -f "$WS/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$WS/install/setup.bash"
  fi
  set -u
}
rmem_max() { cat /proc/sys/net/core/rmem_max 2>/dev/null || echo 0; }
# -R follows the symlink /etc/apt/sources.list.d/ros2.sources of the ros2-apt-source package
ros_apt_source() { grep -Rqs 'packages.ros.org' /etc/apt/sources.list /etc/apt/sources.list.d/; }
pkg_version() { dpkg-query -W -f='${Version}' "$1" 2>/dev/null || true; }
sudo_ready() {
  [ "$DRY" = 1 ] && return 0
  if [ "$(id -u)" = 0 ]; then return 0; fi
  has sudo || die "this step needs sudo"
  sudo -v || die "sudo failed"
}
SUDO=sudo
[ "$(id -u)" = 0 ] && SUDO=""
write_root_file() {  # <path> <content>: write with sudo only when the content differs
  local path=$1 content=$2
  if [ -f "$path" ] && [ "$(cat "$path")" = "$content" ]; then
    ok "$path is already set"
    return 0
  fi
  echo "+ write $path:"
  # shellcheck disable=SC2001
  echo "$content" | sed 's/^/    /'
  [ "$DRY" = 1 ] && return 0
  printf '%s\n' "$content" | $SUDO tee "$path" >/dev/null
}

if [ -z "$JOBS" ]; then
  mem_gib=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo)
  JOBS=$((mem_gib / 3))
  [ "$JOBS" -lt 1 ] && JOBS=1
  [ "$JOBS" -gt "$(nproc)" ] && JOBS=$(nproc)
fi
PY="$VENV/bin/python"
LIB="$REPO/cpp/ucslam/lib/libORB_SLAM3.so"
SLAM_NODE="$WS/install/ucslam_ros/lib/ucslam_ros/slam_node"

echo "setup_jetson.sh $(date '+%Y-%m-%d %H:%M:%S') steps=$STEPS dry_run=$DRY jobs=$JOBS"
echo "repo=$REPO ws=$WS venv=$VENV hf_home=$HF_DIR log=$LOG"

# ---------------------------------------------------------------------------------------------------- check
foreign_cache() {  # <CMakeCache.txt> <expected source dir>: true when the cache was made for another tree
  local cache=$1 src=$2 home
  [ -f "$cache" ] || return 1
  home=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache")
  [ -n "$home" ] && [ "$home" != "$src" ]
}

step_check() {
  step "check (read only)"
  local arch l4t rev rmem py swap_kib mem_kib free_gb f
  arch=$(uname -m)
  if [ "$arch" = aarch64 ]; then ok "architecture aarch64"; else warn "architecture $arch: not a Jetson (a test run)"; fi

  # shellcheck disable=SC1091
  . /etc/os-release
  if [ "${VERSION_ID:-}" = "22.04" ]; then ok "Ubuntu 22.04"; else fail "OS is ${PRETTY_NAME:-unknown}, JetPack 6 and ROS 2 Humble need Ubuntu 22.04"; fi

  if is_jetson; then
    l4t=$(head -1 /etc/nv_tegra_release)
    rev=$(echo "$l4t" | sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9.]*\).*/\1.\2/p')
    info "L4T: $l4t"
    case "$rev" in
      36.3*) warn "L4T $rev is JetPack 6.0 (CUDA 12.2, TensorRT 8.6): not tested with UC-SLAM. The torch wheels of $JAI_INDEX are CUDA 12.6 builds. JetPack 6.1 or newer is recommended" ;;
      36.4.0) ok "L4T $rev = JetPack 6.1" ;;
      36.4.3) ok "L4T $rev = JetPack 6.2" ;;
      36.4.4) ok "L4T $rev = JetPack 6.2.1" ;;
      36.*) warn "L4T $rev: a JetPack 6 release not listed in runJetson.md, check CUDA and TensorRT below" ;;
      *) fail "L4T $rev is not JetPack 6 (L4T R36)" ;;
    esac
    info "nvidia-jetpack package: $(pkg_version nvidia-jetpack)"
  else
    warn "no /etc/nv_tegra_release: not a Jetson, the JetPack checks are skipped"
  fi

  if [ -x /usr/local/cuda/bin/nvcc ]; then
    info "CUDA: $(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  elif is_jetson; then
    warn "no /usr/local/cuda/bin/nvcc (CUDA toolkit of JetPack)"
  fi
  for f in libnvinfer10 libnvinfer8; do
    [ -n "$(pkg_version $f)" ] && info "TensorRT package $f $(pkg_version $f)"
  done
  if python3 -c "import tensorrt" >/dev/null 2>&1; then
    ok "system python sees tensorrt $(python3 -c 'import tensorrt; print(tensorrt.__version__)')"
  elif is_jetson; then
    warn "system python3 cannot import tensorrt (apt package python3-libnvinfer)"
  fi
  if [ -x /usr/src/tensorrt/bin/trtexec ]; then ok "trtexec at /usr/src/tensorrt/bin/trtexec"
  elif is_jetson; then warn "no /usr/src/tensorrt/bin/trtexec: build_trt_engine.sh then uses the TensorRT Python API"; fi

  if has pkg-config && pkg-config --exists opencv4; then
    info "OpenCV for builds (pkg-config opencv4): $(pkg-config --modversion opencv4)"
  else
    warn "no OpenCV development files (pkg-config opencv4): the deps step installs them"
  fi
  local runtimes
  runtimes=$(dpkg-query -W -f='${Package} ${Version}\n' 'libopencv-core*' 'libopencv' 2>/dev/null | grep -v '^\S* $' || true)
  [ -n "$runtimes" ] && info "OpenCV runtime packages: $(echo "$runtimes" | tr '\n' ';')"
  if [ "$(grep -c 'libopencv-core[0-9]' <<<"$runtimes" || true)" -gt 1 ]; then
    info "two OpenCV runtimes side by side are fine as long as slam_node links one (ws and verify steps)"
  fi

  py=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  if [ "$py" = "3.10" ]; then ok "python3 $py"; else fail "python3 is $py, the Jetson wheels and rclpy of Humble need 3.10"; fi

  if [ -f /opt/ros/humble/setup.bash ]; then ok "ROS 2 Humble in /opt/ros/humble"; else warn "no ROS 2 Humble yet: the deps step installs it"; fi
  if ros_apt_source; then ok "apt source packages.ros.org"
  else warn "no packages.ros.org apt source yet: the deps step adds it"; fi
  if grep -Rqsi 'isaac' /etc/apt/sources.list /etc/apt/sources.list.d/; then
    warn "an Isaac apt source is configured. Install ROS 2 from packages.ros.org on the host (runJetson.md section 3)"
  fi

  if has nvpmodel; then
    info "power mode: $(nvpmodel -q 2>/dev/null | tr '\n' ' ' || echo 'run: sudo nvpmodel -q')"
    info "this script does not change it. MAXN is mode 0: sudo nvpmodel -m 0, then sudo jetson_clocks (runJetson.md section 2)"
  fi
  mem_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
  swap_kib=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
  info "memory $((mem_kib / 1024)) MiB, swap $((swap_kib / 1024)) MiB, build jobs $JOBS"
  if [ "$mem_kib" -lt 10000000 ] && [ "$swap_kib" -lt 8000000 ]; then
    warn "8 GB module with less than 8 GiB swap: add a swap file on the NVMe before the build (runJetson.md section 2)"
  fi
  for f in "$REPO" "$HOME"; do
    [ -d "$f" ] || { warn "$f does not exist"; continue; }
    free_gb=$(df -Pk "$f" | awk 'NR==2 {printf "%d", $4 / 1048576}')
    if [ "$free_gb" -ge 20 ]; then ok "$free_gb GiB free under $f"; else warn "only $free_gb GiB free under $f (build, venv, engine and bags need more than 20 GiB)"; fi
  done

  rmem=$(rmem_max)
  if [ "$rmem" = 0 ]; then warn "cannot read /proc/sys/net/core/rmem_max (a container without network?)"
  elif [ "$rmem" -ge "$RMEM_MAX" ]; then ok "net.core.rmem_max $rmem"
  elif [ "$rmem" -lt "$ONE_IMAGE_BYTES" ]; then warn "net.core.rmem_max $rmem is smaller than one 848x480 color image ($ONE_IMAGE_BYTES bytes): DDS loses images. The sysctl step raises it"
  else info "net.core.rmem_max $rmem (the sysctl step sets $RMEM_MAX)"; fi
  info "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>} CYCLONEDDS_URI=${CYCLONEDDS_URI:-<unset>}"

  if id -nG 2>/dev/null | grep -qw dialout; then ok "user $ME is in group dialout"; else warn "user $ME is not in group dialout (serial port to PX4). The deps step adds it, then log in again"; fi
  # shellcheck disable=SC2012
  info "serial ports: $(ls /dev/ttyTHS* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | tr '\n' ' ' || true)"
  if has lsusb; then info "USB Intel devices: $(lsusb | grep -i intel | tr '\n' ';' || true)"; fi
  if has rs-enumerate-devices; then info "rs-enumerate-devices is installed (run it to see the D455 and its firmware)"; fi

  if [ -d "$REPO/.git" ] || [ -f "$REPO/.git" ]; then ok "repository $REPO is a git checkout"
  else warn "$REPO has no .git: ./build.sh needs it for 'git submodule update' (copy .git too, runJetson.md section 5)"; fi
  if [ -f "$REPO/third_party/Pangolin/CMakeLists.txt" ]; then ok "Pangolin source present"
  else warn "third_party/Pangolin is empty: ./build.sh fetches the submodules (needs internet)"; fi
  if [ -f "$REPO/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz" ] || [ -f "$REPO/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt" ]; then
    ok "ORB vocabulary present"
  else warn "no ORB vocabulary in third_party/ORB_SLAM3: ./build.sh fetches the submodules (needs internet)"; fi
  if foreign_cache "$REPO/cpp/ucslam/build/CMakeCache.txt" "$REPO/cpp/ucslam"; then
    fail "cpp/ucslam/build was configured for another folder (copied from another machine, or the folder was renamed). Delete it: rm -rf $REPO/cpp/ucslam/build $REPO/cpp/ucslam/lib $REPO/cpp/ucslam/Thirdparty/*/build $REPO/cpp/ucslam/Thirdparty/*/lib"
  fi
  if foreign_cache "$REPO/third_party/Pangolin/build/CMakeCache.txt" "$REPO/third_party/Pangolin"; then
    fail "third_party/Pangolin/build was configured in another folder. Delete it: rm -rf $REPO/third_party/Pangolin/build $REPO/third_party/Pangolin/install"
  fi
  f=$(find "$REPO/third_party/Pangolin/install/lib" -maxdepth 1 -name '*.so' 2>/dev/null | head -1 || true)
  if [ -n "$f" ] && has file && [ "$arch" = aarch64 ] && file -L "$f" | grep -q x86-64; then
    fail "$f is an x86-64 library (copied from a desktop). Delete it: rm -rf $REPO/third_party/Pangolin/build $REPO/third_party/Pangolin/install"
  fi
  if [ -d "$HF_DIR/hub/models--depth-anything--Depth-Anything-V2-Metric-Indoor-Base-hf" ]; then
    ok "depth model folder in $HF_DIR/hub (the model step checks the pinned revision)"
  else
    info "no depth model in $HF_DIR/hub yet (model step)"
  fi
}

# ---------------------------------------------------------------------------------------------------- deps
step_deps() {
  step "deps (sudo)"
  sudo_ready
  if ! ros_apt_source; then
    # The commands of the ROS 2 Humble install guide (ros2_documentation, Installation/_Apt-Repositories.rst).
    info "adding the ROS 2 apt source of packages.ros.org"
    run $SUDO apt-get update
    run $SUDO apt-get install -y --no-remove software-properties-common curl
    run $SUDO add-apt-repository -y universe
    local ver codename
    # shellcheck disable=SC1091
    codename=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")
    if [ "$DRY" = 1 ]; then
      ver='<latest ros-apt-source release>'
    else
      ver=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
            | grep -F "tag_name" | awk -F'"' '{print $4}')
      [ -n "$ver" ] || die "cannot read the latest ros-apt-source release from api.github.com"
    fi
    run curl -L -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${codename}_all.deb"
    run $SUDO dpkg -i /tmp/ros2-apt-source.deb
  else
    ok "apt source packages.ros.org already set"
  fi
  run $SUDO apt-get update
  # No 'apt upgrade' here: on a Jetson it can also upgrade the L4T packages (runJetson.md section 3.1).
  info "this step does not run 'apt upgrade'. Check 'apt list --upgradable' yourself (runJetson.md section 3.1)"

  local pkgs=(
    ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp ros-humble-tf2-ros ros-humble-tf2-ros-py
    ros-humble-message-filters ros-humble-diagnostic-msgs ros-humble-std-srvs ros-humble-nav-msgs
    ros-humble-rosbag2-storage-mcap ros-humble-rosbag2-py python3-colcon-common-extensions
    ros-humble-mavros ros-humble-mavros-extras
    python3-venv python3-pip rsync tmux
    build-essential cmake git pkg-config libeigen3-dev libboost-serialization-dev libssl-dev
    libglew-dev libgl1-mesa-dev libegl1-mesa-dev libjpeg-dev libpng-dev libepoxy-dev libopenblas-dev)
  [ "$NO_RS_APT" = 1 ] || pkgs+=(ros-humble-realsense2-camera)
  # OpenCV: keep what JetPack installed (NVIDIA 4.8). Only when no development files exist, add NVIDIA's on a
  # Jetson, Ubuntu's elsewhere. --no-remove: apt stops instead of removing a conflicting package.
  if ! (has pkg-config && pkg-config --exists opencv4); then
    if is_jetson; then pkgs+=(nvidia-opencv-dev); else pkgs+=(libopencv-dev); fi
  fi
  if is_jetson && ! python3 -c "import tensorrt" >/dev/null 2>&1; then pkgs+=(python3-libnvinfer); fi
  run $SUDO apt-get install -y --no-remove --no-install-recommends "${pkgs[@]}"

  # GeographicLib datasets for MAVROS (mavros README). Skipped when the geoids folder is already filled.
  if [ -n "$(ls -A /usr/share/GeographicLib/geoids 2>/dev/null || true)" ]; then
    ok "GeographicLib datasets present in /usr/share/GeographicLib"
  elif [ -f /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh ] || [ "$DRY" = 1 ]; then
    run $SUDO bash /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
  else
    fail "no /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh after installing ros-humble-mavros"
  fi

  if id -nG 2>/dev/null | grep -qw dialout; then ok "user $ME is in group dialout"
  else run $SUDO usermod -aG dialout "$ME"; info "log out and in again for the dialout group"; fi
}

# ---------------------------------------------------------------------------------------------------- sysctl
step_sysctl() {
  step "sysctl (sudo) and Cyclone DDS configuration"
  sudo_ready
  write_root_file "$SYSCTL_FILE" "net.core.rmem_max=$RMEM_MAX"
  run $SUDO sysctl -p "$SYSCTL_FILE"
  if [ -f "$CYCLONE_XML" ]; then
    if grep -q SocketReceiveBufferSize "$CYCLONE_XML"; then ok "$CYCLONE_XML exists with SocketReceiveBufferSize"
    else warn "$CYCLONE_XML exists without SocketReceiveBufferSize. Not changed: add <SocketReceiveBufferSize min=\"10MB\"/> under Domain/Internal"; fi
  else
    echo "+ write $CYCLONE_XML"
    if [ "$DRY" = 0 ]; then
      mkdir -p "$(dirname "$CYCLONE_XML")"
      cat > "$CYCLONE_XML" <<'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<!-- Written by scripts/jetson/setup_jetson.sh. From the Cyclone DDS section of the ROS 2 DDS tuning guide. -->
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config
https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain id="any">
        <Internal>
            <SocketReceiveBufferSize min="10MB"/>
        </Internal>
    </Domain>
</CycloneDDS>
EOF
    fi
  fi
  info "every shell that runs ROS needs CYCLONEDDS_URI=file://$CYCLONE_XML (set by $WS/ucslam_env.sh)"
}

# ---------------------------------------------------------------------------------------------------- build
step_build() {
  step "build UC-SLAM (./build.sh, $JOBS jobs)"
  [ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || die "$REPO is not a git checkout: ./build.sh runs 'git submodule update'. Clone the repository or copy it with .git (runJetson.md section 5)"
  if foreign_cache "$REPO/cpp/ucslam/build/CMakeCache.txt" "$REPO/cpp/ucslam" || \
     foreign_cache "$REPO/third_party/Pangolin/build/CMakeCache.txt" "$REPO/third_party/Pangolin"; then
    die "a build folder was configured for another folder (see the check step for the rm command)"
  fi
  # The RealSense examples of ORB-SLAM3 are left out as in the tested build. No ROS paths in this build.
  run env -u CMAKE_PREFIX_PATH -u AMENT_PREFIX_PATH JOBS="$JOBS" \
      UCSLAM_CMAKE_ARGS=-DCMAKE_DISABLE_FIND_PACKAGE_realsense2=TRUE bash "$REPO/build.sh"
  [ "$DRY" = 1 ] && return 0
  [ -f "$LIB" ] || die "no $LIB after ./build.sh"
  ok "$LIB"
  info "libORB_SLAM3 OpenCV: $(grep '^OpenCV_DIR:' "$REPO/cpp/ucslam/build/CMakeCache.txt" || echo '?')"
}

# ---------------------------------------------------------------------------------------------------- ws
check_ldd() {  # one libopencv_core, no cv_bridge, no missing library in slam_node
  local out n
  [ -x "$SLAM_NODE" ] || { fail "no $SLAM_NODE (ws step)"; return 0; }
  # With ROS sourced: the RUNPATH of slam_node covers only its direct libraries, not the ROS ones behind them.
  out=$(ros_source && ldd "$SLAM_NODE") || true
  n=$(grep -c 'libopencv_core' <<<"$out" || true)
  if [ "$n" = 1 ]; then ok "slam_node links one OpenCV: $(grep 'libopencv_core' <<<"$out" | awk '{print $1, $3}')"
  else fail "slam_node links $n libopencv_core libraries: $(grep libopencv_core <<<"$out" | tr '\n' ';')"; fi
  if grep -q 'cv_bridge' <<<"$out"; then fail "slam_node links cv_bridge"; else ok "slam_node does not link cv_bridge"; fi
  if grep -q 'not found' <<<"$out"; then fail "slam_node has missing libraries: $(grep 'not found' <<<"$out" | tr '\n' ';')"
  else ok "slam_node finds all its libraries"; fi
}

write_env_file() {
  local env="$WS/ucslam_env.sh" content rs_line=""
  if [ -f "$RS_WS_SETUP" ]; then
    rs_line="source \"$RS_WS_SETUP\"   # RealSense built from source (runJetson.md section 4.3)"
    info "chaining the RealSense source workspace $RS_WS_SETUP"
  fi
  content=$(cat <<EOF
# Written by scripts/jetson/setup_jetson.sh. Source it in every shell that runs UC-SLAM:
#   source $env
source /opt/ros/humble/setup.bash
$rs_line
source "$WS/install/setup.bash"
export UCSLAM_REPO="$REPO"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if [ -f "$CYCLONE_XML" ]; then export CYCLONEDDS_URI="file://$CYCLONE_XML"; fi
export HF_HOME="$HF_DIR"
export HF_HUB_OFFLINE=1
# depth_node runs with the first python3 on PATH: the venv
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
EOF
)
  if [ -f "$env" ] && [ "$(cat "$env")" = "$content" ]; then ok "$env is up to date"; return 0; fi
  echo "+ write $env"
  [ "$DRY" = 1 ] || printf '%s\n' "$content" > "$env"
}

step_ws() {
  step "ROS 2 workspace $WS ($JOBS jobs)"
  [ -f /opt/ros/humble/setup.bash ] || die "no ROS 2 Humble (deps step)"
  [ -f "$LIB" ] || [ "$DRY" = 1 ] || die "no $LIB (build step)"
  run mkdir -p "$WS"
  if [ "$DRY" = 1 ]; then
    echo "+ (cd $WS && colcon build --base-paths $REPO/ros/ros2 --parallel-workers $JOBS --cmake-args -DCMAKE_BUILD_TYPE=Release)"
  else
    (ros_source && cd "$WS" && MAKEFLAGS="-j$JOBS" colcon build --base-paths "$REPO/ros/ros2" \
        --parallel-workers "$JOBS" --cmake-args -DCMAKE_BUILD_TYPE=Release) || die "colcon build failed (log: $WS/log/latest_build)"
    local line
    line=$(grep -h 'slam_node uses OpenCV' "$WS/log/latest_build/ucslam_ros/stdout.log" 2>/dev/null | tail -1 || true)
    if [ -n "$line" ]; then info "${line#-- }"; else info "slam_node was not reconfigured in this build (no OpenCV line in the log)"; fi
    check_ldd
  fi
  write_env_file
}

# ---------------------------------------------------------------------------------------------------- venv
py_ver() {  # <module>: version or empty
  "$PY" -c "import importlib, sys; m = importlib.import_module(sys.argv[1]); print(m.__version__.split('+')[0])" "$1" 2>/dev/null || true
}

venv_checks() {
  local v
  v=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  if [ "$v" = 3.10 ]; then ok "venv python $v"; else fail "venv python is $v, needs 3.10"; fi
  v=$(py_ver numpy)
  case "$v" in 1.*) ok "numpy $v" ;; *) fail "numpy is '$v', needs < 2 (ROS Humble modules)" ;; esac
  v=$(py_ver torch)
  if [ "$v" = "$TORCH_VER" ]; then ok "torch $v"; else fail "torch is '$v', expected $TORCH_VER"; fi
  v=$(py_ver torchvision)
  if [ "$v" = "$TORCHVISION_VER" ]; then ok "torchvision $v"; else fail "torchvision is '$v', expected $TORCHVISION_VER (without it transformers uses the PIL processor)"; fi
  v=$(py_ver transformers)
  if [ "$v" = "$TRANSFORMERS_VER" ]; then ok "transformers $v"; else fail "transformers is '$v', expected $TRANSFORMERS_VER"; fi
  if "$PY" -c "from transformers.utils import is_torch_available; import sys; sys.exit(0 if is_torch_available() else 1)" 2>/dev/null; then
    ok "transformers accepts this torch (is_torch_available)"
  else
    fail "transformers does not accept this torch (it disables torch below 2.5.0, also for pre-release wheels like 2.5.0a0)"
  fi
  v=$("$PY" -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)" 2>/dev/null || echo "error")
  case "$v" in
    True*) ok "torch CUDA available, CUDA ${v#True }" ;;
    *) if is_jetson; then fail "torch sees no CUDA ($v): not the Jetson wheel?"; else warn "torch sees no CUDA ($v)"; fi ;;
  esac
  v=$("$PY" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || true)
  if [ -n "$v" ]; then ok "tensorrt $v visible in the venv"
  elif is_jetson; then fail "the venv cannot import tensorrt (python3-libnvinfer, venv needs --system-site-packages)"
  else warn "no tensorrt in the venv (normal off the Jetson)"; fi
  if [ -f /opt/ros/humble/setup.bash ]; then
    if (ros_source && "$PY" -c "import rclpy") >/dev/null 2>&1; then ok "the venv imports rclpy"
    else fail "the venv cannot import rclpy with ROS sourced"; fi
  fi
}

step_venv() {
  step "Python venv $VENV"
  if [ ! -x "$PY" ]; then
    run python3 -m venv --system-site-packages "$VENV"
  else
    ok "venv exists"
  fi
  local cons="$VENV/pip-constraints.txt"
  if [ ! -f "$cons" ] || [ "$(cat "$cons")" != "numpy<2" ]; then
    echo "+ write $cons (numpy<2)"
    [ "$DRY" = 1 ] || echo "numpy<2" > "$cons"
  fi
  local have_torch have_tv have_tf
  have_torch=$([ -x "$PY" ] && py_ver torch || true)
  have_tv=$([ -x "$PY" ] && py_ver torchvision || true)
  have_tf=$([ -x "$PY" ] && py_ver transformers || true)
  if [ "$have_torch" = "$TORCH_VER" ] && [ "$have_tv" = "$TORCHVISION_VER" ] && [ "$have_tf" = "$TRANSFORMERS_VER" ]; then
    ok "torch $have_torch, torchvision $have_tv, transformers $have_tf already installed"
  else
    run "$PY" -m pip install -c "$cons" "numpy<2"
    # torch and torchvision from the Jetson index only (--no-deps): with PyPI as an extra index, pip could pick
    # PyPI's aarch64 torch, which is not built for Jetson. The second call takes their dependencies from PyPI
    # and keeps the installed torch, because torch==$TORCH_VER is already satisfied.
    run "$PY" -m pip install -c "$cons" --no-deps --index-url "$JAI_INDEX" \
        "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER"
    run "$PY" -m pip install -c "$cons" --index-url "$PYPI_INDEX" \
        "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" "transformers==$TRANSFORMERS_VER" pyyaml scipy
  fi
  [ "$DRY" = 1 ] && [ ! -x "$PY" ] && return 0
  venv_checks
}

# ---------------------------------------------------------------------------------------------------- model
model_info() {  # prints: <model key> <hf id> <revision>
  "$PY" - "$REPO" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from experiments.precompute_depth import DEFAULT_CONFIG, load_config, precompute_params
from ucslam.depthnet import _DA2_IDS, _DA2_REVISIONS
m = precompute_params(load_config(DEFAULT_CONFIG))["model"]
hf = _DA2_IDS[m]
print(m, hf, _DA2_REVISIONS[hf])
EOF
}

step_model() {
  step "depth model weights in $HF_DIR"
  if [ ! -x "$PY" ]; then
    [ "$DRY" = 1 ] && { info "dry run: no venv yet, the model step needs it"; return 0; }
    die "no venv python $PY (venv step)"
  fi
  local info_line key hf rev snap
  info_line=$(model_info | tail -1) || die "cannot read the model name from the repository"
  read -r key hf rev <<<"$info_line"
  snap="$HF_DIR/hub/models--${hf//\//--}/snapshots/$rev"
  info "model $key = $hf at revision $rev"
  if [ -f "$snap/model.safetensors" ] && [ -f "$snap/config.json" ] && [ -f "$snap/preprocessor_config.json" ]; then
    ok "weights present: $snap"
    return 0
  fi
  warn "weights missing in $snap: downloading (needs internet). Offline: copy the cache folder from the desktop (runJetson.md section 5)"
  echo "+ HF_HOME=$HF_DIR HF_HUB_OFFLINE=0 $PY -c 'snapshot_download($hf, revision=$rev)'"
  [ "$DRY" = 1 ] && return 0
  HF_HOME="$HF_DIR" HF_HUB_OFFLINE=0 "$PY" -c \
    "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" \
    "$hf" "$rev" || { fail "download failed"; return 0; }
  if [ -f "$snap/model.safetensors" ]; then ok "weights present: $snap"; else fail "still no $snap/model.safetensors"; fi
}

# ---------------------------------------------------------------------------------------------------- verify
net_test() {  # <backend> [engine]
  local backend=$1 engine=${2:-}
  info "network test: backend $backend${engine:+, engine $engine}, 848x480, 3 warm-up and 10 timed runs"
  (ros_source; HF_HOME="$HF_DIR" HF_HUB_OFFLINE=1 "$PY" - "$REPO" "$backend" "$engine" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from ucslam.live_depth import LiveDepth
backend, engine = sys.argv[2], sys.argv[3] or None
ld = LiveDepth(None, "cuda", backend=backend, engine=engine, log=lambda m: print("processor warning:", m))
ld.check_stream_size(480, 848)
img = np.tile(np.linspace(30, 220, 848, dtype=np.float32), (480, 1)).astype(np.uint8)
ld.warmup((480, 848), 3)
ms = []
for _ in range(10):
    z, desc, t = ld(img)
    ms.append(t)
z = np.asarray(z, np.float32)
print(f"model {ld.model}, map {z.shape[1]}x{z.shape[0]}, finite {np.isfinite(z).all()}, "
      f"valid {(z > 0).mean():.3f}, net_ms p50 {np.median(ms):.1f} max {max(ms):.1f}")
print("processor ok" if ld.processor_warning is None else "processor NOT torchvision")
sys.exit(0 if np.isfinite(z).all() and ld.processor_warning is None else 1)
EOF
  )
}

step_verify() {
  step "verify"
  local n
  check_ldd
  if [ ! -f /opt/ros/humble/setup.bash ]; then fail "no ROS 2 Humble"; return 0; fi
  if [ ! -f "$WS/install/setup.bash" ]; then fail "no $WS/install/setup.bash (ws step)"; return 0; fi
  for p in ucslam_msgs ucslam_ros ucslam_depth ucslam_px4; do
    if (ros_source && ros2 pkg prefix "$p") >/dev/null 2>&1; then ok "package $p"; else fail "package $p not found in $WS"; fi
  done
  for p in realsense2_camera mavros mavros_extras rosbag2_storage_mcap rmw_cyclonedds_cpp; do
    if (ros_source && ros2 pkg prefix "$p") >/dev/null 2>&1; then ok "package $p"; else warn "package $p not installed (deps step)"; fi
  done
  for f in "ucslam_ros ucslam.launch.py" "ucslam_ros ucslam_d455.launch.py" "ucslam_px4 px4_bridge.launch.py"; do
    # shellcheck disable=SC2086
    if n=$(ros_source && UCSLAM_REPO="$REPO" ros2 launch $f --show-args 2>&1 | grep -c "^    '"); then
      ok "ros2 launch $f --show-args: $n arguments"
    else
      fail "ros2 launch $f --show-args failed"
    fi
  done
  if [ -f "$WS/ucslam_env.sh" ]; then ok "$WS/ucslam_env.sh"; else warn "no $WS/ucslam_env.sh (ws step)"; fi
  n=$(rmem_max)
  if [ "$n" = 0 ]; then warn "cannot read /proc/sys/net/core/rmem_max"
  elif [ "$n" -ge "$RMEM_MAX" ]; then ok "net.core.rmem_max $n"; else warn "net.core.rmem_max $n is below $RMEM_MAX (sysctl step)"; fi
  if [ -f "$CYCLONE_XML" ]; then ok "$CYCLONE_XML"; else warn "no $CYCLONE_XML (sysctl step)"; fi

  if [ ! -x "$PY" ]; then fail "no venv python $PY (venv step)"; return 0; fi
  venv_checks
  if (ros_source && "$PY" -c "import ucslam_msgs.msg, ucslam_depth, ucslam_px4") >/dev/null 2>&1; then
    ok "the venv imports ucslam_msgs, ucslam_depth and ucslam_px4"
  else
    fail "the venv cannot import the workspace Python packages (source $WS/install/setup.bash)"
  fi
  if [ "$NET_TEST" = 0 ]; then info "network test skipped (--no-net-test)"; return 0; fi
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    warn "no CUDA in the venv: network test skipped"
    return 0
  fi
  if net_test torch-fp32; then ok "torch-fp32 network test"; else fail "torch-fp32 network test"; fi
  if [ -n "$ENGINE" ]; then
    if net_test "$BACKEND" "$ENGINE"; then ok "$BACKEND network test"; else fail "$BACKEND network test"; fi
  fi
}

# ---------------------------------------------------------------------------------------------------- main
for s in $ALL_STEPS; do
  want "$s" || continue
  "step_$s"
  if [ "$s" = check ] && [ "$NFAIL" -gt 0 ] && [ "$STEPS" != check ]; then
    die "$NFAIL check(s) failed: fix them first, then run again"
  fi
done
echo
echo "==== summary: $NFAIL failed, $NWARN warnings. Log: $LOG"
if [ "$NFAIL" -gt 0 ]; then exit 1; fi
