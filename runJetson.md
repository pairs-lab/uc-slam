# Run UC-SLAM on a Jetson Orin NX UAV (D455 + PX4 through MAVROS)

This guide takes you from a fresh Jetson Orin NX to a test flight. The Jetson runs UC-SLAM with an Intel RealSense D455 and sends the pose to a PX4 flight controller through MAVROS.

**Read this first.** Nothing in this guide has run on a Jetson yet. All tests so far ran on an x86 desktop (Ubuntu 20.04 host, ROS 2 Humble in the Docker image `ucslam:humble`). These parts are **not tested anywhere**:

| Not tested | Where this guide checks it |
|---|---|
| The Jetson itself (JetPack 6, Orin GPU, memory, heat) | sections 2, 6, 7, 8, 11 |
| The aarch64 build of UC-SLAM and of the ROS 2 nodes | section 6 |
| TensorRT 8.6 (JetPack 6.0) | section 8 (use JetPack 6.1 or newer) |
| `trtexec` (the engine build path used on a Jetson) | section 8.2 |
| A real D455 (streams, IMU rate, emitter off, stamps) | sections 4, 9, 11 |
| Stereo-inertial `slam_node` on D455 data (PinHole stereo file). On the desktop it tracked a EuRoC MH_01 stereo replay with every frame and no map reset when the CPU kept up (keyframe ATE 4.12 cm, plain runner 4.83 cm, N=1). On the same overloaded desktop at twice that speed, images were lost and the map reset 7 times | sections 11.1, 14 |
| librealsense with the RSUSB backend built from source | section 4.3 |
| The `deps` and `sysctl` steps of `setup_jetson.sh` run for real | sections 3, 2.4, 6.1 |
| MAVROS with PX4 (hardware. SITL is not covered) | sections 12, 13 |
| The bridge `odometry` output and the `use_tf` mount | section 13 |
| Flight | section 14 |
| Autostart with tmux or systemd | section 17 |

Each of these is an explicit check step below, marked **CHECK ON THE JETSON**. Do not skip them.

Numbers quoted from tests come from the x86 desktop (RTX 4070 Ti, shared GPU, high CPU load from other jobs). They are not Orin numbers. Facts about JetPack, PX4, MAVROS and the D455 come from a web scan of 2026-10-27/28. Their sources are linked where you need them.

Rules for all commands: run them on the Jetson unless the step says **on the desktop**. The repository is at `~/ucslam` and the ROS 2 workspace at `~/ucslam_ws`. After section 6, open every new terminal with:

```bash
source ~/ucslam_ws/ucslam_env.sh
```

## Contents

0. [What you get and the status of each part](#0-what-you-get-and-the-status-of-each-part)
1. [Hardware checklist](#1-hardware-checklist)
2. [Prepare the Jetson](#2-prepare-the-jetson)
3. [ROS 2 Humble and MAVROS](#3-ros-2-humble-and-mavros)
4. [RealSense D455](#4-realsense-d455)
5. [Get the code and the model weights onto the Jetson](#5-get-the-code-and-the-model-weights-onto-the-jetson)
6. [Build](#6-build)
7. [Python environment of the depth node](#7-python-environment-of-the-depth-node)
8. [TensorRT engine](#8-tensorrt-engine)
9. [Calibrate and write the settings](#9-calibrate-and-write-the-settings)
10. [Fit the depth constants of the d455color domain](#10-fit-the-depth-constants-of-the-d455color-domain)
11. [Bench tests with the propellers off](#11-bench-tests-with-the-propellers-off)
12. [PX4 and MAVROS](#12-px4-and-mavros)
13. [The bridge](#13-the-bridge)
14. [Flight test ladder](#14-flight-test-ladder)
15. [After a flight](#15-after-a-flight)
16. [Troubleshooting](#16-troubleshooting)
17. [Optional autostart](#17-optional-autostart)

---

## 0 What you get and the status of each part

### 0.1 Nodes and topics

```text
 D455 (USB 3)                                                  PX4 flight controller
    |                                                                  ^
    v                                                                  | MAVLink over UART or USB
 realsense2_camera                                                  mavros (px4.launch)
    | /camera/camera/color/image_raw (mono mode, rgb8 848x480 30 Hz)    ^
    | /camera/camera/infra1|infra2/image_rect_raw (stereo mode)         | /mavros/vision_pose/pose (default)
    | /camera/camera/imu (gyro + accel united, 200 Hz)                  | or /mavros/odometry/out
    v                                                                  |
 /ucslam/slam_node  --- ~/odom, ~/status (1 Hz) --->  /ucslam/px4_bridge
    |        ^                                        holds until ~/status is OK, re-anchors jumps,
    |        | /ucslam/depth_prior                    turns the IMU pose into a base_link pose
    v        |
 /ucslam/depth_node (mono mode only: torch-fp32, torch-fp16, trt-fp16 or trt-fp32)
    /ucslam/depth_request (one per keyframe)
```

One launch file starts all of it: `ros2 launch ucslam_ros ucslam_d455.launch.py`. MAVROS runs in its own terminal (section 12).

### 0.2 Modes of the D455 launch

| `mode:=` | Images | Depth prior | Settles after | Use |
|---|---|---|---|---|
| `stereo` | infra1 + infra2 848x480 30 Hz, IR emitter off | off (domain `d455ir`, no fitted constants) | 16 s of map time | **closed-loop flight**: the planned flight mode (needs no depth constants). Tested on a EuRoC stereo replay on the desktop, not yet on D455 data |
| `mono` | color 848x480 30 Hz (global shutter) | on when the settings have fitted `d455color` constants | 76 s of map time | research mode, depth prior on |
| `calib` | color + depth aligned to color + IMU | none, no SLAM | n/a | recordings for the settings and the depth fit |

"Map time" runs only while the camera moves (ros/README.md). A camera that stands still settles later.

### 0.3 Status of each part

| Part | Tested here (x86 desktop) | To check on the Jetson |
|---|---|---|
| `slam_node` (ROS 2): no cv_bridge, own image conversion, IMU gap fill, short-frame guard, latency in `~/status`, `UCSLAM.DepthBackend` check, `depth:=off` | Built in `ucslam:humble`. `ldd`: one `libopencv_core`, no cv_bridge. 282 image conversion cases equal the old cv_bridge output. EuRoC MH_01 mono-inertial replays (N=1 each, loaded host, not an accuracy test): keyframe ATE 3.63 to 9.90 cm, one run with a tracking loss. EuRoC MH_01 stereo-inertial replay, depth off: every frame tracked, no map reset, keyframe ATE 4.12 cm when the CPU kept up. 7 map resets when images were lost under CPU overload | aarch64 build, NVIDIA OpenCV 4.8, `ldd` check, live D455, stereo-inertial tracking |
| Launch files `ucslam.launch.py`, `ucslam_d455.launch.py`, `px4_bridge.launch.py` | `--show-args` lists 42, 49 and 12 arguments. A backend mismatch is refused by `slam_node` and by the D455 launch. The includes start the camera driver, `slam_node` and the bridge | a real camera driver run |
| Depth backends, `ucslam/trt_depth.py`, `export_depth_onnx.py`, `build_trt_engine.sh`, `depth_parity.py` | TensorRT 10.3 on x86. Engine 518x910 for 848x480 images. Default engine (FP16 with FP32 LayerNorm and Softmax) passed parity against torch-fp32: p99 \|d log z\| 0.0093 on 30 OpenLORIS frames. Plain FP16 failed: 0.0220 > 0.02. Only the Python builder branch ran | TensorRT on the Orin, `trtexec` branch, TensorRT 8.6, timing, memory, trt-* inside the ROS node |
| `make_d455_settings.py`, `configs/d455/` | unit tests on synthetic inputs | real D455 bags, the emitter-off config file, the IMU rate |
| `fit_depth_constants.py` | reproduced the paper's OpenLORIS constants with XFeat keypoints: log bias 0.1530 vs 0.1532 | trt-* fits (not run anywhere), D455 data |
| `ucslam_px4` bridge | 27 unit tests. Synthetic live checks. EuRoC MH_01 mono-inertial replays through the bridge: nothing sent before `~/status` OK, z up, continuous output | MAVROS, PX4 hardware, `odometry` output, `use_tf` mount, stereo-inertial input |
| `scripts/jetson/setup_jetson.sh` | `bash -n`, shellcheck (0 findings), `--dry-run` of all steps, and the steps check, build, ws, venv (already installed path), model and verify in the `ucslam:humble` container (x86) on a copy made with the rsync of section 5.2: 0 failures, `ldd` one OpenCV, a torch-fp32 network run with the torchvision processor | every step on the Jetson, deps and sysctl for real |

Known limits from the tests:

- **Images got lost in DDS** on the desktop: `net.core.rmem_max` was 212992 bytes, smaller than one image. The Jetson must raise it (section 2.4).
- The output rate on EuRoC (20 Hz camera) was 17 to 18.5 Hz. PX4 asks for 30 to 50 Hz. The D455 gives 30 Hz. Not measured with a D455.
- Latency on the loaded desktop was p50 106 ms and p90 198 ms (`frame_drop` oldest). This is not an Orin number. Measure it in section 11.

### 0.4 Quick path

1. Copy the code and the weights (section 5).
2. Set the power mode and add swap by hand (2.2, 2.3). Do this before the build: on an 8 GB module the compiler can run out of memory without swap.
3. Run `scripts/jetson/setup_jetson.sh` as your user (sections 2.4, 3, 5.3, 6 and 7 in one script, section 6.1).
4. Do by hand: RealSense check (4), engine (8), calibration (9, 10), bench (11), PX4 (12, 13), flights (14).

---

## 1 Hardware checklist

| Item | Need | How to check |
|---|---|---|
| Jetson | Orin NX 16 GB (recommended) or 8 GB, on a carrier with a 40-pin header or a free USB port for PX4 | `cat /proc/meminfo \| head -1` |
| Memory note | 8 GB is tight: SLAM, torch, the RealSense driver and MAVROS share it, and GPU memory cannot use swap | section 11 watches it with `tegrastats` |
| Storage | NVMe SSD for the root file system, swap and bags. Raw color records at 36.6 MB/s (848x480x3 bytes x 30 Hz) | `df -h /` shows `/dev/nvme0n1p1` |
| Cooling | a fan on the module. MAXN modes throttle when power or heat exceed the limit | `tegrastats` during section 11 |
| Power modes | MAXN_SUPER and 40W need JetPack 6.2 and a flash with the super config, and a carrier rated for it | section 2.2 |
| Camera | Intel RealSense D455. IMU chip BMI055 (SKU K83122-100) or BMI085 (SKU K83122-110/111, D455f). Firmware 5.17.3.10 or newer (recommended by librealsense 2.58.1) | section 4 |
| USB | a short USB 3 cable on a USB 3 port | `lsusb -t` shows the camera at `5000M` |
| Flight controller | PX4. A TELEM port wired to the Jetson UART: TX to RX, RX to TX, GND to GND. Check that both sides use the same logic level | section 12.1 |
| Jetson UART | on the Orin Nano devkit carrier (used by many NX kits) header pins 8 and 10 are `/dev/ttyTHS1` on JetPack 6. Other carriers differ | section 12.1 |
| Plan B link | a USB-serial adapter (`/dev/ttyUSB0`) or the flight controller's USB port (`/dev/ttyACM0`) | section 12.1 |
| Vehicle | a PX4 vehicle that already flies well in Stabilized and Altitude mode under RC. Sensor calibration done. RC loss action and kill switch set and tested. This guide does not cover vehicle setup | fly it under RC before section 14 |
| RC | an RC transmitter for the safety pilot, with a 3-position mode switch (Position, Altitude, Stabilized) and a kill switch | section 12.4 |
| Ground station | a laptop with QGroundControl and a link to the flight controller (USB or telemetry radio) | sections 12.2, 12.3, 13.4 |
| Jetson power | a regulator from the flight battery rated for the power mode of the Jetson (up to 25 W, or 40 W in super mode, table of 2.2), with margin | `tegrastats` under load in section 11 |
| Camera mount | rigid: the camera to IMU transform must not change. Vibration damped: the D455 IMU is inside the camera, so propeller vibration reaches it directly | section 9.4 |

Sources: power modes and super mode from the [Jetson Linux r36.4.3 power guide](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html) and the [JetPack 6.2 notes](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/release-notes/index.html). D455 data from the [RealSense D400 datasheet](https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf). Firmware from the [librealsense releases](https://github.com/realsenseai/librealsense/releases). UART name from the NVIDIA forum ([299825](https://forums.developer.nvidia.com/t/orin-nano-jetpack-6-0-uart1-issue/299825)).

---

## 2 Prepare the Jetson

### 2.1 Check JetPack

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack
uname -m
grep VERSION_ID /etc/os-release
```

Expected: a line starting with `# R36 (release), REVISION: 4`, a `nvidia-jetpack 6.x` version, `aarch64` and `VERSION_ID="22.04"`.

| JetPack | L4T | CUDA | TensorRT | OpenCV | Status for UC-SLAM |
|---|---|---|---|---|---|
| 6.0 | 36.3 | 12.2 | 8.6.2 | 4.8.0 | not recommended: TensorRT 8.6 untested, and the torch wheels of section 7 are CUDA 12.6 builds |
| 6.1 | 36.4.0 | 12.6 | 10.3.0 | 4.8.0 | use this or newer |
| 6.2 | 36.4.3 | 12.6 | 10.3.0 | 4.8.0 | use this or newer, super modes |
| 6.2.1 | 36.4.4 | 12.6 | 10.3.0 | 4.8.0 | use this or newer |

Source: NVIDIA apt indexes and release notes ([6.0](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-60/release-notes/index.html), [6.1](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-61/release-notes/index.html), [6.2](https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-62/release-notes/index.html), [6.2.1](https://docs.nvidia.com/jetson/jetpack/release-notes/index.html)).

### 2.2 Power mode and clocks

```bash
sudo nvpmodel -q
sudo nvpmodel -m 0
sudo jetson_clocks
sudo jetson_clocks --show
```

Check: `nvpmodel -q` prints `MAXN` (or `MAXN_SUPER` on a super flash). `jetson_clocks --show` lists CPU, GPU and EMC at their maximum.

| Module, flash | ID 0 | ID 1 | ID 2 | ID 3 | ID 4 | Default |
|---|---|---|---|---|---|---|
| NX 16 GB, standard | MAXN (8 CPU, GPU 918 MHz) | 10W | 15W | 25W (GPU 408 MHz) | n/a | 15W |
| NX 16 GB, super | MAXN_SUPER (GPU 1173 MHz) | 10W | 15W | 25W | 40W | 40W |
| NX 8 GB, standard | MAXN (6 CPU, GPU 765 MHz) | 10W | 15W | 20W (GPU 408 MHz) | n/a | 15W |
| NX 8 GB, super | MAXN_SUPER (GPU 1173 MHz) | 10W | 15W | 20W | 40W | 40W |

Notes:

- `nvpmodel` keeps its mode across reboots. A mode that changes the GPU TPC mask asks for a reboot.
- After `jetson_clocks` you must reboot before you change the nvpmodel mode again.
- Run `sudo jetson_clocks` again after each boot (section 17 can do it).
- The 25W and 20W modes cap the GPU at 408 MHz. Use mode 0 for the depth network.
- `setup_jetson.sh` never changes the power mode. It only prints it.

### 2.3 Swap on the NVMe for the build

The build of UC-SLAM needs much memory per compile job. Swap helps the CPU build only. GPU memory cannot be swapped.

```bash
df -h /
free -h
sudo fallocate -l 16G /swapfile_ucslam
sudo chmod 600 /swapfile_ucslam
sudo mkswap /swapfile_ucslam
sudo swapon /swapfile_ucslam
echo '/swapfile_ucslam none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

Check: `df -h /` shows the root on `/dev/nvme0n1...`. The last `free -h` shows about 16 GiB more swap. Run the block once only (it appends to `/etc/fstab`).

Optional: JetPack also uses zram swap in RAM. To turn it off, run `sudo systemctl disable nvzramconfig` and reboot. Source: [Jetson AI Lab RAM optimization](https://www.jetson-ai-lab.com/tutorials/ram-optimization) (unofficial).

### 2.4 Socket buffers for DDS

The default `net.core.rmem_max` (212992 bytes on the test desktop) is smaller than one 848x480 color image (1,221,120 bytes). DDS then drops images without an error. In the desktop replays about 230 of 3682 images never reached `slam_node`, and the odometry had gaps of 0.6 to 0.7 s. That is longer than the 0.4 s after which PX4 stops vision fusion.

`setup_jetson.sh --steps sysctl` does this step. By hand:

```bash
echo 'net.core.rmem_max=2147483647' | sudo tee /etc/sysctl.d/10-cyclone-max.conf
sudo sysctl -p /etc/sysctl.d/10-cyclone-max.conf
mkdir -p ~/.ros
cat > ~/.ros/cyclonedds.xml <<'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config
https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain id="any">
        <Internal>
            <SocketReceiveBufferSize min="10MB"/>
        </Internal>
    </Domain>
</CycloneDDS>
EOF
cat /proc/sys/net/core/rmem_max
```

Check: the last command prints `2147483647`. The file in `/etc/sysctl.d/` keeps the value after a reboot. Every ROS shell needs `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and `CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml`. The file `~/ucslam_ws/ucslam_env.sh` (section 6) sets both.

Source: the Cyclone DDS section of the ROS 2 [DDS tuning guide](https://docs.ros.org/en/humble/How-To-Guides/DDS-tuning.html).

### 2.5 Optional headless mode

The desktop uses about 800 MB of RAM. On an 8 GB module turn it off for flights:

```bash
sudo systemctl set-default multi-user.target
sudo systemctl disable nvargus-daemon
```

Undo with `sudo systemctl set-default graphical.target`. `nvargus-daemon` serves CSI cameras only, the D455 is USB. Both take effect after a reboot. Check after the reboot: `free -h` shows more available memory.

---

## 3 ROS 2 Humble and MAVROS

`setup_jetson.sh --steps deps` does 3.1 to 3.4, without the `apt upgrade` of 3.1 (it skips it on purpose, see below). It calls `apt-get install` with `--no-remove`: apt then stops with an error instead of removing a package. If that happens, read the list of packages apt wants to remove, and stop if it names any `nvidia` or `opencv` package.

### 3.1 The ROS 2 apt source

Install ROS 2 from `packages.ros.org`. Do not add the Isaac ROS apt repository on the host: a Humble install on JetPack 6.0 broke that way ([forum](https://forums.developer.nvidia.com/t/ros-humble-desktop-installation-broken-in-jetpack-6-0-on-jetson-agx-orin/295029)). These are the commands of the [ROS 2 Humble install guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html):

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update && sudo apt install -y curl
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update
```

The ROS guide asks for `apt upgrade` next, because old `systemd` and `udev` packages can make the ROS install remove system packages. On a Jetson the upgrade can also move the L4T packages: JetPack 6.1, 6.2 and 6.2.1 share the `r36.4` apt index. Look first:

```bash
apt list --upgradable 2>/dev/null | grep nvidia-l4t
```

If this lists packages, the upgrade moves L4T. Do it on purpose only, then reboot and redo 2.1. Then:

```bash
sudo apt upgrade
```

Check: `apt-cache policy ros-humble-ros-base` shows a candidate from `packages.ros.org`.

### 3.2 Packages

Without `-y`, apt shows its plan and asks before it does anything. With `--no-remove`, apt stops with an error when the plan would remove a package. Then read the list it prints and stop if it names any `nvidia` or `opencv` package.

```bash
sudo apt-get install --no-remove --no-install-recommends \
  ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp ros-humble-tf2-ros ros-humble-tf2-ros-py \
  ros-humble-message-filters ros-humble-diagnostic-msgs ros-humble-std-srvs ros-humble-nav-msgs \
  ros-humble-rosbag2-storage-mcap ros-humble-rosbag2-py python3-colcon-common-extensions \
  ros-humble-mavros ros-humble-mavros-extras ros-humble-realsense2-camera \
  python3-venv python3-pip rsync tmux \
  build-essential cmake git pkg-config libeigen3-dev libboost-serialization-dev libssl-dev \
  libglew-dev libgl1-mesa-dev libegl1-mesa-dev libjpeg-dev libpng-dev libepoxy-dev libopenblas-dev
```

Check:

```bash
source /opt/ros/humble/setup.bash
for p in mavros_extras realsense2_camera rmw_cyclonedds_cpp; do ros2 pkg prefix $p; done
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

Expected: three paths under `/opt/ros/humble`, then `10.3.0` (JetPack 6.1 or newer). If `tensorrt` is missing: `sudo apt install python3-libnvinfer`.

The `vision_pose` and `odometry` plugins of MAVROS are in `mavros_extras` (MAVROS 2.16.0 on the ROS arm64 index). The package names above were resolved by `apt-get install -s` in the x86 Humble container. They were not installed on arm64.

### 3.3 GeographicLib datasets for MAVROS

The [MAVROS README](https://github.com/mavlink/mavros/blob/ros2/mavros/README.md) runs the script with `ros2 run`. Under `sudo` the ROS setup must be sourced inside the root shell:

```bash
sudo bash -c 'source /opt/ros/humble/setup.bash && ros2 run mavros install_geographiclib_datasets.sh'
ls /usr/share/GeographicLib
```

Check: the folder lists `geoids` and other datasets. The same script by its path, as in the [Seeed wiki](https://wiki.seeedstudio.com/control_px4_with_recomputer_jetson/) (unofficial), and as `setup_jetson.sh` runs it: `sudo bash /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh`.

### 3.4 Serial port access

```bash
sudo usermod -aG dialout $USER
```

Log out and in again. Check: `id -nG` lists `dialout`.

### 3.5 OpenCV: keep JetPack's

- JetPack ships NVIDIA OpenCV 4.8.0 (`nvidia-opencv`). Keep it. Do not purge it and do not install Ubuntu's `libopencv-dev`.
- `ros-humble-realsense2-camera` pulls `cv_bridge`, which pulls Ubuntu's OpenCV 4.5.4 runtime (`*.so.4.5d`). Both runtimes then sit side by side. That is fine: they live in different processes.
- `slam_node` no longer uses cv_bridge. Its CMake reads the OpenCV of `libORB_SLAM3` from `cpp/ucslam/build/CMakeCache.txt` and links only that one. It stops with an error on a mismatch.
- The check is `ldd` on `slam_node` after the build (section 6.4).

```bash
pkg-config --modversion opencv4
dpkg-query -W 'libopencv-core*'
```

Expected: `4.8.0` for the build. The second command may list both 4.8 and 4.5d runtime packages.

---

## 4 RealSense D455

### 4.1 Try the apt driver first

Section 3.2 installed `ros-humble-realsense2-camera` 4.58.4 with `ros-humble-librealsense2` 2.58.4. Its backend is not documented. Reports say JetPack 6 kernels have the HID sensor drivers off, so the native backend may give images but **no IMU**. The RSUSB backend works without kernel changes ([forum 298870](https://forums.developer.nvidia.com/t/realsense-d435i-not-working-with-jetson-orin-nano-8gb-jetpack-6-0/298870), unofficial).

**CHECK ON THE JETSON.** Plug the D455 into a USB 3 port and run:

```bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true rgb_camera.color_profile:=848x480x30 enable_depth:=false enable_gyro:=true enable_accel:=true gyro_fps:=200 accel_fps:=250 unite_imu_method:=2
```

In a second terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/imu
```

Expected: about 30 Hz for the color topic and about 200 Hz for the IMU topic. If the IMU topic is silent, try `accel_fps:=200` (BMI085, section 4.2). If it stays silent, go to 4.3.

### 4.2 Find the IMU chip

The two chips share no accelerometer rate (D455 datasheet tables 4-25 and 4-26):

| Chip | Units | Accelerometer rates | Gyroscope rates | `imu_chip:=` |
|---|---|---|---|---|
| BMI055 | SKU K83122-100 | 62.5 or 250 Hz | 200 or 400 Hz | `bmi055` (accel 250 Hz) |
| BMI085 | SKU K83122-110/111, D455f | 100 or 200 Hz | 200 or 400 Hz | `bmi085` (accel 200 Hz) |

With the librealsense tools installed, `rs-enumerate-devices` lists the Accel profiles: 63/250 Hz means BMI055, 100/200 Hz means BMI085. The real test is the `accel_fps` that works in 4.1. Write the chip down. Every D455 launch below uses `imu_chip:=bmi055`. Replace it with `bmi085` if that is your chip. `imu_chip:=auto` (the default) needs `pyrealsense2`, which this guide does not install.

### 4.3 Fallback: librealsense with the RSUSB backend from source

Use this only if 4.1 gave no IMU. Remove the apt driver first. Read the list that `apt` prints and stop if it names any OpenCV package:

```bash
sudo apt remove ros-humble-realsense2-camera ros-humble-librealsense2
```

Build librealsense 2.58.x with RSUSB and CUDA ([installation_jetson.md](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md) lists the build dependencies):

```bash
git clone https://github.com/realsenseai/librealsense.git ~/librealsense
git -C ~/librealsense tag --list '*2.58*'
```

The next command checks out the newest tag of that list. Check that it is a 2.58 release, then build:

```bash
git -C ~/librealsense checkout "$(git -C ~/librealsense tag --list '*2.58*' | sort -V | tail -1)"
mkdir -p ~/librealsense/build && cd ~/librealsense/build
cmake .. -DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j2
sudo make install
cd ~/librealsense && sudo ./scripts/setup_udev_rules.sh
```

Then build realsense-ros 4.58.x from source in its own workspace:

```bash
mkdir -p ~/rs_ws/src && cd ~/rs_ws/src
git clone -b ros2-master https://github.com/realsenseai/realsense-ros.git
git -C realsense-ros tag --list '*4.58*'
git -C realsense-ros checkout "$(git -C realsense-ros tag --list '*4.58*' | sort -V | tail -1)"
cd ~/rs_ws && source /opt/ros/humble/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Check: `rs-enumerate-devices` lists the D455 with a Motion Module. Then repeat the test of 4.1 with `source ~/rs_ws/install/setup.bash` instead of `/opt/ros/humble/setup.bash`.

Keep the workspace at `~/rs_ws`. When `~/rs_ws/install/setup.bash` exists, the `ws` step of `setup_jetson.sh` sources it before the colcon build and writes it into `~/ucslam_ws/ucslam_env.sh`, so the UC-SLAM workspace chains to it. Use `--no-realsense-apt` with the script, so the deps step does not install the apt driver again. After the `ws` step (section 6), check:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 pkg prefix realsense2_camera
```

Expected: a path under `~/rs_ws/install`, not `/opt/ros/humble`. This path is not tested. The librealsense doc names a trade-off of RSUSB: "performance and functional limitations (e.g. multi-cam)".

### 4.4 Firmware

`rs-enumerate-devices` (librealsense tools) and `realsense-viewer` show the firmware version. librealsense 2.58.1 recommends D455 firmware 5.17.3.10 or newer. Update with the RealSense firmware tool if it is older. Some users fixed IMU freezes with a firmware change (forum reports, unofficial).

### 4.5 What to note for later

| Item | Your value | Used in |
|---|---|---|
| IMU chip | bmi055 or bmi085 | every D455 launch (`imu_chip:=`) |
| Real IMU rate of `/camera/camera/imu` | about 200 Hz | written into the settings by `make_d455_settings.py` |
| Color rate | about 30 Hz | sections 9 and 11 |

---

## 5 Get the code and the model weights onto the Jetson

### 5.1 Option A: clone

```bash
git clone --recurse-submodules https://github.com/pairs-lab/uc-slam.git ~/ucslam
```

At the time of writing, the Jetson files (`scripts/jetson/`, `configs/d455/`, `ros/ros2/ucslam_px4/`, `ucslam/trt_depth.py` and others) are not committed yet. Commit and push them first, or use option B. The repository is private: the Jetson needs your GitHub credentials.

### 5.2 Option B: copy from the desktop

**On the desktop.** Write the exclude list once. It leaves out data, run outputs, other baselines and every build folder (an x86 build folder breaks the aarch64 build):

```bash
cat > ~/ucslam_rsync_exclude.txt <<'EOF'
/data/
/runs/
/papers/
/.venv/
/.ruff_cache/
/.pytest_cache/
__pycache__/
/third_party/baselines/
/third_party/ORB_SLAM3_dump/
/third_party/Pangolin/build/
/third_party/Pangolin/install/
/third_party/ORB_SLAM3/build/
/third_party/ORB_SLAM3/lib/
/third_party/ORB_SLAM3/Thirdparty/*/build/
/third_party/ORB_SLAM3/Thirdparty/*/lib/
/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt
/cpp/ucslam/build/
/cpp/ucslam/lib/
/cpp/ucslam/Thirdparty/*/build/
/cpp/ucslam/Thirdparty/*/lib/
/cpp/ucslam/Vocabulary/
TUM_IMU/
EuRoC_IMU/
TUM_TimeStamps/
EuRoC_TimeStamps/
EOF
```

Then copy. Set `JETSON` to your login on the Jetson, for example `JETSON=me@orin.local`:

```bash
JETSON=me@orin.local
rsync -a --info=progress2 --exclude-from=$HOME/ucslam_rsync_exclude.txt ~/ucslam/ $JETSON:ucslam/
```

The copy keeps `.git`, because `./build.sh` runs `git submodule update`. Measured on the desktop on 2026-10-28 (`rsync --stats`): about 750 MB, 8400 files.

Check on the Jetson:

```bash
du -sh ~/ucslam
ls ~/ucslam/scripts/jetson
git -C ~/ucslam status --short | grep -v '^ D' | head -20
```

Expected: about 750M, the six files of `scripts/jetson`, and the new and changed files of the desktop. `git status` also lists the excluded IMU and timestamp files as deleted (` D`). They are benchmark data and are not needed here.

**The first `./build.sh` needs internet** (option A and B). It runs `git submodule update --init --recursive`, which fetches the `pybind11` submodule of Pangolin. The build does not use it, but the command fails without it.

### 5.3 The model weights

UC-SLAM uses `da2-metric-indoor-base` = `depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf` at revision `c6d9784685727bfc6d0a7b5452ce94afaee1e7f5` (pinned in `ucslam/depthnet.py`). The cache folder is 372 MB.

With internet on the Jetson, `setup_jetson.sh --steps model` downloads it (section 7 first, it needs the venv). Without internet, copy it **on the desktop**:

```bash
ssh $JETSON mkdir -p .cache/huggingface/hub
rsync -a ~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Metric-Indoor-Base-hf $JETSON:.cache/huggingface/hub/
```

Check on the Jetson:

```bash
ls -L ~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Metric-Indoor-Base-hf/snapshots/c6d9784685727bfc6d0a7b5452ce94afaee1e7f5
```

Expected: `config.json  model.safetensors  preprocessor_config.json`. The nodes run with `HF_HOME=~/.cache/huggingface` and `HF_HUB_OFFLINE=1` (set by `ucslam_env.sh`), so they never go online.

---

## 6 Build

### 6.1 With the script

```bash
cd ~/ucslam
scripts/jetson/setup_jetson.sh --steps check
```

Read every `[FAIL]` and `[WARN]` line. Then run the rest. Run it as your user. Do not run it with `sudo`: the script then stops with exit 2, because `HOME` would be `/root`. `deps` and `sysctl` call sudo themselves and ask for your password:

```bash
scripts/jetson/setup_jetson.sh
```

The default runs every step in order. A `[FAIL]` in the check step stops the run before anything is changed. Each step can run again without harm. The log of each run is in `~/ucslam_logs/setup/`.

| Step | What it does | sudo |
|---|---|---|
| `check` | read only: JetPack, CUDA, TensorRT, OpenCV, Python 3.10, ROS, apt sources, power mode, memory and swap, disk, `rmem_max`, dialout, serial ports, repository state (copied build folders, x86 libraries) | no |
| `deps` | section 3: ROS apt source, packages, MAVROS, GeographicLib, RealSense apt driver, dialout. No `apt upgrade` (see 3.1) | yes |
| `sysctl` | section 2.4: `/etc/sysctl.d/10-cyclone-max.conf` and `~/.ros/cyclonedds.xml` | yes |
| `build` | `./build.sh` (Pangolin, vocabulary, `libORB_SLAM3`) | no |
| `ws` | `colcon build` of `ros/ros2` into `~/ucslam_ws` (chained to `~/rs_ws` when it exists, 4.3), the `ldd` check, writes `~/ucslam_ws/ucslam_env.sh` | no |
| `venv` | section 7 | no |
| `model` | section 5.3: checks or downloads the pinned weights | no |
| `verify` | `ldd`, the four packages, the three launch files, MAVROS and RealSense packages, Python imports, the transformers gate, the torchvision processor, a short network run | no |

| Option | Default | Meaning |
|---|---|---|
| `--steps a,b` | all | run only these steps (always in the order above) |
| `--dry-run` | off | print the commands that change something, run only read-only checks |
| `--jobs N` | RAM in GiB / 3 | parallel compile jobs |
| `--repo DIR` | the repository of this script | UC-SLAM repository |
| `--ws DIR` | `~/ucslam_ws` | colcon workspace |
| `--venv DIR` | `~/ucslam-venv` | Python venv of the depth node |
| `--hf-home DIR` | `$HF_HOME` or `~/.cache/huggingface` | Hugging Face home |
| `--engine FILE` | none | verify also runs this TensorRT engine once |
| `--backend B` | `trt-fp16` | backend of `--engine` |
| `--no-realsense-apt` | off | deps skips `ros-humble-realsense2-camera` (section 4.3 users) |
| `--no-net-test` | off | verify skips the network run |
| `--log-dir DIR` | `~/ucslam_logs/setup` | log folder |
| `-h`, `--help` | | print the options and exit |

The script installs with `apt-get install --no-remove`, so apt stops instead of removing a package (3). It never purges a package, never runs `apt upgrade`, never reboots, and never changes `nvpmodel` or `jetson_clocks`. Exit status: 0 when no check failed, 1 otherwise, 2 for a usage error or a run under sudo.

### 6.2 Jobs by memory

| Module | RAM seen by Linux | Default `--jobs` | If the build is killed (OOM) |
|---|---|---|---|
| Orin NX 8 GB | about 7 GiB | 2 | add swap (2.3), then `--jobs 1` |
| Orin NX 16 GB | about 15 GiB | 5 | `--jobs 3` |

This rule (RAM in GiB divided by 3) is a starting point, not a measurement. The build time on the Orin is **not measured**. For scale only: in the x86 test container (8 jobs, host load average about 40) Pangolin took 52 s, the UC-SLAM library about 15 min and the workspace 2 min 14 s.

### 6.3 By hand

```bash
cd ~/ucslam
env -u CMAKE_PREFIX_PATH -u AMENT_PREFIX_PATH JOBS=2 UCSLAM_CMAKE_ARGS=-DCMAKE_DISABLE_FIND_PACKAGE_realsense2=TRUE ./build.sh
ls -l ~/ucslam/cpp/ucslam/lib/libORB_SLAM3.so
grep '^OpenCV_DIR:' ~/ucslam/cpp/ucslam/build/CMakeCache.txt
```

Expected: the library exists and `OpenCV_DIR` points to NVIDIA's OpenCV 4.8 (`/usr/lib/aarch64-linux-gnu/cmake/opencv4` or similar). The CMake argument leaves out the RealSense examples of ORB-SLAM3, as in the tested build.

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/ucslam_ws && cd ~/ucslam_ws
MAKEFLAGS=-j2 colcon build --base-paths ~/ucslam/ros/ros2 --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release
grep -h 'slam_node uses OpenCV' ~/ucslam_ws/log/latest_build/ucslam_ros/stdout.log
```

Expected: `4 packages finished` and a line `slam_node uses OpenCV 4.8.0 from ... (libORB_SLAM3: 4.8.0 from ...)`.

### 6.4 The one-OpenCV check (CHECK ON THE JETSON)

```bash
source ~/ucslam_ws/ucslam_env.sh
ldd ~/ucslam_ws/install/ucslam_ros/lib/ucslam_ros/slam_node | grep -c libopencv_core
ldd ~/ucslam_ws/install/ucslam_ros/lib/ucslam_ros/slam_node | grep -E 'cv_bridge|not found'
```

Expected: `1`, then no output. Without the `source` line, `ldd` prints the ROS libraries behind `slam_node` as `not found`. In the x86 container the one library is `libopencv_core.so.4.5d`. On the Jetson it should be the 4.8 library. Two lines mean two OpenCV versions in one process: rebuild UC-SLAM and the workspace against one OpenCV. Never pass another `OpenCV_DIR` to colcon, the node CMake forces the library's.

`-march=native` compiles for the Orin CPU. Results on the Jetson are therefore not the paper's benchmark numbers.

### 6.5 The environment file

The `ws` step writes `~/ucslam_ws/ucslam_env.sh`. By hand, create it with this content (adapt the paths):

```bash
source /opt/ros/humble/setup.bash
# only with the RealSense source build of 4.3:
# source "$HOME/rs_ws/install/setup.bash"
source "$HOME/ucslam_ws/install/setup.bash"
export UCSLAM_REPO="$HOME/ucslam"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if [ -f "$HOME/.ros/cyclonedds.xml" ]; then export CYCLONEDDS_URI="file://$HOME/.ros/cyclonedds.xml"; fi
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
if [ -f "$HOME/ucslam-venv/bin/activate" ]; then source "$HOME/ucslam-venv/bin/activate"; fi
```

`depth_node` runs with the first `python3` on `PATH`. That must be the venv. So always source this file before a launch.

---

## 7 Python environment of the depth node

`setup_jetson.sh --steps venv` does this section.

| Package | Version | Where from | Why |
|---|---|---|---|
| Python | 3.10 | JetPack (Ubuntu 22.04) | rclpy of Humble and the Jetson wheels are cp310 |
| numpy | below 2 | PyPI | ROS Humble modules are built against numpy 1.x |
| torch | 2.11.0 | [Jetson AI Lab index](https://pypi.jetson-ai-lab.io/jp6/cu126) `jp6/cu126` | same version as `ros/docker/Dockerfile` (the depth cache environment) |
| torchvision | 0.26.0 | same index | the HF processor of the depth cache needs it |
| transformers | 5.17.0 | PyPI | pinned in the repository |
| pyyaml, scipy | any | PyPI | depth node and fit tool |
| tensorrt | 10.3.0 | JetPack apt `python3-libnvinfer`, seen through `--system-site-packages` | trt-* backends |

The Jetson AI Lab index is community run. NVIDIA supports only the default JetPack environment, and packages were removed from these indexes before (Oct 2025, [forum](https://forums.developer.nvidia.com/t/packages-removed-from-pypi-jetson-ai-lab-io-indexes/348376)). Keep a copy of the wheels (7.3).

### 7.1 Install

```bash
python3 -m venv --system-site-packages ~/ucslam-venv
echo 'numpy<2' > ~/ucslam-venv/pip-constraints.txt
~/ucslam-venv/bin/python -m pip install -c ~/ucslam-venv/pip-constraints.txt "numpy<2"
~/ucslam-venv/bin/python -m pip install -c ~/ucslam-venv/pip-constraints.txt --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.11.0 torchvision==0.26.0
~/ucslam-venv/bin/python -m pip install -c ~/ucslam-venv/pip-constraints.txt --index-url https://pypi.org/simple torch==2.11.0 torchvision==0.26.0 transformers==5.17.0 pyyaml scipy
```

Why three calls: torch and torchvision must come from the Jetson index only. With PyPI as an extra index, pip could pick PyPI's aarch64 torch, which is not a Jetson build. The third call finds torch 2.11.0 already installed and only adds its dependencies and transformers from PyPI.

### 7.2 Checks (CHECK ON THE JETSON)

```bash
source ~/ucslam_ws/ucslam_env.sh
python -c "import torch, torchvision, transformers, numpy; print(torch.__version__, torchvision.__version__, transformers.__version__, numpy.__version__, torch.cuda.is_available())"
python -c "from transformers.utils import is_torch_available; print(is_torch_available())"
python -c "import tensorrt, rclpy; print(tensorrt.__version__)"
```

Expected: `2.11.0 0.26.0 5.17.0 1.x True`, then `True`, then `10.3.0`.

**The transformers torch gate.** transformers 5.17 turns torch off when torch is below 2.5.0. NVIDIA's own JetPack 6.1 wheel is `2.5.0a0+872d972e41.nv24.08`, and a pre-release `2.5.0a0` counts as below 2.5.0. With that wheel `is_torch_available()` prints `False` and the model load fails. torch 2.11.0 passes the gate.

**The torchvision processor.** Without torchvision, transformers silently uses the PIL processor. The depth maps then differ a little from the ones the constants were fitted on (p99 d log z about 1.6e-3 on EuRoC). The depth node logs one warning (`not the torchvision processor`). Check:

```bash
python - <<'EOF'
import os, sys
sys.path.insert(0, os.environ["UCSLAM_REPO"])
from ucslam.live_depth import LiveDepth
ld = LiveDepth(None, "cuda")
print("processor ok" if ld.processor_warning is None else ld.processor_warning)
EOF
```

Expected: `processor ok`. `setup_jetson.sh --steps verify` runs this check and a short network run.

### 7.3 Keep the wheels

```bash
mkdir -p ~/ucslam_wheels
~/ucslam-venv/bin/python -m pip download --no-deps -d ~/ucslam_wheels --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.11.0 torchvision==0.26.0
ls -lh ~/ucslam_wheels
```

To reinstall from the copy: `pip install --no-deps ~/ucslam_wheels/*.whl`.

---

## 8 TensorRT engine

The trt-fp16 backend is several times faster than torch-fp32 on the desktop (net_ms p50 36 ms vs 78 to 148 ms, noisy, x86). Orin numbers are not measured. An estimate from AGX Orin numbers puts the Base model at 70 to 100 ms on an NX 16 GB in MAXN with FP16 (an inference, not a measurement). Keyframes come at a few Hz, so this can fit.

An engine runs only on the GPU and TensorRT version it was built on, and at one input size. 848x480 images give a 518x910 network input.

### 8.1 Export the ONNX file

**On the desktop** (tested, 35 s, 389 MB):

```bash
cd ~/ucslam
mkdir -p ~/ucslam_engines
HF_HUB_OFFLINE=1 uv run --frozen --with onnx==1.23.2 python scripts/jetson/export_depth_onnx.py --width 848 --height 480 --out ~/ucslam_engines/da2b_518x910.onnx
ssh $JETSON mkdir -p ucslam_engines
rsync -a ~/ucslam_engines/da2b_518x910.onnx ~/ucslam_engines/da2b_518x910.onnx.json $JETSON:ucslam_engines/
```

Expected: `image 848x480 -> network input (1, 3, 518, 910)` and `opset 17`. Copy the `.onnx.json` record too.

Or on the Jetson (not tested on aarch64): `source ~/ucslam_ws/ucslam_env.sh && pip install onnx==1.23.2`, then the same `export_depth_onnx.py` command with `python` instead of `uv run ...`.

### 8.2 Build the engine on the Jetson (CHECK ON THE JETSON)

Stop UC-SLAM, set MAXN (2.2), then:

```bash
source ~/ucslam_ws/ucslam_env.sh
cd ~/ucslam
scripts/jetson/build_trt_engine.sh ~/ucslam_engines/da2b_518x910.onnx ~/ucslam_engines/da2b_518x910_fp16n.engine
```

Expected lines: `precision: fp16 (LayerNormalization and Softmax in FP32)`, `TensorRT version: 10.3.0` and `use it with: backend:=trt-fp16 engine:=...`. On the desktop the build took about 4 min and the engine was 191 to 194 MB.

| Option | Meaning | Status |
|---|---|---|
| (default) | FP16 with LayerNormalization and Softmax in FP32. Backend `trt-fp16` | passed parity on x86 (p99 0.0093) |
| `--plain-fp16` | FP16 everywhere | failed parity on x86 (p99 0.0220 > 0.02). Do not fly it |
| `--fp32` | strict FP32, TF32 off. Backend `trt-fp32` | slower |
| `--workspace 1024` | lower builder memory (default 2048 MiB) | use on an 8 GB module if the build fails |
| `--no-trtexec` | Python builder even when `trtexec` exists | the branch tested on x86 |
| `--python /usr/bin/python3` | Python with the `tensorrt` package | use if the venv cannot import it |

On the Jetson the script uses `/usr/src/tensorrt/bin/trtexec`. **That branch never ran with a real trtexec.** It passes about 40 layer names with `--layerPrecisions`, and the TensorRT fusion may ignore some. So the parity check in 8.3 is required. If the `trtexec` build fails, build again with `--no-trtexec`. Keep `<engine>.json` next to the engine: the backend refuses an engine whose record names another model or precision. Give each build its own file name.

### 8.3 Parity check on D455 frames

Parity compares the engine with torch-fp32 on the same frames, through the live pipeline. You need D455 frames. Record 40 s with the camera (60 frames at one in 15 need 900 images, 30 s at 30 Hz), moving it slowly (section 9.1 explains the launch):

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch ucslam_ros ucslam_d455.launch.py mode:=calib record:=true imu_chip:=bmi055
```

Stop with Ctrl-C. Extract 60 gray frames, one every 15 images, converted like the node does (`cvtColor` RGB to gray):

```bash
BAG=$(ls -d ~/ucslam_bags/d455_calib_* | tail -1)
python3 - $BAG ~/d455_frames <<'EOF'
import os, sys
sys.path.insert(0, os.path.join(os.environ["UCSLAM_REPO"], "scripts", "jetson"))
import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from make_d455_settings import open_bag
bag, out = sys.argv[1], sys.argv[2]
topic = sys.argv[3] if len(sys.argv) > 3 else "/camera/camera/color/image_raw"
os.makedirs(out, exist_ok=True)
reader, types = open_bag(bag, [topic])
if not types:
    sys.exit(f"{topic} is not in {bag}")
k = saved = 0
while reader.has_next() and saved < 60:
    _, data, _ = reader.read_next()
    k += 1
    if k % 15:
        continue
    m = deserialize_message(data, Image)
    ch = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(m.encoding)
    if ch is None:
        sys.exit(f"encoding {m.encoding} not handled")
    img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * ch].reshape(m.height, m.width, ch)
    if ch == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if m.encoding == "rgb8" else cv2.COLOR_BGR2GRAY)
    stamp = m.header.stamp.sec * 1000000000 + m.header.stamp.nanosec
    cv2.imwrite(os.path.join(out, f"{stamp}.png"), img.reshape(m.height, m.width))
    saved += 1
print(f"saved {saved} gray frames in {out}")
EOF
```

Expected: `saved 60 gray frames in /home/<you>/d455_frames`. (Tested on the x86 container with a mono8 EuRoC bag and a synthetic rgb8 message.)

Run the parity check:

```bash
cd ~/ucslam
python scripts/jetson/depth_parity.py --backend trt-fp16 --engine ~/ucslam_engines/da2b_518x910_fp16n.engine --images ~/d455_frames --n 30 --json ~/ucslam_engines/parity_d455_fp16n.json
echo "exit $?"
```

| Result | Limit | x86 desktop, 30 OpenLORIS 848x480 frames, default engine |
|---|---|---|
| \|median d log z\| | at most 0.005 | -6.1e-4 |
| p99 \|d log z\| | at most 0.02 | 0.0093 |
| valid-mask agreement per frame | at least 0.999 | 1.000 |
| exit status | 0 pass, 1 parity fail, 2 error | 0, `PASS` |

Read `net_ms trt-fp16: p50` and `net_ms torch-fp32: p50` in the output. These are the first Orin timing numbers. Write them down.

### 8.4 If parity fails

1. Check that you did not use `--plain-fp16`. Rebuild with the default.
2. Rebuild with `--no-trtexec` (the tested builder) and run parity again.
3. Use `--fp32` (backend `trt-fp32`) or the torch backends (`torch-fp16`, `torch-fp32`). Run parity for any non-default backend.
4. A mask agreement below 0.999 means NaN or inf in the engine output. Never fly that engine.

Whatever backend you choose, the depth constants of section 10 must be fitted with that exact backend and engine. A passing backend still changes the maps a little (mean d log z -1.5e-3 on MH_01).

---

## 9 Calibrate and write the settings

`scripts/jetson/make_d455_settings.py` writes both settings files into `configs/d455/` from a bag or the live camera. It reads the intrinsics from `camera_info`, `IMU.T_b_c1` from `/tf_static` (or Kalibr), and measures the IMU rate. It refuses to write a file that fails its checks.

All D455 recordings go to `~/ucslam_bags/d455_<mode>_<time>` (MCAP).

| Recording | Topics with images | Rate on disk (computed) |
|---|---|---|
| `mode:=calib` | color rgb8 + aligned depth 16UC1 | about 61 MB/s |
| `mode:=mono` | color rgb8 | about 37 MB/s |
| `mode:=stereo` | infra1 + infra2 mono8 | about 24 MB/s |
| any mode with `record_images:=false` | none | small |

### 9.1 The stereo file first

Stereo-inertial is the planned flight mode. It needs no depth constants. In a EuRoC MH_01 stereo replay through `slam_node` it tracked every frame with no map reset when the CPU kept up. When images were lost because the CPU was overloaded, the map reset 7 times. It has not run on D455 data yet (section 11.1 tests it).

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo slam:=false record:=true imu_chip:=bmi055
```

Hold the camera level and still for 10 s, then stop with Ctrl-C.

```bash
cd ~/ucslam
BAG=$(ls -d ~/ucslam_bags/d455_stereo_* | tail -1)
python3 scripts/jetson/make_d455_settings.py --bag $BAG --which stereo --imu-rate 200
```

Check the printout:

| Printed item | Expected |
|---|---|
| IMU sampling rate | about 200 Hz, few missing samples, no long gaps |
| mean accelerometer | y near -9.8 (optical IMU frame, camera level and still) |
| stereo baseline cross-check against `/tf_static` | about 95 mm (D455 datasheet), agreeing |
| written file | `configs/d455/d455_stereo_inertial.yaml` |

**CHECK ON THE JETSON:** the IR emitter must be off in stereo mode (the dot pattern moves with the camera and breaks tracking). The launch passes `configs/d455/realsense_stereo.yaml` with `depth_module.emitter_enabled: 0`. Look at the infra1 image (for example with `rqt_image_view` if installed, or one frame from the bag) and check that it shows no dots.

The stereo file uses `Camera.type: "PinHole"` with zero distortion and `Stereo.T_c1_c2`, not `"Rectified"`. ORB-SLAM3 v1.0 crashes on a Rectified stereo-inertial file (`slam_node` exit -11). Do not edit it back. The stock `RealSense_D435i.yaml` stereo-inertial example crashes `slam_node` for the same reason.

The file has domain `d455ir` with the depth prior off (`UCSLAM.PriorNet 0`). No infrared constants exist.

### 9.2 The mono file (no constants yet)

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch ucslam_ros ucslam_d455.launch.py mode:=calib record:=true imu_chip:=bmi055
```

Record 30 to 60 s, moving slowly, then Ctrl-C.

```bash
cd ~/ucslam
BAG=$(ls -d ~/ucslam_bags/d455_calib_* | tail -1)
python3 scripts/jetson/make_d455_settings.py --bag $BAG --which mono --imu-rate 200
```

Expected: `configs/d455/d455_mono_inertial.yaml` with a warning block. Without constants it has `UCSLAM.PriorNet 0`, `UseDepthFactor 0` and `MaxDepthPriorPoints 0`. An unfitted prior is never switched on.

Live instead of a bag: launch with `slam:=false`, then `make_d455_settings.py --live --duration 10`.

```bash
grep -E '^(Camera.type|Camera.width|Camera.height|Camera.fps|IMU.Frequency|UCSLAM.Dataset|UCSLAM.PriorNet|UCSLAM.DepthBackend)' configs/d455/*.yaml
```

Check: width 848, height 480, fps 30, `IMU.Frequency` near 200, domains `d455color` and `d455ir`. The launch refuses a stream whose size differs from `Camera.width x height x fps`.

### 9.3 Kalibr before flight (recommended)

`/tf_static` holds factory design values for the camera to IMU transform, not a per-unit calibration. Wrong extrinsics break the IMU initialization. Run a Kalibr camera-IMU calibration and pass its `camchain-imucam.yaml`. The two file names below are examples: use your Kalibr output files (one for color with the IMU, one for the infrared pair with the IMU).

```bash
python3 scripts/jetson/make_d455_settings.py --bag $BAG --which mono --kalibr ~/kalibr/camchain-imucam.yaml --kalibr-intrinsics
python3 scripts/jetson/make_d455_settings.py --bag $(ls -d ~/ucslam_bags/d455_stereo_* | tail -1) --which stereo --kalibr ~/kalibr/camchain-imucam-ir.yaml --kalibr-cam cam0
```

| Rule | Detail |
|---|---|
| one file per call | `--which mono` with the color camchain, or `--which stereo` with `--kalibr-cam` naming the infra1 camera of the Kalibr file |
| size must match | the Kalibr resolution (and rostopic, when present) must match the stream |
| difference warning | more than 2 cm or 2 deg from `/tf_static` prints a warning |
| time offset | the script prints `imu_time_offset:=<value>` (minus the Kalibr timeshift). Pass it to every launch. Without Kalibr it is `0.0`, the default |
| color intrinsics | `--kalibr-intrinsics` is the safe choice. Without it the D455 color distortion from `camera_info` (inverse Brown-Conrady) is written as OpenCV k1 k2 p1 p2 k3. That is believed equivalent but not checked on hardware |

### 9.4 IMU noise

| Key | Start value (D435i values of the stock file) | Option |
|---|---|---|
| `IMU.NoiseGyro` | 1e-3 rad/s^0.5 | `--noise-gyro` |
| `IMU.NoiseAcc` | 1e-2 m/s^1.5 | `--noise-acc` |
| `IMU.GyroWalk` | 1e-6 rad/s^1.5 | `--gyro-walk` |
| `IMU.AccWalk` | 1e-4 m/s^2.5 | `--acc-walk` |
| white-noise factor | 1.0 on the bench | `--imu-noise-scale` |

On the UAV, propeller vibration raises the IMU noise. Mount the vehicle with the motors on and the propellers off, record (`record_images:=false`), and look at the accelerometer spread. This means an armed vehicle on the bench: clamp the vehicle, keep the kill switch in hand, and keep everybody away from the motors. Start with `--imu-noise-scale 2` to `5` for flight. Props off gives less vibration than flight: record the IMU again during ladder step 2 of section 14 (`record_images:=false`) and check the noise scale. Allan variance values, when you have them, go in with `--noise-*` and `--*-walk`.

### 9.5 The UCSLAM.DepthBackend link

A mono file written with fitted constants (section 10) ends with:

```text
UCSLAM.DepthBackend: "trt-fp16"
D455.FitEngine: "/home/<you>/ucslam_engines/da2b_518x910_fp16n.engine"
```

| Who | What it does |
|---|---|
| `ucslam_d455.launch.py` | `backend:=` empty takes `UCSLAM.DepthBackend`, `engine:=` empty takes `D455.FitEngine`. Another `backend:=` stops the launch, also with `depth:=off`. Another engine only gives a warning |
| `slam_node` | stops at start with a FATAL message when its `depth_backend` differs and depth is live or file. With `depth:=off` it only logs the key |
| without the key | the backend defaults to `torch-fp32` |

The engine path must not contain spaces, quotes or `#`.

---

## 10 Fit the depth constants of the d455color domain

The depth prior needs four constants per camera domain: `UCSLAM.LogBias`, `SigmaNetC0`, `SigmaNetC1` and `NeffOverN`. The D455 color camera is a new domain, `d455color`. Fit them with the **backend and engine you fly with**. Another backend or engine needs another fit. `configs/default.yaml` is never changed. The constants live in `configs/d455/`.

### 10.1 Record

Record one bag per room of the flight area, 30 to 60 s each, moving slowly, with 0.3 to 5 m of scene depth. At least three rooms: two for the fit and one held out.

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch ucslam_ros ucslam_d455.launch.py mode:=calib record:=true imu_chip:=bmi055
```

Repeat per room. Check the disk: calib bags grow at about 61 MB/s.

### 10.2 Fit (CHECK ON THE JETSON: trt-* fits never ran)

```bash
cd ~/ucslam
mapfile -t BAGS < <(ls -d ~/ucslam_bags/d455_calib_*)
printf '%s\n' "${BAGS[@]}"
python3 scripts/jetson/fit_depth_constants.py --bags "${BAGS[@]}" --fit "$(basename "${BAGS[0]}")" "$(basename "${BAGS[1]}")" --backend trt-fp16 --engine ~/ucslam_engines/da2b_518x910_fp16n.engine --settings configs/d455/d455_mono_inertial.yaml
```

The first two bags are the fit split, the others are held out. Remove the bag of 8.3 from the list if it is not a room of the flight area.

| Option | Default | Meaning |
|---|---|---|
| `--keyframes` | 40 | frames per recording |
| `--tol-ms` | 5 | color to depth stamp tolerance. Raise it if no pairs are found |
| `--z-min`, `--z-max` | 0.3, 5 m | reference depth range |
| `--ref-noise` | 0.01 (per metre of depth) | reference noise, the paper's D435i value |
| `--keypoints` | `orb` | ORB like ORB-SLAM3 (what PriorNet samples in flight). `xfeat` reproduces the paper's OpenLORIS fit |
| `--sample` | `full` | full-resolution prediction (paper). `map`: the half-resolution map of the node |
| `--device` | `cuda` | a CPU fit needs `--device cpu`, never chosen silently |

Outputs: `configs/d455/d455color_depth_fit.json` (every number) and `configs/d455/d455color_depth_constants.yaml`.

Checks:

| Check | Expected |
|---|---|
| held-out "inside 1 sigma" in the printed table | near 0.68 |
| bias per room | stable between rooms |
| no `no color frame has a depth within` message | color and aligned depth share a frameset stamp |

Reference (desktop, D435i OpenLORIS, torch-fp32): XFeat keypoints gave log bias 0.1530 vs 0.1532 in the paper. ORB keypoints give 0.1690 on the same data, because ORB corners sit on depth edges more often. ORB is the right choice for the D455, because it matches what PriorNet samples.

### 10.3 Write the mono file again with the constants

Use the same options as in 9.2 and 9.3 (Kalibr, noise scale), plus `--constants`:

```bash
python3 scripts/jetson/make_d455_settings.py --bag "${BAGS[0]}" --which mono --imu-rate 200 --constants configs/d455/d455color_depth_constants.yaml
grep -E '^(UCSLAM.PriorNet|UCSLAM.LogBias|UCSLAM.SigmaNetC0|UCSLAM.SigmaNetC1|UCSLAM.NeffOverN|UCSLAM.DepthBackend|D455.FitEngine)' configs/d455/d455_mono_inertial.yaml
```

Expected: `UCSLAM.PriorNet: 1`, the four constants of the YAML, `UCSLAM.DepthBackend: "trt-fp16"` and the engine path.

The Metric-Indoor network is outside its training domain outdoors. Points beyond 5 m are dropped (40 % on the OpenLORIS market scenes). The files in `configs/d455/` are specific to your unit and are not tracked by git.

---

## 11 Bench tests with the propellers off

Remove the propellers. Do not connect MAVROS yet.

### 11.1 Stereo mode

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo imu_chip:=bmi055
```

In a second terminal:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 topic echo --once /ucslam/slam_node/status | grep -E 'message|key|value'
ros2 topic hz /ucslam/slam_node/odom
```

Move the camera by hand (rotate, translate a few metres). `~/status` should go `waiting for IMU initialization`, then `settling`, then `tracking` (level OK). Repeat the echo every few seconds.

| Key of `/ucslam/slam_node/status` | Meaning | Expected on the bench |
|---|---|---|
| `state` | 2 tracking, 3 recently lost, 4 lost | 2 |
| `imu_initialized` | IMU initialization done | `true` after some motion |
| `map_time_s`, `settling` | settling until 16 s (stereo) or 76 s (mono) of map time | `settling` false after enough motion |
| `map_changes` | loop closures and other map changes | small |
| `frames`, `dropped_frames`, `old_frames` | images received, dropped by the full queue, skipped because their stamp is not newer | `frames` grows about 30 per second |
| `imu_rejected`, `imu_wait_timeouts` | IMU samples not newer than the last one, frames with no IMU sample at or after their stamp within `imu_wait_ms` | 0 or slow growth |
| `imu_filled` | IMU samples added by the gap fill (on by default in the D455 launch: gap 20 ms, period 5 ms) | 0 or slow growth. Fast growth means IMU messages get lost |
| `imu_fill_skipped` | gaps longer than `imu_fill_max_ms` (1000 ms) not filled | 0 |
| `imu_short_frames` | frames not tracked: fewer than two IMU samples since the last frame | 0. Steady growth means no IMU or an IMU rate below twice the camera rate |
| `bad_images` | dropped images (encoding, step) | 0 |
| `depth_requests`, `depth_pushed`, `depth_refused` | depth prior traffic (mono with live depth) | 0 in stereo mode |
| `track_ms_p50`, `track_ms_p90` | tracking time per frame | below 33 ms for 30 Hz |
| `latency_ms_p50`, `latency_ms_p90` | node clock at publish minus image stamp, last second | positive, `latency_ms_p90` below 200 ms. **Write them down** for section 12.3 |

**CHECK ON THE JETSON:** latency needs camera stamps on the Jetson clock (RealSense global time). A negative latency logs a warning. Negative, or above 1 s: the stamps are not on the Jetson clock. **Do not connect PX4 then.** The librealsense option `<module>.global_time_enabled` puts the stamps on host time and should stay on ([librealsense source](https://github.com/realsenseai/librealsense/blob/master/src/ds/ds-motion-common.cpp). That it fixes the stamps is an inference, not tested). See what the driver exposes:

```bash
ros2 param list /camera/camera | grep global_time
```

Also watch the system:

```bash
tegrastats
```

Note RAM use, GPU load and temperatures.

The odometry rate should be close to 30 Hz. The bridge warns below 20 Hz, and PX4 asks for 30 to 50 Hz.

**Stereo pass test (CHECK ON THE JETSON, required before section 12).** Stereo-inertial `slam_node` has tracked EuRoC replays on the desktop, but never D455 data on the Jetson. Prove it here. It fails when images are lost: `frames` must grow at the camera rate (about 30 per second) and `track_ms_p90` must stay below the frame interval (33 ms at 30 Hz). If it does not, raise `rmem_max` (2.4), add `image_qos_reliable:=true`, or lower the camera rate to 15 Hz and write the matching settings file. Mark a start point. Start the launch above at the mark and move the camera until `settling` is false. Put it back on the mark and read the position (first read). Then carry it by hand for 5 minutes (walk, turn, lift), put it back on the mark and read again:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 topic echo --once /ucslam/slam_node/odom --field pose.pose.position
```

| Check | Pass |
|---|---|
| `imu_initialized` | `true` |
| `settling` | `false` |
| `state` | 2 through the whole 5 minutes. Echo the status every 30 s |
| second read minus first read (same mark) | a few cm in x, y and z |
| `imu_short_frames`, `bad_images` | 0 |

If any check fails, do not go on to section 12. Fix stereo mode first (section 16) or stay with recording only.

### 11.2 Mono mode

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch ucslam_ros ucslam_d455.launch.py mode:=mono imu_chip:=bmi055
```

With fitted constants the depth node starts (`depth:=auto`) with the backend and engine of the settings, warmed up at 848x480. Without constants it runs without the depth prior. Settling takes 76 s of map time, so keep moving the camera.

The depth node logs a line every 10 s: `served`, `dropped_old`, `dropped_full`, `failed`, `waiting`, `net_ms_p50`, `net_ms_p99`.

| Check | Expected |
|---|---|
| `depth_pushed` vs `depth_requests` | close to each other |
| `depth_refused` | 0. Otherwise the domain or backend differs from the settings |
| depth node `failed` | 0 |
| depth node `net_ms_p50` | write it down (Orin timing) |
| `tegrastats` RAM | free memory left (8 GB modules) |

A keyframe whose depth comes after its mapping step gets no depth-prior points. On the desktop with torch-fp32 most keyframes got their depth too late (ros/README.md). The trt-fp16 engine is faster.

### 11.3 Recording

Add `record:=true` to any launch. Use `record_images:=false` on long runs (no images, only IMU, tf, UC-SLAM and PX4 topics). Check the free space with `df -h ~` first.

### 11.4 Stop and save

Press Ctrl-C once in the launch terminal and wait. `slam_node` finishes local mapping and saves the trajectories, which can take up to a minute.

```bash
ls -l ~/ucslam_out
```

Expected: `f_ucslam.txt` (all frames), `kf_ucslam.txt` (keyframes, EuRoC format) and `ucslam_settings.yaml` (the settings the run used). Use `output_dir:=` and `trajectory_name:=` to keep runs apart.

When `slam_node` exits for any reason, the whole launch stops: camera, depth node and bridge. PX4 then sees the vision data stop.

---

## 12 PX4 and MAVROS

**CHECK ON THE JETSON:** nothing in this section ran with MAVROS or PX4. SITL is not covered here. The required check is the bench test of 13.3 and 13.4 with the real flight controller, props off.

### 12.1 The serial link

```bash
dmesg | grep -i ttyTHS
ls -l /dev/ttyTHS* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

Loopback test of the header UART: disconnect PX4, connect pin 8 to pin 10, then:

```bash
stty -F /dev/ttyTHS1 921600 raw -echo
cat /dev/ttyTHS1 &
echo hello > /dev/ttyTHS1
sleep 1
kill %1
```

Expected: `hello` is printed. If not, check the device name for your carrier and L4T release (it was `ttyTHS0` on JetPack 5, and one report says the header UART broke after R36.5). Plan B: a USB-serial adapter (`/dev/ttyUSB0`) or the flight controller's USB port (`/dev/ttyACM0`).

### 12.2 PX4 MAVLink parameters

Set in QGroundControl (Vehicle Setup, Parameters). For the TELEM port wired to the Jetson (example: TELEM 2):

| Parameter | Value | Note |
|---|---|---|
| `MAV_1_CONFIG` | TELEM 2 (the port you wired) | |
| `MAV_1_MODE` | Onboard | |
| `SER_TEL2_BAUD` | 921600 | the `SER_TELn_BAUD` of that port |
| `UXRCE_DDS_CFG` | Disabled, if it uses the same port | MAVROS and uXRCE-DDS cannot share it |

Source: [Seeed wiki](https://wiki.seeedstudio.com/control_px4_with_recomputer_jetson/) for PX4 1.15 (unofficial). Reboot the flight controller after the change.

### 12.3 EKF2 external vision

Check your PX4 version first (QGroundControl, or `ver all` in the MAVLink console). **The default of `EKF2_EV_CTRL` changed from 15 (1.14, 1.15) to 0 (1.16 and newer).**

For the default bridge output `vision_pose` (indoor, no GPS, magnetometer off, vision yaw fused):

| Parameter | Meaning | Default 1.14 / 1.15 | Default 1.16 / 1.17 | Set |
|---|---|---|---|---|
| `EKF2_EV_CTRL` | bit 0 horizontal position, bit 1 vertical position, bit 2 velocity, bit 3 yaw | 15 | 0 | 11 (position and yaw, no velocity: the bridge sends no velocity) |
| `EKF2_MAG_TYPE` | 0 auto, 1 heading, 5 none | 0 | 0 | 5 |
| `EKF2_HGT_REF` | 0 baro, 1 GPS, 2 range, 3 vision (reboot) | 1 | not checked | 3 indoors over a flat floor (see the first hover note below) |
| `EKF2_GPS_CTRL` | GPS aiding bits | 7 | not checked | 0 indoors |
| `EKF2_BARO_CTRL` | baro height aiding | 1 | not checked | 1 (keep as backup) |
| `EKF2_EV_DELAY` | ms, vision delay against the IMU (reboot in 1.15) | 0 | 0 | 0 to start, see below |
| `EKF2_EV_POS_X/Y/Z` | vision sensor position in the body frame (FRD), m | 0 | 0 | 0: the bridge already sends the base_link pose (lever arm applied). Setting both would apply it twice |
| `EKF2_EV_NOISE_MD` | 0: message variance with a parameter floor | 0 | 0 | 0 |
| `EKF2_EVP_NOISE`, `EKF2_EVV_NOISE`, `EKF2_EVA_NOISE` | floors 0.1 m, 0.1 m/s, 0.1 rad | 0.1 | 0.1 | keep. `slam_node` sends 0.01, so the floor wins |
| `EKF2_EV_QMIN` | minimum quality, 0 off | 0 | 0 | 0 (MAVROS sends no quality) |

Sources: PX4 parameter files [v1.14.3](https://github.com/PX4/PX4-Autopilot/blob/v1.14.3/src/modules/ekf2/ekf2_params.c), [v1.15.4](https://github.com/PX4/PX4-Autopilot/blob/v1.15.4/src/modules/ekf2/module.yaml), [v1.16.0](https://github.com/PX4/PX4-Autopilot/blob/v1.16.0/src/modules/ekf2/params_external_vision.yaml), [v1.17.0](https://github.com/PX4/PX4-Autopilot/blob/v1.17.0/src/modules/ekf2/params_external_vision.yaml), and the [external position estimation guide](https://docs.px4.io/main/en/ros/external_position_estimation.html). The "Set" column is a suggestion from reading these sources, not a tested setup.

Reboot the flight controller (`EKF2_HGT_REF` needs it, and `EKF2_EV_DELAY` on 1.15). Then read the values back in QGroundControl.

**Altitude mode is not free of vision with these values.** With `EKF2_HGT_REF` 3 and bit 1 of `EKF2_EV_CTRL` set, the height of Altitude mode comes from vision. A bad vision height then still moves the vehicle in Altitude mode. Only Stabilized is free of vision. For the first hovers (ladder step 2) you can choose `EKF2_HGT_REF` 0 (baro) and `EKF2_EV_CTRL` 9 (bit 0 horizontal position and bit 3 yaw, no vertical position). Then the height comes from the barometer. This choice is derived from the bit table above and is not tested.

With the magnetometer heading kept, use `px4_output:=odometry` and `EKF2_EV_CTRL` 3 (no vision yaw). Do not use `vision_pose` then: PX4 reads it as NED and would take the arbitrary x axis of the SLAM map as North.

**`EKF2_EV_DELAY` from the measured latency.** The pose stamp is the image capture time, and MAVROS time sync maps it to the PX4 clock. So start with 0. `latency_ms_p50` and `latency_ms_p90` of section 11.1 give the age of the pose when it leaves the Jetson (MAVROS and the serial link add more). If the logged vision innovations grow with fast motion, try `EKF2_EV_DELAY` near `latency_ms_p50` and keep the value with the smaller innovations. Log with `SDLOG_PROFILE` bit 7 to see the vision data (PX4 doc). Not tested.

PX4 starts vision fusion only when samples come less than 200 ms apart, and stops it after 400 ms without data.

### 12.4 Failsafes

| Parameter | Default in v1.15.4 | Meaning |
|---|---|---|
| `COM_POSCTL_NAVL` | 0 | on position loss: Altitude or Manual (1 is Land or Descend). Keep 0 with a pilot on RC |
| `COM_POS_FS_EPH` | 5 m | horizontal accuracy limit for position |
| `COM_VEL_FS_EVH` | 1 m/s | velocity accuracy limit |
| `COM_POS_FS_DELAY` | 1 s | delay before the position failsafe |
| `NAV_DLL_ACT` | 0 | action on data link loss |

Source: [commander_params.c v1.15.4](https://github.com/PX4/PX4-Autopilot/blob/v1.15.4/src/modules/commander/commander_params.c). Check them in your version.

The failsafe of `COM_POSCTL_NAVL` acts only when PX4 judges the position invalid (`COM_POS_FS_EPH`, `COM_POS_FS_DELAY`). After vision stops, the position drifts first, which can take seconds. The pilot must not wait for it (section 14).

Flight modes on the RC: in QGroundControl, Vehicle Setup, Flight Modes, assign Stabilized, Altitude and Position to the 3-position switch, and set the kill switch. With the props off, flip each switch and check that QGroundControl shows the mode change and the kill switch state.

### 12.5 Launch MAVROS

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch mavros px4.launch fcu_url:=serial:///dev/ttyTHS1:921600
```

Over USB use `fcu_url:=serial:///dev/ttyACM0:921600`. In a second terminal:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 topic echo --once /mavros/state
```

Expected: `connected: true`. Indoors without GPS, PX4 has no horizontal position yet, so `/mavros/local_position/pose` may be silent or invalid here. Section 13.4 checks it after the bridge sends. Source: [px4.launch](https://github.com/mavlink/mavros/blob/ros2/mavros/launch/px4.launch).

---

## 13 The bridge

The package `ucslam_px4` (node `/ucslam/px4_bridge`) reads `/ucslam/slam_node/odom` and `/ucslam/slam_node/status` and publishes to MAVROS. Its defaults are in `ros/ros2/ucslam_px4/config/px4_bridge.yaml`.

### 13.1 Mount parameters

UC-SLAM publishes the pose of the D455 IMU in a gravity aligned, z-up map with an arbitrary yaw. The bridge turns it into the pose of `base_link` (FLU). With the D455 settings, the body frame is `camera_imu_optical_frame`: x right, y down, z forward.

| Parameter | Default (yaml) | Meaning |
|---|---|---|
| `base_x`, `base_y`, `base_z` | 0, 0, 0 | position of the `base_link` origin in the IMU optical axes, metres |
| `base_roll`, `base_pitch`, `base_yaw` | 1.5707963, -1.5707963, 0 | orientation of `base_link` in the IMU frame, radians, R = Rz(yaw) Ry(pitch) Rx(roll). The default is a D455 that looks forward and level |
| `use_tf` | false | read the mount from TF (`imu` to `base_link`) instead |

How to measure the lever arm: find the point of `base_link` (usually the flight controller or the centre of the frame). Measure from the D455 IMU to it along the camera axes: right is +x, down is +y, forward is +z. Example from the bridge notes: `base_link` 5 cm behind and 3 cm below the IMU gives `base_x:=0.0 base_y:=0.03 base_z:=-0.05`. The launch passes only the `base_*` values you set. The others keep the yaml values.

For a camera that is not forward and level, publish the mount as a static transform and use `px4_use_tf:=true` (not tested live):

```bash
ros2 run tf2_ros static_transform_publisher --x 0.0 --y 0.03 --z -0.05 --roll 1.5707963 --pitch -1.5707963 --yaw 0.0 --frame-id imu --child-frame-id base_link
```

### 13.2 Start in stereo mode

Terminal 1: MAVROS (12.5). Terminal 2:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo imu_chip:=bmi055 imu_time_offset:=0.0 px4:=true base_x:=0.0 base_y:=0.03 base_z:=-0.05 record:=true
```

Use your measured `base_*` values (13.1) and the `imu_time_offset` of 9.3 (0.0 without Kalibr). Terminal 3:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 topic echo --once /ucslam/px4_bridge/status | grep -E 'message|key|value'
ros2 topic hz /mavros/vision_pose/pose
```

| Key of `/ucslam/px4_bridge/status` | Meaning | Expected |
|---|---|---|
| `state` | `sending`, or `hold: <reason>` | `hold: status: settling` until UC-SLAM is OK, then `sending` |
| `output` | `vision_pose` or `odometry` | as launched |
| `sent`, `held`, `received`, `old`, `invalid` | message counters | `sent` grows, `invalid` 0 |
| `jumps`, `resets`, `segments` | re-anchored jumps, gaps over 0.5 s, sending segments | `jumps` and `resets` 0 at rest, `segments` 1 once sending starts |
| `rate_hz` | output rate | near 30. WARN below 20 |
| `latency_ms_p50`, `latency_ms_p90` | receive time minus image stamp | close to the `slam_node` values |
| `correction_yaw_deg` | yaw correction from re-anchoring | 0 until a jump |

The bridge sends only when the last `slam_node` status is newer than 3 s, says `imu_initialized` true and `settling` false, with state 2 (or state 3 with a normal covariance). A new segment needs 3 messages in a row, each within 0.2 s.

### 13.3 Axis checks with the propellers off

Watch `/mavros/vision_pose/pose` (ENU on the ROS side):

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 topic echo /mavros/vision_pose/pose
```

To read the yaw in degrees, copy `orientation.x`, `y`, `z` and `w` from one message into:

```bash
python3 -c "import math,sys;x,y,z,w=map(float,sys.argv[1:]);print(math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))))" <x> <y> <z> <w>
```

| Action | Expected |
|---|---|
| hold the vehicle level and still | `orientation.x` and `orientation.y` near 0 (roll and pitch near 0) |
| tilt it forward, then sideways | roll and pitch of `/mavros/vision_pose/pose` match `/mavros/local_position/pose` (PX4's own attitude) within a few degrees |
| carry it forward | x and y move along the vehicle heading |
| lift it | z goes up |
| yaw it to the left | the printed yaw grows (counter clockwise in ENU). The QGroundControl heading decreases (NED) |

If roll or pitch is near 90 deg, the mount is wrong (for example the identity mount: the bridge logs a WARN at start when the mount rotation is the identity).

### 13.4 Check that PX4 fuses vision

In the MAVLink console of QGroundControl:

```text
listener vehicle_visual_odometry
listener vehicle_local_position
listener estimator_status_flags
```

Expected: `vehicle_visual_odometry` updates with the pose you send. `vehicle_local_position` follows it. `estimator_status_flags` shows the vision position and vision yaw flags true (field names `cs_ev_pos` and `cs_ev_yaw`, check them in your PX4 version). On ROS, `/mavros/local_position/pose` should follow `/mavros/vision_pose/pose` while you carry the vehicle:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 topic hz /mavros/local_position/pose
ros2 topic echo --once /mavros/local_position/pose --field pose.position
ros2 topic echo --once /mavros/vision_pose/pose --field pose.position
```

Expected: a steady rate, and the two positions close to each other.

**Time sync.** The pose carries the image stamp, and MAVLink timesync maps it to the PX4 clock. A broken sync gives a wrong delay without any error. In `listener vehicle_visual_odometry`, `timestamp` minus `timestamp_sample` should be close to the bridge `latency_ms_p50` (tens of ms), never negative and never seconds. Check the field names in your PX4 version.

**Heading.** With `EKF2_MAG_TYPE` 5 and vision yaw, the heading in QGroundControl is the x axis of the SLAM map at start, not North. This is expected.

`MAV_ODOM_LP` (odometry loopback) is gone since PX4 1.14, even though older docs still mention it.

Two points come from reading code, not from a run: PX4 ignores the NaN twist of the `odometry` output, and MAVROS publishes the frames it needs for that output itself. Check them with `listener vehicle_visual_odometry` before flight.

---

## 14 Flight test ladder

Stereo mode is the planned mode for closed-loop flight. It settles after 16 s of map time and needs no depth constants. **It has not run on D455 data yet** (only on a EuRoC replay on the desktop): pass the stereo test of 11.1 first. Fly mono mode first with `px4:=false` (record only, PX4 on its own estimate) until it is proven. Indoors PX4 then has no position. Fly it in Altitude or Stabilized mode only.

**Gate before any flight:** the vehicle already flies well in Stabilized and Altitude mode under RC, sensor calibration is done, and the RC loss action and the kill switch are set and tested (section 1). This guide does not cover vehicle setup.

### 14.0 Pre-flight sequence

1. Battery in, props on, vehicle **disarmed**, kill switch on.
2. Start MAVROS (12.5) and the stereo launch with `px4:=true` (13.2).
3. Hold the frame by hand, away from the props. Move and rotate it for 20 to 30 s until the bridge says `sending`:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 topic echo --once /ucslam/px4_bridge/status | grep -E 'message|key|value'
```

4. Put it on the take-off spot. Check that `/mavros/local_position/pose` follows `/mavros/vision_pose/pose` and is still (commands of 13.4).
5. Check that QGroundControl offers Position mode while disarmed.
6. Arm in Altitude mode.

Repeat steps 3 to 5 after any restart of the launch.

### 14.1 Ladder

| Step | Setup | Pass before the next step | Stop when |
|---|---|---|---|
| 1 Hand-carried walk | props off, disarmed, bridge on, record on | `/mavros/local_position/pose` follows the walk, no jumps, rate near 30 Hz, the path closes when you return to the start | any axis is wrong, `rate_hz` below 20, `resets` grow |
| 2 Hover in Altitude mode | gate above passed. Tethered or in a net, vision fused, pilot on RC. Record the IMU for 9.4 (`record_images:=false`) | the PX4 position estimate stays still while the vehicle hovers still, no estimator warnings | position estimate drifts or jumps. Switch to Stabilized if the height also misbehaves |
| 3 Short Position mode | same place, pilot ready to switch to Altitude on RC, or to Stabilized if the height also misbehaves | holds position, small moves follow the sticks | any jump, oscillation or `hold:` state in the bridge |
| 4 Longer flights | larger area, loop back to the start | the end position matches the start, latency and temperature stay in range | `tegrastats` shows throttling, battery low |

Rules:

- **Never send the pose before `~/status` is OK.** The bridge enforces it. Never run it with `require_status_ok:=false` on a vehicle (that is a test setting).
- Take off only when the bridge says `sending`.
- **Loss.** On `RECENTLY_LOST` or `LOST` the bridge holds. PX4 stops fusing vision after 0.4 s. The position then drifts. The `COM_POSCTL_NAVL` failsafe acts only when PX4 judges the position invalid, which can take seconds. Do not wait for it: switch to Altitude (or Stabilized if the height also misbehaves) at once and land.
- **After any `hold:`, jump or reset in flight**, do not switch back to Position mode in that flight. Land.
- **Jumps.** A loop closure moves the map. The bridge detects steps above `max_speed_mps` x dt + 0.05 m or `max_yaw_rate_dps` x dt + 3 deg, and a rise of `map_changes`. It re-anchors: the output keeps the last position and yaw, and one frame of motion is lost. Each jump is a WARN in the log. MAVROS cannot send the PX4 reset counter, so PX4 never sees a "reset", only the re-anchored stream.
- **Small corrections.** A correction below the step limit (about 0.2 m at 30 Hz with the default `max_speed_mps` 5) reaches PX4 as a step. The `map_changes` check comes up to 1 s late. Expect small twitches after loop closures. For slow indoor flights, copy `ros/ros2/ucslam_px4/config/px4_bridge.yaml`, lower `max_speed_mps` in the copy and pass it with `px4_params_file:=<copy>`.
- **Resets.** A gap of more than 0.5 s starts a new segment without re-anchoring. The new segment can start in a new map. That PX4 resets its position to the new stream comes from reading code and is not tested.
- **Thermal.** Watch `tegrastats`. MAXN throttles when hot.
- **Battery.** The Jetson and the camera draw from the flight battery. The Jetson draws up to its power mode limit. A brownout stops SLAM in flight (vision timeout). Land with margin.
- **ROS domain.** During flights keep ground computers off the Jetson's ROS domain, or give the Jetson its own `ROS_DOMAIN_ID` (`export ROS_DOMAIN_ID=<n>` in every Jetson shell, after the `source` line). A ground computer that subscribes to images loads the Jetson and the Wi-Fi.
- **If `slam_node` dies**, the whole launch stops (bridge included). PX4 sees the vision timeout.

---

## 15 After a flight

1. Land, disarm, then press Ctrl-C once in the launch terminal and wait up to a minute for the save.
2. Collect the files:

| What | Where |
|---|---|
| trajectories and settings copy | `~/ucslam_out/` (`f_<name>.txt`, `kf_<name>.txt`, `ucslam_settings.yaml`) |
| bags | `~/ucslam_bags/d455_<mode>_<time>/` (MCAP) with the UC-SLAM, bridge and MAVROS topics |
| PX4 logs (`.ulg`) | the SD card of the flight controller, or QGroundControl Analyze Tools, Log Download |

Give each flight its own names: `output_dir:=$HOME/ucslam_out/flight01 trajectory_name:=flight01`.

3. Copy to the desktop. **On the desktop:**

```bash
rsync -a --info=progress2 $JETSON:ucslam_out/ ~/ucslam_flights/out/
rsync -a --info=progress2 $JETSON:ucslam_bags/ ~/ucslam_flights/bags/
```

4. Evaluate. A flight has no ground truth. Useful checks:

| Check | How |
|---|---|
| loop closure error | fly back to the start point. The end position in `f_<name>.txt` minus the start position is the drift |
| SLAM vs PX4 | compare `/ucslam/slam_node/odom` (or `/mavros/vision_pose/pose`) with `/mavros/local_position/pose` in the bag |
| replay | a bag with images can be replayed on the desktop or the Jetson. Terminal 1: `ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo camera:=false use_sim_time:=true imu_chip:=bmi055`. Terminal 2: `ros2 bag play <bag> --clock 200 --topics /camera/camera/infra1/image_rect_raw /camera/camera/infra2/image_rect_raw /camera/camera/imu /tf_static` (mono: `/camera/camera/color/image_raw` instead of the two infra topics). Play only these topics: the bag also holds the old UC-SLAM topics, depth priors and `/tf`. The D455 launch keeps `frame_drop` oldest and best-effort images, so a loaded host drops frames |

---

## 16 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No `/camera/camera/imu` messages | JetPack 6 kernel without HID sensor drivers (native backend), or an accel rate the chip does not support | try `imu_chip:=bmi085` (or `bmi055`). Else librealsense with RSUSB (4.3) |
| `imu_chip:=auto cannot read the IMU` | `pyrealsense2` not installed | pass `imu_chip:=bmi055` or `bmi085` |
| Wrong IMU rate (not about 200 Hz) | driver setting or USB load | `ros2 topic hz /camera/camera/imu`. Write the settings again so `IMU.Frequency` equals the real rate |
| `imu_short_frames` rising | no IMU on the topic, or IMU gaps longer than `imu_wait_ms` (50 ms) | check the IMU topic rate and `imu_filled`. Use a USB 3 port and a short cable |
| `imu_filled` rising fast | IMU messages lost on USB or in DDS | section 2.4 socket buffers. Check the cable |
| Images lost (`frames` grows slower than 30/s, odometry gaps) | `rmem_max` too small, best-effort image QoS under load | section 2.4, then check `cat /proc/sys/net/core/rmem_max`. Add `image_qos_reliable:=true` (and `imu_qos_reliable:=true`) to the `ucslam_d455.launch.py` command. If `track_ms_p90` is above the frame interval, the CPU is too slow for the camera rate: lower it to 15 Hz |
| `ldd` shows two `libopencv_core` | UC-SLAM and the node built against different OpenCVs | rebuild UC-SLAM (clean build command below this table) and the workspace in one environment. Do not pass `OpenCV_DIR` to colcon |
| `OpenCV mismatch` or `OpenCV_DIR ... does not exist` in colcon | the library was built elsewhere or with another OpenCV | clean build (command below this table), then colcon again (6.3) |
| Compiler killed during the build (OOM) | too many jobs for the RAM | swap on NVMe (2.3), `--jobs 1` or `2`, headless mode (2.5) |
| `[FAIL] ... configured for another folder` | an x86 or renamed build folder was copied | delete the folders the check prints, build again |
| `./build.sh` fails at `git submodule update` | no internet (Pangolin's pybind11 submodule) or no `.git` | connect once, or copy with `.git` (5.2) |
| `import torch` fails with a missing CUDA library | wheel and JetPack do not match | use JetPack 6.1 or newer. NVIDIA's own wheels need cuSPARSELt ([install guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)) |
| `is_torch_available()` is False | torch below 2.5.0 or a pre-release like 2.5.0a0 | install torch 2.11.0 (section 7) |
| depth node warns `not the torchvision processor` | torchvision missing or broken | install torchvision 0.26.0 from the Jetson index |
| depth node refuses the engine (size) | the engine size does not fit the stream (518x910 is for 848x480) | export and build for the stream size, or set width and height to 848x480 |
| depth node refuses the engine (record) | `<engine>.json` names another model or precision | use the engine with its own record, or build again |
| launch stops: `backend:=... but the depth constants ... were fitted with ...` | backend differs from `UCSLAM.DepthBackend` | leave `backend:=` empty or fit again with that backend |
| `slam_node` FATAL `depth_backend ... UCSLAM.DepthBackend` | the same, when `ucslam.launch.py` is used directly | `backend:=` must equal the key |
| `depth_refused` grows | domain or backend label differs from the settings | `domain` must equal `UCSLAM.Dataset`, `backend` must equal `depth_backend` |
| No pose before init | normal: nothing is published before the IMU initialization | move the camera (rotate and translate) |
| `settling` stays true | map time runs only while the camera moves (16 s stereo, 76 s mono) | keep moving. Do not fly before it ends |
| Bridge `jumps` or `resets` rise | loop closures, tracking loss, gaps over 0.5 s | check `frames` and `imu_*` counters, the lighting and the texture of the scene |
| High latency (`latency_ms_p90`) | slow tracking, CPU load, a slow depth network, queueing | MAXN and `jetson_clocks`, trt-fp16, close other programs, `tegrastats` |
| Negative latency warning | camera stamps not on the Jetson clock | keep the RealSense global time on |
| UART silent | wrong device, wrong baud, TX/RX swapped, console on the port, user not in dialout | loopback test (12.1), `MAV_1_*` and `SER_TELn_BAUD` (12.2), USB-serial plan B |
| `/mavros/state` `connected: false` | link or PX4 MAVLink settings | 12.1 and 12.2 |
| PX4 does not fuse vision | `EKF2_EV_CTRL` 0 (1.16+ default), rate too low, gaps over 200 ms, frames or yaw settings | 12.3, `listener vehicle_visual_odometry`, `rate_hz` near 30 |
| Vision fused but heading wrong | `vision_pose` with the magnetometer fused | `EKF2_MAG_TYPE` 5 with vision yaw, or `px4_output:=odometry` with `EKF2_EV_CTRL` 3 |
| Stereo `slam_node` exits with -11 | a `Rectified` stereo-inertial settings file | use the file of `make_d455_settings.py` (PinHole) |
| Ctrl-C prints a KeyboardInterrupt traceback from the depth node | cosmetic, during `import rclpy` | none |

**Reliable images (stereo).** Add the two QoS arguments to the usual launch:

```bash
source ~/ucslam_ws/ucslam_env.sh
ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo imu_chip:=bmi055 image_qos_reliable:=true imu_qos_reliable:=true
```

Check: `frames` in `/ucslam/slam_node/status` grows about 30 per second, and `track_ms_p90` stays below 33 ms. On the desktop, a stereo replay that lost images under CPU load reset the map 7 times, and the same replay at a pace the CPU could follow tracked every frame with no reset.

**Clean build of UC-SLAM.** Run it without the ROS paths, as the tested build and the `build` step do (the RealSense examples of ORB-SLAM3 stay out):

```bash
cd ~/ucslam
env -u CMAKE_PREFIX_PATH -u AMENT_PREFIX_PATH UCSLAM_CMAKE_ARGS=-DCMAKE_DISABLE_FIND_PACKAGE_realsense2=TRUE ./build.sh --clean
```

Then rebuild the workspace (6.3) and repeat the check of 6.4.

---

## 17 Optional autostart

Start by hand until the flight ladder of section 14 has passed. Then use tmux or systemd.

### 17.1 tmux

```bash
tmux new-session -d -s ucslam
tmux send-keys -t ucslam 'source ~/ucslam_ws/ucslam_env.sh && ros2 launch mavros px4.launch fcu_url:=serial:///dev/ttyTHS1:921600' C-m
tmux split-window -t ucslam
tmux send-keys -t ucslam 'source ~/ucslam_ws/ucslam_env.sh && ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo imu_chip:=bmi055 imu_time_offset:=0.0 px4:=true base_x:=0.0 base_y:=0.03 base_z:=-0.05 record:=true record_images:=false' C-m
tmux attach -t ucslam
```

Replace `imu_time_offset` and the `base_*` values with yours (9.3, 13.1). Detach with Ctrl-b d. To stop, attach and press Ctrl-C in each pane, then wait for the save.

### 17.2 systemd

Write `/etc/systemd/system/ucslam.service` (replace `me` by your user name, and `imu_time_offset` and the `base_*` values by yours):

```ini
[Unit]
Description=UC-SLAM D455 stereo with the PX4 bridge
After=network-online.target

[Service]
User=me
ExecStartPre=+jetson_clocks
ExecStart=/bin/bash -c 'source /home/me/ucslam_ws/ucslam_env.sh && exec ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo imu_chip:=bmi055 imu_time_offset:=0.0 px4:=true base_x:=0.0 base_y:=0.03 base_z:=-0.05 record:=true record_images:=false'
KillSignal=SIGINT
TimeoutStopSec=150
Restart=no

[Install]
WantedBy=multi-user.target
```

MAVROS needs its own unit of the same form. Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ucslam.service
journalctl -u ucslam.service -f
```

**Exit code caveat.** `ucslam.launch.py` exits with code 0 when `slam_node` fails at start, so systemd cannot see a failed start from the exit code. Check `journalctl` and `~/status`, not the unit state. Keep `Restart=no`: an automatic restart in flight gives a new map with a new origin and a new settling period. `SIGINT` lets `slam_node` save its trajectories. Not tested.
