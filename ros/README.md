# UC-SLAM on ROS 1 and ROS 2

UC-SLAM can run on a robot as two nodes. `slam_node` runs the visual-inertial SLAM (C++). `depth_node` runs the depth network on each keyframe image (Python, GPU). The network runs the same code that writes the offline depth cache, so a live depth map should equal the cached map of the same image (the parity test is still to be run).

| Folder | Content |
|---|---|
| `common/ucslam_node_core.h` | The ROS-free core shared by both nodes: input queues, IMU rules, the tracking thread, the depth requests |
| `ros1/ucslam_ros1/` | ROS 1 Noetic package: `slam_node`, `depth_node.py`, messages, `launch/ucslam.launch` |
| `ros2/ucslam_msgs/` | ROS 2 Humble messages `DepthRequest` and `DepthPrior` |
| `ros2/ucslam_ros/` | ROS 2 `slam_node`, `launch/ucslam.launch.py` and `scripts/asl_to_bag.py` (EuRoC or TUM-VI folder to a ROS 2 bag) |
| `ros2/ucslam_depth/` | ROS 2 `depth_node` |
| `docker/` | ROS 2 Humble image with CUDA torch (`Dockerfile`) and the in-container build (`build_ws.sh`) |

## How it works

```text
camera + IMU --> slam_node --- DepthRequest (keyframe stamp + image) ---> depth_node (network on the GPU)
                 Track* in one thread  <--- DepthPrior (metric depth map) ---
                 |
                 +--> ~odom, ~pose, tf map -> imu, ~status, ~tracked_points
```

- `slam_node` sends one `DepthRequest` per new keyframe with the image it was made from. `depth_node` answers with a `DepthPrior` that carries the same stamp. The depth reaches the keyframe in the PriorNet thread of UC-SLAM, so tracking never waits for the network.
- The pose is published only after the IMU initialization (before it, a monocular map has no metric scale) and while tracking is OK or recently lost. It is the IMU (body) pose in the map frame of UC-SLAM, whose z axis points up after the IMU initialization.
- `~status` reports `settling` until the last inertial refinement of the base system has run: 76 s of map time (`map_time_s`, the time since the first keyframe of the map) in monocular-inertial mode and 16 s in stereo-inertial mode. Scale and gravity can still move until then. The refinements run on an internal clock that advances only while the camera moves, so on a platform that stands still they come later than this. A loop closure moves the map: `map_changes` counts the frames on which `System::MapChanged()` reported such a change.
- On shutdown (Ctrl-C) the node waits for local mapping and saves `f_ucslam.txt` (all frames) and `kf_ucslam.txt` (keyframes) in `output_dir`, in the EuRoC format of the benchmark examples. It also keeps `ucslam_settings.yaml`, the settings file it ran with.

## Requirements and limits

- **Settings.** Use a settings file of `configs/` (it needs `File.version: "1.0"`). The node copies it to `output_dir` with the depth source of the run. Relative paths in the node parameters are taken from the working directory of the node.
- **Camera domain.** The depth constants (`UCSLAM.SigmaNetC0`, `SigmaNetC1`, `LogBias`, `NeffOverN`) hold for one camera domain. The provided values are for EuRoC, TUM-VI and the OpenLORIS D435i color camera. A new camera, such as RealSense infrared, needs its own values from reference depth before its results mean anything. `depth_node` sends `domain` and `backend`, and `slam_node` refuses a depth map whose domain differs from `UCSLAM.Dataset` of the settings.
- **IMU rate.** `IMU.Frequency` must equal the real rate of the IMU topic.
- **Late depth.** A keyframe whose depth arrives after its mapping step gets no depth-prior points. Two optional rules exist, both off by default: `dpp_wait_ms` (the mapper waits up to that long) and `dpp_defer_max_kfs` (the points are made in a later mapping step, but they skip the fuse into neighbour keyframes and meet the culling test one step sooner). They are not yet tuned.
- **Status.** First version, tested on bag replays (see the end of this file), not yet on a robot.

## ROS 1 (Noetic, Ubuntu 20.04)

**Build.** First build UC-SLAM with `./build.sh` at the repository root. Then link the package into a catkin workspace, from the repository root:
```bash
export UCSLAM_REPO=$PWD          # the launch file and the depth node find the repository through it
mkdir -p ~/catkin_ws/src && ln -s $UCSLAM_REPO/ros/ros1/ucslam_ros1 ~/catkin_ws/src/
cd ~/catkin_ws && source /opt/ros/noetic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```
Use the devel space. An install space also works when `UCSLAM_REPO` is set.

**Python of the depth node.** It runs in the repository's Python environment (`uv sync`, Python 3.11 with torch), not in the Python 3.8 of Noetic. Install the two ROS helper packages next to it once:
```bash
uv pip install --python .venv/bin/python --target ros/ros1/pydeps rospkg catkin_pkg
```
The launch file starts it through `scripts/venv_python.sh`. With `ROS_HOSTNAME=localhost`, rospy also needs `netifaces`: add it to the line above.

**Run on a EuRoC bag** (topics `/cam0/image_raw`, `/imu0`), with the network live:
```bash
roslaunch --sigint-timeout=120 ucslam_ros1 ucslam.launch frame_drop:=none output_dir:=$HOME/ucslam_out
rosbag play MH_01_easy.bag          # in a second terminal
```
With the precomputed depth cache instead of the network, add `depth:=file` (the cache must exist under `data/depth` of the repository).

**Run on another camera:**
```bash
roslaunch --sigint-timeout=120 ucslam_ros1 ucslam.launch settings:=/path/to/settings.yaml domain:=<UCSLAM.Dataset> \
    image_topic:=/camera/infra1/image_rect_raw imu_topic:=/camera/imu output_dir:=$HOME/ucslam_out
```
Stereo-inertial: `stereo:=true image_right_topic:=...` (both images must carry the same stamp). TUM-VI bags: `settings:=$UCSLAM_REPO/configs/tumvi_mono_inertial.yaml domain:=tumvi clahe:=true view:=tumvi tumvi_seq_dir:=data/tumvi/<sequence>` (a relative `tumvi_seq_dir` is taken from the repository; 16-bit images are converted like the benchmark, by keeping the high byte).

## ROS 2 (Humble, Docker)

The image holds ROS 2 Humble, the RealSense driver, CycloneDDS, Pangolin v0.6 and the CUDA build of torch. It needs the NVIDIA container toolkit on the host.
From the repository root:
```bash
docker build -t ucslam:humble ros/docker
docker run --rm -it --name ucslam --gpus all --net=host --ipc=host \
    -v $PWD:/ws/ucslam:ro -v ucslam_build:/ws/build \
    -v $HOME/.cache/huggingface:/opt/hf:ro -v $HOME/ucslam_out:/ws/out ucslam:humble
```
Inside the container, build once. UC-SLAM is compiled in the volume `/ws/build`, and the repository stays read only.
```bash
/ws/ucslam/ros/docker/build_ws.sh
source /ws/build/install/setup.bash
```
Convert a EuRoC sequence of the host's `data/` folder to a ROS 2 bag, then launch and play it. Open the second shell with `docker exec -it ucslam bash`; it has ROS and the workspace sourced.
```bash
ros2 run ucslam_ros asl_to_bag.py --seq /ws/ucslam/data/euroc/MH_01_easy --out /ws/out/bags/MH_01_easy
ros2 launch ucslam_ros ucslam.launch.py vocabulary:=/ws/build/stock/Vocabulary/ORBvoc.txt \
    output_dir:=/ws/out/MH_01_easy frame_drop:=none
ros2 bag play /ws/out/bags/MH_01_easy        # second shell
```
The launch arguments are the ones of ROS 1 (`ros2 launch ucslam_ros ucslam.launch.py --show-args`). The Hugging Face cache must hold the depth model weights (they are downloaded on the first `precompute_depth.py` run on the host). `RMW_IMPLEMENTATION` is CycloneDDS.

## Jetson Orin (JetPack 6, native ROS 2 Humble)

Not yet tested on a Jetson. The steps follow the Docker image, which uses the same Ubuntu 22.04, compiler and ROS 2 distribution.

1. **System.** JetPack 6 (Ubuntu 22.04) and ROS 2 Humble from the ROS apt repository, then:
   ```bash
   sudo apt install ros-humble-ros-base ros-humble-cv-bridge ros-humble-message-filters ros-humble-tf2-ros \
       ros-humble-diagnostic-msgs ros-humble-realsense2-camera ros-humble-rmw-cyclonedds-cpp \
       python3-colcon-common-extensions python3-venv build-essential cmake git pkg-config libeigen3-dev \
       libopencv-dev libboost-serialization-dev libssl-dev libglew-dev libgl1-mesa-dev
   ```
   UC-SLAM, cv_bridge and the node must use one OpenCV, the Ubuntu OpenCV 4.5 that cv_bridge is built with. `ldd` on the node must list a single `libopencv_core`.
2. **UC-SLAM.** `./build.sh` at the repository root (Pangolin v0.6, then UC-SLAM, with `-march=native` for the Orin CPU). Run it in a shell where ROS is not sourced, or set `UCSLAM_CMAKE_ARGS=-DCMAKE_DISABLE_FIND_PACKAGE_realsense2=TRUE`, so that the RealSense examples of ORB-SLAM3 are left out as in the tested build.
3. **ROS 2 packages**, in a workspace outside the repository:
   ```bash
   source /opt/ros/humble/setup.bash
   mkdir -p ~/ucslam_ws && cd ~/ucslam_ws
   colcon build --base-paths <repository>/ros/ros2 --cmake-args -DCMAKE_BUILD_TYPE=Release
   ```
   The packages find `cpp/ucslam` and the Pangolin of `./build.sh` from their own location.
4. **Python of the depth node.** A venv that also sees rclpy, with the PyTorch wheel NVIDIA builds for your JetPack version, then the other packages with numpy below 2:
   ```bash
   python3 -m venv --system-site-packages ~/ucslam-venv && source ~/ucslam-venv/bin/activate
   pip install <NVIDIA PyTorch wheel for JetPack 6> && pip install "numpy<2" transformers==5.17.0 pyyaml scipy
   python -c "import torch, transformers, rclpy; print(torch.cuda.is_available())"
   ```
   Copy the Hugging Face cache that holds `models--depth-anything--Depth-Anything-V2-Metric-Indoor-Base-hf` from a machine with internet, and set `HF_HOME` to it and `HF_HUB_OFFLINE=1`.
5. **Run** with the venv active, `UCSLAM_REPO=<repository>` and `source ~/ucslam_ws/install/setup.bash`, as in the ROS 2 section, with the topics of the camera driver.
6. **Speed.** Use the maximum power mode (`sudo nvpmodel -m 0` and `sudo jetson_clocks`). The network runs in PyTorch FP32 and is several times slower on an Orin NX than on a desktop GPU. Then more keyframes get their depth after their mapping step (see `dpp_defer_max_kfs`). `depth_node` logs `net_ms`, and `~status` gives `track_ms_p50` and `track_ms_p90`. A faster backend such as TensorRT FP16 changes the depth values, so the depth constants must then be fitted again with that backend and `backend` and `depth_backend` set to its name.

## On a drone

- **Calibrate first.** The camera intrinsics, the camera-IMU transform `IMU.T_b_c1` (for example with Kalibr) and the IMU noise (Allan variance) go into a settings file made from `configs/`. The depth constants of a new camera, such as RealSense infrared, must be fitted from reference depth before the depth prior can be trusted. Give the camera its own `UCSLAM.Dataset` name and pass it as `domain`.
- **Start.** The monocular-inertial map needs motion to initialize, and nothing is published before that. `~status` reports `settling` until the last inertial refinement (76 s of map time in monocular-inertial mode), while scale and gravity can still move. Do not feed the pose to the flight controller before `~status` is OK.
- **Jumps.** A loop closure moves the map. `~status` counts these as `map_changes`. The flight controller must accept a reset of the pose.
- **Frames.** `~odom` is the pose of the IMU in the UC-SLAM map frame, whose z axis points up after the initialization and whose yaw is arbitrary. Publish a static transform from `body_frame` to your `base_link`. PX4 or ArduPilot need a conversion to their frames, for example through MAVROS or the uXRCE-DDS bridge.
- **Stop.** Ctrl-C saves the trajectories, which can take up to a minute.

## Topics, services and parameters

| Name | Type | Direction |
|---|---|---|
| `image`, `image_right` | sensor_msgs/Image (mono8, or any encoding cv_bridge converts) | in |
| `imu` | sensor_msgs/Imu | in |
| `depth_request` | DepthRequest | slam_node to depth_node |
| `depth_prior` | DepthPrior (32FC1 metres, 0 = no depth) | depth_node to slam_node |
| `~odom` | nav_msgs/Odometry (fixed pose covariance, twist not provided) | out |
| `~pose` | geometry_msgs/PoseStamped | out |
| tf `map_frame` to `body_frame` | the IMU pose | out |
| `~tracked_points` | sensor_msgs/PointCloud2, with `publish_points` | out |
| `~status` | diagnostic_msgs/DiagnosticArray, 1 Hz | out |
| `~reset`, `~reset_active_map` | std_srvs/Trigger | service |
| `~localization_mode` | std_srvs/SetBool | service |

Main `slam_node` parameters (defaults in brackets; `~status` also reports `map_time_s`): `vocabulary`, `settings`, `output_dir`, `stereo` [false], `frame_drop` [oldest: drop the oldest frame when tracking falls behind; none: track every frame, for bags], `frame_queue` [3], `imu_wait_ms` [50: wait for an IMU sample at or after each image], `imu_time_offset` [0 s, added to the IMU stamps], `clahe` [false], `depth_source` [live, or file], `depth_dir`, `depth_backend` [torch-fp32], `depth_wait_ms` [2000], `dpp_wait_ms` [0], `dpp_defer_max_kfs` [0], `map_frame` [map], `body_frame` [imu], `publish_tf` [true], `publish_points` [false], `pose_std_m` and `pose_std_rad` [0.01], `viewer` [false].

Main `depth_node` parameters: `domain` [euroc], `backend` [torch-fp32], `model` [the model of `configs/default.yaml`], `device` [cuda], `view` [camera, or tumvi], `max_age_s` [5: older requests are dropped], `warmup_width` and `warmup_height` [0: warm up on the first request].

## Test status

| Check | Status |
|---|---|
| Benchmark path unchanged (the live-depth code is off by default) | code review done, run check pending |
| ROS 1 `slam_node` and the shared core compile (syntax check), ROS 1 launch file loads | done |
| ROS 1 depth node message conversions, round trip | done |
| ROS 2 API use checked against the Humble sources (review, no build) | done |
| ROS 2 image and in-container build (Ubuntu 22.04, g++ 11, Eigen 3.4, OpenCV 4.5) | done |
| ROS 2 replay of EuRoC MH_01 mono-inertial, depth cache, one run | done: keyframe ATE 4.84 cm (the 10 benchmark runs: 2.89 to 10.28 cm), every keyframe got its depth |
| ROS 2 replay of EuRoC MH_01 mono-inertial, live depth on an RTX 4070 Ti, one run | done: keyframe ATE 4.55 cm, every keyframe got depth, network 90 to 95 ms per image (median), depth reaches a keyframe about 110 ms after it is made (median), so no keyframe got depth-prior points |
| Full catkin build and ROS 1 replay | to do |
| Jetson Orin | not tested |
