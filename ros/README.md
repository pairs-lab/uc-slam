# UC-SLAM on ROS 1 and ROS 2

UC-SLAM can run on a robot as two nodes. `slam_node` runs the visual-inertial SLAM (C++). `depth_node` runs the depth network on each keyframe image (Python, GPU). The network runs the same code that writes the offline depth cache, so a live depth map should equal the cached map of the same image (the parity test is still to be run). On ROS 2 a third package, `ucslam_px4`, sends the pose to a PX4 flight controller through MAVROS.

**Jetson Orin NX on a UAV (D455, PX4, MAVROS): see [runJetson.md](../runJetson.md) at the repository root.** It covers JetPack, the build, the TensorRT engine, the D455 calibration, PX4 and the flight tests. Nothing has run on a Jetson yet.

| Folder | Content |
|---|---|
| `common/ucslam_node_core.h` | The ROS-free core shared by both nodes: input queues, IMU rules (gap fill, short-frame guard), the tracking thread, the depth requests |
| `common/image_convert.h` | Image to 8-bit gray conversion of the ROS 2 node (no cv_bridge) |
| `ros1/ucslam_ros1/` | ROS 1 Noetic package: `slam_node`, `depth_node.py`, messages, `launch/ucslam.launch` |
| `ros2/ucslam_msgs/` | ROS 2 Humble messages `DepthRequest` and `DepthPrior` |
| `ros2/ucslam_ros/` | ROS 2 `slam_node`, `launch/ucslam.launch.py`, `launch/ucslam_d455.launch.py` (RealSense D455) and `scripts/asl_to_bag.py` (EuRoC or TUM-VI folder to a ROS 2 bag) |
| `ros2/ucslam_depth/` | ROS 2 `depth_node` (backends torch-fp32, torch-fp16, trt-fp16, trt-fp32) |
| `ros2/ucslam_px4/` | ROS 2 `px4_bridge`: UC-SLAM odometry to MAVROS, with gating and jump handling |
| `docker/` | ROS 2 Humble image with CUDA torch (`Dockerfile`) and the in-container build (`build_ws.sh`) |

## How it works

```text
camera + IMU --> slam_node --- DepthRequest (keyframe stamp + image) ---> depth_node (network on the GPU)
                 Track* in one thread  <--- DepthPrior (metric depth map) ---
                 |
                 +--> ~odom, ~pose, tf map -> imu, ~status, ~tracked_points
                 |
                 +--> px4_bridge (ucslam_px4) --> /mavros/vision_pose/pose or /mavros/odometry/out
```

- `slam_node` sends one `DepthRequest` per new keyframe with the image it was made from. `depth_node` answers with a `DepthPrior` that carries the same stamp. The depth reaches the keyframe in the PriorNet thread of UC-SLAM, so tracking never waits for the network.
- The pose is published only after the IMU initialization (before it, a monocular map has no metric scale) and while tracking is OK or recently lost. It is the IMU (body) pose in the map frame of UC-SLAM, whose z axis points up after the IMU initialization.
- `~status` reports `settling` until the last inertial refinement of the base system has run: 76 s of map time (`map_time_s`, the time since the first keyframe of the map) in monocular-inertial mode and 16 s in stereo-inertial mode. Scale and gravity can still move until then. The refinements run on an internal clock that advances only while the camera moves, so on a platform that stands still they come later than this. A loop closure moves the map: `map_changes` counts the frames on which `System::MapChanged()` reported such a change.
- On shutdown (Ctrl-C) the node waits for local mapping and saves `f_ucslam.txt` (all frames) and `kf_ucslam.txt` (keyframes) in `output_dir`, in the EuRoC format of the benchmark examples. It also keeps `ucslam_settings.yaml`, the settings file it ran with.
- ROS 2: when `slam_node` exits for any reason (a failed start, a crash, Ctrl-C), `ucslam.launch.py` ends the whole launch. A launch that includes it (`ucslam_d455.launch.py`) then stops too: camera driver, depth node and PX4 bridge. The launch exits with code 0 even when `slam_node` failed at start, so a script or a systemd unit cannot see the failure from the exit code.

## Requirements and limits

- **Settings.** Use a settings file of `configs/` (it needs `File.version: "1.0"`). The node copies it to `output_dir` with the depth source of the run. Relative paths in the node parameters are taken from the working directory of the node. D455 settings are written by `scripts/jetson/make_d455_settings.py` into `configs/d455/`.
- **Camera domain.** The depth constants (`UCSLAM.SigmaNetC0`, `SigmaNetC1`, `LogBias`, `NeffOverN`) hold for one camera domain. The provided values are for EuRoC, TUM-VI and the OpenLORIS D435i color camera. A new camera, such as RealSense infrared, needs its own values from reference depth before its results mean anything (`scripts/jetson/fit_depth_constants.py` fits the D455 color domain `d455color`). `depth_node` sends `domain` and `backend`, and `slam_node` refuses a depth map whose domain differs from `UCSLAM.Dataset` of the settings.
- **Depth backend.** The constants also hold for one network backend only (section "Depth backends"). Settings with fitted D455 constants name it in `UCSLAM.DepthBackend`.
- **IMU rate.** `IMU.Frequency` must equal the real rate of the IMU topic.
- **Late depth.** A keyframe whose depth arrives after its mapping step gets no depth-prior points. Two optional rules exist, both off by default: `dpp_wait_ms` (the mapper waits up to that long) and `dpp_defer_max_kfs` (the points are made in a later mapping step, but they skip the fuse into neighbour keyframes and meet the culling test one step sooner). They are not yet tuned.
- **Status.** Tested on bag replays (see the end of this file), not yet on a robot.

## ROS 1 (Noetic, Ubuntu 20.04)

The ROS 1 node is unchanged by the ROS 2 work below. It still uses cv_bridge, does not accept `depth_source` off, does not expose the IMU fill parameters and does not check `UCSLAM.DepthBackend`.

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
Stereo-inertial: `stereo:=true image_right_topic:=...` (both images must carry the same stamp). TUM-VI bags: `settings:=$UCSLAM_REPO/configs/tumvi_mono_inertial.yaml domain:=tumvi clahe:=true view:=tumvi tumvi_seq_dir:=data/tumvi/<sequence>` (a relative `tumvi_seq_dir` is taken from the repository). 16-bit images are converted like the benchmark, by keeping the high byte.

## ROS 2 (Humble, Docker)

The image holds ROS 2 Humble, the RealSense driver, CycloneDDS, Pangolin v0.6 and the CUDA build of torch. It needs the NVIDIA container toolkit on the host.
From the repository root:
```bash
docker build -t ucslam:humble ros/docker
docker run --rm -it --name ucslam --gpus all --net=host --ipc=host \
    -v $PWD:/ws/ucslam:ro -v ucslam_build:/ws/build \
    -v $HOME/.cache/huggingface:/opt/hf:ro -v $HOME/ucslam_out:/ws/out ucslam:humble
```
Inside the container, build once. UC-SLAM is compiled in the volume `/ws/build`, and the repository stays read only. Build UC-SLAM before the ROS 2 packages: the `slam_node` CMake links the OpenCV that `libORB_SLAM3` was built with (below).
```bash
/ws/ucslam/ros/docker/build_ws.sh
source /ws/build/install/setup.bash
```
Convert a EuRoC sequence of the host's `data/` folder to a ROS 2 bag, then launch and play it. Open the second shell with `docker exec -it ucslam bash`. It has ROS and the workspace sourced. `asl_to_bag.py` imports cv2: outside the image, install `python3-opencv` to run it.
```bash
ros2 run ucslam_ros asl_to_bag.py --seq /ws/ucslam/data/euroc/MH_01_easy --out /ws/out/bags/MH_01_easy
ros2 launch ucslam_ros ucslam.launch.py vocabulary:=/ws/build/stock/Vocabulary/ORBvoc.txt \
    output_dir:=/ws/out/MH_01_easy frame_drop:=none
ros2 bag play /ws/out/bags/MH_01_easy        # second shell
```
The Hugging Face cache must hold the depth model weights (they are downloaded on the first `precompute_depth.py` run on the host). `RMW_IMPLEMENTATION` is CycloneDDS.

**Bag replays with sim time.** Launch with `use_sim_time:=true frame_drop:=none image_qos_reliable:=true` and play with `ros2 bag play <bag> --rate 1.0 --clock 200`. The pose latency in `~/status` is then measured on `/clock` (resolution 5 ms at 200 Hz).

**Socket buffers.** The Linux default `net.core.rmem_max` (212992 bytes) is smaller than one camera image. Under CPU load, DDS then drops images without any error: `frames` in `~/status` stops growing. In the desktop replays about 230 of 3682 images were lost this way, even with a reliable image subscription. Raise the buffer (`sudo sysctl -w net.core.rmem_max=2147483647`) and set the CycloneDDS `SocketReceiveBufferSize`. [runJetson.md](../runJetson.md) section 2.4 has the persistent setup.

### OpenCV and cv_bridge

The ROS 2 `slam_node` no longer uses cv_bridge. `common/image_convert.h` converts `mono8`, `8UC1`, `mono16`, `16UC1`, `rgb8`, `bgr8`, `rgba8`, `bgra8`, `yuv422` (UYVY) and `yuv422_yuy2` (YUYV) to 8-bit gray. The color conversions are the cv_bridge ones (`cvtColor`), so the gray image equals the old node's (282 cases, 0 mismatches). 16-bit images keep the high byte. `8UC1` is now taken as gray (the old cv_bridge path refused it). Other encodings, a bad step and failed conversions are dropped, counted in `bad_images` of `~status` and logged (throttled). They never end the node.

The node links the OpenCV of `libORB_SLAM3`: CMake reads `OpenCV_DIR` from `cpp/ucslam/build/CMakeCache.txt`, uses that OpenCV, and stops when the version differs or the folder does not exist. The colcon log prints `slam_node uses OpenCV <ver> from <dir> (libORB_SLAM3: <ver> from <dir>)`. Check after the build:
```bash
ldd install/ucslam_ros/lib/ucslam_ros/slam_node | grep -c libopencv_core     # must print 1
```
If the library was built with another OpenCV, rebuild the library. Do not pass a different `OpenCV_DIR` to colcon: the node CMake forces the library's. On JetPack this keeps NVIDIA's OpenCV 4.8 in `slam_node` although `ros-humble-realsense2-camera` pulls cv_bridge and Ubuntu's OpenCV 4.5.4 into its own process. Every `scripts/*.py` of `ucslam_ros` is installed by a glob: a new script needs a colcon rebuild but no CMake edit.

### Launch arguments of ucslam.launch.py

`ros2 launch ucslam_ros ucslam.launch.py --show-args` lists all 42. Defaults keep the behaviour of the first version.

| Group | Arguments [default] |
|---|---|
| Files | `repo` [$UCSLAM_REPO], `settings` [configs/euroc_mono_inertial.yaml], `vocabulary` [third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt], `output_dir` [~/ucslam_out], `trajectory_name` [ucslam] |
| Topics | `image_topic` [/cam0/image_raw], `image_right_topic` [/cam1/image_raw], `imu_topic` [/imu0], `stereo` [false] |
| Depth | `depth` [live: depth_node answers each keyframe. file: the depth cache under `depth_dir`. off: no depth_node and no depth prior], `depth_dir`, `domain` [euroc], `view` [camera], `tumvi_seq_dir`, `device` [cuda], `backend` [torch-fp32], `engine` [''], `model` ['': the default of the depth config], `warmup_width`, `warmup_height` [0], `max_age_s` [5.0], `depth_wait_ms` [2000.0], `dpp_defer_max_kfs` [0], `dpp_wait_ms` [0.0] |
| Tracking | `frame_drop` [oldest, or none for bags], `frame_queue` [3], `clahe` [false], `viewer` [false], `publish_points` [false] |
| IMU | `imu_wait_ms` [50.0], `imu_time_offset` [0.0 s], `imu_fill_gap_ms` [0.0: off], `imu_fill_period_ms` [5.0], `imu_fill_max_ms` [1000.0], `imu_qos_reliable` [false] |
| Images | `image_qos_reliable` [false] |
| Output | `publish_tf` [true], `map_frame` [map], `body_frame` [imu], `pose_std_m` [0.01], `pose_std_rad` [0.01], `use_sim_time` [false] |

**`depth:=off`** starts no depth_node. `slam_node` then writes `UCSLAM.PriorNet 0` into its settings copy, so no PriorNet thread runs, no depth request is sent and keyframes never wait for depth. If the settings had `UCSLAM.PriorNet 1`, the node logs a WARN that the run has no depth prior.

**IMU gap fill.** With `imu_fill_gap_ms` above 0, a new IMU sample more than `imu_fill_gap_ms` after the last kept one gets linearly interpolated samples every `imu_fill_period_ms`, at a + k x period, up to half a period before the new sample. This is the rule the paper used offline on OpenLORIS-Scene (gap 20 ms, period 2.5 ms at 400 Hz). The node checks 0.1 <= `imu_fill_period_ms` <= `imu_fill_gap_ms`. Gaps longer than `imu_fill_max_ms` (at most 10000) are not filled and are counted in `imu_fill_skipped` with a WARN (clock jump or long dropout). For the D455 (200 Hz gyro with `unite_imu_method` 2) use `imu_fill_gap_ms:=20.0 imu_fill_period_ms:=5.0`, the defaults of `ucslam_d455.launch.py`.

**Short-frame guard.** A frame with fewer than two IMU samples since the last tracked frame is not tracked. Its samples carry over to the next frame. It is counted in `imu_short_frames` with a WARN. A gap longer than `imu_wait_ms` therefore costs the frames inside it. A steadily growing count means no IMU on the topic, or an IMU rate below twice the camera rate.

**`image_qos_reliable`, `imu_qos_reliable`.** The subscriptions are best effort by default. `true` makes them reliable, which needs a reliable publisher (the RealSense ROS 2 driver's default, and bag replays). Use it when images or IMU samples get lost (`frames` stops growing, `imu_filled` rises). `ucslam_d455.launch.py` takes and passes both arguments.

**Backend check.** When the settings contain `UCSLAM.DepthBackend` and `backend` differs, `slam_node` stops at start with a FATAL message (depth live or file). With `depth:=off` the key is only logged.

### Depth backends

The `depth_node` parameter `backend` picks the network. The launch argument `backend` feeds both `depth_node` `backend` and `slam_node` `depth_backend`, which must be equal, or every prior is refused.

| Backend | Network | Needs |
|---|---|---|
| `torch-fp32` | PyTorch FP32, the backend of the depth cache and of every provided constant | nothing more |
| `torch-fp16` | PyTorch FP16 | its own constant fit |
| `trt-fp16` | TensorRT FP16 engine, LayerNorm and Softmax in FP32 by default | `engine`, `warmup_width` and `warmup_height`, its own constant fit |
| `trt-fp32` | TensorRT FP32 engine (TF32 off) | the same |

An engine has a fixed input size: 848x480 images give 518x910, 640x480 give 518x686, 752x480 give 518x812. A `trt-*` backend refuses to start without the warm-up size, or when the size does not fit the engine. If the stream size changes at run time, every request fails, with a warning at most every 10 s. The engine record `<engine>.json` (written by the build) must name the same model and precision. Tools: `scripts/jetson/export_depth_onnx.py`, `build_trt_engine.sh` and `depth_parity.py` (pass limits: |median d log z| <= 0.005, p99 <= 0.02, valid-mask agreement >= 0.999). The trt-* backends still load torch, torchvision and transformers for the processor and the resample. Without torchvision, transformers uses the PIL processor and the node logs one warning (`not the torchvision processor`): the maps then differ a little from the fitted ones. Ctrl-C ends the node cleanly with exit 0, also while the network loads. Engine steps: [runJetson.md](../runJetson.md) section 8.

### RealSense D455: ucslam_d455.launch.py

One launch starts the RealSense driver, UC-SLAM, optionally the PX4 bridge and a bag. `ros2 launch ucslam_ros ucslam_d455.launch.py --show-args` lists all 47. It needs `UCSLAM_REPO` and `realsense2_camera` 4.54 or newer.

| `mode:=` | Streams | UC-SLAM | Settles after |
|---|---|---|---|
| `mono` (default) | color 848x480x30 + united IMU | mono-inertial, `configs/d455/d455_mono_inertial.yaml`, domain `d455color`, depth prior when the file has fitted constants | 76 s of map time |
| `stereo` | infra1 + infra2 848x480x30, IR emitter off (`configs/d455/realsense_stereo.yaml`), united IMU | stereo-inertial, `configs/d455/d455_stereo_inertial.yaml`, domain `d455ir`, no depth prior. The planned mode for closed-loop flight. Tested on a EuRoC MH_01 stereo replay (every frame tracked, no map reset, keyframe ATE 4.12 cm when the CPU kept up), not yet on D455 data | 16 s of map time |
| `calib` | color + depth aligned to color (`enable_sync`) + IMU | none. For `make_d455_settings.py` and `fit_depth_constants.py`, use with `record:=true` | n/a |

Main arguments: `slam` [true], `settings` [the file of the mode], `depth` [auto: live when the settings have `UCSLAM.PriorNet 1`, else off. Stereo refuses live], `domain` ['': `UCSLAM.Dataset` of the settings], `backend` and `engine` ['': `UCSLAM.DepthBackend` and `D455.FitEngine` of the settings. Another backend than `UCSLAM.DepthBackend` stops the launch, also with depth off], `imu_chip` [auto: asks the device with pyrealsense2. bmi055: accel 250 Hz. bmi085: accel 200 Hz], `gyro_fps` [200], `accel_fps`, `width`/`height`/`fps` [848/480/30, must equal `Camera.*` of the settings], `imu_fill_gap_ms` [20.0], `imu_fill_period_ms` [5.0], `imu_time_offset` [0.0], `auto_exposure` [true], `initial_reset` [false], `camera` [true. false: topics from a bag replay], `record` [false], `record_images` [true], `bag_dir` [~/ucslam_bags], `px4` [false], `px4_output` [vision_pose], `px4_params_file`, `px4_use_tf`, `base_x` .. `base_yaw` ['': keep the bridge's yaml mount]. `frame_drop` is always `oldest`. Bags are MCAP (`ros-humble-rosbag2-storage-mcap`) under `bag_dir/d455_<mode>_<time>`.

The stereo settings file uses `Camera.type: "PinHole"` with zero distortion and `Stereo.T_c1_c2`, not `"Rectified"`: ORB-SLAM3 v1.0 crashes on a Rectified stereo-inertial file (`slam_node` exit -11, `Settings.cc` prints the unset second camera).

### ucslam_px4: the PX4 bridge

`ros2 launch ucslam_px4 px4_bridge.launch.py` (12 arguments, each empty argument keeps the value of `config/px4_bridge.yaml`), or `px4:=true` in the D455 launch. Pure Python, no torch. MAVROS comes from `ros-humble-mavros` and `ros-humble-mavros-extras` (the `vision_pose` and `odometry` plugins are in mavros_extras). `tf2_ros_py` is needed only for `use_tf`.

- **In:** `/ucslam/slam_node/odom` and `/ucslam/slam_node/status`. **Out:** `vision_pose` (default): `geometry_msgs/PoseStamped` on `/mavros/vision_pose/pose`, frame `map`, for PX4 with vision yaw fused and the magnetometer off. `odometry`: `nav_msgs/Odometry` on `/mavros/odometry/out`, frame `odom`, child `base_link`, covariance copied and rotated, twist NaN, usable with the magnetometer heading kept. The stamp is the image stamp.
- **Mount.** `base_x`, `base_y`, `base_z` (m) and `base_roll`, `base_pitch`, `base_yaw` (rad, R = Rz Ry Rx) give the pose of `base_link` (FLU) in the UC-SLAM IMU body frame. With the D455 settings the body is `camera_imu_optical_frame` (x right, y down, z forward). The yaml default 1.5707963, -1.5707963, 0 is a D455 that looks forward and level. Without the yaml the node default is the identity (a WARN at start). `use_tf:=true` reads `imu` -> `base_link` from TF instead.
- **Gating.** A pose is sent only when it is finite, the last status is newer than `status_timeout_s` (3 s) and says IMU initialized, not settling and state 2 (or state 3 with a normal covariance), the position variance is not above 10 x the smallest seen, and `start_frames` (3) messages arrived in a row within 0.2 s of each other. `require_status_ok:=false` keeps only the covariance gate (tests only).
- **Jumps and resets.** A position step above `max_speed_mps` x dt + `jump_margin_m` (5 m/s, 0.05 m), a yaw step above `max_yaw_rate_dps` x dt + `jump_margin_deg` (180 deg/s, 3 deg), a rise of `map_changes` or a flip of `imu_initialized` re-anchors the output to the last sent position and yaw (roll and pitch stay from UC-SLAM). A gap above `gap_reset_s` (0.5 s) starts a new segment without re-anchoring. Each event is a WARN.
- **`~/status`** (`/ucslam/px4_bridge/status`, 1 Hz): `state` (`sending` or `hold: <reason>`), `output`, `sent`, `held`, `jumps`, `resets`, `segments`, `received`, `old`, `invalid`, `rate_hz`, `latency_ms_p50`, `latency_ms_p90`, `correction_yaw_deg`. Level OK while sending at `publish_rate_min_hz` (20) or more.
- Not run against MAVROS or PX4. PX4 settings: [runJetson.md](../runJetson.md) sections 12 and 13.

## Topics, services and parameters

| Name | Type | Direction |
|---|---|---|
| `image`, `image_right` | sensor_msgs/Image (ROS 2: mono8, 8UC1, mono16, 16UC1, rgb8, bgr8, rgba8, bgra8, yuv422, yuv422_yuy2. ROS 1: any encoding cv_bridge converts) | in |
| `imu` | sensor_msgs/Imu | in |
| `depth_request` | DepthRequest | slam_node to depth_node |
| `depth_prior` | DepthPrior (32FC1 metres, 0 = no depth) | depth_node to slam_node |
| `~odom` | nav_msgs/Odometry (fixed pose covariance, x100 when recently lost, twist not provided) | out |
| `~pose` | geometry_msgs/PoseStamped | out |
| tf `map_frame` to `body_frame` | the IMU pose | out |
| `~tracked_points` | sensor_msgs/PointCloud2, with `publish_points` | out |
| `~status` | diagnostic_msgs/DiagnosticArray, 1 Hz | out |
| `~reset`, `~reset_active_map` | std_srvs/Trigger | service |
| `~localization_mode` | std_srvs/SetBool | service |

Main `slam_node` parameters (defaults in brackets): `vocabulary`, `settings`, `output_dir`, `stereo` [false], `frame_drop` [oldest: drop the oldest frame when tracking falls behind. none: track every frame, for bags], `frame_queue` [3], `imu_wait_ms` [50: wait for an IMU sample at or after each image], `imu_time_offset` [0 s, added to the IMU stamps], `imu_fill_gap_ms` [0: off], `imu_fill_period_ms` [5], `imu_fill_max_ms` [1000], `imu_qos_reliable` and `image_qos_reliable` [false], `clahe` [false], `depth_source` [live, file or off], `depth_dir`, `depth_backend` [torch-fp32], `depth_wait_ms` [2000], `dpp_wait_ms` [0], `dpp_defer_max_kfs` [0], `map_frame` [map], `body_frame` [imu], `publish_tf` [true], `publish_points` [false], `pose_std_m` and `pose_std_rad` [0.01], `trajectory_name` [ucslam], `viewer` [false]. The fill parameters, the QoS parameters, `depth_source` off and the backend check are ROS 2 only.

Main `depth_node` parameters: `domain` [euroc], `backend` [torch-fp32, or torch-fp16, trt-fp16, trt-fp32], `engine` [TensorRT engine for trt-*, relative paths from the repository], `model` [the model of `configs/default.yaml`], `device` [cuda], `view` [camera, or tumvi], `max_age_s` [5: older requests are dropped], `warmup_width` and `warmup_height` [0: warm up on the first request. Required by trt-*]. Every 10 s it logs `served`, `dropped_old`, `dropped_full`, `failed`, `waiting`, `net_ms_p50` and `net_ms_p99`.

### Keys of ~status (ROS 2)

| Key | Meaning |
|---|---|
| `state` | tracking state: 1 not initialized, 2 OK, 3 recently lost, 4 lost |
| `imu_initialized`, `map_time_s`, `settling`, `map_changes` | IMU init done, map time since the first keyframe, settling period, map changes |
| `frames`, `dropped_frames`, `old_frames` | images received, dropped by the full queue (`frame_drop` oldest), skipped because their stamp is not after the last frame |
| `imu_rejected`, `imu_wait_timeouts` | IMU samples whose stamp is not after the last one, frames that found no IMU sample at or after their stamp within `imu_wait_ms` |
| `imu_filled` | IMU samples added by the gap fill. Rising: the IMU stream loses messages (USB or DDS) |
| `imu_fill_skipped` | gaps longer than `imu_fill_max_ms`, not filled |
| `imu_short_frames` | frames not tracked for lack of IMU samples |
| `bad_images` | images dropped (encoding, step, conversion) |
| `depth_requests`, `depth_pushed`, `depth_refused` | depth prior traffic |
| `track_ms_p50`, `track_ms_p90` | tracking time per frame over the last second |
| `latency_ms_p50`, `latency_ms_p90` | node clock at publish minus the image stamp, over the poses published in the last second (0 when none). Needs camera stamps on the system clock. Excludes MAVROS and PX4 transport |

Level OK (`tracking`) when state 2, IMU initialized and not settling. WARN while waiting for the IMU initialization, settling or recently lost. ERROR when lost or before the first image.

## Jetson Orin

See [runJetson.md](../runJetson.md). It replaces the Jetson notes of earlier versions of this file (those used cv_bridge with Ubuntu's OpenCV, which is no longer needed). The helper script is `scripts/jetson/setup_jetson.sh`.

## On a drone

- **Calibrate first.** The camera intrinsics, the camera-IMU transform `IMU.T_b_c1` (for example with Kalibr) and the IMU noise (Allan variance) go into a settings file. The depth constants of a new camera must be fitted from reference depth, with the backend you fly with, before the depth prior can be trusted. Give the camera its own `UCSLAM.Dataset` name and pass it as `domain`. For the D455 the tools in `scripts/jetson/` do this.
- **Start.** The monocular-inertial map needs motion to initialize, and nothing is published before that. `~status` reports `settling` until the last inertial refinement (76 s of map time mono-inertial, 16 s stereo-inertial), while scale and gravity can still move. Do not feed the pose to the flight controller before `~status` is OK. `ucslam_px4` enforces this.
- **Jumps.** A loop closure moves the map. `~status` counts these as `map_changes`. `ucslam_px4` re-anchors its output on a jump.
- **Frames.** `~odom` is the pose of the IMU in the UC-SLAM map frame, whose z axis points up after the initialization and whose yaw is arbitrary. `ucslam_px4` turns it into a `base_link` pose for MAVROS.
- **Stop.** Ctrl-C saves the trajectories, which can take up to a minute.

## Test status

| Check | Status |
|---|---|
| Benchmark path unchanged (the live-depth code is off by default) | code review done, run check pending |
| ROS 1 `slam_node` and the shared core compile (syntax check), ROS 1 launch file loads | done (syntax only after the core changes, no catkin build) |
| ROS 1 depth node message conversions, round trip | done |
| ROS 2 image and in-container build (Ubuntu 22.04, g++ 11, Eigen 3.4, OpenCV 4.5) | done: clean build of the 4 packages, `ldd slam_node`: one `libopencv_core.so.4.5d`, no cv_bridge |
| ROS 2 image conversion without cv_bridge | done: 282 cases equal the cv_bridge output |
| ROS 2 replay of EuRoC MH_01 mono-inertial, depth cache, one run | done: keyframe ATE 4.84 cm (the 10 benchmark runs: 2.89 to 10.28 cm), every keyframe got its depth |
| ROS 2 replay of EuRoC MH_01 mono-inertial, live depth on an RTX 4070 Ti, one run | done: keyframe ATE 4.55 cm, every keyframe got depth, network 90 to 95 ms per image (median), depth reaches a keyframe about 110 ms after it is made (median), so no keyframe got depth-prior points |
| ROS 2 replays after the node changes (IMU fill, guard, latency), depth cache | done, N=1 each, on a heavily loaded host (load 25 to 60): keyframe ATE 5.56 and 3.63 cm (fill on), later 9.90 cm with one tracking loss. Not an accuracy check |
| ROS 2 replay through `px4_bridge`, EuRoC MH_01 | done, N=1 each, loaded host: no pose before `~status` OK, z up (tilt 0.8 deg), no NaN, keyframe ATE 7.07 cm (`frame_drop` none) and 3.95 cm (oldest). Images lost in DDS (`rmem_max` 212992): output 17 to 18.5 Hz, gaps up to 0.7 s, 1 reset per run. Latency p50 106 ms, p90 198 ms (oldest), not valid for an Orin |
| `UCSLAM.DepthBackend` check | done: mismatch refused by `slam_node` and by `ucslam_d455.launch.py`, match and missing key start |
| Launch files `--show-args` | done: 42, 49 and 12 arguments |
| `ucslam_px4` unit tests (`colcon test`) | done: 27 tests, 0 failures. Synthetic live checks done |
| `ucslam_px4` with MAVROS and PX4 (SITL or hardware) | not run |
| TensorRT backends (x86, TensorRT 10.3, Python builder) | done: default engine parity p99 0.0093 to 0.0096 (pass), plain FP16 0.0220 on OpenLORIS office1-1 (fail). trt-* not run inside the ROS node |
| `trtexec` engine build, TensorRT 8.6 | not run |
| D455 tools (`make_d455_settings.py`, `fit_depth_constants.py`) | unit tests. The fit reproduced the paper's OpenLORIS constants with XFeat (log bias 0.1530 vs 0.1532). trt-* fits not run |
| Real D455, live camera | not run |
| Stereo-inertial tracking of `slam_node`, EuRoC MH_01 replay, depth off, `image_qos_reliable` | done, N=1 each, loaded host: at bag rate 0.25 every frame tracked, no map reset, keyframe ATE 4.12 cm (the plain `stereo_inertial_euroc` runner with the same settings: 4.83 cm, no reset). At bag rate 0.5 the CPU could not keep up, images were lost and the map reset 7 times. Not run on D455 data |
| Full catkin build and ROS 1 replay | to do |
| Jetson Orin (aarch64 build, OpenCV 4.8, TensorRT, D455, MAVROS, flight) | not tested. Steps in [runJetson.md](../runJetson.md) |
