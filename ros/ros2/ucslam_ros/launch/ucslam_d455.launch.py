"""UC-SLAM with an Intel RealSense D455: the camera driver, UC-SLAM, and optionally the PX4 bridge and a bag.

Modes (launch argument mode):

    mono    mono-inertial on the global-shutter color camera, with the depth prior. The research mode. Needs
            configs/d455/d455_mono_inertial.yaml with fitted d455color constants. Settles after 76 s of map time.
    stereo  stereo-inertial on infra1 + infra2 with the IR emitter off and no depth prior (depth off). Settles
            after 16 s and needs no fitted constants: the safe mode for closed-loop flight.
            Needs configs/d455/d455_stereo_inertial.yaml.
    calib   color + depth aligned to color + IMU, for scripts/jetson/fit_depth_constants.py and
            make_d455_settings.py. No SLAM. Use with record:=true.

Both settings files are written by scripts/jetson/make_d455_settings.py. With depth:=auto the depth node starts only
when the settings have UCSLAM.PriorNet 1. The domain is UCSLAM.Dataset of the settings (d455color or d455ir).

Depth backend: a mono file with fitted constants names the backend of the fit in UCSLAM.DepthBackend (and the
engine in D455.FitEngine). backend:= empty takes that backend, and the launch stops when backend:= names another
one, because the constants are only valid for the backend they were fitted with (slam_node checks the same key).
Without UCSLAM.DepthBackend the default is torch-fp32.

IMU chip: imu_chip:=auto (default) reads the accelerometer rates of the attached D455 with pyrealsense2 and takes
the highest one (250 Hz for a BMI055, 200 Hz for a BMI085). Without pyrealsense2 or without a camera it stops
with a message: then give imu_chip:=bmi055 or bmi085 (rs-enumerate-devices lists the Accel profiles). The two chips
share no accelerometer rate, and a rate the unit does not support leaves the driver without an accel stream.

The stream size (width, height, fps) must equal Camera.width, Camera.height and Camera.fps of the settings.

PX4 bridge: base_x .. base_yaw empty (default) keep the mount of ucslam_px4/config/px4_bridge.yaml (or of
px4_params_file), which is a D455 that looks forward and level (base_roll pi/2, base_pitch -pi/2). Each base_*
argument that is set overrides that one value only. px4_use_tf empty keeps use_tf of the same file. The bridge gets
use_sim_time of this launch.

Examples:

    ros2 launch ucslam_ros ucslam_d455.launch.py mode:=calib record:=true
    ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo slam:=false        # camera only, for the settings
    ros2 launch ucslam_ros ucslam_d455.launch.py mode:=mono engine:=$HOME/engines/da2b.plan
    ros2 launch ucslam_ros ucslam_d455.launch.py mode:=stereo px4:=true base_z:=-0.05 record:=true

Needs realsense2_camera 4.54 or newer (launch arguments rgb_camera.color_profile, depth_module.depth_profile,
depth_module.infra_profile). UCSLAM_REPO must point to the repository.
"""
import datetime
import os
import re

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, LogInfo, \
    OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

REPO = os.environ.get("UCSLAM_REPO", "")
HERE = os.path.dirname(os.path.abspath(__file__))
D455 = os.path.join(REPO, "configs", "d455")
ACCEL_FPS = {"bmi055": "250", "bmi085": "200"}   # D455 datasheet tables 4-25 and 4-26, the higher rate
BASE = ("base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw")
BACKENDS = ("torch-fp32", "torch-fp16", "trt-fp16", "trt-fp32")

ARGS = [
    ("mode", "mono", "mono, stereo or calib, see the file docstring"),
    ("slam", "true", "start UC-SLAM. false: camera only (to record, or to run make_d455_settings.py --live)"),
    ("settings", "", "UC-SLAM settings, empty: configs/d455/d455_mono_inertial.yaml or d455_stereo_inertial.yaml"),
    ("vocabulary", "", "ORB vocabulary, empty: the default of ucslam.launch.py"),
    ("output_dir", os.path.join(os.path.expanduser("~"), "ucslam_out"), "trajectories and the settings copy"),
    ("depth", "auto", "auto: live when the settings have UCSLAM.PriorNet 1, else off. live or off to force it"),
    ("domain", "", "depth domain, empty: UCSLAM.Dataset of the settings"),
    ("backend", "", "depth backend: torch-fp32, torch-fp16, trt-fp16 or trt-fp32. Empty: UCSLAM.DepthBackend of "
                    "the settings, else torch-fp32. Another backend than UCSLAM.DepthBackend is refused"),
    ("engine", "", "TensorRT engine file, backend trt-* only. Empty: D455.FitEngine of the settings"),
    ("model", "", "depth model, empty: the default"),
    ("device", "cuda", "device of the depth network"),
    ("imu_fill_gap_ms", "20.0", "fill IMU gaps longer than this (the RealSense IMU over USB can drop samples)"),
    ("imu_fill_period_ms", "5.0", "spacing of the filled IMU samples (5 ms = 200 Hz)"),
    ("imu_time_offset", "0.0", "added to every IMU stamp [s], for example minus the Kalibr timeshift_cam_imu"),
    ("imu_wait_ms", "50.0", "wait for an IMU sample at or after each image"),
    ("image_qos_reliable", "false", "reliable image subscriptions, so a lost UDP fragment is resent instead of losing "
                                    "the image. Needs a reliable publisher (the RealSense driver by default)"),
    ("imu_qos_reliable", "false", "reliable IMU subscription"),
    ("depth_wait_ms", "2000.0", "late-depth rule UCSLAM.DepthWaitMs"),
    ("dpp_defer_max_kfs", "0", "late-depth rule UCSLAM.DppDeferMaxKFs"),
    ("dpp_wait_ms", "0.0", "late-depth rule UCSLAM.DppWaitMs"),
    ("frame_queue", "3", "frames waiting for tracking, the oldest is dropped beyond"),
    ("publish_tf", "true", "UC-SLAM broadcasts the tf map -> imu"),
    ("viewer", "false", "Pangolin viewer"),
    ("trajectory_name", "ucslam", "saved files f_<name>.txt and kf_<name>.txt"),
    ("use_sim_time", "false", "UC-SLAM nodes use /clock (replays). The camera is never started with it"),
    ("camera", "true", "start the RealSense driver. false: the topics come from a bag replay"),
    ("camera_namespace", "camera", "realsense2_camera camera_namespace"),
    ("camera_name", "camera", "realsense2_camera camera_name"),
    ("width", "848", "stream width (color in mono and calib, infrared in stereo)"),
    ("height", "480", "stream height"),
    ("fps", "30", "stream rate"),
    ("imu_chip", "auto", "IMU of this D455: bmi055 (accel 250 Hz), bmi085 (accel 200 Hz, D455f and newer units) "
                         "or auto (ask the device with pyrealsense2)"),
    ("accel_fps", "", "accelerometer rate, empty: from imu_chip"),
    ("gyro_fps", "200", "gyroscope rate, the rate of the united IMU topic and of IMU.Frequency"),
    ("auto_exposure", "true", "color auto exposure (mono and calib)"),
    ("initial_reset", "false", "reset the camera at start, helps after a USB error"),
    ("record", "false", "record a bag (mcap) of the camera, IMU, tf and UC-SLAM topics"),
    ("record_images", "true", "with record: also the images (about 2 GB per minute in mono and calib)"),
    ("bag_dir", os.path.join(os.path.expanduser("~"), "ucslam_bags"), "parent folder of the recorded bags"),
    ("px4", "false", "start the ucslam_px4 bridge to MAVROS"),
    ("px4_output", "vision_pose", "vision_pose (/mavros/vision_pose/pose) or odometry (/mavros/odometry/out)"),
    ("px4_params_file", "", "parameter file of the bridge, empty: config/px4_bridge.yaml of ucslam_px4"),
    ("px4_use_tf", "", "true: the bridge reads the base_link mount from TF instead of base_*. Empty: use_tf of the "
                       "bridge's parameter file"),
    ("base_x", "", "position of base_link in the IMU body frame of UC-SLAM [m]. Empty: the value of the bridge's "
                   "parameter file (forward-looking D455)"),
    ("base_y", "", "see base_x"),
    ("base_z", "", "see base_x"),
    ("base_roll", "", "orientation of base_link in the IMU body frame [rad]. Empty: the value of the bridge's "
                      "parameter file (pi/2 for a forward-looking D455)"),
    ("base_pitch", "", "see base_roll (-pi/2 for a forward-looking D455)"),
    ("base_yaw", "", "see base_roll"),
]


def _share(pkg):
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    try:
        return get_package_share_directory(pkg)
    except PackageNotFoundError:
        return None


def _setting(path, key):
    """Value text of a top-level key of an OpenCV YAML settings file, or None."""
    with open(path) as f:
        for line in f:
            m = re.match(r"^" + re.escape(key) + r":\s*\"?([^\"#\s]*)", line)
            if m:
                return m.group(1)
    return None


def _detect_accel_fps():
    """(accelerometer rate for the first attached RealSense with an IMU, None) or (None, reason).

    250 Hz (BMI055) or 200 Hz (BMI085) when the unit has it, so a BMI085 with the newer 400 Hz firmware stays at 200."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None, "pyrealsense2 is not installed"
    try:
        for dev in rs.context().query_devices():
            rates = {int(p.fps()) for s in dev.query_sensors() for p in s.get_stream_profiles()
                     if p.stream_type() == rs.stream.accel}
            if rates:
                return str(next((r for r in (250, 200) if r in rates), max(rates))), None
    except Exception as e:   # noqa: BLE001 (any librealsense error means: no answer)
        return None, f"pyrealsense2 failed: {e}"
    return None, "no RealSense with an accelerometer is attached"


def _int_setting(path, key):
    s = _setting(path, key)
    try:
        return int(round(float(s)))
    except (TypeError, ValueError):
        return None


def _include(path, args):
    """Include a launch file with plain string arguments, in a scope that sees none of this file's arguments."""
    inc = IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=list(args.items()))
    return GroupAction([inc], scoped=True, forwarding=False)


def _setup(context):
    v = {n: LaunchConfiguration(n).perform(context) for n, _, _ in ARGS}
    flag = lambda n: v[n].strip().lower() == "true"  # noqa: E731
    mode = v["mode"]
    if not REPO or not os.path.isdir(D455):
        raise RuntimeError(f"UCSLAM_REPO is {REPO!r}, and {D455} is not a folder. Set UCSLAM_REPO to the root of the "
                           "UC-SLAM repository (export UCSLAM_REPO=$HOME/ucslam) and launch again")
    if mode not in ("mono", "stereo", "calib"):
        raise RuntimeError(f"mode must be mono, stereo or calib, got {mode!r}")
    if v["imu_chip"] not in ACCEL_FPS and v["imu_chip"] != "auto" and not v["accel_fps"]:
        raise RuntimeError(f"imu_chip must be auto, bmi055 or bmi085, got {v['imu_chip']!r}")
    profile = f"{v['width']}x{v['height']}x{v['fps']}"
    cam = f"/{v['camera_namespace']}/{v['camera_name']}" if v["camera_namespace"] else f"/{v['camera_name']}"
    t_color, t_imu = f"{cam}/color/image_raw", f"{cam}/imu"
    t_ir1, t_ir2 = f"{cam}/infra1/image_rect_raw", f"{cam}/infra2/image_rect_raw"
    t_depth = f"{cam}/aligned_depth_to_color/image_raw"
    actions = [LogInfo(msg=f"ucslam_d455: mode {mode}, profile {profile}, gyro {v['gyro_fps']} Hz")]

    # ---- RealSense driver
    if flag("camera"):
        rs_share = _share("realsense2_camera")
        if rs_share is None:
            raise RuntimeError("realsense2_camera is not installed (ros-humble-realsense2-camera or a source build)")
        accel = v["accel_fps"] or ACCEL_FPS.get(v["imu_chip"], "")
        if not accel:
            accel, why = _detect_accel_fps()
            if accel is None:
                raise RuntimeError(f"imu_chip:=auto cannot read the IMU of the camera ({why}). Give imu_chip:=bmi055 "
                                   "(Accel 63/250 Hz in rs-enumerate-devices) or imu_chip:=bmi085 (Accel 100/200 Hz)")
            actions.append(LogInfo(msg=f"ucslam_d455: imu_chip auto, accel_fps {accel} from the device"))
        rs = {"camera_namespace": v["camera_namespace"], "camera_name": v["camera_name"],
              "initial_reset": v["initial_reset"], "publish_tf": "true",
              "enable_gyro": "true", "enable_accel": "true", "gyro_fps": v["gyro_fps"],
              "accel_fps": accel, "unite_imu_method": "2",
              "enable_infra1": "false", "enable_infra2": "false"}
        if mode in ("mono", "calib"):
            rs.update({"enable_color": "true", "rgb_camera.color_profile": profile,
                       "rgb_camera.enable_auto_exposure": v["auto_exposure"]})
        if mode == "mono":
            rs.update({"enable_depth": "false"})
        elif mode == "calib":
            rs.update({"enable_depth": "true", "depth_module.depth_profile": profile, "align_depth.enable": "true",
                       "enable_sync": "true"})
        else:
            rs.update({"enable_color": "false", "enable_depth": "false", "enable_infra1": "true",
                       "enable_infra2": "true", "depth_module.infra_profile": profile,
                       "config_file": os.path.join(D455, "realsense_stereo.yaml")})
        actions.append(_include(os.path.join(rs_share, "launch", "rs_launch.py"), rs))

    # ---- UC-SLAM
    run_slam = flag("slam") and mode != "calib"
    if run_slam:
        settings = v["settings"] or os.path.join(
            D455, "d455_stereo_inertial.yaml" if mode == "stereo" else "d455_mono_inertial.yaml")
        if not os.path.exists(settings):
            raise RuntimeError(f"settings {settings} not found. Write it with scripts/jetson/make_d455_settings.py "
                               "(UCSLAM_REPO must point to the repository)")
        prior = _setting(settings, "UCSLAM.PriorNet")
        domain = v["domain"] or _setting(settings, "UCSLAM.Dataset") or ""
        size = tuple(_int_setting(settings, k) for k in ("Camera.width", "Camera.height", "Camera.fps"))
        if flag("camera") and size != (int(v["width"]), int(v["height"]), int(v["fps"])):
            raise RuntimeError(f"the stream {profile} differs from Camera.width x height x fps = "
                               f"{'x'.join(str(s) for s in size)} of {settings}. The intrinsics hold for one size "
                               "only: write the settings again at this size or launch with the size of the settings")
        depth = v["depth"]
        if depth == "auto":
            depth = "live" if prior == "1" else "off"
        if depth not in ("live", "off"):
            raise RuntimeError(f"depth must be auto, live or off here, got {depth!r}")
        if mode == "stereo" and depth == "live":
            raise RuntimeError("stereo mode has no fitted infrared depth constants: use depth:=off")
        if depth == "live" and prior != "1":
            raise RuntimeError(f"depth:=live but {settings} has UCSLAM.PriorNet {prior}: fit the constants first")
        if prior != "1":
            actions.append(LogInfo(msg=f"ucslam_d455: {os.path.basename(settings)} has UCSLAM.PriorNet {prior}, "
                                       "running without the depth prior"))
        # The depth constants hold for the backend of the fit only (UCSLAM.DepthBackend, written by
        # make_d455_settings). slam_node refuses another depth_backend too, so the check runs with depth off as well.
        fit_backend = _setting(settings, "UCSLAM.DepthBackend") or ""
        fit_engine = _setting(settings, "D455.FitEngine") or ""
        backend, engine = v["backend"] or fit_backend or "torch-fp32", v["engine"]
        if backend not in BACKENDS:
            raise RuntimeError(f"backend must be one of {', '.join(BACKENDS)}, got {backend!r}")
        if fit_backend and backend != fit_backend:
            raise RuntimeError(f"backend:={backend}, but the depth constants of {settings} were fitted with "
                               f"{fit_backend} (UCSLAM.DepthBackend). Launch with backend:={fit_backend} or leave "
                               f"backend empty, or fit the constants again with {backend}")
        if depth == "live":
            if not fit_backend:
                actions.append(LogInfo(msg=f"ucslam_d455: WARNING {os.path.basename(settings)} has no "
                                           f"UCSLAM.DepthBackend, so backend {backend} cannot be checked against the "
                                           "constant fit"))
            if backend.startswith("trt"):
                if not engine:
                    engine = fit_engine
                elif fit_engine and os.path.realpath(engine) != os.path.realpath(fit_engine):
                    actions.append(LogInfo(msg=f"ucslam_d455: WARNING engine {engine} is not the engine of the fit "
                                               f"({fit_engine}). Another engine needs another fit"))
                if not engine:
                    raise RuntimeError(f"backend {backend} needs engine:=<TensorRT engine file>")
                if not os.path.exists(engine):
                    raise RuntimeError(f"engine {engine} not found")
            actions.append(LogInfo(msg=f"ucslam_d455: depth backend {backend}"
                                       + (f", engine {engine}" if backend.startswith("trt") else "")
                                       + (f" (fit backend {fit_backend})" if fit_backend else "")))
        u = {"settings": settings, "output_dir": v["output_dir"], "stereo": "true" if mode == "stereo" else "false",
             "image_topic": t_ir1 if mode == "stereo" else t_color, "image_right_topic": t_ir2, "imu_topic": t_imu,
             "depth": depth, "domain": domain, "backend": backend, "engine": engine, "model": v["model"],
             "device": v["device"], "frame_drop": "oldest", "frame_queue": v["frame_queue"],
             "imu_fill_gap_ms": v["imu_fill_gap_ms"], "imu_fill_period_ms": v["imu_fill_period_ms"],
             "imu_time_offset": v["imu_time_offset"], "imu_wait_ms": v["imu_wait_ms"],
             "image_qos_reliable": v["image_qos_reliable"], "imu_qos_reliable": v["imu_qos_reliable"],
             "depth_wait_ms": v["depth_wait_ms"], "dpp_defer_max_kfs": v["dpp_defer_max_kfs"],
             "dpp_wait_ms": v["dpp_wait_ms"], "publish_tf": v["publish_tf"], "viewer": v["viewer"],
             "trajectory_name": v["trajectory_name"], "use_sim_time": v["use_sim_time"]}
        if depth == "live" and size[0] and size[1]:
            u.update({"warmup_width": str(size[0]), "warmup_height": str(size[1])})
        if v["vocabulary"]:
            u["vocabulary"] = v["vocabulary"]
        share = _share("ucslam_ros")
        path = os.path.join(share, "launch", "ucslam.launch.py") if share else os.path.join(HERE, "ucslam.launch.py")
        actions.append(_include(path, u))

    # ---- PX4 bridge
    if flag("px4"):
        if not run_slam:
            raise RuntimeError("px4:=true needs UC-SLAM (slam:=true and mode mono or stereo)")
        share = _share("ucslam_px4")
        if share is None:
            raise RuntimeError("ucslam_px4 is not installed: colcon build the packages of ros/ros2")
        p = {"odom_topic": "/ucslam/slam_node/odom", "status_topic": "/ucslam/slam_node/status",
             "output": v["px4_output"], "use_sim_time": v["use_sim_time"]}
        if v["px4_params_file"]:
            p["params_file"] = v["px4_params_file"]
        if v["px4_use_tf"].strip():
            p["use_tf"] = v["px4_use_tf"].strip()
        # Only the base_* that are set. An empty value keeps the mount of the bridge's parameter file: passing
        # "0.0" here would replace the forward-looking mount (roll pi/2, pitch -pi/2) with the identity.
        p.update({k: v[k].strip() for k in BASE if v[k].strip()})
        mount = ", ".join(f"{k} {p[k]}" for k in BASE + ("use_tf",) if k in p) or \
            "the values of the bridge's parameter file"
        actions.append(LogInfo(msg=f"ucslam_d455: PX4 bridge, output {v['px4_output']}, mount {mount}"))
        actions.append(_include(os.path.join(share, "launch", "px4_bridge.launch.py"), p))

    # ---- bag
    if flag("record"):
        topics = [f"{cam}/color/camera_info", f"{cam}/infra1/camera_info", f"{cam}/infra2/camera_info",
                  f"{cam}/aligned_depth_to_color/camera_info", t_imu, "/tf_static", "/tf"]
        if flag("record_images"):
            topics += {"mono": [t_color], "stereo": [t_ir1, t_ir2], "calib": [t_color, t_depth]}[mode]
        if run_slam:
            topics += ["/ucslam/slam_node/odom", "/ucslam/slam_node/pose", "/ucslam/slam_node/status",
                       "/ucslam/depth_request", "/ucslam/depth_prior"]
        if flag("px4"):
            topics += ["/ucslam/px4_bridge/status", "/mavros/vision_pose/pose", "/mavros/odometry/out",
                       "/mavros/local_position/pose", "/mavros/state"]
        out = os.path.join(os.path.expanduser(v["bag_dir"]), f"d455_{mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        actions.append(LogInfo(msg=f"ucslam_d455: recording {out}"))
        actions.append(ExecuteProcess(cmd=["ros2", "bag", "record", "-s", "mcap", "-o", out] + topics,
                                      output="screen", sigterm_timeout="20"))
    return actions


def generate_launch_description():
    decl = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
    return LaunchDescription(decl + [OpaqueFunction(function=_setup)])
