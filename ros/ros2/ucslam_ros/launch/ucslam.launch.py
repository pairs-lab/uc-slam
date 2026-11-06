"""UC-SLAM on ROS 2: slam_node and, with depth:=live, depth_node. Examples in ros/README.md.

depth:=live starts depth_node (the network answers each keyframe), depth:=file reads the depth cache under depth_dir,
depth:=off starts no depth_node and runs without the depth prior (slam_node then sets UCSLAM.PriorNet 0 in its
settings copy).

When slam_node exits (also a failed start), the whole launch stops.
Ctrl-C stops both nodes. slam_node then waits for local mapping and saves f_<name>.txt and kf_<name>.txt in
output_dir, which can take up to a minute, so the SIGTERM timeout is raised to 120 s.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO = os.environ.get("UCSLAM_REPO", "")

ARGS = [
    ("repo", REPO, "UC-SLAM repository root"),
    ("settings", os.path.join(REPO, "configs", "euroc_mono_inertial.yaml"), "UC-SLAM settings YAML"),
    ("vocabulary", os.path.join(REPO, "third_party", "ORB_SLAM3", "Vocabulary", "ORBvoc.txt"), "ORB vocabulary"),
    ("output_dir", os.path.join(os.path.expanduser("~"), "ucslam_out"), "trajectories and the settings copy"),
    ("stereo", "false", "stereo-inertial instead of mono-inertial"),
    ("image_topic", "/cam0/image_raw", "left (or only) image: mono8, mono16, rgb8, bgr8, rgba8, bgra8 or yuv422"),
    ("image_right_topic", "/cam1/image_raw", "right image, stereo only"),
    ("imu_topic", "/imu0", "IMU, each message with both gyroscope and accelerometer"),
    ("depth", "live", "live: depth_node answers each keyframe. file: the depth cache under depth_dir. "
                      "off: no depth_node, no depth prior"),
    ("depth_dir", os.path.join(REPO, "data", "depth"), "depth cache, depth:=file only"),
    ("domain", "euroc", "must equal UCSLAM.Dataset of the settings"),
    ("view", "camera", "camera, or tumvi for the TUM-VI pinhole view"),
    ("tumvi_seq_dir", "", "TUM-VI sequence folder with the cam0 calibration, view:=tumvi only"),
    ("device", "cuda", "device of the depth network"),
    ("backend", "torch-fp32", "depth backend: torch-fp32, torch-fp16, trt-fp16 or trt-fp32. The depth constants "
                              "of the domain must be fitted with the same backend. slam_node stops at start when "
                              "the settings name another one in UCSLAM.DepthBackend (not with depth:=off)"),
    ("engine", "", "TensorRT engine file, backend:=trt-* only"),
    ("model", "", "depth model name, empty: the default of the depth config"),
    ("warmup_width", "0", "depth_node runs the network once at this size before serving, 0: no warm-up"),
    ("warmup_height", "0", "see warmup_width"),
    ("max_age_s", "5.0", "depth_node skips requests older than this"),
    ("frame_drop", "oldest", "oldest: drop frames when tracking falls behind (camera). none: bag replay"),
    ("frame_queue", "3", "frames waiting for tracking, frame_drop:=oldest only"),
    ("clahe", "false", "CLAHE before tracking, as the TUM-VI benchmark examples"),
    ("viewer", "false", "Pangolin viewer"),
    ("publish_points", "false", "publish ~/tracked_points"),
    ("publish_tf", "true", "broadcast the tf map_frame -> body_frame"),
    ("map_frame", "map", "frame of the published pose"),
    ("body_frame", "imu", "child frame of the published pose: the IMU body of the settings"),
    ("pose_std_m", "0.01", "position standard deviation in ~/odom [m], x10 when recently lost"),
    ("pose_std_rad", "0.01", "orientation standard deviation in ~/odom [rad], x10 when recently lost"),
    ("trajectory_name", "ucslam", "saved files f_<name>.txt and kf_<name>.txt"),
    ("depth_wait_ms", "2000.0", "late-depth rule UCSLAM.DepthWaitMs, depth:=live only"),
    ("dpp_defer_max_kfs", "0", "late-depth rule UCSLAM.DppDeferMaxKFs"),
    ("dpp_wait_ms", "0.0", "late-depth rule UCSLAM.DppWaitMs"),
    ("imu_wait_ms", "50.0", "wait for an IMU sample at or after each image. A frame inside an IMU gap longer than "
                            "this is not tracked when it has fewer than two IMU samples (status imu_short_frames)"),
    ("imu_time_offset", "0.0", "added to every IMU stamp [s]"),
    ("imu_fill_gap_ms", "0.0", "fill IMU gaps longer than this by linear interpolation, 0: off"),
    ("imu_fill_period_ms", "5.0", "spacing of the filled IMU samples"),
    ("imu_fill_max_ms", "1000.0", "IMU gaps longer than this are not filled (clock jump or bad stamp)"),
    ("imu_qos_reliable", "false", "reliable IMU subscription (needs a reliable publisher)"),
    ("image_qos_reliable", "false", "reliable image subscription (needs a reliable publisher). Use it for bag "
                                    "replays and when images get lost (frames stop growing in ~/status)"),
    ("use_sim_time", "false", "use /clock, for bag replays played with --clock"),
]


def _nodes(context):
    v = {n: LaunchConfiguration(n).perform(context) for n, _, _ in ARGS}
    flag = lambda n: v[n].strip().lower() == "true"  # noqa: E731
    sim = flag("use_sim_time")
    if v["depth"] not in ("live", "file", "off"):
        raise RuntimeError(f"launch argument depth must be live, file or off, got {v['depth']!r}")

    slam = Node(
        package="ucslam_ros", executable="slam_node", name="slam_node", namespace="ucslam", output="screen",
        parameters=[{
            "vocabulary": v["vocabulary"], "settings": v["settings"], "output_dir": v["output_dir"],
            "stereo": flag("stereo"), "viewer": flag("viewer"),
            "frame_drop": v["frame_drop"], "frame_queue": int(v["frame_queue"]),
            "clahe": flag("clahe"), "publish_points": flag("publish_points"), "publish_tf": flag("publish_tf"),
            "imu_qos_reliable": flag("imu_qos_reliable"), "image_qos_reliable": flag("image_qos_reliable"),
            "map_frame": v["map_frame"], "body_frame": v["body_frame"],
            "pose_std_m": float(v["pose_std_m"]), "pose_std_rad": float(v["pose_std_rad"]),
            "trajectory_name": v["trajectory_name"],
            "depth_source": v["depth"], "depth_dir": v["depth_dir"], "depth_backend": v["backend"],
            "depth_wait_ms": float(v["depth_wait_ms"]),
            "dpp_wait_ms": float(v["dpp_wait_ms"]), "dpp_defer_max_kfs": int(v["dpp_defer_max_kfs"]),
            "imu_wait_ms": float(v["imu_wait_ms"]), "imu_time_offset": float(v["imu_time_offset"]),
            "imu_fill_gap_ms": float(v["imu_fill_gap_ms"]), "imu_fill_period_ms": float(v["imu_fill_period_ms"]),
            "imu_fill_max_ms": float(v["imu_fill_max_ms"]),
            "use_sim_time": sim,
        }],
        remappings=[("image", v["image_topic"]), ("image_right", v["image_right_topic"]), ("imu", v["imu_topic"])],
        sigterm_timeout="120", sigkill_timeout="10",
        # Without slam_node the other nodes are useless: a failed start (for example a backend that does not match
        # UCSLAM.DepthBackend) or a crash ends the whole launch.
        on_exit=Shutdown(reason="slam_node exited"))
    nodes = [slam]

    if v["depth"] == "live":
        p = {
            "repo": v["repo"], "domain": v["domain"], "backend": v["backend"], "engine": v["engine"],
            "device": v["device"], "view": v["view"], "tumvi_seq_dir": v["tumvi_seq_dir"],
            "max_age_s": float(v["max_age_s"]),
            "warmup_width": int(v["warmup_width"]), "warmup_height": int(v["warmup_height"]),
            "use_sim_time": sim,
        }
        if v["model"]:
            p["model"] = v["model"]
        nodes.append(Node(package="ucslam_depth", executable="depth_node", name="depth_node", namespace="ucslam",
                          output="screen", parameters=[p]))
    return nodes


def generate_launch_description():
    decl = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
    return LaunchDescription(decl + [OpaqueFunction(function=_nodes)])
