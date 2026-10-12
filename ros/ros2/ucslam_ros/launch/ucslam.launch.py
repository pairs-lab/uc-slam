"""UC-SLAM on ROS 2: slam_node and, with depth:=live, depth_node. Examples in ros/README.md.

Ctrl-C stops both nodes. slam_node then waits for local mapping and saves f_<name>.txt and kf_<name>.txt in
output_dir, which can take up to a minute, so the SIGTERM timeout is raised to 120 s.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

REPO = os.environ.get("UCSLAM_REPO", "")


def generate_launch_description():
    args = [
        ("repo", REPO, "UC-SLAM repository root"),
        ("settings", os.path.join(REPO, "configs", "euroc_mono_inertial.yaml"), "UC-SLAM settings YAML"),
        ("vocabulary", os.path.join(REPO, "third_party", "ORB_SLAM3", "Vocabulary", "ORBvoc.txt"), "ORB vocabulary"),
        ("output_dir", os.path.join(os.path.expanduser("~"), "ucslam_out"), "trajectories and the settings copy"),
        ("stereo", "false", "stereo-inertial instead of mono-inertial"),
        ("image_topic", "/cam0/image_raw", "left (or only) image, mono8"),
        ("image_right_topic", "/cam1/image_raw", "right image, stereo only"),
        ("imu_topic", "/imu0", "IMU"),
        ("depth", "live", "live: depth_node answers each keyframe. file: the depth cache under depth_dir"),
        ("depth_dir", os.path.join(REPO, "data", "depth"), "depth cache, depth:=file only"),
        ("domain", "euroc", "must equal UCSLAM.Dataset of the settings"),
        ("view", "camera", "camera, or tumvi for the TUM-VI pinhole view"),
        ("tumvi_seq_dir", "", "TUM-VI sequence folder with the cam0 calibration, view:=tumvi only"),
        ("device", "cuda", "device of the depth network"),
        ("frame_drop", "oldest", "oldest: drop frames when tracking falls behind (camera). none: bag replay"),
        ("clahe", "false", "CLAHE before tracking, as the TUM-VI benchmark examples"),
        ("viewer", "false", "Pangolin viewer"),
        ("publish_points", "false", "publish ~/tracked_points"),
        ("dpp_defer_max_kfs", "0", "late-depth rule UCSLAM.DppDeferMaxKFs"),
        ("dpp_wait_ms", "0.0", "late-depth rule UCSLAM.DppWaitMs"),
        ("imu_qos_reliable", "false", "reliable IMU subscription (needs a reliable publisher)"),
    ]
    decl = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
    c = {n: LaunchConfiguration(n) for n, _, _ in args}

    slam = Node(
        package="ucslam_ros", executable="slam_node", name="slam_node", namespace="ucslam", output="screen",
        parameters=[{
            "vocabulary": c["vocabulary"], "settings": c["settings"], "output_dir": c["output_dir"],
            "stereo": PythonExpression(["'", c["stereo"], "' == 'true'"]),
            "viewer": PythonExpression(["'", c["viewer"], "' == 'true'"]),
            "frame_drop": c["frame_drop"],
            "clahe": PythonExpression(["'", c["clahe"], "' == 'true'"]),
            "publish_points": PythonExpression(["'", c["publish_points"], "' == 'true'"]),
            "imu_qos_reliable": PythonExpression(["'", c["imu_qos_reliable"], "' == 'true'"]),
            "depth_source": c["depth"], "depth_dir": c["depth_dir"], "depth_backend": "torch-fp32",
            "depth_wait_ms": 2000.0,
            "dpp_wait_ms": PythonExpression(["float('", c["dpp_wait_ms"], "')"]),
            "dpp_defer_max_kfs": PythonExpression(["int('", c["dpp_defer_max_kfs"], "')"]),
        }],
        remappings=[("image", c["image_topic"]), ("image_right", c["image_right_topic"]), ("imu", c["imu_topic"])],
        sigterm_timeout="120", sigkill_timeout="10")

    depth = Node(
        package="ucslam_depth", executable="depth_node", name="depth_node", namespace="ucslam", output="screen",
        condition=IfCondition(PythonExpression(["'", c["depth"], "' == 'live'"])),
        parameters=[{
            "repo": c["repo"], "domain": c["domain"], "backend": "torch-fp32", "device": c["device"],
            "view": c["view"], "tumvi_seq_dir": c["tumvi_seq_dir"], "max_age_s": 5.0,
        }])

    return LaunchDescription(decl + [slam, depth])
