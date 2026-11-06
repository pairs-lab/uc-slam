"""Starts the UC-SLAM to PX4 bridge (node px4_bridge in namespace ucslam).

  ros2 launch ucslam_px4 px4_bridge.launch.py output:=vision_pose base_x:=0.05

The parameters come from params_file (default: config/px4_bridge.yaml of this package). Each argument below
that is not empty overrides the value of the file, so an empty argument keeps the file value:
  odom_topic, status_topic, output (vision_pose or odometry), use_tf,
  base_x, base_y, base_z (m), base_roll, base_pitch, base_yaw (rad): pose of base_link in the IMU body frame.
use_sim_time is false by default (true for bag replays).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"not a boolean: {v!r}")


OVERRIDES = [
    ("odom_topic", str, "UC-SLAM odometry topic"),
    ("status_topic", str, "UC-SLAM status topic"),
    ("output", str, "vision_pose (/mavros/vision_pose/pose) or odometry (/mavros/odometry/out)"),
    ("use_tf", _bool, "read the base_link mount from TF instead of base_*"),
    ("base_x", float, "base_link x in the IMU body frame [m]"),
    ("base_y", float, "base_link y in the IMU body frame [m]"),
    ("base_z", float, "base_link z in the IMU body frame [m]"),
    ("base_roll", float, "base_link roll in the IMU body frame [rad]"),
    ("base_pitch", float, "base_link pitch in the IMU body frame [rad]"),
    ("base_yaw", float, "base_link yaw in the IMU body frame [rad]"),
]


def _node(context):
    over = {}
    for name, kind, _ in OVERRIDES:
        v = LaunchConfiguration(name).perform(context).strip()
        if v:
            over[name] = kind(v)
    over["use_sim_time"] = _bool(LaunchConfiguration("use_sim_time").perform(context))
    return [Node(package="ucslam_px4", executable="px4_bridge", name="px4_bridge", namespace="ucslam",
                 output="screen", parameters=[LaunchConfiguration("params_file").perform(context), over])]


def generate_launch_description():
    share = get_package_share_directory("ucslam_px4")
    args = [DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "px4_bridge.yaml"),
                                  description="parameter file of the bridge"),
            DeclareLaunchArgument("use_sim_time", default_value="false")]
    args += [DeclareLaunchArgument(n, default_value="", description=d + " (empty: value of params_file)")
             for n, _, d in OVERRIDES]
    return LaunchDescription(args + [OpaqueFunction(function=_node)])
