"""Bridge from UC-SLAM to PX4 through MAVROS (ROS 2 Humble).

Reads the odometry and status of slam_node, converts the pose of the UC-SLAM IMU body to base_link (FLU) in an
ENU world, holds it while UC-SLAM is not ready, re-anchors it after a jump, and sends it to MAVROS:
  output vision_pose  geometry_msgs/PoseStamped on /mavros/vision_pose/pose (VISION_POSITION_ESTIMATE)
  output odometry     nav_msgs/Odometry on /mavros/odometry/out (ODOMETRY), frame odom, child base_link
The stamp is the image stamp of UC-SLAM, unchanged: MAVROS timesync maps it to the PX4 clock.
The logic lives in bridge_core.py (no ROS). Parameters: config/px4_bridge.yaml.
"""
import math
import signal

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from ucslam_px4.bridge_core import BridgeConfig, BridgeCore, Pose, StatusReport, mount_from_params


def _as_bool(v) -> bool:
    if isinstance(v, str):
        if v.strip().lower() in ("true", "1", "yes"):
            return True
        if v.strip().lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"not a boolean: {v!r}")
    return bool(v)


class Px4Bridge(Node):
    def __init__(self):
        super().__init__("px4_bridge")
        # dynamic_typing: a yaml value 5 (int) for a float parameter must not stop the node. Each value is
        # converted to the type of its default.
        dyn = ParameterDescriptor(dynamic_typing=True)

        def p(name, default):
            v = self.declare_parameter(name, default, dyn).value
            if v is None:
                return default
            if isinstance(default, bool):
                return _as_bool(v)
            return type(default)(v)

        odom_topic = str(p("odom_topic", "/ucslam/slam_node/odom"))
        status_topic = str(p("status_topic", "/ucslam/slam_node/status"))
        self.output = str(p("output", "vision_pose"))
        if self.output not in ("vision_pose", "odometry"):
            raise ValueError(f"parameter output must be vision_pose or odometry, got {self.output!r}")
        vision_pose_topic = str(p("vision_pose_topic", "/mavros/vision_pose/pose"))
        odometry_topic = str(p("odometry_topic", "/mavros/odometry/out"))
        mount = [p(n, 0.0) for n in ("base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw")]
        self.use_tf = p("use_tf", False)
        self.base_frame = str(p("base_frame", "base_link"))
        cfg = BridgeConfig(
            max_speed_mps=p("max_speed_mps", 5.0),
            max_yaw_rate_dps=p("max_yaw_rate_dps", 180.0),
            jump_margin_m=p("jump_margin_m", 0.05),
            jump_margin_deg=p("jump_margin_deg", 3.0),
            gap_reset_s=p("gap_reset_s", 0.5),
            require_status_ok=p("require_status_ok", True),
            status_timeout_s=p("status_timeout_s", 3.0),
            start_frames=p("start_frames", 3))
        self.rate_min_hz = p("publish_rate_min_hz", 20.0)
        self.core = BridgeCore(cfg, None if self.use_tf else mount_from_params(*mount))
        self.body_frame = ""   # child frame of the UC-SLAM odometry, read from the first message

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        if self.output == "vision_pose":
            self.pub = self.create_publisher(PoseStamped, vision_pose_topic, qos)
            out_topic = vision_pose_topic
        else:
            self.pub = self.create_publisher(Odometry, odometry_topic, qos)
            out_topic = odometry_topic
        self.pub_status = self.create_publisher(DiagnosticArray, "~/status", qos)
        self.create_subscription(Odometry, odom_topic, self.on_odom, qos)
        self.create_subscription(DiagnosticArray, status_topic, self.on_status, qos)
        if self.use_tf:
            from tf2_ros import Buffer, TransformListener   # only needed for use_tf
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(1.0, self.on_timer)
        mount_text = "from TF" if self.use_tf else \
            "x %.3f y %.3f z %.3f m, roll %.4f pitch %.4f yaw %.4f rad" % tuple(mount)
        self.get_logger().info(f"{odom_topic} -> {out_topic} ({self.output}), base_link mount {mount_text}, "
                               f"status gate {'on' if cfg.require_status_ok else 'off'}")
        if not self.use_tf and all(abs(a) < 1e-9 for a in mount[3:]):
            self.get_logger().warn("base_link mount rotation is the identity: the UC-SLAM body axes are sent as "
                                   "FLU. With the D455 settings the body is the IMU optical frame (x right, y down, "
                                   "z forward). For a camera that looks forward and level, set base_roll 1.5707963 "
                                   "base_pitch -1.5707963 base_yaw 0 (the yaml defaults)")

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_status(self, msg: DiagnosticArray) -> None:
        for s in msg.status:
            if s.name == "ucslam":
                values = {kv.key: kv.value for kv in s.values}
                level = s.level if isinstance(s.level, int) else int.from_bytes(s.level, "little")
                self.core.on_status(StatusReport(level, s.message, values), self.now_s())
                self.log_events()
                return

    def on_odom(self, msg: Odometry) -> None:
        if not self.body_frame:
            self.body_frame = msg.child_frame_id
        if self.use_tf and self.core.mount is None:
            self.lookup_mount()
        pp, qq = msg.pose.pose.position, msg.pose.pose.orientation
        try:
            raw = Pose.from_quat((pp.x, pp.y, pp.z), (qq.x, qq.y, qq.z, qq.w))
        except ValueError as e:   # zero or non-finite quaternion: counted and logged by the core
            self.core.on_invalid(str(e), self.now_s())
            self.log_events()
            return
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        out = self.core.on_odom(stamp, raw, list(msg.pose.covariance), self.now_s())
        self.log_events()
        if out is None:
            return
        q = out.pose.quat()
        if self.output == "vision_pose":
            m = PoseStamped()
            m.header.stamp = msg.header.stamp
            m.header.frame_id = "map"
            pose = m.pose
        else:
            m = Odometry()
            m.header.stamp = msg.header.stamp
            m.header.frame_id = "odom"
            m.child_frame_id = "base_link"
            pose = m.pose.pose
            m.pose.covariance = [float(v) for v in out.covariance]
            # UC-SLAM gives no velocity. PX4 ignores a non-finite ODOMETRY velocity (mavlink_receiver.cpp,
            # isAllFinite check), so NaN keeps it from fusing a zero velocity.
            nan = float("nan")
            m.twist.twist.linear.x = m.twist.twist.linear.y = m.twist.twist.linear.z = nan
            m.twist.twist.angular.x = m.twist.twist.angular.y = m.twist.twist.angular.z = nan
        pose.position.x, pose.position.y, pose.position.z = out.pose.p
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        self.pub.publish(m)

    def lookup_mount(self) -> None:
        body = self.body_frame or "imu"
        try:
            t = self.tf_buffer.lookup_transform(body, self.base_frame, Time())
        except Exception as e:   # tf2 raises several exception types, all mean "not yet"
            self.get_logger().warn(f"waiting for TF {body} -> {self.base_frame}: {e}", throttle_duration_sec=5.0)
            return
        tr, ro = t.transform.translation, t.transform.rotation
        self.core.mount = Pose.from_quat((tr.x, tr.y, tr.z), (ro.x, ro.y, ro.z, ro.w))
        self.get_logger().info(f"base_link mount from TF {body} -> {self.base_frame}: "
                               f"({tr.x:.3f}, {tr.y:.3f}, {tr.z:.3f}) m, q ({ro.x:.4f}, {ro.y:.4f}, "
                               f"{ro.z:.4f}, {ro.w:.4f})")

    def log_events(self) -> None:
        # rclpy refuses two severities from one call site, so each severity has its own line.
        for level, text in self.core.pop_events():
            if level == "warn":
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)

    def on_timer(self) -> None:
        now = self.now_s()
        self.core.tick(now)
        st = self.core.stats(now)
        state = self.core.state
        s = DiagnosticStatus()
        s.name = "px4_bridge"
        s.hardware_id = "ucslam"
        # the first window of a segment is partly empty, so its rate is not judged
        slow = state == "sending" and st["segment_age_s"] > 1.5 and st["rate_hz"] < self.rate_min_hz
        if state == "sending" and not slow:
            s.level, s.message = DiagnosticStatus.OK, "sending"
        elif slow:
            s.level, s.message = DiagnosticStatus.WARN, f"sending below {self.rate_min_hz:.0f} Hz"
            self.get_logger().warn(f"output rate {st['rate_hz']:.1f} Hz is below publish_rate_min_hz "
                                   f"{self.rate_min_hz:.1f} (PX4 wants 30 to 50 Hz)", throttle_duration_sec=5.0)
        else:
            s.level, s.message = DiagnosticStatus.WARN, state
        values = [("state", state), ("output", self.output)]
        for k in ("sent", "held", "jumps", "resets", "segments", "received", "old", "invalid"):
            values.append((k, str(int(st[k]))))
        for k in ("rate_hz", "latency_ms_p50", "latency_ms_p90", "correction_yaw_deg"):
            values.append((k, "%.2f" % st[k] if math.isfinite(st[k]) else "nan"))
        s.values = [KeyValue(key=k, value=v) for k, v in values]
        a = DiagnosticArray()
        a.header.stamp = self.get_clock().now().to_msg()
        a.status.append(s)
        self.pub_status.publish(a)


def main(args=None):
    rclpy.init(args=args)
    node = Px4Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl-C reaches the node twice (from the terminal process group and from ros2 launch). A second
        # KeyboardInterrupt inside destroy_node gave a traceback, so further SIGINT is ignored while cleaning up.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
