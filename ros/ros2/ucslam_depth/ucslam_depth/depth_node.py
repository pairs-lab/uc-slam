"""UC-SLAM depth node for ROS 2 (Humble).

Answers each DepthRequest of slam_node (one per keyframe) with the metric depth map of the depth cache code
(ucslam/live_depth.py of the repository), as a DepthPrior. It needs torch, so it runs with the python of the
venv in ros/docker/Dockerfile (see ros/README.md).

Parameters:
  repo         repository root (default: the environment variable UCSLAM_REPO)
  domain       must equal UCSLAM.Dataset of the SLAM settings (euroc, tumvi, openloris, ...)
  backend      label of the network backend the per-domain constants were fitted with (torch-fp32)
  model        network name, empty: the model of configs/default.yaml
  device       cuda or cpu
  view         camera (the image as recorded) or tumvi (the TUM-VI pinhole rectification, needs tumvi_seq_dir)
  max_age_s    a request that waited longer is dropped (5.0)
  warmup_width, warmup_height   stream size for the warm-up at start, 0: the size of the first request
"""
import array
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from ucslam_msgs.msg import DepthPrior, DepthRequest


def image_to_gray(msg: Image) -> np.ndarray:
    if msg.encoding not in ("mono8", "8UC1"):
        raise ValueError(f"depth request image must be mono8, got {msg.encoding}")
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
    return np.ascontiguousarray(a)


def depth_to_image(zmap: np.ndarray, header) -> Image:
    z = np.ascontiguousarray(zmap.astype(np.float32))   # float16 values are exact in float32
    m = Image()
    m.header = header
    m.height, m.width = int(z.shape[0]), int(z.shape[1])
    m.encoding = "32FC1"
    m.is_bigendian = int(sys.byteorder == "big")
    m.step = 4 * m.width
    m.data = array.array("B", z.tobytes())   # an array.array skips the per-byte checks of a list or bytes
    return m


class DepthNode(Node):
    def __init__(self):
        super().__init__("depth_node")
        repo = self.declare_parameter("repo", os.environ.get("UCSLAM_REPO", "")).value
        if not repo or not os.path.isdir(os.path.join(repo, "ucslam")):
            raise RuntimeError(f"parameter repo (or UCSLAM_REPO) must name the UC-SLAM repository, got {repo!r}")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from ucslam.live_depth import DEFAULT_CONFIG, DepthWorker, LiveDepth, make_live_view

        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.domain = str(p("domain", "euroc"))
        self.backend = str(p("backend", "torch-fp32"))
        config = str(p("config", DEFAULT_CONFIG))
        seq_dir = str(p("tumvi_seq_dir", ""))
        if seq_dir and not os.path.isabs(seq_dir):   # relative paths are taken from the repository
            seq_dir = os.path.join(repo, seq_dir)
        view = make_live_view(str(p("view", "camera")), seq_dir, config)
        model, device = str(p("model", "")), str(p("device", "cuda"))
        max_age_s = float(p("max_age_s", 5.0))
        w, h = int(p("warmup_width", 0)), int(p("warmup_height", 0))
        self.get_logger().info("loading the network")
        self.live = LiveDepth(model or None, device, config, view=view)
        qos = QoSProfile(depth=16, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(DepthPrior, "depth_prior", qos)
        self.worker = DepthWorker(self.live, self.on_result, max_age_s=max_age_s, log=self.get_logger().info,
                                  warmup_shape=(h, w) if w > 0 and h > 0 else None)
        self.sub = self.create_subscription(DepthRequest, "depth_request", self.on_request, qos)
        self.timer = self.create_timer(10.0, lambda: self.get_logger().info(str(self.worker.stats())))
        self.get_logger().info(f"serving model {self.live.model} on {self.live.net.device}, domain {self.domain}")

    def on_request(self, msg: DepthRequest) -> None:
        try:
            gray = image_to_gray(msg.image)
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=10.0)
            return
        self.worker.submit(msg.header, gray)

    def on_result(self, header, zmap, desc, net_ms, queue_ms) -> None:
        if not rclpy.ok():
            return
        m = DepthPrior()
        m.header = header
        m.model = desc["model"]
        m.model_source = str(getattr(self.live.net, "source", ""))
        m.backend = self.backend
        m.domain = self.domain
        m.view = int(desc["kind"])
        m.full_width = int(desc["full_w"])
        m.full_height = int(desc["full_h"])
        m.k_view = np.asarray(desc["k_view"], np.float32)
        m.depth = depth_to_image(zmap, header)
        m.net_ms = float(net_ms)
        m.queue_ms = float(queue_ms)
        self.pub.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.worker.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
