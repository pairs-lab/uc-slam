"""UC-SLAM depth node for ROS 2 (Humble).

Answers each DepthRequest of slam_node (one per keyframe) with the metric depth map of the depth cache code
(ucslam/live_depth.py of the repository), as a DepthPrior. It needs torch, so it runs with the python of the
venv in ros/docker/Dockerfile (see ros/README.md).

Parameters:
  repo         repository root (default: the environment variable UCSLAM_REPO)
  domain       must equal UCSLAM.Dataset of the SLAM settings (euroc, tumvi, openloris, ...)
  backend      network backend: torch-fp32 (default, the backend of the depth cache), torch-fp16, trt-fp16 or
               trt-fp32. It is also the DepthPrior.backend label, which slam_node compares with its depth_backend
               (and with UCSLAM.DepthBackend of the settings when present). The per-domain constants hold for the
               backend they were fitted with
  engine       TensorRT engine file for trt-fp16 and trt-fp32 (relative paths are taken from the repository).
               An engine has a fixed input size: the stream size must match it
  model        network name, empty: the model of configs/default.yaml
  device       cuda or cpu
  view         camera (the image as recorded) or tumvi (the TUM-VI pinhole rectification, needs tumvi_seq_dir)
  max_age_s    a request that waited longer is dropped (5.0)
  warmup_width, warmup_height   stream size for the warm-up at start, 0: the size of the first request.
               Required by trt-fp16 and trt-fp32: the node checks the engine size against it and refuses to start
               on a mismatch

Ctrl-C (or a shutdown from the launch file) ends the node without a traceback, also while the network loads.
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

try:
    from rclpy.executors import ExternalShutdownException
except ImportError:   # older rclpy: spin ends with KeyboardInterrupt only
    class ExternalShutdownException(Exception):
        pass


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
        from ucslam.live_depth import BACKENDS, DEFAULT_CONFIG, DepthWorker, LiveDepth, make_live_view

        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.domain = str(p("domain", "euroc"))
        self.backend = str(p("backend", "torch-fp32"))
        if self.backend not in BACKENDS:
            raise RuntimeError(f"parameter backend must be one of {BACKENDS}, got {self.backend!r}")
        engine = os.path.expanduser(str(p("engine", "")))
        if engine and not os.path.isabs(engine):   # relative paths are taken from the repository
            engine = os.path.join(repo, engine)
        if self.backend.startswith("trt-") and not os.path.isfile(engine):
            raise RuntimeError(f"backend {self.backend} needs parameter engine (a TensorRT engine file), "
                               f"got {engine!r}")
        config = str(p("config", DEFAULT_CONFIG))
        seq_dir = str(p("tumvi_seq_dir", ""))
        if seq_dir and not os.path.isabs(seq_dir):   # relative paths are taken from the repository
            seq_dir = os.path.join(repo, seq_dir)
        view = make_live_view(str(p("view", "camera")), seq_dir, config)
        model, device = str(p("model", "")), str(p("device", "cuda"))
        max_age_s = float(p("max_age_s", 5.0))
        w, h = int(p("warmup_width", 0)), int(p("warmup_height", 0))
        if self.backend.startswith("trt-") and (w <= 0 or h <= 0):
            # An engine takes one image size. Without the stream size a wrong stream fails every keyframe.
            raise RuntimeError(f"backend {self.backend} needs warmup_width and warmup_height set to the camera stream "
                               f"size (e.g. 848 and 480), so the engine size is checked at start")
        self.get_logger().info(f"loading the network, backend {self.backend}" + (f", engine {engine}" if engine else ""))
        self.worker = None
        self.live = LiveDepth(model or None, device, config, view=view, backend=self.backend, engine=engine or None,
                              log=self.get_logger().warn)
        if w > 0 and h > 0:
            self.live.check_stream_size(h, w)   # a fixed-size engine: refuse a wrong stream size now
        qos = QoSProfile(depth=16, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(DepthPrior, "depth_prior", qos)
        self.worker = DepthWorker(self.live, self.on_result, max_age_s=max_age_s, log=self.get_logger().info,
                                  warmup_shape=(h, w) if w > 0 and h > 0 else None, warn=self.get_logger().warn)
        self.sub = self.create_subscription(DepthRequest, "depth_request", self.on_request, qos)
        self.timer = self.create_timer(10.0, lambda: self.get_logger().info(str(self.worker.stats())))
        self.get_logger().info(f"serving model {self.live.model} on {self.live.net.device}, backend {self.backend}, "
                               f"domain {self.domain}")

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
        try:
            self.pub.publish(m)
        except Exception:
            if rclpy.ok():
                raise
            # the context shut down (Ctrl-C) between the check above and the publish: drop the result

    def close(self) -> None:
        """Stop the worker thread, then the node. Safe after a failed start and after a shutdown."""
        if self.worker is not None:
            self.worker.stop()
        self.destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DepthNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass   # Ctrl-C, also while the network loads, or a shutdown from outside: end without a traceback
    finally:
        try:
            if node is not None:
                node.close()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:   # a second Ctrl-C during the cleanup
            pass


if __name__ == "__main__":
    main()
