#!/usr/bin/env python3
"""UC-SLAM depth node for ROS 1 (Noetic).

Answers each DepthRequest of slam_node (one per keyframe) with the metric depth map of the depth cache code
(ucslam/live_depth.py), as a DepthPrior. It needs torch, so it runs in the repository's Python environment through
scripts/venv_python.sh (see ros/README.md), not in the system Python of Noetic.

Parameters (private):
  domain       must equal UCSLAM.Dataset of the SLAM settings (euroc, tumvi, openloris, ...)
  backend      label of the network backend the per-domain constants were fitted with (torch-fp32)
  model        network name, empty: the model of configs/default.yaml
  device       cuda or cpu
  view         camera (the image as recorded) or tumvi (the TUM-VI pinhole rectification, needs tumvi_seq_dir)
  max_age_s    a request that waited longer is dropped (5.0)
  warmup_width, warmup_height   stream size for the warm-up at start, 0: the size of the first request
"""
import os
import sys

import numpy as np
import rospy
from sensor_msgs.msg import Image

from ucslam_ros1.msg import DepthPrior, DepthRequest

REPO = os.environ.get("UCSLAM_REPO") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from ucslam.live_depth import DEFAULT_CONFIG, DepthWorker, LiveDepth, make_live_view  # noqa: E402


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
    m.data = z.tobytes()
    return m


class DepthNode:
    def __init__(self):
        p = lambda name, default: rospy.get_param("~" + name, default)  # noqa: E731
        self.domain = str(p("domain", "euroc"))
        self.backend = str(p("backend", "torch-fp32"))
        config = str(p("config", DEFAULT_CONFIG))
        seq_dir = str(p("tumvi_seq_dir", ""))
        if seq_dir and not os.path.isabs(seq_dir):   # relative paths are taken from the repository
            seq_dir = os.path.join(REPO, seq_dir)
        view = make_live_view(str(p("view", "camera")), seq_dir, config)
        rospy.loginfo("depth_node: loading the network")
        self.live = LiveDepth(str(p("model", "")) or None, str(p("device", "cuda")), config, view=view)
        w, h = int(p("warmup_width", 0)), int(p("warmup_height", 0))
        self.pub = rospy.Publisher("depth_prior", DepthPrior, queue_size=16)
        self.worker = DepthWorker(self.live, self.on_result, max_age_s=float(p("max_age_s", 5.0)),
                                  log=rospy.loginfo, warmup_shape=(h, w) if w > 0 and h > 0 else None)
        self.sub = rospy.Subscriber("depth_request", DepthRequest, self.on_request, queue_size=16,
                                    buff_size=1 << 24, tcp_nodelay=True)
        self.timer = rospy.Timer(rospy.Duration(10.0), self.on_timer)
        rospy.loginfo(f"depth_node: serving model {self.live.model} on {self.live.net.device}, domain {self.domain}")

    def on_request(self, msg: DepthRequest) -> None:
        try:
            gray = image_to_gray(msg.image)
        except ValueError as e:
            rospy.logwarn_throttle(10.0, f"depth_node: {e}")
            return
        self.worker.submit(msg.header, gray)

    def on_result(self, header, zmap, desc, net_ms, queue_ms) -> None:
        if rospy.is_shutdown():
            return
        m = DepthPrior()
        m.header = header
        m.model = desc["model"]
        m.model_source = getattr(self.live.net, "source", "")
        m.backend = self.backend
        m.domain = self.domain
        m.view = int(desc["kind"])
        m.full_width = int(desc["full_w"])
        m.full_height = int(desc["full_h"])
        m.k_view = [float(v) for v in desc["k_view"]]
        m.depth = depth_to_image(zmap, header)
        m.net_ms = float(net_ms)
        m.queue_ms = float(queue_ms)
        self.pub.publish(m)

    def on_timer(self, _event) -> None:
        rospy.loginfo(f"depth_node: {self.worker.stats()}")


def main() -> None:
    rospy.init_node("depth_node")
    node = DepthNode()
    rospy.spin()
    node.worker.stop()


if __name__ == "__main__":
    main()
