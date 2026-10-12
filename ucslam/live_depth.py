"""Live metric depth for the ROS 1 and ROS 2 depth nodes.

The node must give the SLAM system the same map the offline depth cache would hold for the same image, because the
per-domain constants of the depth factor were fitted on that map. So this module calls the code of the cache
(experiments/precompute_depth.py) and of the network wrapper (ucslam/depthnet.py), unchanged:

    1. MetricDepth.predict on the gray image (copied to three channels inside),
    2. for a pinhole view (TUM-VI fisheye) the same rectification, otherwise the image as recorded,
    3. downscale_map: block average by ``downscale`` with INTER_AREA, 0 at or below z_min, clip at z_max, float16.

The result is the float16 map, the network input size and the view. The C++ side (PriorNet::PushDepthMap) applies
the same float16-to-float32 conversion as its cache reader.
"""
from __future__ import annotations

import collections
import os
import sys
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from experiments.precompute_depth import (DEFAULT_CONFIG, downscale_map, load_config,  # noqa: E402
                                          make_view, net_input, precompute_params)
from ucslam.depthnet import MetricDepth  # noqa: E402

VIEW_CAMERA = 1
VIEW_PINHOLE = 2


class LiveDepth:
    """One network on one device. ``__call__`` maps a gray image to a float16 depth map."""

    def __init__(self, model: Optional[str] = None, device: str = "cuda", config: str = DEFAULT_CONFIG,
                 view: Optional[Dict] = None):
        cfg = load_config(config)
        self.params = precompute_params(cfg)
        self.model = model or self.params["model"]
        self.net = MetricDepth(self.model, device)
        # The view of make_view: {"view": "camera"} for EuRoC-like and RealSense cameras, or the TUM-VI pinhole
        # view with its remap maps. None means "camera" at the size of each image.
        self.view = view

    def warmup(self, shape: Tuple[int, int], n: int = 3) -> None:
        """Run the network n times at the stream size before serving (first calls are slow)."""
        img = np.zeros(shape, np.uint8)
        for _ in range(n):
            self(img)

    def __call__(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict, float]:
        """gray: HxW uint8. Returns the float16 map, the view description and the network time in ms."""
        if gray.ndim != 2:
            raise ValueError(f"expected a gray image, got shape {gray.shape}")
        view = self.view or {"view": "camera", "maps": None}
        x = net_input(gray, view)
        t0 = time.perf_counter()
        z = self.net.predict(x)
        net_ms = 1000.0 * (time.perf_counter() - t0)
        zmap = downscale_map(z, self.params["downscale"], float(self.params["z_min_m"]), float(self.params["z_max_m"]))
        h, w = x.shape[:2]
        desc = {"kind": VIEW_PINHOLE if view.get("view") == "pinhole" else VIEW_CAMERA,
                "full_w": int(w), "full_h": int(h),
                "k_view": [float(v) for v in view.get("K_view", [0.0, 0.0, 0.0, 0.0])],
                "model": self.model}
        return zmap, desc, net_ms


def make_live_view(kind: str = "camera", seq_dir: str = "", config: str = DEFAULT_CONFIG) -> Optional[Dict]:
    """The view of the depth cache for a live stream.

    ``camera``: the image as recorded (EuRoC, OpenLORIS, RealSense). Returns None, the default of LiveDepth.
    ``tumvi``: the pinhole rectification of the TUM-VI cache, built by make_view from cam0 of the sequence
    folder ``seq_dir`` (for a replay of a TUM-VI sequence).
    """
    if kind == "camera":
        return None
    if kind == "tumvi":
        from ucslam import calib
        return make_view("tumvi", calib.load_tumvi_camera(seq_dir, "cam0"), load_config(config))
    raise ValueError(f"unknown view {kind!r}, expected camera or tumvi")


class DepthWorker:
    """The request queue of a depth node, without ROS.

    One worker thread runs the network on the requests in arrival order (FIFO). When ``max_queue`` requests wait,
    the oldest is dropped. A request that waited more than ``max_age_s`` is dropped too, because the local BA has
    moved past its keyframe by then. The network warms up at ``warmup_shape`` when the thread starts, or else at
    the size of the first request.
    ``on_result(key, zmap, desc, net_ms, queue_ms)`` runs in the worker thread.
    """

    def __init__(self, live: "LiveDepth", on_result: Callable, max_age_s: float = 5.0, max_queue: int = 16,
                 log: Callable[[str], None] = print, warmup_shape: Optional[Tuple[int, int]] = None):
        self.live = live
        self.on_result = on_result
        self.max_age_s = float(max_age_s)
        self.max_queue = max(1, int(max_queue))
        self.log = log
        self.served = self.dropped_old = self.dropped_full = self.failed = 0
        self.net_ms: collections.deque = collections.deque(maxlen=1000)
        self._q: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._stop = False
        self._warm = False
        self._warmup_shape = warmup_shape   # (h, w) of the stream if known, else the first request's size
        self._thread = threading.Thread(target=self._run, name="depth_worker", daemon=True)
        self._thread.start()

    def submit(self, key, gray: np.ndarray) -> None:
        with self._cv:
            if len(self._q) >= self.max_queue:
                self._q.popleft()
                self.dropped_full += 1
            self._q.append((key, gray, time.monotonic()))
            self._cv.notify()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout)

    def stats(self) -> Dict:
        ms = sorted(self.net_ms)
        p = (lambda q: float(ms[min(len(ms) - 1, int(q * (len(ms) - 1) + 0.5))]) if ms else float("nan"))
        with self._cv:
            waiting = len(self._q)
        return {"served": self.served, "dropped_old": self.dropped_old, "dropped_full": self.dropped_full,
                "failed": self.failed, "waiting": waiting, "net_ms_p50": p(0.5), "net_ms_p99": p(0.99)}

    def _warmup(self, shape) -> None:
        self.log(f"depth worker: warming up the network at {shape[1]}x{shape[0]}")
        try:
            self.live.warmup(shape)
        except Exception as e:  # serve anyway: a real failure shows up as failed requests
            self.log(f"depth worker: warm-up failed: {e!r}")
        self._warm = True

    def _run(self) -> None:
        if self._warmup_shape is not None:
            self._warmup(self._warmup_shape)
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait(0.1)
                if self._stop:
                    return
                key, gray, t_in = self._q.popleft()
            if 1000.0 * (time.monotonic() - t_in) > 1000.0 * self.max_age_s:
                self.dropped_old += 1
                continue
            if not self._warm:
                self._warmup(gray.shape)
            queue_ms = 1000.0 * (time.monotonic() - t_in)
            try:
                zmap, desc, net_ms = self.live(gray)
            except Exception as e:  # a bad image must not stop the node
                self.failed += 1
                self.log(f"depth worker: inference failed: {e!r}")
                continue
            self.served += 1
            self.net_ms.append(net_ms)
            self.on_result(key, zmap, desc, net_ms, queue_ms)
