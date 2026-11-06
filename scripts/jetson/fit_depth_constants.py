#!/usr/bin/env python3
"""Fit the four depth-prior constants of a camera domain (default d455color) from color + aligned sensor depth.

UC-SLAM reads four constants per camera domain: UCSLAM.LogBias, SigmaNetC0, SigmaNetC1 and NeffOverN. This script
fits them with the estimators of the paper (keyframe_table, neff_ratio, sequence_stats, fit_affine, coverage,
split_summary, _centred_pool), copied unchanged into depth_fit_lib.py from the paper's experiment code, and follows
the OpenLORIS fit of the paper (a D435i with aligned depth):

    frames        depth.keyframes_per_seq (40) frames per recording, spread evenly over the color frames that have
                  an aligned depth within --tol-ms (5 ms)
    reference     the aligned depth at the nearest pixel of each keypoint, kept between --z-min and --z-max
                  (0.3 to 5 m), 0 = no depth, reference log noise sigma_ref = --ref-noise * z (0.01 z)
    residual      r = log z_hat - log z_ref, per-keyframe median removed for (c0, c1)
    log_bias      median of r over the fit points
    c0, c1        deflated binned fit of sigma_net(log z_hat), z_ref 1 m
    neff_over_n   median over the fit recordings of N_eff / N per keyframe

Differences to the paper fit, on purpose:

    keypoints     ORB keypoints like ORB-SLAM3 (--keypoints orb, the default): nFeatures, scale factor, levels and
                  FAST thresholds of the settings, FAST per 35 px cell with the low threshold as fallback, and the
                  strongest keypoints spread over a grid on each level. PriorNet samples the depth at these keypoints
                  on the robot. In flight it keeps at most UCSLAM.MaxDepthPriorPoints (100) of them per keyframe,
                  while the fit uses every keypoint with a reference depth (several hundred per keyframe). The paper
                  fit does the same (OpenLORIS, 716 per keyframe). The paper's OpenLORIS fit used XFeat (top 1024),
                  available as --keypoints xfeat. N_eff / N depends on the number of points per keyframe, so the
                  two give different ratios.
    backend       the YAML records --backend and --engine. make_d455_settings.py writes them as
                  UCSLAM.DepthBackend and D455.FitEngine, and slam_node and ucslam_d455.launch.py refuse another
                  backend. Fit with the backend you fly with.
    network       the backend used in flight (--backend torch-fp32, torch-fp16, trt-fp16 or trt-fp32 with --engine),
                  through ucslam.live_depth.LiveDepth, on the gray image the depth node receives (cv_bridge mono8 of
                  the color stream). --sample full (default, as the paper) samples the full-resolution prediction.
                  --sample map samples the half-resolution float16 map that the depth node publishes.
    input         ROS 2 bags (--bags, the color topic and /camera/camera/aligned_depth_to_color/image_raw in 16UC1
                  millimetres) or OpenLORIS-style folders (--folders, color/ and aligned_depth/ with color.txt and
                  aligned_depth.txt). The fit/held-out split is by recording (--fit).

Outputs: a JSON with every number (--out-json) and a small YAML (--out-yaml, default
configs/d455/<domain>_depth_constants.yaml) that scripts/jetson/make_d455_settings.py --constants reads.

Usage (on the Jetson, with the venv of the depth node and ROS 2 sourced for bags):

    python3 scripts/jetson/fit_depth_constants.py --bags ~/ucslam_bags/calib_room1 ~/ucslam_bags/calib_room2 \\
        ~/ucslam_bags/calib_room3 --fit calib_room1 calib_room2 --backend trt-fp16 --engine ~/engines/da2b.plan

Record each bag with ucslam_d455.launch.py mode:=calib record:=true, in several rooms, moving slowly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from depth_fit_lib import (_centred_pool, _jsonable, coverage, fit_affine,  # noqa: E402,F401
                           keyframe_table, load_config, nearest_frame, neff_ratio, read_list, sequence_stats,
                           split_summary)

DOMAIN = "d455color"
BACKENDS = ("torch-fp32", "torch-fp16", "trt-fp16", "trt-fp32")
COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
OUT_DIR = os.path.join(REPO, "configs", "d455")
# ORB-SLAM3 ORBextractor constants (cpp/ucslam/src/ORBextractor.cc): border and FAST cell size.
EDGE_THRESHOLD = 19
CELL = 35


# ---------------------------------------------------------------- keypoints --

class OrbSlamKeypoints:
    """Keypoints placed like ORB-SLAM3's ORBextractor, in level-0 pixels.

    Per level the feature count follows the geometric series of ORBextractor. FAST runs with the low threshold
    once, and in every 35 px cell that has a corner of the high threshold only those corners are kept (the
    extractor's per-cell fallback). The kept corners are spread over a grid with about as many cells as wanted
    features, the strongest of each cell first. The extractor's quadtree does the same job. This is an
    approximation of the extractor, not a copy: the descriptors and the exact quadtree split are not needed for a
    depth fit."""

    def __init__(self, n_features: int = 1250, scale_factor: float = 1.2, n_levels: int = 8,
                 ini_th_fast: int = 20, min_th_fast: int = 7):
        import cv2
        self.cv2 = cv2
        f = 1.0 / scale_factor
        n_per = n_features * (1 - f) / (1 - f ** n_levels)
        counts, total = [], 0
        for _ in range(n_levels - 1):
            c = int(round(n_per))
            counts.append(c)
            total += c
            n_per *= f
        counts.append(max(n_features - total, 0))
        self.counts = counts
        self.scales = [scale_factor ** k for k in range(n_levels)]
        self.ini, self.min = int(ini_th_fast), int(min_th_fast)
        self.fast = cv2.FastFeatureDetector_create(threshold=self.min, nonmaxSuppression=True)
        self.params = {"n_features": n_features, "scale_factor": scale_factor, "n_levels": n_levels,
                       "ini_th_fast": ini_th_fast, "min_th_fast": min_th_fast}

    @staticmethod
    def spread(xy: np.ndarray, resp: np.ndarray, n: int, w: int, h: int) -> np.ndarray:
        """Indices of n points, the best of each grid cell first, then the second best, and so on."""
        if xy.shape[0] <= n:
            return np.arange(xy.shape[0])
        side = max(1.0, math.sqrt(w * h / float(n)))
        ncol = int(math.ceil(w / side))
        cell = (xy[:, 1] // side).astype(np.int64) * ncol + (xy[:, 0] // side).astype(np.int64)
        order = np.lexsort((-resp, cell))
        sc = cell[order]
        first = np.r_[True, sc[1:] != sc[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(order.size), 0))
        rank = np.arange(order.size) - start
        return order[np.lexsort((-resp[order], rank))][:n]

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        h0, w0 = gray.shape
        out = []
        img = gray
        for level, (n, s) in enumerate(zip(self.counts, self.scales)):
            if level > 0:   # each level from the one above, as ORBextractor::ComputePyramid
                img = cv2.resize(img, (int(round(w0 / s)), int(round(h0 / s))), interpolation=cv2.INTER_LINEAR)
            if n <= 0:
                continue
            h, w = img.shape
            b = EDGE_THRESHOLD - 3
            kps = self.fast.detect(img, None)
            if not kps:
                continue
            xy = np.array([k.pt for k in kps], np.float64)
            resp = np.array([k.response for k in kps], np.float64)
            inside = (xy[:, 0] >= b) & (xy[:, 0] < w - b) & (xy[:, 1] >= b) & (xy[:, 1] < h - b)
            xy, resp = xy[inside], resp[inside]
            cell = ((xy[:, 1] - b) // CELL).astype(np.int64) * 100000 + ((xy[:, 0] - b) // CELL).astype(np.int64)
            strong = resp >= self.ini
            has_strong = np.isin(cell, np.unique(cell[strong]))
            keep = strong | ~has_strong
            xy, resp = xy[keep], resp[keep]
            sel = self.spread(xy, resp, n, w, h)
            out.append(xy[sel] * s)
        return np.concatenate(out) if out else np.zeros((0, 2))


class XFeatKeypoints:
    """The paper's OpenLORIS keypoints: XFeat top-k on the gray frame (experiments/openloris_depth_calib.py)."""

    def __init__(self, topk: int, device: str):
        from ucslam import backbone
        self.bb = backbone.XFeatBackbone(device=device)
        self.topk, self.device = int(topk), device
        self.params = {"topk": self.topk}

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(np.ascontiguousarray(gray)).float().div_(255.0)[None, None].to(self.device)
        with torch.no_grad():
            kps, _ = self.bb.detect(x, topk=self.topk)
        return kps.detach().cpu().numpy().astype(np.float64)


def orb_params_from_settings(path: str) -> dict:
    """ORBextractor.* of a UC-SLAM settings file, read with OpenCV as ORB-SLAM3 does."""
    import cv2
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    g = lambda k: fs.getNode(k).real()  # noqa: E731
    out = {"n_features": int(g("ORBextractor.nFeatures")), "scale_factor": float(g("ORBextractor.scaleFactor")),
           "n_levels": int(g("ORBextractor.nLevels")), "ini_th_fast": int(g("ORBextractor.iniThFAST")),
           "min_th_fast": int(g("ORBextractor.minThFAST"))}
    fs.release()
    return out


# ------------------------------------------------------------------- images --

def image_array(encoding: str, height: int, width: int, step: int, data, is_bigendian: bool = False) -> np.ndarray:
    """A sensor_msgs/Image buffer as a numpy array (H, W) or (H, W, C), respecting the row step."""
    enc = encoding.lower()
    if enc in ("rgb8", "bgr8"):
        dtype, ch = np.uint8, 3
    elif enc in ("rgba8", "bgra8"):
        dtype, ch = np.uint8, 4
    elif enc in ("mono8", "8uc1"):
        dtype, ch = np.uint8, 1
    elif enc in ("mono16", "16uc1"):
        dtype, ch = np.uint16, 1
    elif enc in ("yuv422", "yuv422_yuy2", "uyvy", "yuyv"):
        dtype, ch = np.uint8, 2
    elif enc == "32fc1":
        dtype, ch = np.float32, 1
    else:
        raise ValueError(f"image encoding {encoding} is not handled")
    item = np.dtype(dtype).itemsize
    buf = np.frombuffer(bytes(data), np.uint8)
    if step < width * ch * item or buf.size < step * height:
        raise ValueError(f"image buffer too small: step {step}, {buf.size} bytes for {width}x{height} {encoding}")
    rows = buf[: step * height].reshape(height, step)[:, : width * ch * item]
    dt_ = np.dtype(dtype).newbyteorder(">" if is_bigendian else "<")
    arr = np.ascontiguousarray(rows).view(dt_).reshape(height, width, ch) if ch > 1 else \
        np.ascontiguousarray(rows).view(dt_).reshape(height, width)
    return arr.astype(dtype)


def to_gray(arr: np.ndarray, encoding: str) -> np.ndarray:
    """The mono8 image slam_node makes (cv_bridge toCvCopy "mono8", cv::cvtColor), sent to the depth node."""
    import cv2
    enc = encoding.lower()
    if enc in ("mono8", "8uc1"):
        return arr
    if enc in ("mono16", "16uc1"):
        return (arr >> 8).astype(np.uint8)   # slam_node keeps the high byte (HighByte)
    code = {"rgb8": cv2.COLOR_RGB2GRAY, "bgr8": cv2.COLOR_BGR2GRAY, "rgba8": cv2.COLOR_RGBA2GRAY,
            "bgra8": cv2.COLOR_BGRA2GRAY, "yuv422": cv2.COLOR_YUV2GRAY_UYVY, "uyvy": cv2.COLOR_YUV2GRAY_UYVY,
            "yuv422_yuy2": cv2.COLOR_YUV2GRAY_YUY2, "yuyv": cv2.COLOR_YUV2GRAY_YUY2}.get(enc)
    if code is None:
        raise ValueError(f"no gray conversion for {encoding}")
    return cv2.cvtColor(arr, code)


def depth_mm(arr: np.ndarray, encoding: str) -> np.ndarray:
    """Aligned depth in uint16 millimetres (16UC1 as is, 32FC1 metres converted)."""
    if encoding.lower() in ("16uc1", "mono16"):
        return arr
    if encoding.lower() == "32fc1":
        z = np.where(np.isfinite(arr) & (arr > 0), arr * 1000.0, 0.0)
        return np.clip(np.rint(z), 0, 65535).astype(np.uint16)
    raise ValueError(f"aligned depth encoding {encoding} is not 16UC1 or 32FC1")


# --------------------------------------------------------------- recordings --

class FolderRecording:
    """OpenLORIS-style folder: color.txt and aligned_depth.txt ("t rel" lines) with the PNGs they list."""

    def __init__(self, path: str, color_list: str = "color.txt", depth_list: str = "aligned_depth.txt"):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path.rstrip("/"))
        self.t_color, self.p_color = read_list(os.path.join(self.path, color_list))
        self.t_depth, self.p_depth = read_list(os.path.join(self.path, depth_list))

    def frames(self, pairs: Sequence[Tuple[int, int]]) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        import cv2
        for i, j in pairs:
            im = cv2.imread(self.p_color[i], cv2.IMREAD_UNCHANGED)
            dep = cv2.imread(self.p_depth[j], cv2.IMREAD_UNCHANGED)
            if im is None or dep is None:
                raise FileNotFoundError(f"{self.p_color[i]} or {self.p_depth[j]}")
            if im.ndim == 3:   # cv2 reads BGR(A). The converter of the paper used cv2 BGR2GRAY, as here.
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY if im.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
            if dep.dtype != np.uint16:
                raise ValueError(f"{self.p_depth[j]} is not 16-bit millimetres")
            yield int(self.t_color[i]), im, dep


def cdr_header_stamp(data) -> Optional[int]:
    """Header stamp in ns of a serialized message that starts with std_msgs/Header (sensor_msgs/Image), or None.

    CDR: a 4-byte encapsulation (0x00 0x00 big endian, 0x00 0x01 little endian), then builtin_interfaces/Time as
    int32 sec and uint32 nanosec. Reading 12 bytes saves deserializing a whole image just for its stamp."""
    b = bytes(data[:12])
    if len(b) < 12 or b[0] != 0 or b[1] not in (0, 1):
        return None
    sec = int.from_bytes(b[4:8], "little" if b[1] == 1 else "big", signed=True)
    nsec = int.from_bytes(b[8:12], "little" if b[1] == 1 else "big", signed=False)
    if nsec >= 1_000_000_000:
        return None
    return sec * 1_000_000_000 + nsec


class BagRecording:
    """ROS 2 bag with the color topic and the aligned depth topic. Pairs use the header stamps.

    The first pass reads only the stamps (12 bytes of each message), the second deserializes the chosen frames."""

    def __init__(self, uri: str, color_topic: str = COLOR_TOPIC, depth_topic: str = DEPTH_TOPIC):
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from make_d455_settings import open_bag
        self.uri = os.path.abspath(uri)
        self.name = os.path.basename(self.uri.rstrip("/"))
        if self.name.endswith((".mcap", ".db3")):
            self.name = os.path.splitext(self.name)[0]
        self.topics = (color_topic, depth_topic)
        self._open, self._deser = open_bag, deserialize_message
        reader, types = open_bag(self.uri, list(self.topics))
        for t in self.topics:
            if t not in types:
                raise SystemExit(f"{uri} has no topic {t} (record with ucslam_d455.launch.py mode:=calib record:=true)")
        self._cls = {t: get_message(types[t]) for t in self.topics}
        stamps: Dict[str, List[Tuple[int, int]]] = {t: [] for t in self.topics}
        count = {t: 0 for t in self.topics}
        self.stamps_parsed = 0
        while reader.has_next():
            topic, data, _t = reader.read_next()
            t_ns = cdr_header_stamp(data)
            if t_ns is None:   # not plain CDR: deserialize as before
                msg = deserialize_message(data, self._cls[topic])
                t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            else:
                self.stamps_parsed += 1
            stamps[topic].append((t_ns, count[topic]))
            count[topic] += 1
        del reader
        c = sorted(stamps[color_topic])
        d = sorted(stamps[depth_topic])
        self.t_color = np.array([s for s, _ in c], np.int64)
        self.i_color = [k for _, k in c]
        self.t_depth = np.array([s for s, _ in d], np.int64)
        self.i_depth = [k for _, k in d]

    def frames(self, pairs: Sequence[Tuple[int, int]]) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        want = {self.topics[0]: {self.i_color[i] for i, _ in pairs},
                self.topics[1]: {self.i_depth[j] for _, j in pairs}}
        got: Dict[Tuple[str, int], np.ndarray] = {}
        reader, _types = self._open(self.uri, list(self.topics))
        count = {t: 0 for t in self.topics}
        while reader.has_next():
            topic, data, _t = reader.read_next()
            k = count[topic]
            count[topic] += 1
            if k not in want[topic]:
                continue
            m = self._deser(data, self._cls[topic])
            arr = image_array(m.encoding, m.height, m.width, m.step, m.data, bool(m.is_bigendian))
            got[(topic, k)] = to_gray(arr, m.encoding) if topic == self.topics[0] else depth_mm(arr, m.encoding)
        del reader
        for i, j in pairs:
            yield int(self.t_color[i]), got[(self.topics[0], self.i_color[i])], got[(self.topics[1], self.i_depth[j])]


def select_frames(t_color: np.ndarray, t_depth: np.ndarray, tol_ns: int, n_kf: int) -> List[Tuple[int, int]]:
    """(color index, depth index) of n_kf frames spread evenly over the color frames with a depth within tol_ns."""
    j = nearest_frame(t_depth, t_color, int(tol_ns))
    usable = np.nonzero(j >= 0)[0]
    if usable.size == 0:
        return []
    k = min(int(n_kf), usable.size)
    pick = np.unique(np.linspace(0, usable.size - 1, k).round().astype(int))
    return [(int(usable[p]), int(j[usable[p]])) for p in pick]


def reference_points(dep_mm: np.ndarray, uv: np.ndarray, z_min: float, z_max: float):
    """Aligned depth at the nearest pixel of each keypoint (openloris_depth_calib.reference_points with a z range)."""
    h, w = dep_mm.shape
    ui = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    z = dep_mm[vi, ui].astype(np.float64) / 1000.0
    has = z > 0
    keep = has & (z >= z_min) & (z <= z_max)
    return keep, z, {"no_depth": int((~has).sum()), "out_of_range": int((has & ~keep).sum())}


# ------------------------------------------------------------------ network --

def make_live(backend: str, engine: Optional[str], model: Optional[str], device: str):
    """ucslam.live_depth.LiveDepth with the flight backend. torch-fp32 is today's MetricDepth path."""
    from ucslam.live_depth import LiveDepth
    if backend == "torch-fp32" and not engine:
        return LiveDepth(model, device)
    if backend.startswith("trt") and not engine:
        raise SystemExit(f"backend {backend} needs --engine (a TensorRT engine built on this device)")
    return LiveDepth(model, device, backend=backend, engine=engine)


class Predictor:
    """Network depth at keypoints: the full-resolution prediction, or the published half-resolution map."""

    def __init__(self, live, sample: str = "full"):
        self.live, self.sample = live, sample
        self.net = getattr(live, "net", None)
        if sample == "full" and self.net is None:
            raise SystemExit("LiveDepth has no .net predictor, use --sample map")
        self.downscale = int(live.params["downscale"]) if hasattr(live, "params") else 2
        self.latency_s: List[float] = []

    def __call__(self, gray: np.ndarray, uv: np.ndarray) -> np.ndarray:
        from ucslam.depthnet import sample_depth
        t0 = time.perf_counter()
        if self.sample == "full":
            depth = self.net.predict(gray)
            lat = getattr(self.net, "last_latency_s", float("nan"))
            self.latency_s.append(float(lat) if math.isfinite(float(lat)) else time.perf_counter() - t0)
            return sample_depth(depth, uv)
        zmap, _desc, net_ms = self.live(gray)
        self.latency_s.append(net_ms / 1000.0)
        z = np.asarray(zmap, np.float32).astype(np.float64)
        z[z <= 0] = np.nan
        return sample_depth(z, (np.asarray(uv, np.float64) + 0.5) / self.downscale - 0.5)


# ---------------------------------------------------------------------- fit --

def process(rec, pairs, keypoints, predict, model: str, z_min: float, z_max: float, ref_noise: float) -> dict:
    """Per-point arrays of one recording in the layout ``sequence_stats`` reads."""
    cols = {c: [] for c in ("kf", "t_ns", "u", "v", "z_ref", "sigma_ref", f"zhat_{model}")}
    counts = {"frames": len(pairs), "keypoints": 0, "no_depth": 0, "out_of_range": 0, "net_nan": 0, "points": 0}
    t_kf = []
    t_start = time.time()
    for kf, (t_ns, gray, dep) in enumerate(rec.frames(pairs)):
        t_kf.append(t_ns)
        if dep.shape != gray.shape:
            raise ValueError(f"{rec.name}: aligned depth {dep.shape} and image {gray.shape} differ")
        uv = keypoints(gray)
        counts["keypoints"] += uv.shape[0]
        keep, z, c = reference_points(dep, uv, z_min, z_max)
        counts["no_depth"] += c["no_depth"]
        counts["out_of_range"] += c["out_of_range"]
        uv, z = uv[keep], z[keep]
        if uv.shape[0] == 0:
            continue
        zh = predict(gray, uv)
        counts["net_nan"] += int((~np.isfinite(zh)).sum())
        n = uv.shape[0]
        cols["kf"].append(np.full(n, kf, np.int64))
        cols["t_ns"].append(np.full(n, t_ns, np.int64))
        cols["u"].append(uv[:, 0])
        cols["v"].append(uv[:, 1])
        cols["z_ref"].append(z)
        cols["sigma_ref"].append(ref_noise * z)
        cols[f"zhat_{model}"].append(zh)
        counts["points"] += n
    out = {c: (np.concatenate(v) if v else np.zeros(0)) for c, v in cols.items()}
    out["kf_t_ns"] = np.asarray(t_kf, np.int64)
    tc = np.asarray(rec.t_color, np.int64)
    out["duration_s"] = float((tc[-1] - tc[0]) / 1.0e9) if tc.size else 0.0
    out["counts"] = counts
    out["wall_s"] = time.time() - t_start
    return out


def fit_constants(per_rec: Dict[str, dict], fit_keys: List[str], ho_keys: List[str], model: str, dcfg: dict) -> dict:
    """The constants and their checks, with the statistics of experiments/depth_prior_bench.py, as the paper fit."""
    stats = {s: sequence_stats(d, model, dcfg) for s, d in per_rec.items()}
    fit, ho = split_summary(stats, fit_keys), split_summary(stats, ho_keys)
    if not fit.get("n_points"):
        raise SystemExit("no reference point in the fit recordings")
    z_ref, n_bins = float(dcfg["z_ref_m"]), int(dcfg["fit_bins"])
    x, dd, sr = _centred_pool(stats, fit_keys)
    aff = fit_affine(x, dd, sr, n_bins, z_ref)
    if not aff.get("ok"):
        raise SystemExit(f"affine fit failed: {aff}")
    c0, c1 = aff["sigma_net"]["c0"], aff["sigma_net"]["c1"]
    xh, ddh, srh = _centred_pool(stats, ho_keys)
    cov = {"fit": coverage(c0, c1, x, dd, sr, z_ref), "heldout": coverage(c0, c1, xh, ddh, srh, z_ref)}
    return {"constants": {"log_bias": fit["bias"], "sigma_net_c0": c0, "sigma_net_c1": c1,
                          "neff_over_n": fit["neff_ratio_median"]},
            "fit_split": list(fit_keys), "heldout_split": list(ho_keys), "fit": fit, "heldout": ho,
            "affine_fit": aff, "coverage": cov,
            "per_sequence": {s: {k: v for k, v in st.items() if not k.startswith("_")} for s, st in stats.items()}}


def _git() -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, "describe", "--always", "--dirty"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def write_yaml(path: str, res: dict, a: argparse.Namespace) -> None:
    c = res["constants"]
    m = res["meta"]
    doc = {"domain": a.domain, "backend": a.backend, "engine": os.path.abspath(a.engine) if a.engine else "",
           "model": m["model"], "keypoints": a.keypoints, "sample": a.sample,
           "sigma_net_c0": float(c["sigma_net_c0"]), "sigma_net_c1": float(c["sigma_net_c1"]),
           "log_bias": float(c["log_bias"]), "neff_over_n": float(c["neff_over_n"]),
           "fit_split": res["fit_split"], "heldout_split": res["heldout_split"], "partial": False,
           "n_points_fit": int(res["fit"]["n_points"]), "n_per_kf_mean": float(res["fit"]["n_per_kf_mean"]),
           "coverage_heldout_1sigma": float(res["coverage"]["heldout"].get("frac_1sigma", float("nan"))),
           "date": m["date"], "json": os.path.abspath(a.out_json)}
    head = (f"# Depth-prior constants of domain {a.domain}. Written by scripts/jetson/fit_depth_constants.py.\n"
            "# Read by scripts/jetson/make_d455_settings.py --constants. Valid only for the backend named here.\n")
    with open(path, "w") as f:
        f.write(head + yaml.safe_dump(doc, sort_keys=False))


def summary(res: dict) -> str:
    c = res["constants"]
    L = [f"constants: log_bias {c['log_bias']:.4f}, sigma_net_c0 {c['sigma_net_c0']:.4f}, "
         f"sigma_net_c1 {c['sigma_net_c1']:.5f}, neff_over_n {c['neff_over_n']:.4f}",
         "| split | recordings | points | keyframes | bias | spread raw | N per kf | Neff/N median | inside 1 sigma |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, s in (("fit", res["fit"]), ("held-out", res["heldout"])):
        if s.get("n_seq"):
            cov = res["coverage"]["fit" if name == "fit" else "heldout"]
            L.append(f"| {name} | {s['n_seq']} | {s['n_points']} | {s['n_kf_used']} | {s['bias']:.4f} | "
                     f"{s['spread_raw']:.4f} | {s['n_per_kf_mean']:.1f} | {s['neff_ratio_median']:.4f} | "
                     f"{cov.get('frac_1sigma', float('nan')):.3f} |")
    return "\n".join(L)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", nargs="*", default=[], help="ROS 2 bags (folder, .mcap or .db3)")
    ap.add_argument("--folders", nargs="*", default=[], help="folders with color.txt and aligned_depth.txt")
    ap.add_argument("--fit", nargs="*", default=None,
                    help="recording names (folder or bag basenames) of the fit split, the others are held out. "
                         "Default: every recording when there is one, else required")
    ap.add_argument("--domain", default=DOMAIN)
    ap.add_argument("--backend", choices=BACKENDS, default="torch-fp32")
    ap.add_argument("--engine", default=None, help="TensorRT engine, backend trt-*")
    ap.add_argument("--model", default=None, help="depth model, default ucslam.depth_model of configs/default.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample", choices=("full", "map"), default="full",
                    help="full: full-resolution prediction (paper). map: the half-resolution map of the depth node")
    ap.add_argument("--keypoints", choices=("orb", "xfeat"), default="orb")
    ap.add_argument("--settings", default=None, help="take the ORBextractor parameters from this settings file")
    ap.add_argument("--n-features", type=int, default=1250, help="ORBextractor.nFeatures when --settings is not given")
    ap.add_argument("--topk", type=int, default=None, help="XFeat keypoints, default depth.topk (1024)")
    ap.add_argument("--keyframes", type=int, default=None, help="frames per recording, default depth.keyframes_per_seq")
    ap.add_argument("--tol-ms", type=float, default=5.0, help="color to depth stamp tolerance")
    ap.add_argument("--z-min", type=float, default=0.3)
    ap.add_argument("--z-max", type=float, default=5.0)
    ap.add_argument("--ref-noise", type=float, default=0.01, help="reference log-depth noise per metre of depth")
    ap.add_argument("--color-topic", default=COLOR_TOPIC)
    ap.add_argument("--depth-topic", default=DEPTH_TOPIC)
    ap.add_argument("--color-list", default="color.txt")
    ap.add_argument("--depth-list", default="aligned_depth.txt")
    ap.add_argument("--out-json", default=None, help="default configs/d455/<domain>_depth_fit.json")
    ap.add_argument("--out-yaml", default=None, help="default configs/d455/<domain>_depth_constants.yaml")
    ap.add_argument("--threads", type=int, default=4)
    return ap


def main(argv=None, live=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_json = a.out_json or os.path.join(OUT_DIR, f"{a.domain}_depth_fit.json")
    a.out_yaml = a.out_yaml or os.path.join(OUT_DIR, f"{a.domain}_depth_constants.yaml")
    cfg = load_config()
    dcfg = cfg["depth"]
    n_kf = a.keyframes or int(dcfg["keyframes_per_seq"])
    recs = [FolderRecording(p, a.color_list, a.depth_list) for p in a.folders]
    recs += [BagRecording(p, a.color_topic, a.depth_topic) for p in a.bags]
    if not recs:
        print("give --bags or --folders")
        return 2
    names = [r.name for r in recs]
    if len(set(names)) != len(names):
        print(f"recording names must differ: {names}")
        return 2
    fit_keys = a.fit if a.fit is not None else (names if len(names) == 1 else None)
    if not fit_keys:
        print("several recordings: name the fit split with --fit (the others are the held-out check)")
        return 2
    unknown = [k for k in fit_keys if k not in names]
    if unknown:
        print(f"--fit names {unknown} are not among the recordings {names}")
        return 2
    ho_keys = [n for n in names if n not in fit_keys]
    if not ho_keys:
        print("WARNING: no held-out recording, the coverage check runs on the fit data only")
    import torch
    torch.set_num_threads(a.threads)
    if a.device != "cpu" and not torch.cuda.is_available():
        # A CPU fit runs another numeric path than the flight network on the GPU, so it is never chosen silently.
        print(f"--device {a.device} but torch {torch.__version__} sees no CUDA device (a CPU-only wheel?). Install "
              "the CUDA torch of the depth node, or give --device cpu to fit on the CPU on purpose")
        return 2
    device = a.device
    if a.keypoints == "orb":
        kp_params = orb_params_from_settings(a.settings) if a.settings else {"n_features": a.n_features}
        keypoints = OrbSlamKeypoints(**kp_params)
    else:
        keypoints = XFeatKeypoints(a.topk or int(dcfg["topk"]), device)
    live = live if live is not None else make_live(a.backend, a.engine, a.model, device)
    model = getattr(live, "model", None) or a.model or cfg["ucslam"]["depth_model"]
    predict = Predictor(live, a.sample)
    print(f"{model} backend {a.backend} on {device}, keypoints {a.keypoints} {keypoints.params}, sample {a.sample}, "
          f"{len(recs)} recordings, {n_kf} frames each", flush=True)
    per_rec, counts, walls, lat = {}, {}, {}, {}
    for r in recs:
        pairs = select_frames(r.t_color, r.t_depth, int(a.tol_ms * 1e6), n_kf)
        if not pairs:
            off = nearest_frame(r.t_depth, r.t_color, 10**12)
            gap = np.abs(r.t_depth[off[off >= 0]] - r.t_color[off >= 0]) * 1e-6 if r.t_depth.size else np.zeros(0)
            print(f"  {r.name}: no color frame has a depth within {a.tol_ms} ms "
                  f"(median offset {np.median(gap) if gap.size else float('nan'):.1f} ms)")
            return 3
        n_lat = len(predict.latency_s)
        d = process(r, pairs, keypoints, predict, model, a.z_min, a.z_max, a.ref_noise)
        per_rec[r.name], counts[r.name], walls[r.name] = d, d["counts"], d["wall_s"]
        new = predict.latency_s[n_lat:]
        lat[r.name] = 1e3 * float(np.median(new)) if new else None
        print(f"  {r.name}: {d['counts']} {d['wall_s']:.0f} s", flush=True)
    res = fit_constants(per_rec, fit_keys, ho_keys, model, dcfg)
    res["counts"] = counts
    res["meta"] = {"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "domain": a.domain, "model": model,
                   "model_source": getattr(getattr(live, "net", None), "source", "?"), "backend": a.backend,
                   "engine": os.path.abspath(a.engine) if a.engine else "", "device": device,
                   "keypoints": a.keypoints, "keypoint_params": keypoints.params, "sample": a.sample,
                   "keyframes_per_rec": n_kf, "tol_ms": a.tol_ms, "z_min_m": a.z_min, "z_max_m": a.z_max,
                   "ref_lognoise_per_m": a.ref_noise, "recordings": {r.name: getattr(r, "path", getattr(r, "uri", ""))
                                                                     for r in recs},
                   "wall_s": walls, "latency_median_ms": lat, "git": _git(), "python": platform.python_version(),
                   "machine": platform.machine(), "torch": torch.__version__}
    for p in (a.out_json, a.out_yaml):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(_jsonable(res), f, indent=1)
    write_yaml(a.out_yaml, res, a)
    print(summary(res))
    print(f"wrote {a.out_json} and {a.out_yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
