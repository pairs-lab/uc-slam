#!/usr/bin/env python3
"""Write UC-SLAM settings files for an Intel RealSense D455 from a ROS 2 bag or the live camera.

Two files, written to configs/d455/ by default:

    d455_mono_inertial.yaml     mono-inertial on the color camera, UCSLAM block of the adopted variant a11r_capgatex,
                                domain d455color. The depth prior is on only when fitted constants are given.
    d455_stereo_inertial.yaml   stereo-inertial on infra1 + infra2, UCSLAM block of variant depth_fix2 with the depth
                                prior off (PriorNet 0), domain d455ir. Camera.type PinHole with zero distortion and
                                Stereo.T_c1_c2 = translation b, not the stock "Rectified": ORB-SLAM3 v1.0 Settings
                                crash on a Rectified stereo-inertial file (operator<< reads the unset Camera 2).

What comes from where:

    Camera1.fx fy cx cy, k1 k2 p1 p2 (k3)   camera_info of the color (or infra1) stream
    Camera.width, Camera.height              camera_info
    Stereo.T_c1_c2                           camera_info of infra2: b = -P[3] / P[0] (the driver writes Tx = -fx b)
    IMU.T_b_c1                               /tf_static: pose of the camera optical frame (frame_id of camera_info)
                                             in the frame of the IMU messages (camera_imu_optical_frame with
                                             unite_imu_method 2). Or a Kalibr camchain-imucam file (--kalibr),
                                             for one file at a time (--which mono or stereo). Its resolution
                                             and rostopic must match the stream, and it is compared with
                                             /tf_static (warning above 2 cm or 2 deg).
    UCSLAM.DepthBackend, D455.FitEngine      the backend and engine of the constant fit (mono file with
                                             --constants, from the backend and engine keys of the constants
                                             YAML). slam_node refuses another depth_backend when the key is
                                             present. ucslam_d455.launch.py takes both as its defaults and
                                             refuses another backend. ORB-SLAM3 does not read them.
    IMU.Frequency                            measured on the IMU topic (mean rate over the stamps)
    IMU.NoiseGyro, NoiseAcc, GyroWalk,       the D435i values of the stock RealSense file (same Bosch IMU family),
    AccWalk                                  times --imu-noise-scale for the white noise (propeller vibration)
    UCSLAM.*                                 scripts/gen_ucslam_yaml.py generate() on an in-memory copy of
                                             configs/default.yaml with an added per_domain entry. default.yaml is
                                             never written, because the paper tools list its domains.
    UCSLAM.SigmaNetC0 C1, LogBias, NeffOverN the YAML of scripts/jetson/fit_depth_constants.py (--constants)

Without --constants the mono file is written with UCSLAM.PriorNet 0, UseDepthFactor 0 and MaxDepthPriorPoints 0,
a warning block at the top and a console warning. An unfitted depth prior is never switched on by mistake.

The stock files are cpp/ucslam/Examples/{Monocular,Stereo}-Inertial/RealSense_D435i.yaml. Every key that this
script does not name keeps its stock value (1250 ORB features, FAST thresholds, Stereo.ThDepth 40, viewer).

Usage (ROS 2 sourced, so that rosbag2_py and rclpy import):

    python3 scripts/jetson/make_d455_settings.py --bag ~/ucslam_bags/d455_calib_<time>
    python3 scripts/jetson/make_d455_settings.py --live --duration 10
    python3 scripts/jetson/make_d455_settings.py --bag <bag> --constants configs/d455/d455color_depth_constants.yaml

The bag or the camera must carry the camera_info of the chosen streams, /tf_static and the united IMU topic
(launch ucslam_d455.launch.py with mode:=calib or slam:=false, see its docstring).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import math
import os
import re
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(REPO, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "scripts"))

import gen_ucslam_yaml as G  # noqa: E402

STOCK_DIRS = (os.path.join(REPO, "cpp", "ucslam", "Examples"),
              os.path.join(REPO, "third_party", "ORB_SLAM3", "Examples"))
STOCK_MONO = "Monocular-Inertial/RealSense_D435i.yaml"
STOCK_STEREO = "Stereo-Inertial/RealSense_D435i.yaml"
OUT_DIR = os.path.join(REPO, "configs", "d455")
MONO_NAME = "d455_mono_inertial.yaml"
STEREO_NAME = "d455_stereo_inertial.yaml"
DOMAIN_COLOR = "d455color"
DOMAIN_IR = "d455ir"
MONO_VARIANT = "a11r_capgatex"
STEREO_VARIANT = "depth_fix2"
# The three keys that switch the depth prior off. With them, depth_fix2 becomes the ablation abl2_nodepth.
NO_DEPTH = {"PriorNet": 0, "UseDepthFactor": 0, "MaxDepthPriorPoints": 0}
# IMU noise of the stock RealSense_D435i.yaml ("Use those from VINS-mono"). The D455 has the same Bosch IMU family.
D435I_IMU = {"NoiseGyro": 1e-3, "NoiseAcc": 1e-2, "GyroWalk": 1e-6, "AccWalk": 1e-4}
CONSTANT_KEYS = ("sigma_net_c0", "sigma_net_c1", "log_bias", "neff_over_n")
BACKENDS = ("torch-fp32", "torch-fp16", "trt-fp16", "trt-fp32")
# Kalibr against /tf_static: a larger difference is reported, because one of the two is then likely wrong.
KALIBR_TF_TOL_M = 0.02
KALIBR_TF_TOL_DEG = 2.0


def topics(ns: str = "camera", name: str = "camera") -> Dict[str, str]:
    """The realsense2_camera 4.x topics for camera_namespace ns and camera_name name."""
    p = f"/{ns}/{name}" if ns else f"/{name}"
    return {"color_info": f"{p}/color/camera_info", "infra1_info": f"{p}/infra1/camera_info",
            "infra2_info": f"{p}/infra2/camera_info", "imu": f"{p}/imu", "tf_static": "/tf_static"}


# ------------------------------------------------------------------ geometry --

def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_T(t, q_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*q_xyzw)
    T[:3, 3] = np.asarray(t, float)
    return T


def rot_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix in degrees."""
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1) / 2))))


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


class TfTree:
    """Static transforms parent -> child. ``lookup(target, source)`` is the pose of source in target."""

    def __init__(self):
        self.parent: Dict[str, Tuple[str, np.ndarray]] = {}

    def add(self, parent: str, child: str, T_parent_child: np.ndarray) -> None:
        self.parent[child.lstrip("/")] = (parent.lstrip("/"), np.asarray(T_parent_child, float))

    def _to_root(self, frame: str) -> Tuple[str, np.ndarray]:
        T = np.eye(4)
        seen = set()
        f = frame.lstrip("/")
        while f in self.parent:
            if f in seen:
                raise ValueError(f"tf loop at {f}")
            seen.add(f)
            p, Tpc = self.parent[f]
            T = Tpc @ T
            f = p
        return f, T

    def frames(self) -> List[str]:
        return sorted(set(self.parent) | {p for p, _ in self.parent.values()})

    def lookup(self, target: str, source: str) -> np.ndarray:
        rt, T_rt = self._to_root(target)
        rs, T_rs = self._to_root(source)
        known = self.frames()
        for f in (target, source):
            if f.lstrip("/") not in known:
                raise KeyError(f"frame {f} is not in /tf_static (frames: {', '.join(known)})")
        if rt != rs:
            raise KeyError(f"frames {target} and {source} are not connected in /tf_static")
        return inv_T(T_rt) @ T_rs


# ------------------------------------------------------------ sensor inputs --

def camera_from_info(info: dict) -> dict:
    """Pinhole intrinsics and plumb_bob distortion from a camera_info dict (keys of sensor_msgs/CameraInfo)."""
    k = [float(v) for v in info["k"]]
    d = [float(v) for v in info.get("d", [])]
    model = info.get("distortion_model", "plumb_bob") or "plumb_bob"
    if model not in ("plumb_bob", "rational_polynomial") and any(abs(v) > 0 for v in d):
        raise ValueError(f"distortion model {model} is not plumb_bob, UC-SLAM PinHole needs k1 k2 p1 p2 [k3]")
    if any(abs(v) > 0 for v in d[5:]):
        raise ValueError(f"distortion {model} has non-zero coefficients after k3 ({d[5:]}). UC-SLAM PinHole takes "
                         "k1 k2 p1 p2 k3 only. Calibrate with a 5-coefficient model (Kalibr radtan) instead")
    d = (d + [0.0] * 5)[:5]
    cam = {"fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5], "width": int(info["width"]), "height": int(info["height"]),
           "dist": d, "frame_id": info.get("frame_id", ""), "p": [float(v) for v in info.get("p", [0.0] * 12)]}
    if cam["fx"] <= 0 or cam["fy"] <= 0 or cam["width"] <= 0 or cam["height"] <= 0:
        raise ValueError(f"camera_info holds no intrinsics: {info}")
    return cam


def stereo_baseline(infra2: dict) -> float:
    """b = -P[3] / P[0] of the right camera_info (realsense2_camera writes Tx = -fx b)."""
    p = infra2["p"]
    if p[0] <= 0:
        raise ValueError("infra2 camera_info has no projection matrix")
    return -p[3] / p[0]


def imu_rate(t_ns) -> dict:
    """Mean rate, median interval and gaps of a stream of IMU stamps."""
    t = np.asarray(t_ns, np.int64)
    if t.size < 3:
        raise ValueError(f"only {t.size} IMU messages")
    dt_s = np.diff(t).astype(np.float64) * 1e-9
    good = dt_s > 0
    if not good.any():
        raise ValueError(f"the {t.size} IMU stamps never increase (all equal or going back): check the IMU topic "
                         "and the driver clock")
    med = float(np.median(dt_s[good]))
    span = float((t.max() - t.min()) * 1e-9)
    return {"n": int(t.size), "span_s": span, "rate_hz": float((t.size - 1) / span),
            "median_dt_ms": 1e3 * med, "rate_median_hz": 1.0 / med,
            "gaps": int(np.sum(dt_s > 1.5 * med)), "max_gap_ms": 1e3 * float(dt_s.max()),
            "not_increasing": int(np.sum(~good))}


def kalibr_camera(path: str, cam: str = "cam0") -> dict:
    """T_imu_cam, time shift, resolution, rostopic and pinhole intrinsics of one camera of a Kalibr camchain-imucam.

    resolution and rostopic are None when the file does not hold them."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    if cam not in (doc or {}):
        raise ValueError(f"{path} has no camera {cam} (it has {', '.join(sorted(doc or {}))})")
    c = doc[cam]
    T_cam_imu = np.asarray(c["T_cam_imu"], float)
    res = c.get("resolution")
    out = {"T_imu_cam": inv_T(T_cam_imu), "timeshift_cam_imu": float(c.get("timeshift_cam_imu", 0.0)),
           "resolution": (int(res[0]), int(res[1])) if res else None, "rostopic": c.get("rostopic")}
    if "intrinsics" in c and c.get("camera_model", "pinhole") == "pinhole" and \
            c.get("distortion_model", "radtan") == "radtan":
        fu, fv, pu, pv = [float(v) for v in c["intrinsics"]]
        w, h = [int(v) for v in c["resolution"]]
        d = [float(v) for v in c.get("distortion_coeffs", [0, 0, 0, 0])]
        out["camera"] = {"fx": fu, "fy": fv, "cx": pu, "cy": pv, "width": w, "height": h,
                         "dist": (d + [0.0] * 5)[:5], "frame_id": "kalibr:" + cam, "p": [0.0] * 12}
    return out


def load_constants(path: str, domain: str = DOMAIN_COLOR) -> dict:
    """The YAML of fit_depth_constants.py. Refuses another domain, a partial fit split or a non-finite value."""
    with open(path) as f:
        c = yaml.safe_load(f)
    if c.get("domain") != domain:
        raise ValueError(f"{path} holds constants of domain {c.get('domain')!r}, not {domain!r}")
    if c.get("partial"):
        raise ValueError(f"{path} comes from a partial fit split, refit with every recording")
    out = {}
    for k in CONSTANT_KEYS:
        v = c.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
            raise ValueError(f"{path}: {k} is {v!r}, expected a finite number")
        out[k] = float(v)
    if out["sigma_net_c0"] <= 0 or not (0 < out["neff_over_n"] <= 1):
        raise ValueError(f"{path}: sigma_net_c0 must be > 0 and neff_over_n in (0, 1], got {out}")
    backend = c.get("backend")
    if backend not in BACKENDS:
        raise ValueError(f"{path}: backend is {backend!r}, expected one of {', '.join(BACKENDS)}")
    engine = c.get("engine") or ""
    if any(ch.isspace() or ch in "\"#" for ch in engine):
        raise ValueError(f"{path}: engine path {engine!r} has a space, quote or #, which the settings reader of "
                         "ucslam_d455.launch.py does not take. Move the engine to a plain path and fit again")
    return {"values": out, "backend": backend, "engine": engine, "model": c.get("model", "?"),
            "keypoints": c.get("keypoints", "?"), "date": c.get("date", "?"), "path": os.path.abspath(path)}


# ---------------------------------------------------------------- rendering --

def stock_text(rel: str) -> Tuple[str, str]:
    for d in STOCK_DIRS:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            with open(p) as f:
                return f.read(), os.path.relpath(p, REPO)
    raise FileNotFoundError(f"stock file {rel} not found in {STOCK_DIRS}")


def fmt_T(T: np.ndarray) -> str:
    rows = [", ".join(f"{v:.9f}" for v in T[i]) for i in range(4)]
    return ("IMU.T_b_c1: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: f\n   data: [" +
            ",\n         ".join(rows) + "]")


def _sub(text: str, pat: str, rep: str) -> str:
    out, n = re.subn(pat, lambda _m: rep, text, count=1, flags=re.M)
    if n != 1:
        raise ValueError(f"pattern {pat!r} matched {n} times in the stock file")
    return out


def render_camera(stock: str, cam: dict, T_b_c: np.ndarray, imu_hz: float, fps: int, imu_noise: Dict[str, float],
                  baseline: Optional[float] = None) -> str:
    """The stock text with the camera, IMU and (stereo) baseline lines of this D455."""
    f = lambda v: G.fmt_value(float(v), float)  # noqa: E731
    subs = [(r"^Camera1\.fx:.*$", f"Camera1.fx: {f(cam['fx'])}"), (r"^Camera1\.fy:.*$", f"Camera1.fy: {f(cam['fy'])}"),
            (r"^Camera1\.cx:.*$", f"Camera1.cx: {f(cam['cx'])}"),
            (r"^Camera\.width:.*$", f"Camera.width: {int(cam['width'])}"),
            (r"^Camera\.height:.*$", f"Camera.height: {int(cam['height'])}"),
            (r"^Camera\.fps:.*$", f"Camera.fps: {int(fps)}"),
            (r"^IMU\.T_b_c1: !!opencv-matrix\n(?:   .*\n)+?   data: \[[^\]]*\]", fmt_T(T_b_c)),
            (r"^IMU\.NoiseGyro:.*$", f"IMU.NoiseGyro: {f(imu_noise['NoiseGyro'])} # rad/s^0.5"),
            (r"^IMU\.NoiseAcc:.*$", f"IMU.NoiseAcc: {f(imu_noise['NoiseAcc'])} # m/s^1.5"),
            (r"^IMU\.GyroWalk:.*$", f"IMU.GyroWalk: {f(imu_noise['GyroWalk'])} # rad/s^1.5"),
            (r"^IMU\.AccWalk:.*$", f"IMU.AccWalk: {f(imu_noise['AccWalk'])} # m/s^2.5"),
            (r"^IMU\.Frequency:.*$", f"IMU.Frequency: {f(round(imu_hz, 1))}")]
    cy = f"Camera1.cy: {f(cam['cy'])}"
    if baseline is None:
        k1, k2, p1, p2, k3 = cam["dist"]
        subs += [(r"^Camera1\.cy:.*$", cy), (r"^Camera1\.k1:.*$", f"Camera1.k1: {f(k1)}"),
                 (r"^Camera1\.k2:.*$", f"Camera1.k2: {f(k2)}"),
                 (r"^Camera1\.p1:.*$", f"Camera1.p1: {f(p1)}"),
                 (r"^Camera1\.p2:.*$", f"Camera1.p2: {f(p2)}" + (f"\nCamera1.k3: {f(k3)}" if k3 != 0 else ""))]
    else:
        # The stock stereo file says Camera.type "Rectified". ORB-SLAM3 v1.0 crashes on a Rectified stereo-inertial
        # file: Settings::operator<< prints Camera 2 from originalCalib2_, which only PinHole and KannalaBrandt set
        # (cpp/ucslam/src/Settings.cc, readCamera2 and operator<<). So the pair is written as PinHole with zero
        # distortion, the same intrinsics on both sides and Stereo.T_c1_c2 = pure translation b along x. The
        # rectification ORB-SLAM3 then computes is the identity map, as for the EuRoC stereo files.
        if any(abs(v) > 0 for v in cam["dist"]):
            raise ValueError("the infrared pair must have zero distortion (use the rect_raw infrared streams)")
        zero = "\nCamera{n}.k1: 0.0\nCamera{n}.k2: 0.0\nCamera{n}.p1: 0.0\nCamera{n}.p2: 0.0"
        right = (f"\n\n# Right camera (infra2): the same rectified intrinsics, zero distortion\n"
                 f"Camera2.fx: {f(cam['fx'])}\nCamera2.fy: {f(cam['fy'])}\nCamera2.cx: {f(cam['cx'])}\n"
                 f"Camera2.cy: {f(cam['cy'])}" + zero.format(n=2))
        T12 = np.eye(4)
        T12[0, 3] = baseline
        tc = fmt_T(T12).replace("IMU.T_b_c1", "Stereo.T_c1_c2")
        subs += [(r"^Camera\.type:.*$", 'Camera.type: "PinHole"'),
                 (r"^# Rectified Camera calibration \(OpenCV\)$",
                  "# Left camera (infra1), rectified by the camera, so zero distortion. PinHole, not Rectified,\n"
                  "# because ORB-SLAM3 v1.0 Settings crash on a Rectified stereo-inertial file (see the script)."),
                 (r"^Camera1\.cy:.*$", cy + zero.format(n=1) + right),
                 (r"^Stereo\.b:.*$", f"# Pose of infra2 in infra1: the baseline {baseline:.6f} m along x\n" + tc)]
    text = stock
    for pat, rep in subs:
        text = _sub(text, pat, rep)
    return text


def domain_config(cfg: dict, domain: str, constants: Dict[str, float]) -> dict:
    """A deep copy of the loaded default.yaml with ucslam.per_domain.<domain> set. The file is not touched."""
    out = copy.deepcopy(cfg)
    out["ucslam"]["per_domain"][domain] = {k: float(constants[k]) for k in CONSTANT_KEYS}
    return out


def placeholder_constants(cfg: dict) -> Dict[str, float]:
    """Values for a file whose depth prior is off. They are never read with PriorNet 0."""
    pd = cfg["ucslam"]["per_domain"]
    src = pd.get("openloris") or pd["euroc"]
    return {k: float(src[k]) for k in CONSTANT_KEYS}


WARN_UNFITTED = """\
# ################################################################################################
# WARNING: UCSLAM.PriorNet is 0. No fitted {domain} depth constants were given to make_d455_settings.py.
# The depth prior is OFF (PriorNet 0, UseDepthFactor 0, MaxDepthPriorPoints 0). The four constants in the
# UCSLAM block are placeholders copied from another camera and must not be used. Fit them with
# scripts/jetson/fit_depth_constants.py, then write this file again with the constants YAML.
# ################################################################################################
"""


def header(lines: List[str]) -> str:
    return "".join(f"# {s}\n" if s else "#\n" for s in lines)


def build_mono(cfg: dict, cam: dict, T_b_c: np.ndarray, imu_hz: float, fps: int, imu_noise: Dict[str, float],
               constants: Optional[dict], provenance: List[str], domain: str = DOMAIN_COLOR) -> str:
    stock, stock_rel = stock_text(STOCK_MONO)
    body = render_camera(stock, cam, T_b_c, imu_hz, fps, imu_noise)
    overrides = G.variant_overrides(cfg, MONO_VARIANT)
    if constants is None:
        overrides.update(NO_DEPTH)
        consts = placeholder_constants(cfg)
        dep = ["Depth constants: NONE. The depth prior is off, see the warning below."]
    else:
        consts = constants["values"]
        dep = [f"Depth constants: {constants['path']}",
               f"  fitted {constants['date']} with backend {constants['backend']}, model {constants['model']}, "
               f"keypoints {constants['keypoints']}.",
               f"  Valid for backend {constants['backend']} only (UCSLAM.DepthBackend at the end of this file). "
               "Another backend needs another fit."]
    head = header([f"Intel RealSense D455, mono-inertial on the color camera, domain {domain}.",
                   f"Written by scripts/jetson/make_d455_settings.py on {dt.datetime.now():%Y-%m-%d %H:%M}"
                   f" from the stock {stock_rel}."] + provenance + dep)
    if constants is None:
        head += WARN_UNFITTED.format(domain=domain)
    body = "%YAML:1.0\n" + head + body[len("%YAML:1.0\n"):]
    text = G.generate(body, domain_config(cfg, domain, consts), domain, stock_rel, overrides=overrides,
                      variant=MONO_VARIANT, source_label="the D455 file of scripts/jetson/make_d455_settings.py")
    if constants is None:
        text += "# WARNING: depth prior OFF (PriorNet 0). The constants above are placeholders.\n"
    else:
        text += ("\n# Backend and engine of the depth-constant fit. The constants above hold for this backend only.\n"
                 "# slam_node stops at start when its depth_backend differs. ucslam_d455.launch.py takes both as its\n"
                 "# defaults and refuses another backend. ORB-SLAM3 does not read these keys.\n"
                 f'UCSLAM.DepthBackend: "{constants["backend"]}"\nD455.FitEngine: "{constants["engine"]}"\n')
    return text


def build_stereo(cfg: dict, cam: dict, baseline: float, T_b_c: np.ndarray, imu_hz: float, fps: int,
                 imu_noise: Dict[str, float], provenance: List[str], domain: str = DOMAIN_IR) -> str:
    stock, stock_rel = stock_text(STOCK_STEREO)
    body = render_camera(stock, cam, T_b_c, imu_hz, fps, imu_noise, baseline=baseline)
    overrides = G.variant_overrides(cfg, STEREO_VARIANT)
    overrides.update(NO_DEPTH)
    head = header([f"Intel RealSense D455, stereo-inertial on infra1 + infra2 (emitter off), domain {domain}.",
                   f"Written by scripts/jetson/make_d455_settings.py on {dt.datetime.now():%Y-%m-%d %H:%M}"
                   f" from the stock {stock_rel}."] + provenance +
                  ["Variant depth_fix2 with the depth prior off (PriorNet 0, UseDepthFactor 0, MaxDepthPriorPoints 0),",
                   "the same UCSLAM keys as the ablation abl2_nodepth. No infrared depth constants exist.",
                   "The constants in the UCSLAM block are placeholders and are not read with PriorNet 0."])
    body = "%YAML:1.0\n" + head + body[len("%YAML:1.0\n"):]
    return G.generate(body, domain_config(cfg, domain, placeholder_constants(cfg)), domain, stock_rel,
                      overrides=overrides, variant=STEREO_VARIANT,
                      source_label="the D455 file of scripts/jetson/make_d455_settings.py")


# ------------------------------------------------------------------- checks --

REQUIRED = [("File.version", "str"), ("Camera.type", "str"), ("Camera1.fx", "real"), ("Camera1.fy", "real"),
            ("Camera1.cx", "real"), ("Camera1.cy", "real"), ("Camera.width", "int"), ("Camera.height", "int"),
            ("Camera.fps", "int"), ("Camera.RGB", "int"), ("IMU.InsertKFsWhenLost", "int"),
            ("IMU.NoiseGyro", "real"), ("IMU.NoiseAcc", "real"), ("IMU.GyroWalk", "real"), ("IMU.AccWalk", "real"),
            ("IMU.Frequency", "real"), ("ORBextractor.nFeatures", "int"), ("ORBextractor.scaleFactor", "real"),
            ("ORBextractor.nLevels", "int"), ("ORBextractor.iniThFAST", "int"), ("ORBextractor.minThFAST", "int")]


def check_settings(path: str) -> List[str]:
    """Problems that would stop ORB-SLAM3's Settings or the UC-SLAM reader. Empty list: the file is usable.

    The types follow Settings::readParameter (a float must be a real literal, Settings.cc) and the UCSLAM key
    table of scripts/gen_ucslam_yaml.py. The file is parsed with OpenCV's reader, as the C++ side does."""
    import cv2
    bad: List[str] = []
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return [f"OpenCV cannot open {path}"]

    def typed(key: str, kind: str) -> None:
        n = fs.getNode(key)
        if n.empty():
            bad.append(f"{key} missing")
        elif kind == "real" and not n.isReal():
            bad.append(f"{key} must be a real number with a decimal point")
        elif kind == "int" and not n.isInt():
            bad.append(f"{key} must be an integer")
        elif kind == "str" and not n.isString():
            bad.append(f"{key} must be a quoted string")

    for key, kind in REQUIRED:
        typed(key, kind)
    ctype = fs.getNode("Camera.type").string()
    if ctype == "PinHole":
        for key in ("Camera1.k1", "Camera1.k2", "Camera1.p1", "Camera1.p2"):
            typed(key, "real")
        if not fs.getNode("Camera1.k3").empty():
            typed("Camera1.k3", "real")
        if not fs.getNode("Camera2.fx").empty():     # a stereo pair
            for key in ("Camera2.fx", "Camera2.fy", "Camera2.cx", "Camera2.cy", "Stereo.ThDepth"):
                typed(key, "real")
            T12 = fs.getNode("Stereo.T_c1_c2").mat()
            if T12 is None or T12.shape != (4, 4):
                bad.append("Stereo.T_c1_c2 must be a 4x4 opencv-matrix")
            elif not 0.0 < float(np.linalg.norm(T12[:3, 3])) < 1.0:
                bad.append(f"Stereo.T_c1_c2 baseline {np.linalg.norm(T12[:3, 3]):.4f} m is not a stereo baseline")
    elif ctype == "Rectified":
        bad.append('Camera.type "Rectified" crashes ORB-SLAM3 v1.0 Settings in stereo (null Camera 2), use PinHole')
    else:
        bad.append(f"Camera.type {ctype!r} is not PinHole")
    T = fs.getNode("IMU.T_b_c1").mat()
    if T is None or T.shape != (4, 4):
        bad.append("IMU.T_b_c1 must be a 4x4 opencv-matrix")
    else:
        R = T[:3, :3].astype(np.float64)
        if np.abs(R @ R.T - np.eye(3)).max() > 1e-5 or abs(np.linalg.det(R) - 1) > 1e-5:
            bad.append("IMU.T_b_c1 rotation is not orthonormal")
        if np.linalg.norm(T[:3, 3]) > 0.5:
            bad.append(f"IMU.T_b_c1 lever arm {np.linalg.norm(T[:3, 3]):.3f} m is too long for one device")
    types = {k: t for k, _p, t in G.KEYS}
    types.update(dict(G.OPTIONAL_KEYS))
    types["Dataset"] = str
    for key, t in types.items():
        n = fs.getNode(f"UCSLAM.{key}")
        if n.empty():
            if key in ("Dataset",) or key in [k for k, _p, _t in G.KEYS]:
                bad.append(f"UCSLAM.{key} missing")
            continue
        typed(f"UCSLAM.{key}", {float: "real", int: "int", str: "str"}[t])
    backend = fs.getNode("UCSLAM.DepthBackend")
    if not backend.empty() and (not backend.isString() or backend.string() not in BACKENDS):
        bad.append(f"UCSLAM.DepthBackend must be a quoted string, one of {', '.join(BACKENDS)}")
    if fs.getNode("UCSLAM.PriorNet").isInt() and int(fs.getNode("UCSLAM.PriorNet").real()) == 1:
        if fs.getNode("UCSLAM.DepthModel").empty():
            bad.append("UCSLAM.PriorNet is 1 but UCSLAM.DepthModel is missing")
        if backend.empty():
            bad.append("UCSLAM.PriorNet is 1 but UCSLAM.DepthBackend (the backend of the constant fit) is missing")
    fs.release()
    return bad


# ------------------------------------------------------------- collection --

def _info_dict(msg) -> dict:
    return {"frame_id": msg.header.frame_id, "width": msg.width, "height": msg.height,
            "distortion_model": msg.distortion_model, "d": list(msg.d), "k": list(msg.k), "p": list(msg.p)}


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class Collected:
    """What the settings need, from a bag or from the live camera."""

    def __init__(self, source: str):
        self.source = source
        self.info: Dict[str, dict] = {}
        self.tf: List[Tuple[str, str, np.ndarray]] = []
        self.imu_t: List[int] = []
        self.imu_acc: List[Tuple[float, float, float]] = []
        self.imu_frame = ""

    def add_tf(self, msg) -> None:
        for tr in msg.transforms:
            t, q = tr.transform.translation, tr.transform.rotation
            self.tf.append((tr.header.frame_id, tr.child_frame_id, make_T((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))))

    def add_imu(self, msg) -> None:
        self.imu_t.append(_stamp_ns(msg.header))
        a = msg.linear_acceleration
        self.imu_acc.append((a.x, a.y, a.z))
        self.imu_frame = msg.header.frame_id


def bag_storage_id(uri: str) -> str:
    """The storage plugin of a bag: from metadata.yaml, or from the file extension."""
    if os.path.isfile(uri):
        return "mcap" if uri.endswith(".mcap") else "sqlite3"
    meta = os.path.join(uri, "metadata.yaml")
    if os.path.exists(meta):
        with open(meta) as f:
            m = yaml.safe_load(f)
        sid = (m.get("rosbag2_bagfile_information") or {}).get("storage_identifier")
        if sid:
            return sid
    names = os.listdir(uri)
    return "mcap" if any(n.endswith(".mcap") for n in names) else "sqlite3"


def open_bag(uri: str, want: List[str]):
    """A rosbag2_py reader filtered to the topics of ``want`` that the bag has, and their types."""
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id=bag_storage_id(uri)),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    have = [t for t in want if t in types]
    if have:
        reader.set_filter(rosbag2_py.StorageFilter(topics=have))
    return reader, {t: types[t] for t in have}


def collect_bag(uri: str, tp: Dict[str, str], imu_seconds: float) -> Collected:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader, types = open_bag(uri, list(tp.values()))
    for key in ("imu", "tf_static"):
        if tp[key] not in types:
            raise SystemExit(f"{uri} has no {tp[key]} topic. Record it (ucslam_d455.launch.py record:=true).")
    cls = {t: get_message(ty) for t, ty in types.items()}
    c = Collected(f"bag {os.path.abspath(uri)}")
    rev = {v: k for k, v in tp.items()}
    while reader.has_next():
        topic, data, _t = reader.read_next()
        msg = deserialize_message(data, cls[topic])
        key = rev[topic]
        if key == "tf_static":
            c.add_tf(msg)
        elif key == "imu":
            if not c.imu_t or (_stamp_ns(msg.header) - c.imu_t[0]) * 1e-9 <= imu_seconds:
                c.add_imu(msg)
        elif key not in c.info:
            c.info[key] = _info_dict(msg)
    return c


def collect_live(tp: Dict[str, str], seconds: float, need: List[str], timeout_s: float = 30.0) -> Collected:
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Imu
    from tf2_msgs.msg import TFMessage
    rclpy.init()
    node = rclpy.create_node("make_d455_settings")
    c = Collected("live camera")
    latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    subs = [node.create_subscription(TFMessage, tp["tf_static"], c.add_tf, latched),
            node.create_subscription(Imu, tp["imu"], c.add_imu, qos_profile_sensor_data)]
    for key in ("color_info", "infra1_info", "infra2_info"):
        subs.append(node.create_subscription(
            CameraInfo, tp[key], lambda m, k=key: c.info.setdefault(k, _info_dict(m)), qos_profile_sensor_data))
    t0 = time.monotonic()
    print(f"listening for {seconds:.0f} s of IMU data and the camera_info of "
          f"{', '.join(need) or 'every stream that is on'}", flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            span = (c.imu_t[-1] - c.imu_t[0]) * 1e-9 if len(c.imu_t) > 1 else 0.0
            if span >= seconds and all(k in c.info for k in need) and c.tf:
                break
            if time.monotonic() - t0 > timeout_s + seconds:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return c


# --------------------------------------------------------------------- main --

def build_files(c: Collected, a: argparse.Namespace, cfg: Optional[dict] = None) -> Dict[str, str]:
    """Settings texts by output name, from what was collected. Prints the measured values and the warnings."""
    cfg = cfg or G.load_config()
    tree = TfTree()
    for p, ch, T in c.tf:
        tree.add(p, ch, T)
    if not c.imu_t:
        raise SystemExit("no IMU message: is unite_imu_method 2 set and the IMU topic right?")
    try:
        rate = imu_rate(c.imu_t)
    except ValueError as e:
        raise SystemExit(f"IMU: {e}")
    # IMU.Frequency turns the noise densities into the noise of one sample, so it is the sampling rate, 1 / median
    # interval. Dropped samples lower the mean rate but not the sampling rate (and slam_node fills the gaps).
    hz = rate["rate_median_hz"]
    print(f"IMU {c.imu_frame}: {rate['n']} messages over {rate['span_s']:.1f} s, sampling rate {hz:.1f} Hz "
          f"(median interval {rate['median_dt_ms']:.3f} ms), mean delivered rate {rate['rate_hz']:.1f} Hz, "
          f"{rate['gaps']} gaps (longest {rate['max_gap_ms']:.1f} ms)")
    if abs(hz - a.imu_rate) > 0.02 * a.imu_rate:
        print(f"WARNING: the IMU samples at {hz:.1f} Hz, not the requested {a.imu_rate:.0f} Hz (gyro_fps). "
              "IMU.Frequency is written with the measured rate. Check the driver settings.")
    if rate["rate_hz"] < 0.98 * hz:
        print(f"WARNING: {100 * (1 - rate['rate_hz'] / hz):.1f} % of the IMU samples are missing.")
    if rate["gaps"]:
        print(f"WARNING: {rate['gaps']} IMU gaps longer than 1.5 intervals. Launch slam_node with imu_fill_gap_ms "
              "(ucslam_d455.launch.py sets 20 ms).")
    acc = np.asarray(c.imu_acc[: max(3, int(rate["rate_median_hz"]))], float).mean(axis=0)
    print(f"mean accelerometer over the first second, IMU frame: ({acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f}) m/s^2, "
          f"norm {np.linalg.norm(acc):.2f}. Camera level and still: y is near -9.8 in the optical IMU frame.")
    if abs(np.linalg.norm(acc) - 9.81) > 0.5:
        print("WARNING: the accelerometer norm is far from 9.81. Was the camera moving, or is the IMU uncalibrated?")
    noise = {k: v * (a.imu_noise_scale if k in ("NoiseGyro", "NoiseAcc") else 1.0) for k, v in D435I_IMU.items()}
    for k, v in (("NoiseGyro", a.noise_gyro), ("NoiseAcc", a.noise_acc), ("GyroWalk", a.gyro_walk),
                 ("AccWalk", a.acc_walk)):
        if v is not None:
            noise[k] = v
    try:
        kal = kalibr_camera(a.kalibr, a.kalibr_cam) if a.kalibr else None
    except (KeyError, ValueError) as e:
        raise SystemExit(f"--kalibr {a.kalibr}: {e}")
    constants = load_constants(a.constants, a.domain) if a.constants else None
    which = a.which
    if which == "auto":
        which = "both" if ("infra1_info" in c.info and "infra2_info" in c.info and "color_info" in c.info) else (
            "stereo" if "infra1_info" in c.info and "infra2_info" in c.info else "mono")
    if kal is not None and which == "both":
        # One Kalibr camera is one stream: color for the mono file, infra1 for the stereo file.
        raise SystemExit("--kalibr gives the extrinsic of one camera, but both files would be written. Run once with "
                         "--which mono --kalibr <camchain of the color camera> and once with --which stereo "
                         "--kalibr <camchain of the infrared pair> --kalibr-cam <the cam of infra1>")
    if kal is not None and a.kalibr_intrinsics and which != "mono":
        raise SystemExit("--kalibr-intrinsics is for the mono file only (the infrared pair keeps the rectified "
                         "camera_info of the driver)")
    out: Dict[str, str] = {}

    def extrinsic(cam_frame: str, cam: dict, stream: str) -> Tuple[np.ndarray, str]:
        """IMU.T_b_c1 of one stream (color or infra1): /tf_static, or Kalibr checked against the stream and tf."""
        T_tf, tf_err = None, "the IMU messages carry no frame_id"
        if c.imu_frame:
            try:
                T_tf = tree.lookup(c.imu_frame, cam_frame)
            except KeyError as e:
                tf_err = str(e)
        if kal is None:
            if T_tf is None:
                raise SystemExit(f"IMU.T_b_c1 of the {stream} camera: {tf_err}")
            return T_tf, f"IMU.T_b_c1: /tf_static, pose of {cam_frame} in {c.imu_frame} (factory design values)."
        name = f"Kalibr {a.kalibr_cam} of {a.kalibr}"
        topic = kal["rostopic"] or ""
        if topic and f"/{stream}/" not in topic:
            raise SystemExit(f"{name} was calibrated on {topic}, not on the {stream} stream of this file")
        if kal["resolution"] is None:
            raise SystemExit(f"{name} has no resolution, so it cannot be checked against the {stream} stream")
        if kal["resolution"] != (cam["width"], cam["height"]):
            raise SystemExit(f"{name} resolution {kal['resolution'][0]}x{kal['resolution'][1]} differs from the "
                             f"{stream} stream {cam['width']}x{cam['height']}: is it the {stream} camera?")
        T = kal["T_imu_cam"]
        if T_tf is None:
            print(f"WARNING: cannot compare {name} with /tf_static ({tf_err})")
        else:
            d_m = float(np.linalg.norm(T[:3, 3] - T_tf[:3, 3]))
            d_deg = rot_deg(T_tf[:3, :3].T @ T[:3, :3])
            print(f"{name} against /tf_static for the {stream} camera: lever arm {100 * d_m:.2f} cm apart, "
                  f"rotation {d_deg:.2f} deg apart")
            if d_m > KALIBR_TF_TOL_M or d_deg > KALIBR_TF_TOL_DEG:
                print(f"WARNING: {name} and /tf_static differ by more than {100 * KALIBR_TF_TOL_M:.0f} cm or "
                      f"{KALIBR_TF_TOL_DEG:.0f} deg. Check that the Kalibr camera is the {stream} camera and that "
                      "the IMU of the calibration is the united camera IMU (camera_imu_optical_frame)")
        return T, f"IMU.T_b_c1: Kalibr {os.path.abspath(a.kalibr)} ({a.kalibr_cam}), inverse of T_cam_imu."

    def report_T(name: str, T: np.ndarray) -> None:
        print(f"{name}: lever arm ({T[0, 3]:+.4f}, {T[1, 3]:+.4f}, {T[2, 3]:+.4f}) m, "
              f"rotation {rot_deg(T[:3, :3]):.2f} deg")

    if which in ("mono", "both"):
        if "color_info" not in c.info and not (kal and a.kalibr_intrinsics):
            raise SystemExit("no color camera_info: record or stream the color topic")
        cam = camera_from_info(c.info["color_info"]) if "color_info" in c.info else None
        src = f"camera_info of the color stream ({c.source})"
        if kal and a.kalibr_intrinsics:
            if "camera" not in kal:
                raise SystemExit("--kalibr-intrinsics needs a pinhole radtan camera in the Kalibr file")
            if cam is not None and (kal["camera"]["width"], kal["camera"]["height"]) != (cam["width"], cam["height"]):
                raise SystemExit(f"Kalibr resolution {kal['camera']['width']}x{kal['camera']['height']} differs "
                                 f"from the stream {cam['width']}x{cam['height']}")
            frame = cam["frame_id"] if cam else ""
            cam = dict(kal["camera"], frame_id=frame)
            src = f"Kalibr {os.path.abspath(a.kalibr)} ({a.kalibr_cam})"
        T, tsrc = extrinsic(cam["frame_id"], cam, "color")
        report_T("color camera in the IMU frame", T)
        print(f"color {cam['width']}x{cam['height']}: fx {cam['fx']:.2f} fy {cam['fy']:.2f} cx {cam['cx']:.2f} "
              f"cy {cam['cy']:.2f}, distortion {', '.join(f'{v:.5f}' for v in cam['dist'])}")
        prov = [f"Intrinsics and distortion: {src}.", tsrc,
                f"IMU.Frequency: sampling rate {hz:.2f} Hz measured on {c.imu_frame} ({rate['n']} messages, "
                f"requested {a.imu_rate:.0f} Hz).",
                f"IMU noise: D435i start values, white noise x {a.imu_noise_scale:g}"
                + (", overridden by arguments." if any(v is not None for v in (a.noise_gyro, a.noise_acc, a.gyro_walk,
                                                                                 a.acc_walk)) else ".")]
        if constants is None:
            print("WARNING: no --constants given. The mono file is written with UCSLAM.PriorNet 0 (depth prior OFF). "
                  "Run fit_depth_constants.py, then this script again with --constants.")
        else:
            print(f"depth constants {constants['values']} fitted with backend {constants['backend']}: "
                  f"launch with backend:={constants['backend']}")
        out[MONO_NAME] = build_mono(cfg, cam, T, hz, a.fps, noise, constants, prov, a.domain)
    if which in ("stereo", "both"):
        if "infra1_info" not in c.info or "infra2_info" not in c.info:
            raise SystemExit("no infra1/infra2 camera_info: record or stream mode:=stereo")
        cam = camera_from_info(c.info["infra1_info"])
        b = stereo_baseline(camera_from_info(c.info["infra2_info"]))
        if not 0.05 < b < 0.15:
            raise SystemExit(f"stereo baseline {b:.4f} m from infra2 camera_info is not a D455 baseline (0.095 m)")
        try:   # cross-check with /tf_static: pose of the infra2 optical frame in the infra1 optical frame
            T12 = tree.lookup(cam["frame_id"], cam["frame_id"].replace("infra1", "infra2"))
            print(f"tf check: infra2 in infra1 at ({T12[0, 3]:+.5f}, {T12[1, 3]:+.5f}, {T12[2, 3]:+.5f}) m")
            if abs(T12[0, 3] - b) > 0.002 or abs(T12[1, 3]) > 0.002 or abs(T12[2, 3]) > 0.002:
                print("WARNING: /tf_static and the infra2 camera_info disagree on the baseline by more than 2 mm")
        except KeyError:
            pass
        T, tsrc = extrinsic(cam["frame_id"], cam, "infra1")
        report_T("infra1 camera in the IMU frame", T)
        print(f"infra1 {cam['width']}x{cam['height']}: fx {cam['fx']:.2f}, baseline {b:.5f} m")
        prov = [f"Intrinsics: camera_info of infra1, Stereo.b from infra2 ({c.source}).", tsrc,
                f"IMU.Frequency: sampling rate {hz:.2f} Hz measured ({rate['n']} messages, requested "
                f"{a.imu_rate:.0f} Hz).",
                f"IMU noise: D435i start values, white noise x {a.imu_noise_scale:g}."]
        out[STEREO_NAME] = build_stereo(cfg, cam, b, T, hz, a.fps, noise, prov)
    if kal is not None:
        print(f"Kalibr timeshift_cam_imu {kal['timeshift_cam_imu']:+.5f} s: launch with "
              f"imu_time_offset:={-kal['timeshift_cam_imu']:.5f} (t_cam = t_imu - shift)")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bag", help="ROS 2 bag folder (or .mcap / .db3 file)")
    src.add_argument("--live", action="store_true", help="read the running camera driver")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds of IMU data used for the rate")
    ap.add_argument("--camera-namespace", default="camera")
    ap.add_argument("--camera-name", default="camera")
    ap.add_argument("--which", choices=("auto", "mono", "stereo", "both"), default="auto",
                    help="auto: every file whose streams are present")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--constants", help="YAML of fit_depth_constants.py. Without it the mono file has PriorNet 0")
    ap.add_argument("--domain", default=DOMAIN_COLOR, help="UCSLAM.Dataset of the mono file")
    ap.add_argument("--kalibr", help="Kalibr camchain-imucam YAML: IMU.T_b_c1 from it instead of /tf_static")
    ap.add_argument("--kalibr-cam", default="cam0")
    ap.add_argument("--kalibr-intrinsics", action="store_true", help="also take the mono intrinsics from Kalibr")
    ap.add_argument("--fps", type=int, default=30, help="Camera.fps, the stream rate")
    ap.add_argument("--imu-rate", type=float, default=200.0, help="the requested IMU rate (gyro_fps), for the check")
    ap.add_argument("--imu-noise-scale", type=float, default=1.0,
                    help="factor on NoiseGyro and NoiseAcc for propeller vibration (1.0 on the bench)")
    ap.add_argument("--noise-gyro", type=float, help="IMU.NoiseGyro from an Allan variance run [rad/s^0.5]")
    ap.add_argument("--noise-acc", type=float, help="IMU.NoiseAcc [m/s^1.5]")
    ap.add_argument("--gyro-walk", type=float, help="IMU.GyroWalk [rad/s^1.5]")
    ap.add_argument("--acc-walk", type=float, help="IMU.AccWalk [m/s^2.5]")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    tp = topics(a.camera_namespace, a.camera_name)
    if a.bag:
        c = collect_bag(a.bag, tp, a.duration)
    else:
        need = {"mono": ["color_info"], "stereo": ["infra1_info", "infra2_info"],
                "both": ["color_info", "infra1_info", "infra2_info"]}.get(a.which, [])
        c = collect_live(tp, a.duration, need)
    files = build_files(c, a)
    os.makedirs(a.out_dir, exist_ok=True)
    rc = 0
    for name, text in files.items():
        path = os.path.join(a.out_dir, name)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(text)
            tmp = f.name
        problems = check_settings(tmp)
        os.unlink(tmp)
        if problems:
            print(f"NOT written {path}: " + ", ".join(problems))
            rc = 1
            continue
        with open(path, "w") as f:
            f.write(text)
        prior = G.parse_ucslam_keys(text).get("PriorNet")
        print(f"wrote {path} (UCSLAM.Dataset {G.parse_ucslam_keys(text)['Dataset']}, PriorNet {prior})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
