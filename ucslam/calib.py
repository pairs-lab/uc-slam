"""Camera models and dataset calibration for UC-SLAM.

This module holds the two camera models the paper uses and the readers that
build them from the dataset files on disk. Nothing here is fitted. Every
number comes from the dataset calibration file.

Conventions
-----------
Pixels are always DISTORTED image pixels. This module never rectifies an
image. It maps points to the pixels the camera actually produced, which keeps
the noise model in the space where the covariance head predicts it.

A camera-frame point is Xc with the optical axis along +z. Projection is

    x = X / Z,  y = Y / Z            normalized pinhole coordinates
    (x_d, y_d) = distort(x, y)       model dependent
    u = fx * x_d + cx,  v = fy * y_d + cy

Two distortion models are supported.

``pinhole-radtan`` is the OpenCV radial-tangential model with
``dist = [k1, k2, p1, p2]``. With r2 = x^2 + y^2,

    x_d = x (1 + k1 r2 + k2 r2^2) + 2 p1 x y + p2 (r2 + 2 x^2)
    y_d = y (1 + k1 r2 + k2 r2^2) + p1 (r2 + 2 y^2) + 2 p2 x y

``pinhole-equidistant`` is the Kannala-Brandt fisheye model with
``dist = [k1, k2, k3, k4]``. With r = hypot(x, y) and theta = atan2(r, 1),

    theta_d = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
    x_d = (theta_d / r) x,  y_d = (theta_d / r) y

and the limit theta_d / r -> 1 as r -> 0.

Both models are cross-checked against OpenCV to 1e-9 in
``tests/test_calib.py``. The analytic Jacobian is checked against a central
finite difference.

Extrinsics
----------
``Camera.T_bc`` is body-from-camera, so X_body = T_bc @ X_cam. The body frame
is the IMU frame of the dataset, which is the frame the ground truth uses.

Constants
---------
The three numerical constants of this module live in the ``cameras:`` block of
``configs/default.yaml`` and are read at import time. They are z_min, the
tolerance of the undistortion solve, and its iteration cap.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import yaml

__all__ = [
    "PINHOLE_EQUIDISTANT",
    "PINHOLE_RADTAN",
    "Camera",
    "camera_constants",
    "fisheye_rectifier",
    "image_index",
    "invert_se3",
    "load_euroc_camera",
    "load_stereo",
    "load_tumvi_camera",
    "pinhole_intrinsics",
    "project",
    "project_jac",
    "unproject",
]

PINHOLE_RADTAN = "pinhole-radtan"
PINHOLE_EQUIDISTANT = "pinhole-equidistant"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG = os.path.join(_REPO_ROOT, "configs", "default.yaml")

# Fallback values, used only when the config file cannot be read. They are the
# same numbers as the ``cameras:`` block of configs/default.yaml, and
# tests/test_calib.py asserts that they still agree.
_CAMERA_FALLBACK = {
    "z_min": 1.0e-6,
    "unproject_tol_px": 1.0e-7,
    "unproject_max_iter": 20,
}


def camera_constants(path: str | None = None) -> dict:
    """Return the ``cameras:`` block of the parameter file.

    Missing keys and an unreadable file fall back to ``_CAMERA_FALLBACK``.
    """
    out = dict(_CAMERA_FALLBACK)
    try:
        with open(path or _DEFAULT_CONFIG, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return out
    block = cfg.get("cameras") or {}
    for key in out:
        if key in block:
            out[key] = block[key]
    return out


_CONST = camera_constants()
Z_MIN = float(_CONST["z_min"])
UNPROJECT_TOL_PX = float(_CONST["unproject_tol_px"])
UNPROJECT_MAX_ITER = int(_CONST["unproject_max_iter"])

# Below this normalized radius the equidistant model is evaluated with its
# series limit instead of the ratio theta_d / r.
_SMALL_R = 1.0e-8


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------

@dataclass
class Camera:
    """One calibrated camera.

    Attributes:
        model: ``"pinhole-radtan"`` or ``"pinhole-equidistant"``.
        fx, fy, cx, cy: pinhole intrinsics in pixels.
        dist: (4,) distortion, [k1,k2,p1,p2] for radtan, [k1,k2,k3,k4] for
            equidistant.
        width, height: image size in pixels.
        T_bc: (4,4) homogeneous body-from-camera transform.
        name: free text, for example ``"euroc/cam0"``.
    """

    model: str
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray
    width: int
    height: int
    T_bc: np.ndarray
    name: str = ""

    def __post_init__(self):
        if self.model not in (PINHOLE_RADTAN, PINHOLE_EQUIDISTANT):
            raise ValueError(f"unknown camera model {self.model!r}")
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.width = int(self.width)
        self.height = int(self.height)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        if self.dist.shape != (4,):
            raise ValueError("dist must have 4 entries")
        self.T_bc = np.asarray(self.T_bc, dtype=np.float64)
        if self.T_bc.shape != (4, 4):
            raise ValueError("T_bc must be 4x4")

    @property
    def K(self) -> np.ndarray:
        """(3,3) pinhole intrinsic matrix. It ignores the distortion."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def T_cb(self) -> np.ndarray:
        """(4,4) camera-from-body transform, the inverse of T_bc."""
        return invert_se3(self.T_bc)


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform, using the rotation transpose."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


# --------------------------------------------------------------------------
# Distortion, forward
# --------------------------------------------------------------------------

def _normalize(cam: Camera, Xc: np.ndarray):
    """Normalized coordinates and the mask of points that cannot be projected."""
    Xc = np.atleast_2d(np.asarray(Xc, dtype=np.float64))
    if Xc.ndim != 2 or Xc.shape[1] != 3:
        raise ValueError("Xc must have shape (N,3)")
    z = Xc[:, 2]
    with np.errstate(invalid="ignore"):
        bad = ~(z > Z_MIN)          # also True for NaN
    zs = np.where(bad, 1.0, z)
    return Xc[:, 0] / zs, Xc[:, 1] / zs, zs, bad


def _distort_radtan(x, y, d):
    k1, k2, p1, p2 = d
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def _distort_equidistant(x, y, d):
    k1, k2, k3, k4 = d
    r = np.hypot(x, y)
    theta = np.arctan2(r, 1.0)
    t2 = theta * theta
    theta_d = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = np.where(r > _SMALL_R, theta_d / np.where(r > _SMALL_R, r, 1.0), 1.0)
    return scale * x, scale * y


def _distort(cam: Camera, x, y):
    if cam.model == PINHOLE_RADTAN:
        return _distort_radtan(x, y, cam.dist)
    return _distort_equidistant(x, y, cam.dist)


def project(cam: Camera, Xc: np.ndarray) -> np.ndarray:
    """Project camera-frame points to distorted pixels.

    Args:
        cam: the camera.
        Xc: (N,3) points in the camera frame. A single (3,) point is accepted
            and returns a (1,2) array.

    Returns:
        (N,2) distorted pixel coordinates. Rows whose z is not above
        ``Z_MIN`` are NaN, and NaN inputs stay NaN.
    """
    x, y, _, bad = _normalize(cam, Xc)
    xd, yd = _distort(cam, x, y)
    uv = np.stack([cam.fx * xd + cam.cx, cam.fy * yd + cam.cy], axis=1)
    uv[bad] = np.nan
    return uv


# --------------------------------------------------------------------------
# Distortion, Jacobian
# --------------------------------------------------------------------------

def _distort_jac_radtan(x, y, d):
    """(N,2,2) d(x_d,y_d)/d(x,y) for the radial-tangential model."""
    k1, k2, p1, p2 = d
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    # d(radial)/d(r2) times d(r2)/dx and d(r2)/dy
    drad = k1 + 2.0 * k2 * r2
    drad_dx = 2.0 * x * drad
    drad_dy = 2.0 * y * drad
    J = np.empty((x.shape[0], 2, 2))
    J[:, 0, 0] = radial + x * drad_dx + 2.0 * p1 * y + 6.0 * p2 * x
    J[:, 0, 1] = x * drad_dy + 2.0 * p1 * x + 2.0 * p2 * y
    J[:, 1, 0] = y * drad_dx + 2.0 * p1 * x + 2.0 * p2 * y
    J[:, 1, 1] = radial + y * drad_dy + 6.0 * p1 * y + 2.0 * p2 * x
    return J


def _distort_jac_equidistant(x, y, d):
    """(N,2,2) d(x_d,y_d)/d(x,y) for the equidistant model.

    With s = theta_d / r the distorted point is (s x, s y). The derivative of
    s with respect to r divided by r is written c below, which keeps the
    formula finite at r = 0:

        d x_d / dx = s + c x^2,  d x_d / dy = c x y,  d y_d / dy = s + c y^2

    The limit at r -> 0 is s = 1 and c = 2 k1 - 2/3, obtained from the series
    theta_d = r + (k1 - 1/3) r^3 + O(r^5).
    """
    k1, k2, k3, k4 = d
    r = np.hypot(x, y)
    theta = np.arctan2(r, 1.0)
    t2 = theta * theta
    poly = 1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))
    theta_d = theta * poly
    dtd_dtheta = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
    dtheta_dr = 1.0 / (1.0 + r * r)

    big = r > _SMALL_R
    rs = np.where(big, r, 1.0)
    s = np.where(big, theta_d / rs, 1.0)
    c = np.where(big,
                 (dtd_dtheta * dtheta_dr * rs - theta_d) / (rs * rs * rs),
                 2.0 * k1 - 2.0 / 3.0)

    J = np.empty((x.shape[0], 2, 2))
    J[:, 0, 0] = s + c * x * x
    J[:, 0, 1] = c * x * y
    J[:, 1, 0] = c * x * y
    J[:, 1, 1] = s + c * y * y
    return J


def project_jac(cam: Camera, Xc: np.ndarray) -> np.ndarray:
    """Analytic d(pixel)/d(Xc).

    Args:
        cam: the camera.
        Xc: (N,3) points in the camera frame.

    Returns:
        (N,2,3) Jacobian of ``project``. Rows whose z is not above ``Z_MIN``
        are NaN. It matches a central finite difference of ``project`` to
        1e-6 relative, which is checked in tests/test_calib.py.
    """
    x, y, z, bad = _normalize(cam, Xc)
    if cam.model == PINHOLE_RADTAN:
        Jd = _distort_jac_radtan(x, y, cam.dist)
    else:
        Jd = _distort_jac_equidistant(x, y, cam.dist)

    n = x.shape[0]
    Jn = np.zeros((n, 2, 3))          # d(x,y)/d(Xc)
    Jn[:, 0, 0] = 1.0 / z
    Jn[:, 0, 2] = -x / z
    Jn[:, 1, 1] = 1.0 / z
    Jn[:, 1, 2] = -y / z

    J = np.einsum("nij,njk->nik", Jd, Jn)
    J[:, 0, :] *= cam.fx
    J[:, 1, :] *= cam.fy
    J[bad] = np.nan
    return J


# --------------------------------------------------------------------------
# Distortion, inverse
# --------------------------------------------------------------------------

def _undistort_radtan(xd, yd, d, max_iter: int, tol: float):
    """Newton solve of distort(x,y) = (xd,yd) in normalized coordinates.

    Returns the solution and the final absolute residual.
    """
    x = xd.copy()
    y = yd.copy()
    for _ in range(max_iter):
        fx_, fy_ = _distort_radtan(x, y, d)
        ex = fx_ - xd
        ey = fy_ - yd
        if np.nanmax(np.abs(np.stack([ex, ey]))) < tol:
            break
        J = _distort_jac_radtan(x, y, d)
        a, b = J[:, 0, 0], J[:, 0, 1]
        c, e = J[:, 1, 0], J[:, 1, 1]
        det = a * e - b * c
        ok = np.abs(det) > 1e-12
        dets = np.where(ok, det, 1.0)
        dx = np.where(ok, (e * ex - b * ey) / dets, 0.0)
        dy = np.where(ok, (a * ey - c * ex) / dets, 0.0)
        x = x - dx
        y = y - dy
    fx_, fy_ = _distort_radtan(x, y, d)
    res = np.maximum(np.abs(fx_ - xd), np.abs(fy_ - yd))
    return x, y, res


def _solve_equidistant_theta(theta_d, d, max_iter: int, tol: float):
    """Newton solve of theta (1 + k1 th^2 + ... ) = theta_d for theta >= 0."""
    k1, k2, k3, k4 = d
    theta = theta_d.copy()
    for _ in range(max_iter):
        t2 = theta * theta
        g = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
        if np.nanmax(np.abs(g)) < tol:
            break
        dg = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
        ok = np.abs(dg) > 1e-12
        theta = np.where(ok, theta - g / np.where(ok, dg, 1.0), theta)
        theta = np.clip(theta, 0.0, np.pi)
    t2 = theta * theta
    res = np.abs(theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d)
    return theta, res


def unproject(cam: Camera, uv: np.ndarray, max_iter: int = UNPROJECT_MAX_ITER) -> np.ndarray:
    """Turn distorted pixels into unit bearing vectors in the camera frame.

    The distortion is inverted with Newton iterations in normalized
    coordinates. The radial-tangential case solves the 2-D distortion map. The
    equidistant case solves the scalar polynomial for theta, which is exact
    for every field angle up to pi, including the rays behind the image plane
    that a 195 degree fisheye still sees.

    Args:
        cam: the camera.
        uv: (N,2) distorted pixels. A single (2,) pixel is accepted.
        max_iter: Newton iteration cap.

    Returns:
        (N,3) unit bearings. A row is NaN when its input is not finite, and
        when the residual of the solve stays above ten times the tolerance
        after ``max_iter`` steps. For a bearing with a positive z the round
        trip project(cam, bearing * z) reproduces the pixel to better than
        1e-3 px, which is measured on both real cameras in
        tests/test_calib.py.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N,2)")
    if uv.shape[0] == 0:
        return np.zeros((0, 3))
    xd = (uv[:, 0] - cam.cx) / cam.fx
    yd = (uv[:, 1] - cam.cy) / cam.fy
    tol = UNPROJECT_TOL_PX / max(cam.fx, cam.fy)

    finite = np.isfinite(xd) & np.isfinite(yd)
    xd = np.where(finite, xd, 0.0)
    yd = np.where(finite, yd, 0.0)

    if cam.model == PINHOLE_RADTAN:
        x, y, res = _undistort_radtan(xd, yd, cam.dist, max_iter, tol)
        b = np.stack([x, y, np.ones_like(x)], axis=1)
    else:
        theta_d = np.hypot(xd, yd)
        theta, res = _solve_equidistant_theta(theta_d, cam.dist, max_iter, tol)
        big = theta_d > _SMALL_R
        scale = np.where(big, np.sin(theta) / np.where(big, theta_d, 1.0), 1.0)
        b = np.stack([scale * xd, scale * yd, np.cos(theta)], axis=1)

    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    accept = 10.0 * max(tol, 1e-12)
    bad = (~finite) | ~(res < accept) | ~np.isfinite(b).all(axis=1)
    b[bad] = np.nan
    return b


# --------------------------------------------------------------------------
# Dataset readers
# --------------------------------------------------------------------------

def _read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_euroc_camera(seq_dir: str, which: str) -> Camera:
    """Build a Camera from an EuRoC ``mav0/<which>/sensor.yaml``.

    Args:
        seq_dir: sequence directory, for example ``data/euroc/MH_01_easy``.
        which: ``"cam0"`` or ``"cam1"``.

    Returns:
        A ``pinhole-radtan`` Camera with T_bc taken straight from T_BS, which
        EuRoC defines as body-from-sensor.
    """
    if which not in ("cam0", "cam1"):
        raise ValueError("which must be cam0 or cam1")
    path = os.path.join(seq_dir, "mav0", which, "sensor.yaml")
    y = _read_yaml(path)
    if y.get("camera_model") != "pinhole":
        raise ValueError(f"{path}: unexpected camera_model {y.get('camera_model')!r}")
    if y.get("distortion_model") != "radial-tangential":
        raise ValueError(f"{path}: unexpected distortion_model {y.get('distortion_model')!r}")
    fu, fv, cu, cv = [float(v) for v in y["intrinsics"]]
    dist = np.asarray(y["distortion_coefficients"], dtype=np.float64)
    w, h = [int(v) for v in y["resolution"]]
    T_bc = np.asarray(y["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    return Camera(PINHOLE_RADTAN, fu, fv, cu, cv, dist, w, h, T_bc,
                  name=f"euroc/{which}")


def load_tumvi_camera(seq_dir: str, which: str) -> Camera:
    """Build a Camera from a TUM-VI ``dso/camchain.yaml``.

    Args:
        seq_dir: sequence directory, for example ``data/tumvi/room1``.
        which: ``"cam0"`` or ``"cam1"``.

    Returns:
        A ``pinhole-equidistant`` Camera. The chain stores T_cam_imu, which is
        camera-from-body, so T_bc is its inverse.
    """
    if which not in ("cam0", "cam1"):
        raise ValueError("which must be cam0 or cam1")
    path = os.path.join(seq_dir, "dso", "camchain.yaml")
    y = _read_yaml(path)[which]
    if y.get("camera_model") != "pinhole":
        raise ValueError(f"{path}: unexpected camera_model {y.get('camera_model')!r}")
    if y.get("distortion_model") != "equidistant":
        raise ValueError(f"{path}: unexpected distortion_model {y.get('distortion_model')!r}")
    fx, fy, cx, cy = [float(v) for v in y["intrinsics"]]
    dist = np.asarray(y["distortion_coeffs"], dtype=np.float64)
    w, h = [int(v) for v in y["resolution"]]
    T_cb = np.asarray(y["T_cam_imu"], dtype=np.float64)
    return Camera(PINHOLE_EQUIDISTANT, fx, fy, cx, cy, dist, w, h,
                  invert_se3(T_cb), name=f"tumvi/{which}")


def load_stereo(seq_dir: str, dataset: str) -> tuple[Camera, Camera, np.ndarray]:
    """Load both cameras of a sequence and the stereo extrinsic.

    Args:
        seq_dir: sequence directory.
        dataset: ``"euroc"`` or ``"tumvi"``.

    Returns:
        (cam0, cam1, T_c1c0) with T_c1c0 the camera1-from-camera0 transform,
        so X_cam1 = T_c1c0 @ X_cam0. EuRoC derives it from the two T_BS.
        TUM-VI takes it from the camchain entry T_cn_cnm1 of cam1.

    The measured baseline is 0.11008 m on EuRoC MH_01_easy and 0.10109 m on
    TUM-VI room1.
    """
    if dataset == "euroc":
        cam0 = load_euroc_camera(seq_dir, "cam0")
        cam1 = load_euroc_camera(seq_dir, "cam1")
        T_c1c0 = invert_se3(cam1.T_bc) @ cam0.T_bc
    elif dataset == "tumvi":
        cam0 = load_tumvi_camera(seq_dir, "cam0")
        cam1 = load_tumvi_camera(seq_dir, "cam1")
        chain = _read_yaml(os.path.join(seq_dir, "dso", "camchain.yaml"))
        T_c1c0 = np.asarray(chain["cam1"]["T_cn_cnm1"], dtype=np.float64)
    else:
        raise ValueError("dataset must be euroc or tumvi")
    return cam0, cam1, T_c1c0


def image_index(seq_dir: str, which: str) -> tuple[np.ndarray, list[str]]:
    """Read ``mav0/<which>/data.csv``.

    Args:
        seq_dir: sequence directory.
        which: ``"cam0"`` or ``"cam1"``.

    Returns:
        (t_ns, paths) with t_ns an int64 array of nanosecond timestamps sorted
        ascending, and paths the matching absolute image paths. Comment lines
        and blank lines are skipped. Both datasets use the same layout.
    """
    cam_dir = os.path.join(seq_dir, "mav0", which)
    csv_path = os.path.join(cam_dir, "data.csv")
    stamps: list[int] = []
    names: list[str] = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            stamps.append(int(parts[0]))
            names.append(parts[1].strip())
    t = np.asarray(stamps, dtype=np.int64)
    order = np.argsort(t, kind="stable")
    t = t[order]
    data_dir = os.path.abspath(os.path.join(cam_dir, "data"))
    paths = [os.path.join(data_dir, names[i]) for i in order]
    return t, paths


# ------------------------------------------------------------ rectifier --

def pinhole_intrinsics(fov_deg: float, size) -> np.ndarray:
    """K of a square-pixel pinhole view with the given field of view.

    The focal length is ``(w / 2) / tan(fov / 2)`` so that the horizontal
    field of view is ``fov_deg``. The principal point is the centre of the
    pixel grid, ``((w - 1) / 2, (h - 1) / 2)``.
    """
    w, h = int(size[0]), int(size[1])
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[f, 0.0, (w - 1) / 2.0],
                     [0.0, f, (h - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def fisheye_rectifier(cam, fov_deg: float, size) -> dict:
    """Rectification maps from an equidistant fisheye camera to a pinhole view.

    Uses ``cv2.fisheye.initUndistortRectifyMap`` with ``R = I``, so the
    rectified view shares the optical axis of the fisheye camera. The TUM-VI
    equidistant model and the OpenCV fisheye model are the same polynomial.

    Returns:
        dict with ``K`` (3,3) of the rectified view, ``map1`` and ``map2``
        for ``cv2.remap``, ``size`` (w, h), ``fov_deg`` and ``f``.
    """
    import cv2
    if cam.model != PINHOLE_EQUIDISTANT:
        raise ValueError("fisheye_rectifier expects a pinhole-equidistant camera")
    K_new = pinhole_intrinsics(fov_deg, size)
    w, h = int(size[0]), int(size[1])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        cam.K, cam.dist.reshape(4, 1), np.eye(3), K_new, (w, h), cv2.CV_32FC1)
    return {"K": K_new, "map1": map1, "map2": map2, "size": (w, h),
            "fov_deg": float(fov_deg), "f": float(K_new[0, 0])}
