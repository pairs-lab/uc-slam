"""Ground-truth trajectory loading and interpolation for UC-SLAM.

Both supported datasets publish the pose of the body (IMU) frame in a world
frame. EuRoC writes it to ``mav0/state_groundtruth_estimate0/data.csv`` and
TUM-VI writes it to ``dso/gt_imu.csv``. Both files store the quaternion in
``w, x, y, z`` order. This module converts to ``x, y, z, w`` on load and keeps
that order everywhere afterwards.

Conventions:
  - A pose is world-from-body. ``X_world = R @ X_body + p``.
  - ``Trajectory.q`` is a unit quaternion in ``[x, y, z, w]`` order.
  - Timestamps are integer nanoseconds and are sorted.
  - A homogeneous transform ``T_ba`` maps frame-a points into frame-b.

Validity rule. A query time is valid when it lies inside the closed interval
between the first and the last sample, and when it does not fall strictly
inside a sampling gap longer than ``max_gap_ns``. A query that lands exactly
on a stored sample is always valid, even at the edge of a gap, because no
interpolation happens there. Invalid queries return NaN.

The default gap threshold comes from ``configs/default.yaml``, key
``trajectory.max_gap_ns``. Nothing in this module hard-codes it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

__all__ = [
    "DEFAULT_MAX_GAP_NS",
    "Trajectory",
    "camera_pose",
    "image_timestamps",
    "load_trajectory",
    "pose_at",
    "quat_to_rot",
    "relative_pose",
    "valid_mask",
]


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "default.yaml")


def _trajectory_config(path: str | None = None) -> dict:
    """Read the ``trajectory`` block of the parameter file."""
    with open(path or _CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)
    if "trajectory" not in cfg:
        raise KeyError(f"{path or _CONFIG_PATH}: missing the 'trajectory' section")
    return cfg["trajectory"]


_CFG = _trajectory_config()

#: Default sampling gap above which interpolation is refused, in nanoseconds.
DEFAULT_MAX_GAP_NS: int = int(_CFG["max_gap_ns"])

#: Ground-truth file of each dataset, relative to the sequence directory.
GT_REL = {"euroc": str(_CFG["euroc_gt_rel"]), "tumvi": str(_CFG["tumvi_gt_rel"])}

#: Image index file of a camera, relative to the sequence directory.
_IMAGE_CSV_REL = str(_CFG["image_csv_rel"])


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A sampled world-from-body trajectory.

    Attributes:
        t_ns: (M,) int64 timestamps in nanoseconds, strictly increasing.
        p: (M,3) float64 world-from-body translation in metres.
        q: (M,4) float64 unit quaternion in ``[x, y, z, w]`` order.
        name: free text, normally ``"<dataset>/<sequence>"``.
    """

    t_ns: np.ndarray
    p: np.ndarray
    q: np.ndarray
    name: str = ""

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])

    @property
    def duration_s(self) -> float:
        """Length of the trajectory in seconds."""
        if len(self) < 2:
            return 0.0
        return float(self.t_ns[-1] - self.t_ns[0]) * 1e-9


# --------------------------------------------------------------------------
# Quaternion helpers
# --------------------------------------------------------------------------

def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Return unit quaternions. A zero quaternion becomes the identity."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    bad = n[..., 0] <= 1e-12
    out = np.where(n > 1e-12, q / np.where(n > 1e-12, n, 1.0), 0.0)
    if np.any(bad):
        out[bad] = np.array([0.0, 0.0, 0.0, 1.0])
    return out


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrices from ``[x, y, z, w]`` quaternions, shape (...,4) to (...,3,3)."""
    q = _normalize_quat(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Shortest-arc SLERP between two quaternion arrays, shape (N,4) with alpha (N,).

    The second quaternion is negated where the dot product is negative, so the
    interpolation always takes the short way round. When the two quaternions are
    nearly identical the sine ratio is ill conditioned, so the routine falls back
    to a normalized linear blend. The angular error of that fallback is below
    1e-13 rad at the switch point, far below the 1e-9 rad tolerance of the tests.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where(dot[:, None] < 0.0, -q1, q1)
    d = np.clip(np.abs(dot), -1.0, 1.0)

    near = d > 1.0 - 1e-9
    s0 = np.where(near, 1.0 - alpha, 0.0)
    s1 = np.where(near, alpha, 0.0)
    far = ~near
    if np.any(far):
        theta = np.arccos(d[far])
        sin_t = np.sin(theta)
        s0[far] = np.sin((1.0 - alpha[far]) * theta) / sin_t
        s1[far] = np.sin(alpha[far] * theta) / sin_t

    q = s0[:, None] * q0 + s1[:, None] * q1
    return _normalize_quat(q)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _read_stamps(path: str) -> np.ndarray:
    """Read the first column of a csv as exact int64 nanoseconds.

    A nanosecond stamp near 1.5e18 does not fit in the 53-bit mantissa of a
    float64, so parsing it as a float loses up to 128 ns. TUM-VI stamps are hit
    by this on every row. The column is therefore parsed as an integer.
    """
    return np.loadtxt(path, delimiter=",", comments="#", usecols=(0,),
                      dtype=np.int64, ndmin=1)


def _read_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a ground-truth csv, sort it, drop duplicate stamps.

    Returns ``(t_ns (M,) int64, cols (M,C-1) float64)`` where ``cols`` holds
    every column after the timestamp. Duplicate timestamps keep the first row.
    """
    t = _read_stamps(path)
    A = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if A.shape[0] == 0:
        raise ValueError(f"{path}: no rows")
    if A.shape[0] != t.shape[0]:
        raise ValueError(f"{path}: {A.shape[0]} rows but {t.shape[0]} timestamps")
    order = np.argsort(t, kind="stable")
    t, A = t[order], A[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], np.ascontiguousarray(A[keep, 1:], dtype=np.float64)


def load_trajectory(seq_dir: str, dataset: str) -> Trajectory:
    """Load the ground-truth body trajectory of one sequence.

    EuRoC reads ``mav0/state_groundtruth_estimate0/data.csv`` whose columns are
    timestamp, p(3), q_wxyz(4), then velocity and biases. TUM-VI reads
    ``dso/gt_imu.csv`` whose columns are timestamp, t(3), q_wxyz(4). Both give
    the body (IMU) frame in the world frame. Rows are sorted by timestamp and
    duplicate timestamps are dropped, keeping the first of each group.
    Quaternions are normalized and stored as ``[x, y, z, w]``.

    Args:
        seq_dir: sequence directory, for example ``data/euroc/MH_01_easy``.
        dataset: ``"euroc"`` or ``"tumvi"``.

    Returns:
        A Trajectory with at least one sample.
    """
    ds = str(dataset).lower()
    if ds not in GT_REL:
        raise ValueError(f"dataset must be one of {sorted(GT_REL)}, got {dataset!r}")
    path = os.path.join(seq_dir, GT_REL[ds])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    t_ns, A = _read_csv(path)
    if A.shape[1] < 7:
        raise ValueError(f"{path}: expected at least 8 columns, got {A.shape[1] + 1}")
    p = np.ascontiguousarray(A[:, 0:3], dtype=np.float64)
    q_wxyz = A[:, 3:7]
    q = _normalize_quat(np.stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]], axis=1))
    name = f"{ds}/{os.path.basename(os.path.abspath(seq_dir.rstrip(os.sep)))}"
    return Trajectory(t_ns=t_ns, p=p, q=q, name=name)


# --------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------

def _as_query(t_ns) -> tuple[np.ndarray, bool, tuple]:
    """Normalize a query to a flat int64 array. Returns (flat, was_scalar, shape)."""
    t = np.asarray(t_ns)
    shape = t.shape
    if t.dtype.kind == "f":
        t = np.rint(t)
    t = t.astype(np.int64, copy=False)
    return np.atleast_1d(t).reshape(-1), (len(shape) == 0), shape


def _bracket(traj: Trajectory, tq: np.ndarray, max_gap_ns: int):
    """Bracketing index, interpolation weight and validity flag for flat queries."""
    t = traj.t_ns
    M = t.shape[0]
    if M == 0:
        raise ValueError("empty trajectory")
    if M == 1:
        idx = np.zeros(tq.shape, dtype=np.intp)
        alpha = np.zeros(tq.shape, dtype=np.float64)
        return idx, alpha, (tq == t[0])

    idx = np.searchsorted(t, tq, side="right") - 1
    in_range = (tq >= t[0]) & (tq <= t[-1])
    idx = np.clip(idx, 0, M - 2)
    t0 = t[idx]
    dt = (t[idx + 1] - t0).astype(np.float64)
    alpha = (tq - t0).astype(np.float64) / dt
    strictly_inside = (alpha > 0.0) & (alpha < 1.0)
    ok = in_range & ~(strictly_inside & (dt > float(max_gap_ns)))
    return idx, alpha, ok


def pose_at(traj: Trajectory, t_ns, max_gap_ns: int = DEFAULT_MAX_GAP_NS):
    """Interpolate the world-from-body pose at one or many times.

    Position is interpolated linearly. Rotation is interpolated with
    shortest-arc SLERP. A query that lands exactly on a sample returns that
    sample bit for bit.

    Args:
        traj: the trajectory.
        t_ns: a scalar or an array of nanosecond timestamps. A floating point
            query is rounded to the nearest nanosecond.
        max_gap_ns: sampling gaps longer than this are not interpolated.

    Returns:
        ``(R, p)`` with R of shape ``(...,3,3)`` and p of shape ``(...,3)``,
        world-from-body. Both are NaN where the query lies outside the
        trajectory or strictly inside a gap longer than ``max_gap_ns``. A
        scalar query returns ``(3,3)`` and ``(3,)``.
    """
    tq, was_scalar, shape = _as_query(t_ns)
    idx, alpha, ok = _bracket(traj, tq, max_gap_ns)

    j1 = np.minimum(idx + 1, len(traj) - 1)
    p = (1.0 - alpha)[:, None] * traj.p[idx] + alpha[:, None] * traj.p[j1]
    q = _slerp(traj.q[idx], traj.q[j1], alpha)
    R = quat_to_rot(q)

    bad = ~ok
    if np.any(bad):
        R[bad] = np.nan
        p[bad] = np.nan

    if was_scalar:
        return R[0], p[0]
    return R.reshape(shape + (3, 3)), p.reshape(shape + (3,))


def valid_mask(traj: Trajectory, t_ns, max_gap_ns: int = DEFAULT_MAX_GAP_NS) -> np.ndarray:
    """True where :func:`pose_at` returns a finite pose.

    The shape follows the query. A scalar query returns a 0-d boolean array.
    """
    tq, was_scalar, shape = _as_query(t_ns)
    _, _, ok = _bracket(traj, tq, max_gap_ns)
    if was_scalar:
        return np.asarray(ok[0], dtype=bool)
    return ok.reshape(shape)


def camera_pose(traj: Trajectory, t_ns, cam, max_gap_ns: int = DEFAULT_MAX_GAP_NS) -> np.ndarray:
    """World-from-camera pose, ``T_wc = T_wb @ cam.T_bc``.

    Args:
        traj: the trajectory.
        t_ns: a scalar or an array of nanosecond timestamps.
        cam: any object with a ``T_bc`` attribute holding the (4,4) body-from-camera
            transform, for example ``ucslam.calib.Camera``.
        max_gap_ns: sampling gaps longer than this are not interpolated.

    Returns:
        ``(4,4)`` for a scalar query, otherwise ``(...,4,4)``. Rows that have no
        valid ground truth are all NaN.
    """
    T_bc = np.asarray(cam.T_bc, dtype=np.float64)
    if T_bc.shape != (4, 4):
        raise ValueError(f"cam.T_bc must be (4,4), got {T_bc.shape}")

    R, p = pose_at(traj, t_ns, max_gap_ns=max_gap_ns)
    if R.ndim == 2:
        T_wb = np.eye(4)
        T_wb[:3, :3] = R
        T_wb[:3, 3] = p
        return T_wb @ T_bc

    lead = R.shape[:-2]
    T_wb = np.zeros(lead + (4, 4), dtype=np.float64)
    T_wb[..., :3, :3] = R
    T_wb[..., :3, 3] = p
    T_wb[..., 3, 3] = 1.0
    T_wc = T_wb @ T_bc
    bad = ~np.isfinite(R).all(axis=(-2, -1))
    if np.any(bad):
        T_wc[bad] = np.nan
    return T_wc


def relative_pose(T_wa: np.ndarray, T_wb: np.ndarray) -> np.ndarray:
    """Return ``T_ba = inverse(T_wb) @ T_wa``, the transform taking frame-a points to frame-b.

    The inverse uses the rigid form ``R^T`` and ``-R^T t`` instead of a general
    matrix inverse, so a round trip is exact to machine precision. Both
    arguments may carry leading batch dimensions and are broadcast together.
    """
    A = np.asarray(T_wa, dtype=np.float64)
    B = np.asarray(T_wb, dtype=np.float64)
    if A.shape[-2:] != (4, 4) or B.shape[-2:] != (4, 4):
        raise ValueError(f"expected (...,4,4) transforms, got {A.shape} and {B.shape}")

    Rb = B[..., :3, :3]
    tb = B[..., :3, 3]
    Rb_inv = np.swapaxes(Rb, -1, -2)
    B_inv = np.zeros(B.shape, dtype=np.float64)
    B_inv[..., :3, :3] = Rb_inv
    B_inv[..., :3, 3] = -np.einsum("...ij,...j->...i", Rb_inv, tb)
    B_inv[..., 3, 3] = 1.0
    return B_inv @ A


# --------------------------------------------------------------------------
# Small convenience used by the corpus builder and by the tests
# --------------------------------------------------------------------------

def image_timestamps(seq_dir: str, which: str = "cam0") -> np.ndarray:
    """Sorted nanosecond timestamps of one camera, read from ``mav0/<which>/data.csv``.

    This is a thin reader kept here so trajectory coverage can be measured
    without importing the calibration module. ``ucslam.calib.image_index``
    returns the same timestamps together with the image paths.
    """
    path = os.path.join(seq_dir, _IMAGE_CSV_REL.format(which=which))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.unique(_read_stamps(path))
