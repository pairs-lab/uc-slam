"""Stereo landmark harvesting for the real-data covariance gate (plan G-2).

The covariance head needs training targets that come from real images, not
from a synthetic homography. A calibrated stereo rig gives them. The pair
needs no rectification, only a known relative pose. Two views of
the same instant see the same scene from a known relative pose, so a match
that survives the epipolar test yields a metric 3-D point together with a
first-order covariance of that point. Transferring the point and its
covariance into a later frame produces a predicted 2-D observation with a
predicted 2x2 covariance, which is exactly what the head must learn.

Pipeline, in the order the functions appear below:

    match_descriptors    64-D descriptors of view 0 and view 1 to index pairs
    epipolar_residual    angular Sampson error of each pair, in radians
    triangulate          midpoint of the common perpendicular, plus parallax
    triangulation_cov    first-order 3x3 covariance of the triangulated point
    transfer_cov         move a point and its covariance into another frame
    stereo_landmarks     the whole chain with the configured thresholds

Conventions. Every pose is a 4x4 homogeneous matrix. ``T_c1c0`` maps cam0
points to cam1 points, so ``X_c1 = R_c1c0 @ X_c0 + t_c1c0``. Every 3-D point
returned by this module lives in the cam0 frame unless the name says
otherwise. Bearings are unit vectors in a camera frame. Pixels are always
DISTORTED pixels, which is what the detector reports.

Camera models come from ``ucslam.calib``. That module is imported lazily so
that ``match_descriptors``, which needs no camera, works on its own.

Every gate threshold lives in the ``stereo:`` block of ``configs/default.yaml``
and is read from there. The only fixed numbers in the code are the descriptor
width ``DESC_DIM``, which must match ``backbone.desc_dim``, and the 1e-12 floor
that flags a pair of parallel rays in ``triangulate``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import yaml

from ucslam.geometry import hat

log = logging.getLogger(__name__)

DEFAULT_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))

DESC_DIM = 64                      # must equal backbone.desc_dim (XFeat: 64)

_CALIB = None


def _calib():
    """Import ``ucslam.calib`` on first use and cache the module."""
    global _CALIB
    if _CALIB is None:
        from ucslam import calib as _c
        _CALIB = _c
    return _CALIB


def load_stereo_config(path: str | None = None) -> dict:
    """Read the ``stereo:`` block of the parameter file.

    Args:
        path: YAML file to read. Defaults to ``configs/default.yaml``.

    Returns:
        The ``stereo`` mapping. Raises KeyError when the block is absent.
    """
    with open(path or DEFAULT_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    if "stereo" not in cfg:
        raise KeyError("config is missing the 'stereo' section")
    return cfg["stereo"]


# --------------------------------------------------------------- matching --

def match_descriptors(f0, f1, ratio: float = 0.8, mutual: bool = True):
    """Match two sets of backbone features by their descriptors.

    ``f0`` (N,65) and ``f1`` (M,65) are the feature tensors of
    ``backbone.detect``. Only the first ``DESC_DIM`` columns are used, which
    drops the detection score in the last column. Those columns are already
    L2 normalized, so the squared Euclidean distance between descriptors a and
    b is ``2 - 2 a.b`` and the whole distance matrix is one matrix product.
    Ranking by largest inner product is therefore the same as ranking by
    smallest distance, and this function never forms the distance matrix.

    Two tests select a pair (i, j):
      * mutual nearest neighbour. j is the nearest descriptor in view 1 to i
        and i is the nearest descriptor in view 0 to j. Skipped when
        ``mutual`` is False.
      * ratio test on DISTANCES, not on squared distances. The nearest
        distance from i must be at most ``ratio`` times the second nearest
        distance from i. The test is applied on the view-0 side only, because
        mutual nearest neighbour already constrains the view-1 side. It is
        skipped when view 1 holds fewer than two descriptors.

    Memory. The similarity matrix is computed in row chunks whenever N*M
    exceeds ``stereo.match_chunk_entries`` from the config, so peak memory
    stays near that many floats no matter how many keypoints are passed. The
    work runs on whatever device the tensors are on, GPU included.

    Args:
        f0: (N,65) torch tensor.
        f1: (M,65) torch tensor, same dtype and device as ``f0``.
        ratio: nearest over second nearest distance ratio. 1.0 or more
            disables the test.
        mutual: require mutual nearest neighbour.

    Returns:
        (idx0, idx1), two numpy int64 arrays of length K. ``idx0`` indexes
        rows of ``f0`` and is strictly increasing. ``idx1`` indexes rows of
        ``f1``. They are numpy because every geometric function downstream is
        numpy.
    """
    max_entries = int(_MATCH_CHUNK_ENTRIES)
    d0 = torch.as_tensor(f0)[:, :DESC_DIM]
    d1 = torch.as_tensor(f1)[:, :DESC_DIM]
    n, m = int(d0.shape[0]), int(d1.shape[0])
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if n == 0 or m == 0:
        return empty
    d1 = d1.to(device=d0.device, dtype=d0.dtype)

    chunk = n if n * m <= max_entries else max(1, max_entries // m)
    dev, dt = d0.device, d0.dtype
    nn1 = torch.zeros(n, dtype=torch.long, device=dev)      # best j for each i
    dist1 = torch.zeros(n, dtype=dt, device=dev)            # best distance
    dist2 = torch.full((n,), float("inf"), dtype=dt, device=dev)
    col_best = torch.full((m,), -float("inf"), dtype=dt, device=dev)
    col_arg = torch.zeros(m, dtype=torch.long, device=dev)

    k = min(2, m)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = d0[s:e] @ d1.T                                # (C,M) inner products
        vals, idx = torch.topk(sim, k, dim=1)
        nn1[s:e] = idx[:, 0]
        dist1[s:e] = torch.clamp(2.0 - 2.0 * vals[:, 0], min=0.0).sqrt()
        if k >= 2:
            dist2[s:e] = torch.clamp(2.0 - 2.0 * vals[:, 1], min=0.0).sqrt()
        cmax, carg = sim.max(dim=0)
        upd = cmax > col_best
        col_best = torch.where(upd, cmax, col_best)
        col_arg = torch.where(upd, carg + s, col_arg)
        del sim, vals, idx

    ok = torch.ones(n, dtype=torch.bool, device=dev)
    if mutual:
        ok &= col_arg[nn1] == torch.arange(n, device=dev)
    if ratio is not None and ratio < 1.0 and k >= 2:
        ok &= dist1 <= ratio * dist2
    idx0 = torch.nonzero(ok, as_tuple=False).flatten()
    idx1 = nn1[idx0]
    return (idx0.cpu().numpy().astype(np.int64),
            idx1.cpu().numpy().astype(np.int64))


# -------------------------------------------------------------- epipolar --

def epipolar_residual(cam0, cam1, T_c1c0, uv0, uv1) -> np.ndarray:
    """Angular Sampson epipolar error of matched pixels, in radians.

    Both pixel sets are unprojected to unit bearings, so the function is valid
    for the fisheye model as well as for the pinhole model. Write b0 and b1
    for the unit bearings in the cam0 and cam1 frames. With
    ``E = hat(t_c1c0) @ R_c1c0`` the epipolar constraint is

        g(b0, b1) = b1^T E b0 = 0.

    Derivation of the returned quantity. Perturb each bearing inside its own
    tangent plane on the unit sphere, b0 <- normalize(b0 + P0 a) and
    b1 <- normalize(b1 + P1 c), with P0 = I - b0 b0^T and P1 = I - b1 b1^T.
    To first order the perturbed bearing is b + P a, and the length of P a is
    the rotation angle in radians. The linearization of g is

        g(b0 + P0 a, b1 + P1 c) = g(b0, b1) + (P0 E^T b1).a + (P1 E b0).c.

    The smallest joint perturbation that drives g to zero has norm

        |g| / sqrt( ||P0 E^T b1||^2 + ||P1 E b0||^2 ),

    which is the Sampson approximation. Because a and c are angles, that
    number is an angle in radians. It is the first-order angular distance from
    the observed bearing pair to the nearest pair that satisfies the epipolar
    constraint exactly.

    Degenerate case. A pure rotation has t_c1c0 = 0, hence E = 0, and the
    denominator vanishes. Those entries return NaN, and so do entries whose
    bearings are not finite.

    Args:
        cam0, cam1: ``calib.Camera`` for the two views.
        T_c1c0: (4,4) cam1-from-cam0.
        uv0: (K,2) distorted pixels in view 0.
        uv1: (K,2) distorted pixels in view 1.

    Returns:
        (K,) non-negative angular errors in radians, NaN where undefined.
    """
    cal = _calib()
    uv0 = np.atleast_2d(np.asarray(uv0, float))
    uv1 = np.atleast_2d(np.asarray(uv1, float))
    if uv0.shape[0] == 0:
        return np.zeros(0)
    b0 = cal.unproject(cam0, uv0)
    b1 = cal.unproject(cam1, uv1)
    T_c1c0 = np.asarray(T_c1c0, float)
    R, t = T_c1c0[:3, :3], T_c1c0[:3, 3]
    E = hat(t) @ R

    g = np.einsum("ki,ij,kj->k", b1, E, b0)
    Eb0 = b0 @ E.T                       # (K,3), row k is E b0_k
    Etb1 = b1 @ E                        # (K,3), row k is E^T b1_k
    # Project each gradient onto the tangent plane of its own bearing.
    p0 = Etb1 - b0 * np.sum(Etb1 * b0, axis=1, keepdims=True)
    p1 = Eb0 - b1 * np.sum(Eb0 * b1, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(p0 * p0, axis=1) + np.sum(p1 * p1, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.abs(g) / denom
    res[~np.isfinite(denom) | (denom <= 0.0)] = np.nan
    return res


# --------------------------------------------------------- triangulation --

def triangulate(cam0, cam1, T_c1c0, uv0, uv1):
    """Midpoint triangulation of matched pixels, with the parallax angle.

    Both pixel sets become unit bearings, so the fisheye model works too. Ray
    0 leaves the cam0 origin along b0. Ray 1 leaves the cam1 centre, which sits
    at ``c = -R_c1c0^T t_c1c0`` in the cam0 frame, along
    ``d1 = R_c1c0^T b1``. The returned point is the midpoint of the common
    perpendicular of the two rays, that is the midpoint of the segment that
    minimizes ``|| s b0 - (c + u d1) ||``. With ``B = b0.d1``, ``p = b0.c`` and
    ``q = d1.c`` the stationary conditions give

        u = (p B - q) / (1 - B^2),      s = p + u B,
        X0 = 0.5 ( s b0 + c + u d1 ).

    Rays that are close to parallel make ``1 - B^2`` vanish. Those entries
    return NaN instead of a huge point.

    Parallax. The direction from the triangulated point back to camera 0 is
    -b0 and back to camera 1 is -d1, so the angle between the two bearings, in
    a common frame, is the angle subtended at the point. A point whose
    parallax falls below the configured floor carries almost no depth
    information. Its depth uncertainty grows like the inverse of the parallax,
    so a small angular error in either bearing moves the point along the ray
    by an unbounded amount. Such points are useful for rotation but not for
    scale, and ``stereo_landmarks`` rejects them.

    Args:
        cam0, cam1: ``calib.Camera`` for the two views.
        T_c1c0: (4,4) cam1-from-cam0.
        uv0, uv1: (K,2) matched distorted pixels.

    Returns:
        (X0, parallax). ``X0`` is (K,3) in the cam0 frame. ``parallax`` is
        (K,) in radians, in [0, pi].
    """
    cal = _calib()
    uv0 = np.atleast_2d(np.asarray(uv0, float))
    uv1 = np.atleast_2d(np.asarray(uv1, float))
    if uv0.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0)
    b0 = cal.unproject(cam0, uv0)
    b1 = cal.unproject(cam1, uv1)
    T_c1c0 = np.asarray(T_c1c0, float)
    R, t = T_c1c0[:3, :3], T_c1c0[:3, 3]
    c = -R.T @ t                                    # cam1 centre in the cam0 frame
    d1 = b1 @ R                                     # (K,3), row k is R^T b1_k

    B = np.sum(b0 * d1, axis=1)
    p = b0 @ c
    q = d1 @ c
    denom = 1.0 - B * B
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (p * B - q) / denom
        s = p + u * B
        X0 = 0.5 * (s[:, None] * b0 + c[None, :] + u[:, None] * d1)
    bad = ~np.isfinite(denom) | (np.abs(denom) < 1e-12)
    X0[bad] = np.nan
    parallax = np.arccos(np.clip(B, -1.0, 1.0))
    return X0, parallax


def triangulation_cov(cam0, cam1, T_c1c0, uv0, uv1, X0, sigma_px: float,
                      ridge_rel: float | None = None,
                      return_ridged: bool = False):
    """First-order covariance of a triangulated point under pixel noise.

    Model. The two pixel observations are independent and isotropic with
    standard deviation ``sigma_px`` in both views. Stacking the two
    projections of the same cam0 point gives the 4x3 Jacobian

        J = [ Jp0 ; Jp1 @ R_c1c0 ],

    where ``Jp0 = project_jac(cam0, X0)`` and
    ``Jp1 = project_jac(cam1, R_c1c0 X0 + t_c1c0)``. The second block carries
    the rotation because ``d uv1 / d X0 = (d uv1 / d X_c1) (d X_c1 / d X0)``
    and ``d X_c1 / d X0 = R_c1c0``. With ``W = diag(1 / sigma_px^2)`` the
    Gauss-Newton covariance of the point is ``(J^T W J)^-1``.

    This is the covariance of the maximum likelihood point, not of the
    midpoint returned by ``triangulate``. The two agree closely for a well
    conditioned pair. The tests check that by Monte Carlo and measure a
    relative Frobenius gap near one percent.

    Conditioning. ``J^T W J`` is 3x3 and loses rank when the two rays become
    parallel, because the depth direction then carries no information. When
    its smallest eigenvalue falls below ``ridge_rel`` times its largest, a
    ridge of ``ridge_rel * lambda_max`` is added to the diagonal before the
    inverse. Those points are reported through ``return_ridged`` and their
    count is logged at INFO level. ``stereo_landmarks`` rejects them.

    Args:
        cam0, cam1: ``calib.Camera`` for the two views.
        T_c1c0: (4,4) cam1-from-cam0.
        uv0, uv1: (K,2) matched pixels. They are part of the contract
            signature and are not used. The linearization point is ``X0``, so
            the observed pixels do not enter a first-order covariance.
        X0: (K,3) triangulated points in the cam0 frame.
        sigma_px: isotropic pixel standard deviation, the same in both views.
        ridge_rel: relative ridge. Defaults to ``stereo.cov_ridge_rel``.
        return_ridged: when True return ``(Cov, ridged)`` with ``ridged`` a
            (K,) boolean array, True where the ridge was needed.

    Returns:
        (K,3,3) covariance of the point in the cam0 frame. Rows whose ``X0``
        is not finite are NaN, and those rows are never marked ridged.
    """
    cal = _calib()
    X0 = np.atleast_2d(np.asarray(X0, float))
    k = X0.shape[0]
    if ridge_rel is None:
        ridge_rel = float(_STEREO_CFG["cov_ridge_rel"])
    cov = np.full((k, 3, 3), np.nan)
    ridged = np.zeros(k, bool)
    if k == 0:
        return (cov, ridged) if return_ridged else cov

    T_c1c0 = np.asarray(T_c1c0, float)
    R, t = T_c1c0[:3, :3], T_c1c0[:3, 3]
    good = np.all(np.isfinite(X0), axis=1)
    if not np.any(good):
        return (cov, ridged) if return_ridged else cov

    Xg = X0[good]
    X1 = Xg @ R.T + t
    Jp0 = cal.project_jac(cam0, Xg)                     # (G,2,3)
    Jp1 = cal.project_jac(cam1, X1)                     # (G,2,3)
    J = np.concatenate([Jp0, np.einsum("kij,jl->kil", Jp1, R)], axis=1)   # (G,4,3)
    A = np.einsum("kia,kib->kab", J, J) / (float(sigma_px) ** 2)          # (G,3,3)

    finite = np.all(np.isfinite(A.reshape(A.shape[0], -1)), axis=1)
    lam_max = np.full(A.shape[0], np.nan)
    lam_min = np.full(A.shape[0], np.nan)
    if np.any(finite):
        ev = np.linalg.eigvalsh(A[finite])
        lam_min[finite], lam_max[finite] = ev[:, 0], ev[:, -1]
    need = finite & (lam_min <= ridge_rel * lam_max)
    if np.any(need):
        A[need] += (ridge_rel * lam_max[need])[:, None, None] * np.eye(3)[None]
        log.info("triangulation_cov: ridged %d of %d points",
                 int(np.count_nonzero(need)), k)

    out = np.full((A.shape[0], 3, 3), np.nan)
    for i in np.nonzero(finite)[0]:
        try:
            out[i] = np.linalg.inv(A[i])
        except np.linalg.LinAlgError:
            out[i] = np.nan
    cov[good] = out
    ridged[good] = need
    return (cov, ridged) if return_ridged else cov


def transfer_cov(cam_b, T_ba, X_a, Cov_a):
    """Move points and their covariance into another camera frame and project.

    With ``X_b = R_ba X_a + t_ba`` and ``uv_b = project(cam_b, X_b)`` the
    first-order propagation of the point covariance to the image is

        Sigma_uv = Jp R_ba Cov_a R_ba^T Jp^T,    Jp = project_jac(cam_b, X_b).

    The translation drops out because it is deterministic. This is the step
    that turns a stereo landmark into a predicted observation with a predicted
    2x2 covariance, which is the target the covariance head is scored against.

    Args:
        cam_b: ``calib.Camera`` of the destination view.
        T_ba: (4,4) frame-b-from-frame-a.
        X_a: (K,3) points in frame a.
        Cov_a: (K,3,3) covariance of those points in frame a.

    Returns:
        (uv_b, Sigma_uv) with shapes (K,2) and (K,2,2). Points behind the
        camera give NaN pixels from ``project``, and their covariance is NaN
        as well.
    """
    cal = _calib()
    X_a = np.atleast_2d(np.asarray(X_a, float))
    Cov_a = np.asarray(Cov_a, float).reshape(-1, 3, 3)
    if X_a.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0, 2, 2))
    T_ba = np.asarray(T_ba, float)
    R, t = T_ba[:3, :3], T_ba[:3, 3]
    X_b = X_a @ R.T + t
    uv_b = cal.project(cam_b, X_b)
    Jp = cal.project_jac(cam_b, X_b)                      # (K,2,3)
    M = np.einsum("kij,jl->kil", Jp, R)                   # (K,2,3)
    Sigma = np.einsum("kia,kab,kjb->kij", M, Cov_a, M)
    bad = ~np.all(np.isfinite(uv_b), axis=1)
    Sigma[bad] = np.nan
    return uv_b, Sigma


# ------------------------------------------------------------- the chain --

_REJECT_ORDER = ("match", "epipolar", "depth_sign", "depth_range",
                 "parallax", "cov_conditioning")


def stereo_landmarks(bundle: dict) -> dict:
    """Run the whole stereo chain on one image pair and report every rejection.

    Args:
        bundle: a dict with
            cam0, cam1  ``calib.Camera`` for the two views,
            T_c1c0      (4,4) cam1-from-cam0,
            kps0, kps1  (N,2) and (M,2) sub-pixel pixels from the backbone,
            feats0, feats1  (N,65) and (M,65) backbone features,
            cfg         optional. Either the full parsed config, in which case
                        its ``stereo`` block is used, or the ``stereo`` block
                        itself. Defaults to ``configs/default.yaml``.
        Any threshold key may also be given directly in the bundle, where it
        overrides the config. The keys are ratio, mutual, epipolar_rad, z_min,
        z_max, parallax_min_deg, sigma_px, cov_ridge_rel and cov_cond_max.

    Rejection order. Each test runs on the survivors of the previous one and
    is counted separately, so the counts sum with the survivor count to N, the
    number of keypoints in view 0.
        match             no mutual nearest neighbour, or the ratio test
                          failed. Counted as N minus the number of matches.
        epipolar          Sampson angular error above ``epipolar_rad``, or not
                          finite.
        depth_sign        the triangulated z in the cam0 frame is not finite
                          or is not positive. Catches near-parallel rays and
                          points behind the camera.
        depth_range       z outside [``z_min``, ``z_max``].
        parallax          the angle between the two bearings is below
                          ``parallax_min_deg``, so the depth is not observable.
        cov_conditioning  the 3x3 triangulation covariance is not finite, or
                          the information matrix needed the ridge, or the
                          condition number of the covariance exceeds
                          ``cov_cond_max``.

    Returns:
        A dict with
            idx0, idx1        (K,) int64 indices of the survivors into kps0
                              and kps1,
            uv0, uv1          (K,2) the surviving pixels,
            X0                (K,3) points in the cam0 frame,
            parallax          (K,) radians,
            epipolar          (K,) radians,
            cov               (K,3,3) triangulation covariance in cam0,
            n_input           N, the number of view-0 keypoints,
            n_kept            K,
            n_ridge           how many of the points that reached the
                              covariance test needed the ridge. All of them
                              are rejected under cov_conditioning, so this is
                              a diagnostic and not a property of the
                              survivors.
            rejected          dict with one count per reason, in the order
                              above.
    """
    cfg = bundle.get("cfg")
    if cfg is None:
        cfg = dict(_STEREO_CFG)
    elif "stereo" in cfg:
        cfg = dict(cfg["stereo"])
    else:
        cfg = dict(cfg)
    for key in ("ratio", "mutual", "epipolar_rad", "z_min", "z_max",
                "parallax_min_deg", "sigma_px", "cov_ridge_rel", "cov_cond_max"):
        if key in bundle:
            cfg[key] = bundle[key]

    cam0, cam1 = bundle["cam0"], bundle["cam1"]
    T_c1c0 = np.asarray(bundle["T_c1c0"], float)
    kps0 = _to_numpy(bundle["kps0"])
    kps1 = _to_numpy(bundle["kps1"])
    n_input = int(kps0.shape[0])
    rejected = {name: 0 for name in _REJECT_ORDER}

    idx0, idx1 = match_descriptors(bundle["feats0"], bundle["feats1"],
                                   ratio=float(cfg["ratio"]),
                                   mutual=bool(cfg["mutual"]))
    rejected["match"] = n_input - int(idx0.size)
    if idx0.size == 0:
        return _empty_result(n_input, rejected)

    uv0, uv1 = kps0[idx0], kps1[idx1]

    ep = epipolar_residual(cam0, cam1, T_c1c0, uv0, uv1)
    keep = np.isfinite(ep) & (ep <= float(cfg["epipolar_rad"]))
    rejected["epipolar"] = int(np.count_nonzero(~keep))
    idx0, idx1, uv0, uv1, ep = idx0[keep], idx1[keep], uv0[keep], uv1[keep], ep[keep]
    if idx0.size == 0:
        return _empty_result(n_input, rejected)

    X0, parallax = triangulate(cam0, cam1, T_c1c0, uv0, uv1)
    z = X0[:, 2]
    keep = np.all(np.isfinite(X0), axis=1) & (z > 0.0)
    rejected["depth_sign"] = int(np.count_nonzero(~keep))
    idx0, idx1, uv0, uv1, ep = idx0[keep], idx1[keep], uv0[keep], uv1[keep], ep[keep]
    X0, parallax = X0[keep], parallax[keep]
    if idx0.size == 0:
        return _empty_result(n_input, rejected)

    z = X0[:, 2]
    keep = (z >= float(cfg["z_min"])) & (z <= float(cfg["z_max"]))
    rejected["depth_range"] = int(np.count_nonzero(~keep))
    idx0, idx1, uv0, uv1, ep = idx0[keep], idx1[keep], uv0[keep], uv1[keep], ep[keep]
    X0, parallax = X0[keep], parallax[keep]
    if idx0.size == 0:
        return _empty_result(n_input, rejected)

    keep = parallax >= np.deg2rad(float(cfg["parallax_min_deg"]))
    rejected["parallax"] = int(np.count_nonzero(~keep))
    idx0, idx1, uv0, uv1, ep = idx0[keep], idx1[keep], uv0[keep], uv1[keep], ep[keep]
    X0, parallax = X0[keep], parallax[keep]
    if idx0.size == 0:
        return _empty_result(n_input, rejected)

    cov, ridged = triangulation_cov(cam0, cam1, T_c1c0, uv0, uv1, X0,
                                    sigma_px=float(cfg["sigma_px"]),
                                    ridge_rel=float(cfg["cov_ridge_rel"]),
                                    return_ridged=True)
    n_ridge = int(np.count_nonzero(ridged))
    flat = cov.reshape(cov.shape[0], -1)
    finite = np.all(np.isfinite(flat), axis=1)
    cond = np.full(cov.shape[0], np.inf)
    if np.any(finite):
        ev = np.linalg.eigvalsh(cov[finite])
        with np.errstate(divide="ignore", invalid="ignore"):
            cond[finite] = ev[:, -1] / ev[:, 0]
    keep = (finite & ~ridged & np.isfinite(cond) & (cond > 0)
            & (cond <= float(cfg["cov_cond_max"])))
    rejected["cov_conditioning"] = int(np.count_nonzero(~keep))
    idx0, idx1, uv0, uv1, ep = idx0[keep], idx1[keep], uv0[keep], uv1[keep], ep[keep]
    X0, parallax, cov = X0[keep], parallax[keep], cov[keep]

    return {"idx0": idx0, "idx1": idx1, "uv0": uv0, "uv1": uv1,
            "X0": X0, "parallax": parallax, "epipolar": ep, "cov": cov,
            "n_input": n_input, "n_kept": int(idx0.size), "n_ridge": n_ridge,
            "rejected": rejected}


# ---------------------------------------------------------------- helpers --

def _to_numpy(a: Any) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy().astype(np.float64)
    return np.asarray(a, float)


def _empty_result(n_input: int, rejected: dict) -> dict:
    return {"idx0": np.zeros(0, np.int64), "idx1": np.zeros(0, np.int64),
            "uv0": np.zeros((0, 2)), "uv1": np.zeros((0, 2)),
            "X0": np.zeros((0, 3)), "parallax": np.zeros(0),
            "epipolar": np.zeros(0), "cov": np.zeros((0, 3, 3)),
            "n_input": n_input, "n_kept": 0, "n_ridge": 0,
            "rejected": dict(rejected)}


_STEREO_CFG = load_stereo_config()
_MATCH_CHUNK_ENTRIES = int(_STEREO_CFG["match_chunk_entries"])
