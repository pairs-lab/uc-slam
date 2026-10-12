"""Geometry core for UC-SLAM M1 experiments.

Conventions (match the manuscript, Sec. III):
  - Camera pose is T_cw = (R, t):  x_c = R @ X_w + t.
  - Right perturbation on SO(3):   R <- R @ Exp(dtheta).
  - Pose tangent ordering:         xi = [dtheta (3), dt (3)].
  - Pose covariance is expressed in this tangent at the estimate:
        R_true = R_est @ Exp(dtheta),  t_true = t_est + dt.

All Jacobians here are convention-checked by tests/test_geometry.py against
finite differences. Keep it that way: any change to a Jacobian must keep the
tests passing.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# SO(3)
# --------------------------------------------------------------------------

def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that hat(a) @ b = a x b."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def so3_exp(theta: np.ndarray) -> np.ndarray:
    """Exponential map R = Exp(theta), Rodrigues formula, numerically safe."""
    angle = np.linalg.norm(theta)
    if angle < 1e-12:
        return np.eye(3) + hat(theta)
    axis = theta / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map, returns rotation vector theta with R = Exp(theta)."""
    cos_angle = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-9:
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    if np.pi - angle < 1e-6:
        # Near pi: use the symmetric part.
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        # Fix signs using the largest component.
        k = int(np.argmax(axis))
        if axis[k] > 0:
            sign = np.sign(np.array([R[2, 1] - R[1, 2],
                                     R[0, 2] - R[2, 0],
                                     R[1, 0] - R[0, 1]]))
            sign[sign == 0] = 1.0
            axis = axis * sign * np.sign(sign[k])
        return axis / max(np.linalg.norm(axis), 1e-12) * angle
    return angle / (2.0 * np.sin(angle)) * np.array([R[2, 1] - R[1, 2],
                                                     R[0, 2] - R[2, 0],
                                                     R[1, 0] - R[0, 1]])


def random_rotation(rng: np.random.Generator, max_angle_rad: float = np.pi) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0.0, max_angle_rad)
    return so3_exp(axis * angle)


# --------------------------------------------------------------------------
# Pinhole projection
# --------------------------------------------------------------------------

def project(xc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame points (N,3) to pixels (N,2)."""
    xc = np.atleast_2d(xc)
    z = xc[:, 2:3]
    uv = xc[:, :2] / z
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def proj_jacobian(xc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """d pi / d x_c for a single camera-frame point. Returns (2,3)."""
    x, y, z = xc
    fx, fy = K[0, 0], K[1, 1]
    return np.array([[fx / z, 0.0, -fx * x / z ** 2],
                     [0.0, fy / z, -fy * y / z ** 2]])


# --------------------------------------------------------------------------
# Weighted motion-only PnP (the tracking pose optimization of the paper)
# --------------------------------------------------------------------------

def pnp_gauss_newton(X_w: np.ndarray,
                     u_obs: np.ndarray,
                     info: np.ndarray,
                     K: np.ndarray,
                     R0: np.ndarray,
                     t0: np.ndarray,
                     n_iters: int = 15,
                     lm_lambda: float = 1e-4):
    """Minimize sum_i r_i^T W_i r_i with r_i = u_i - pi(R X_i + t).

    Args:
        X_w:   (N,3) world points (assumed exact here; the tracking case).
        u_obs: (N,2) pixel observations.
        info:  (N,2,2) per-observation information matrices W_i = Sigma_i^{-1}.
        K:     (3,3) intrinsics.
        R0,t0: initial pose guess (T_cw).

    Returns:
        R, t, Sigma_pose (6x6 in [dtheta, dt] tangent), n_used_iters
    """
    R = R0.copy()
    t = t0.copy()
    lam = lm_lambda
    prev_cost = np.inf
    H = np.eye(6)
    N = X_w.shape[0]
    fx, fy = K[0, 0], K[1, 1]

    def batch_hat(V):
        Hm = np.zeros((V.shape[0], 3, 3))
        Hm[:, 0, 1] = -V[:, 2]; Hm[:, 0, 2] = V[:, 1]
        Hm[:, 1, 0] = V[:, 2];  Hm[:, 1, 2] = -V[:, 0]
        Hm[:, 2, 0] = -V[:, 1]; Hm[:, 2, 1] = V[:, 0]
        return Hm

    def cost_of(Rc, tc):
        xc = (Rc @ X_w.T).T + tc
        r = u_obs - project(xc, K)
        return float(np.einsum('ni,nij,nj->', r, info, r)), xc, r

    for it in range(n_iters):
        cost, xc, r = cost_of(R, t)

        # Vectorized Jacobians: J_i = -Jpi_i @ [ -R [X_i]^ , I ]  (2,6)
        z = xc[:, 2]
        Jpi = np.zeros((N, 2, 3))
        Jpi[:, 0, 0] = fx / z
        Jpi[:, 0, 2] = -fx * xc[:, 0] / z ** 2
        Jpi[:, 1, 1] = fy / z
        Jpi[:, 1, 2] = -fy * xc[:, 1] / z ** 2
        Jx = np.concatenate([-np.einsum('ij,njk->nik', R, batch_hat(X_w)),
                             np.tile(np.eye(3), (N, 1, 1))], axis=2)  # (N,3,6)
        J = -np.einsum('nij,njk->nik', Jpi, Jx)                        # (N,2,6)

        H = np.einsum('nka,nkl,nlb->ab', J, info, J)
        b = np.einsum('nka,nkl,nl->a', J, info, r)

        # Levenberg damping on a stagnating step.
        try:
            dx = np.linalg.solve(H + lam * np.diag(np.diag(H)), -b)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        R_new = R @ so3_exp(dx[:3])
        t_new = t + dx[3:]
        cost_new, _, _ = cost_of(R_new, t_new)

        if cost_new < cost:
            R, t = R_new, t_new
            lam = max(lam * 0.5, 1e-7)
            if abs(prev_cost - cost_new) < 1e-10 * max(1.0, cost_new):
                prev_cost = cost_new
                break
            prev_cost = cost_new
        else:
            lam *= 10.0

    Sigma_pose = np.linalg.inv(H)
    return R, t, Sigma_pose, it + 1


def pose_error_tangent(R_est: np.ndarray, t_est: np.ndarray,
                       R_gt: np.ndarray, t_gt: np.ndarray) -> np.ndarray:
    """Error xi such that T_gt = T_est (+) xi under the right-perturbation
    convention used for the covariance: R_gt = R_est Exp(dtheta), t_gt = t_est + dt."""
    dtheta = so3_log(R_est.T @ R_gt)
    dt = t_gt - t_est
    return np.concatenate([dtheta, dt])


def nees(R_est, t_est, Sigma_pose, R_gt, t_gt) -> float:
    e = pose_error_tangent(R_est, t_est, R_gt, t_gt)
    return float(e @ np.linalg.solve(Sigma_pose, e))
