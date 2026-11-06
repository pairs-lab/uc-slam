"""Depth-constant fit helpers for scripts/jetson/fit_depth_constants.py.

Each definition below is copied unchanged from the experiment code of the paper, which this repository
does not ship: experiments/depth_prior_bench.py (robust statistics, the affine fit of sigma_net, the
effective sample ratio, the per-sequence and split summaries), experiments/openloris_depth_calib.py
(read_list) and experiments/real_corpus.py (nearest_frame). Only REPO differs: this file sits one folder
deeper. Keep the copies equal to their sources, so a D455 fit uses the estimators of the paper.
"""
import math
import os

import numpy as np
import yaml


# ---- from experiments/depth_prior_bench.py -------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/jetson -> repo
DEFAULT_CONFIG = os.path.join(REPO, "configs", "default.yaml")
MAD_SCALE = 1.4826                  # robust std of a Gaussian from its MAD
GAUSS_MEDIAN_ABS = 0.6744897501960817   # median of |N(0,1)|
SIGMA_NET_FLOOR = 1.0e-3            # floor on an affine sigma_net that turns negative
def load_config(path: str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def robust_spread(x: np.ndarray) -> float:
    """1.4826 times the median absolute deviation about the median."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(MAD_SCALE * np.median(np.abs(x - np.median(x))))


def robust_spread_about_zero(x: np.ndarray) -> float:
    """1.4826 times the median of |x|, for a sample already centred."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(MAD_SCALE * np.median(np.abs(x)))


def deflate(spread: float, ref_var: float) -> float:
    """``sqrt(spread^2 - ref_var)``, or 0 when the reference explains it all."""
    if not np.isfinite(spread) or not np.isfinite(ref_var):
        return float("nan")
    return float(math.sqrt(max(spread * spread - ref_var, 0.0)))


def keyframe_table(r, kf, t_kf, min_points: int, clip_sigmas: float) -> dict:
    """Per-keyframe median, clipped mean, count and time for one sequence.

    Only keyframes with at least ``min_points`` finite residuals appear.
    The clipped mean uses residuals clipped at the sequence median plus or
    minus ``clip_sigmas`` robust spreads. ``r_clipped`` and ``kf_of_point``
    are returned for every finite residual, used or not.
    """
    r = np.asarray(r, float)
    kf = np.asarray(kf)
    ok = np.isfinite(r)
    r, kf = r[ok], kf[ok]
    empty = {"kf": np.zeros(0, int), "median": np.zeros(0), "mean": np.zeros(0),
             "n": np.zeros(0, int), "t_s": np.zeros(0), "r_clipped": r, "kf_of_point": kf}
    if r.size == 0:
        return empty
    med, spr = np.median(r), robust_spread(r)
    rc = np.clip(r, med - clip_sigmas * spr, med + clip_sigmas * spr)
    empty["r_clipped"] = rc
    rows = []
    for k in np.unique(kf):
        sel = kf == k
        n = int(np.count_nonzero(sel))
        if n < min_points:
            continue
        rows.append((int(k), float(np.median(r[sel])), float(np.mean(rc[sel])), n,
                     float(t_kf[int(k)]) / 1.0e9))
    if not rows:
        return empty
    return {"kf": np.array([x[0] for x in rows], int),
            "median": np.array([x[1] for x in rows]),
            "mean": np.array([x[2] for x in rows]),
            "n": np.array([x[3] for x in rows], int),
            "t_s": np.array([x[4] for x in rows]),
            "r_clipped": rc, "kf_of_point": kf}


def neff_ratio(kt: dict) -> dict:
    """N_eff / N of one sequence from the keyframe-mean variance.

    ``rho = (N Var(mean_k) / Var(r) - 1) / (N - 1)`` with N the mean point
    count of the keyframes that entered, clipped to [0, 1], and
    ``N_eff / N = 1 / (1 + (N - 1) rho)``. Needs at least three keyframes.
    """
    base = {"ratio": float("nan"), "rho": float("nan"), "n_mean": float("nan"),
            "n_kf": int(kt["kf"].size)}
    if kt["kf"].size < 3:
        return base
    in_kf = np.isin(kt["kf_of_point"], kt["kf"])
    var_pt = float(np.var(kt["r_clipped"][in_kf], ddof=1))
    var_mean = float(np.var(kt["mean"], ddof=1))
    n_mean = float(np.mean(kt["n"]))
    base["n_mean"] = n_mean
    if var_pt <= 0 or n_mean <= 1:
        return base
    rho = (n_mean * var_mean / var_pt - 1.0) / (n_mean - 1.0)
    rho = float(np.clip(rho, 0.0, 1.0))
    base.update({"ratio": float(1.0 / (1.0 + (n_mean - 1.0) * rho)), "rho": rho,
                 "var_point": var_pt, "var_kf_mean": var_mean})
    return base


def sequence_stats(d: dict, model: str, dcfg: dict) -> dict:
    """All per-sequence quantities of one network on one sequence.

    The ``_points`` entry holds per-point arrays for the pooled fit and is
    not written to the JSON.
    """
    zhat = np.asarray(d[f"zhat_{model}"], float)
    z_ref = np.asarray(d["z_ref"], float)
    sig_ref = np.asarray(d["sigma_ref"], float)
    kf = np.asarray(d["kf"])
    valid = np.isfinite(zhat) & (zhat > 0) & np.isfinite(z_ref) & (z_ref > 0)
    r = np.full(zhat.shape, np.nan)
    r[valid] = np.log(zhat[valid]) - np.log(z_ref[valid])
    kt = keyframe_table(r, kf, d["kf_t_ns"], int(dcfg["min_points_per_kf"]),
                        float(dcfg["clip_sigmas"]))
    rv = r[valid]
    nan = float("nan")
    out = {
        "n_points": int(rv.size),
        "n_kf_total": int(np.unique(kf).size) if kf.size else 0,
        "n_kf_used": int(kt["kf"].size),
        "median_r": float(np.median(rv)) if rv.size else nan,
        "spread_r": robust_spread(rv),
        "ref_var_mean": float(np.mean(sig_ref[valid] ** 2)) if rv.size else nan,
        "ref_lognoise_median": float(np.median(sig_ref[valid])) if rv.size else nan,
        "kf_median_spread": robust_spread(kt["median"]) if kt["kf"].size >= 3 else nan,
        "kf_medians": kt["median"].tolist(), "kf_t_s": kt["t_s"].tolist(),
        "kf_n": kt["n"].tolist(),
        "neff": neff_ratio(kt),
        "duration_s": float(d["duration_s"]),
        "median_zhat": float(np.median(zhat[valid])) if rv.size else nan,
        "median_zref": float(np.median(z_ref[valid])) if rv.size else nan,
    }
    if kt["kf"].size >= 2:
        dm, dt = np.diff(kt["median"]), np.diff(kt["t_s"])
        good = dt > 0
        out["rw_increments"] = (dm[good] / np.sqrt(dt[good])).tolist()
    else:
        out["rw_increments"] = []
    kf_med = dict(zip(kt["kf"].tolist(), kt["median"].tolist()))
    idx = np.nonzero(valid & np.isin(kf, kt["kf"]))[0]
    centred = np.array([r[i] - kf_med[int(kf[i])] for i in idx]) if idx.size else np.zeros(0)
    out["_points"] = {
        "r": rv, "log_zhat": np.log(zhat[valid]), "sigma_ref": sig_ref[valid],
        "d_centred": np.abs(centred),
        "log_zhat_centred": np.log(zhat[idx]) if idx.size else np.zeros(0),
        "sigma_ref_centred": sig_ref[idx] if idx.size else np.zeros(0),
    }
    return out


def fit_affine(log_zhat, d_centred, sigma_ref, n_bins: int, z_ref_m: float) -> dict:
    """Robust binned fit of ``sigma_net(log z) = c0 + c1 (log z - log z_ref)``.

    Bins are quantile bins of ``log_zhat``. In each bin the robust spread of
    the centred residual is 1.4826 times its median, the mean reference
    variance of the bin is subtracted, and a line weighted by the bin counts
    is fitted through the bin spreads. Both the raw and the deflated lines
    are returned. Duplicate quantile edges are merged, so a nearly constant
    abscissa yields fewer bins.
    """
    x = np.asarray(log_zhat, float) - math.log(float(z_ref_m))
    dd = np.asarray(d_centred, float)
    sr = np.asarray(sigma_ref, float)
    ok = np.isfinite(x) & np.isfinite(dd) & np.isfinite(sr)
    x, dd, sr = x[ok], dd[ok], sr[ok]
    if x.size < 2 * n_bins:
        return {"ok": False, "reason": f"only {x.size} centred residuals"}
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return {"ok": False, "reason": "log z_hat is nearly constant"}
    which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        sel = which == b
        n = int(np.count_nonzero(sel))
        if n < 2:
            continue
        raw = float(MAD_SCALE * np.median(dd[sel]))
        refv = float(np.mean(sr[sel] ** 2))
        rows.append({"x_mean": float(np.mean(x[sel])), "n": n, "sigma_raw": raw,
                     "ref_var": refv, "sigma_net": deflate(raw, refv),
                     "x_lo": float(edges[b]), "x_hi": float(edges[b + 1])})
    if len(rows) < 2:
        return {"ok": False, "reason": "fewer than two populated bins"}
    xb = np.array([q["x_mean"] for q in rows])
    wb = np.array([q["n"] for q in rows], float)
    res = {"ok": True, "bins": rows, "n_points": int(x.size), "z_ref_m": float(z_ref_m)}
    for name in ("sigma_raw", "sigma_net"):
        yb = np.array([q[name] for q in rows])
        c1, c0 = np.polyfit(xb, yb, 1, w=np.sqrt(wb))
        pred = c0 + c1 * xb
        res[name] = {"c0": float(c0), "c1": float(c1),
                     "rms_bin_misfit": float(np.sqrt(np.average((yb - pred) ** 2, weights=wb)))}
    return res


def coverage(c0: float, c1: float, log_zhat, d_centred, sigma_ref, z_ref_m: float) -> dict:
    """How the affine model covers the centred residuals of a split.

    ``sigma_tot = sqrt(max(c0 + c1 (log z - log z_ref), floor)^2 + sigma_ref^2)``.
    Returns the fractions inside one and two sigma_tot (0.683 and 0.954 for
    a calibrated Gaussian), the median of d over 0.6745 sigma_tot (1 when
    calibrated) and how often the affine model hit the floor.
    """
    x = np.asarray(log_zhat, float) - math.log(float(z_ref_m))
    dd = np.asarray(d_centred, float)
    sr = np.asarray(sigma_ref, float)
    ok = np.isfinite(x) & np.isfinite(dd) & np.isfinite(sr)
    x, dd, sr = x[ok], dd[ok], sr[ok]
    if x.size == 0:
        return {"n": 0}
    s_net = c0 + c1 * x
    floored = s_net < SIGMA_NET_FLOOR
    s_net = np.maximum(s_net, SIGMA_NET_FLOOR)
    q = dd / np.sqrt(s_net ** 2 + sr ** 2)
    return {"n": int(x.size), "frac_1sigma": float(np.mean(q <= 1.0)),
            "frac_2sigma": float(np.mean(q <= 2.0)),
            "median_ratio": float(np.median(q) / GAUSS_MEDIAN_ABS),
            "frac_floored": float(np.mean(floored))}


def split_summary(seq_stats: dict, keys) -> dict:
    """Pool the per-sequence statistics of ``keys`` into one split summary."""
    keys = [k for k in keys if k in seq_stats and seq_stats[k]["n_points"] > 0]
    if not keys:
        return {"n_seq": 0, "n_points": 0}
    nan = float("nan")
    pts_r = np.concatenate([seq_stats[k]["_points"]["r"] for k in keys])
    pts_sr = np.concatenate([seq_stats[k]["_points"]["sigma_ref"] for k in keys])
    med_s = np.array([seq_stats[k]["median_r"] for k in keys])
    spread = robust_spread(pts_r)
    ref_var = float(np.mean(pts_sr ** 2))
    kf_spreads = np.array([seq_stats[k]["kf_median_spread"] for k in keys])
    kf_spreads = kf_spreads[np.isfinite(kf_spreads)]
    rw = np.concatenate([np.asarray(seq_stats[k]["rw_increments"], float) for k in keys])
    ratios = np.array([seq_stats[k]["neff"]["ratio"] for k in keys])
    rhos = np.array([seq_stats[k]["neff"]["rho"] for k in keys])
    nm = np.array([seq_stats[k]["neff"]["n_mean"] for k in keys])
    fin = np.isfinite(ratios)
    out = {
        "n_seq": len(keys), "sequences": keys,
        "n_points": int(pts_r.size),
        "n_kf_used": int(sum(seq_stats[k]["n_kf_used"] for k in keys)),
        "bias": float(np.median(pts_r)), "spread_raw": spread, "ref_var_mean": ref_var,
        "ref_lognoise_median": float(np.median(pts_sr)),
        "spread_net": deflate(spread, ref_var),
        "per_seq_median": dict(zip(keys, med_s.tolist())),
        "sigma0_std": float(np.std(med_s, ddof=1)) if med_s.size >= 2 else nan,
        "sigma0_rms": float(np.sqrt(np.mean(med_s ** 2))),
        "sigma0_max_abs": float(np.max(np.abs(med_s))),
        "kf_median_spread_median": float(np.median(kf_spreads)) if kf_spreads.size else nan,
        "kf_median_spread_per_seq": {k: seq_stats[k]["kf_median_spread"] for k in keys},
        "sigma_rw_upper": robust_spread_about_zero(rw) if rw.size else nan,
        "n_rw_increments": int(rw.size),
        "neff_ratio_median": float(np.median(ratios[fin])) if fin.any() else nan,
        "neff_ratio_min": float(np.min(ratios[fin])) if fin.any() else nan,
        "neff_ratio_max": float(np.max(ratios[fin])) if fin.any() else nan,
        "rho_median": float(np.median(rhos[fin])) if fin.any() else nan,
        "n_per_kf_mean": float(np.mean(nm[np.isfinite(nm)])) if np.isfinite(nm).any() else nan,
        "neff_ratio_per_seq": dict(zip(keys, ratios.tolist())),
    }
    if np.isfinite(out["neff_ratio_median"]) and np.isfinite(out["n_per_kf_mean"]):
        out["neff_median"] = float(out["neff_ratio_median"] * out["n_per_kf_mean"])
    return out


def _centred_pool(seq_stats: dict, keys):
    keys = [k for k in keys if k in seq_stats and seq_stats[k]["n_points"] > 0]
    if not keys:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    p = [seq_stats[k]["_points"] for k in keys]
    return (np.concatenate([q["log_zhat_centred"] for q in p]),
            np.concatenate([q["d_centred"] for q in p]),
            np.concatenate([q["sigma_ref_centred"] for q in p]))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o



# ---- from experiments/openloris_depth_calib.py ---------------------------------------------------

def read_list(path: str):
    """'t rel' lines (TUM style) -> (t_ns int64 sorted, absolute paths)."""
    from scripts.openloris_convert import t_ns
    ts, ps = [], []
    base = os.path.dirname(path)
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            t, rel = line.split()[:2]
            ts.append(t_ns(t))
            ps.append(os.path.join(base, rel))
    o = np.argsort(ts, kind="stable")
    return np.asarray(ts, np.int64)[o], [ps[i] for i in o]



# ---- from experiments/real_corpus.py -------------------------------------------------------------

def nearest_frame(t_index: np.ndarray, t_query, tol_ns: int) -> np.ndarray:
    """Index of the nearest timestamp in ``t_index``, or -1 when none is close.

    Args:
        t_index: (M,) int64 timestamps, sorted ascending.
        t_query: scalar or (N,) int64 query timestamps.
        tol_ns: a match must be within this many nanoseconds.

    Returns:
        (N,) int64 indices, -1 where no timestamp lies within the tolerance.
        A scalar query still returns a length-1 array.
    """
    t_index = np.asarray(t_index, np.int64)
    q = np.atleast_1d(np.asarray(t_query, np.int64))
    m = t_index.shape[0]
    if m == 0:
        return np.full(q.shape, -1, np.int64)
    if m == 1:
        ok = np.abs(t_index[0] - q) <= int(tol_ns)
        return np.where(ok, 0, -1).astype(np.int64)
    k = np.clip(np.searchsorted(t_index, q), 1, m - 1)
    left = k - 1
    take_left = np.abs(t_index[left] - q) <= np.abs(t_index[k] - q)
    pick = np.where(take_left, left, k)
    ok = np.abs(t_index[pick] - q) <= int(tol_ns)
    return np.where(ok, pick, -1).astype(np.int64)
