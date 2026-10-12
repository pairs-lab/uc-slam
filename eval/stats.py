"""Cell statistics and significance tests for the pre-registered protocol
(docs/protocol.md).

Cells
-----
Since protocol amendment A10 every accuracy cell is ``mean ± std`` over the
successful runs of one (method, sequence), the sample standard deviation
(ddof 1). The run count is stated in the caption. A superscript marks only
failed runs, which stay excluded:

    3.42$\\pm$0.61                  every run passed (no space around the sign, as the ORB-SLAM3 tables print it)
    3.42$\\pm$0.61$^{2\\,\\mathrm{f}}$   2 runs failed (failure rule in protocol.md)
    \\fail                           no successful run

``stat="median_iqr"`` gives the registered form of protocol Section 4,
``median (IQR)`` with the run count as a superscript, which the credential
table keeps (its criterion reads the IQR). IQR = Q3 - Q1 with numpy's default
(linear) quantile interpolation.

Tests
-----
paired_wilcoxon      scipy.stats.wilcoxon, two-sided, method='exact', paired over
                     sequences (n = 11 on EuRoC detects a consistent ~15 % effect
                     at p < 0.05: that is the minimum effect worth claiming).
bootstrap_median_diff_ci  percentile bootstrap CI on the median paired difference.
holm                 Holm step-down adjustment of a family of p-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _st


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def median_iqr(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """(median, q25, q75, iqr) over finite values; NaN quadruple if empty."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return (float(med), float(q25), float(q75), float(q75 - q25))


# Statistic of a table cell, protocol amendment A10: "mean_std" or "median_iqr".
CELL_STAT = "mean_std"


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """(mean, sample std with ddof 1) over finite values; std is NaN with fewer than two values."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (np.nan, np.nan)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else np.nan)


def cell_center(values: Sequence[float], stat: Optional[str] = None) -> float:
    """The central value a cell prints and is ranked by: the mean, or the median under median_iqr."""
    return mean_std(values)[0] if (stat or CELL_STAT) == "mean_std" else median_iqr(values)[0]


def success_count(values: Sequence[float], n_runs: int) -> Tuple[int, int]:
    """(n_success, n_fail): successes are the finite entries; failures = n_runs - successes."""
    n_ok = int(sum(1 for x in values if np.isfinite(x)))
    if n_ok > n_runs:
        raise ValueError(f"{n_ok} successful values but only {n_runs} runs declared")
    return n_ok, n_runs - n_ok


def format_cell(values: Sequence[float], n_runs: int = 10, precision: int = 1,
                scale: float = 1.0, fail_macro: str = r"\fail",
                show_n: bool = True, stat: Optional[str] = None) -> str:
    """One table cell. Under mean_std (A10): 'mean $\\pm$ std', with '$^{k\\,\\mathrm{f}}$' when k runs
    failed. Under median_iqr: 'median (IQR)$^{n_ok}$' or '$^{n_ok/n_fail}$' when there were failures.

    ``values`` holds one number per successful run (NaN/inf marks a failed run);
    ``scale`` converts units for display (e.g. 100 for m -> cm). ``show_n`` False drops the superscript.
    """
    n_ok, n_fail = success_count(values, n_runs)
    if n_ok == 0:
        return fail_macro
    if (stat or CELL_STAT) == "mean_std":
        m, sd = mean_std(values)
        body = f"{m * scale:.{precision}f}" + (rf"$\pm${sd * scale:.{precision}f}" if np.isfinite(sd) else "")
        if show_n and n_fail:
            body += rf"$^{{{n_fail}\,\mathrm{{f}}}}$"
        return body
    med, _, _, iqr = median_iqr(values)
    body = f"{med * scale:.{precision}f} ({iqr * scale:.{precision}f})"
    if not show_n:
        return body
    sup = f"{n_ok}" if n_fail == 0 else f"{n_ok}/{n_fail}"
    return body + f"$^{{{sup}}}$"


# ---------------------------------------------------------------------------
# Paired comparisons across sequences
# ---------------------------------------------------------------------------


@dataclass
class PairedTest:
    n: int
    n_zero_diff: int
    statistic: float
    p_value: float
    method: str
    median_diff: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_boot: int

    def summary(self, unit: str = "") -> str:
        return (f"n={self.n}, W={self.statistic:g}, p={self.p_value:.4g} ({self.method}), "
                f"median diff={self.median_diff:.4g}{unit} "
                f"[{self.ci_low:.4g}, {self.ci_high:.4g}] ({self.ci_level:.0%} bootstrap, B={self.n_boot})")


def paired_wilcoxon(a: Sequence[float], b: Sequence[float],
                    alternative: str = "two-sided") -> Tuple[float, float, str, int]:
    """Paired Wilcoxon signed-rank on per-sequence medians a vs b.

    Returns (statistic, p, method_used, n_zero_diff). Uses scipy's exact
    distribution (method='exact'); pairs with zero difference are discarded
    (scipy zero_method='wilcox'), and their count is returned so the caption
    can say so. Falls back to the asymptotic method only if scipy refuses.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of equal length (one entry per sequence)")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    d = a - b
    n_zero = int(np.sum(d == 0))
    if np.count_nonzero(d) == 0:
        return float("nan"), 1.0, "degenerate (all differences zero)", n_zero
    try:
        res = _st.wilcoxon(a, b, alternative=alternative, method="exact", zero_method="wilcox")
        method = "exact"
    except (ValueError, TypeError):
        res = _st.wilcoxon(a, b, alternative=alternative, method="approx", zero_method="wilcox")
        method = "approx"
    return float(res.statistic), float(res.pvalue), method, n_zero


def bootstrap_median_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                             level: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    """(median(a-b), ci_low, ci_high): percentile bootstrap over the paired differences."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    meds = np.median(d[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return (float(np.median(d)), float(lo), float(hi))


def paired_comparison(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                      level: float = 0.95, seed: int = 0,
                      alternative: str = "two-sided") -> PairedTest:
    """Everything the protocol asks for in one object (a - b is the reported difference)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    stat, p, method, n_zero = paired_wilcoxon(a[mask], b[mask], alternative=alternative)
    med, lo, hi = bootstrap_median_diff_ci(a[mask], b[mask], n_boot=n_boot, level=level, seed=seed)
    return PairedTest(n=int(mask.sum()), n_zero_diff=n_zero, statistic=stat, p_value=p,
                      method=method, median_diff=med, ci_low=lo, ci_high=hi,
                      ci_level=level, n_boot=n_boot)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input), clipped to 1.

    Reject H_i at level alpha iff adjusted p_i <= alpha. Matches R's
    p.adjust(p, "holm").
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adj_sorted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj_sorted[rank] = running
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def holm_reject(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    return holm(p_values) <= alpha


# ---------------------------------------------------------------------------
# Convenience over the shared data contract
# ---------------------------------------------------------------------------

# results: {sequence: {method: [per-run metric, NaN for a failed run]}}
Results = Mapping[str, Mapping[str, Sequence[float]]]


def per_sequence_medians(results: Results, method: str,
                         sequences: Optional[Sequence[str]] = None) -> np.ndarray:
    seqs = list(sequences) if sequences is not None else list(results)
    return np.array([median_iqr(results[s][method])[0] if method in results[s] else np.nan
                     for s in seqs])


def compare_methods(results: Results, method_a: str, method_b: str,
                    sequences: Optional[Sequence[str]] = None, **kw) -> PairedTest:
    """Paired test of per-sequence medians, method_a - method_b."""
    seqs = list(sequences) if sequences is not None else list(results)
    return paired_comparison(per_sequence_medians(results, method_a, seqs),
                             per_sequence_medians(results, method_b, seqs), **kw)


def family_comparison(results: Results, reference: str, methods: Sequence[str],
                      sequences: Optional[Sequence[str]] = None, alpha: float = 0.05,
                      **kw) -> Dict[str, dict]:
    """reference vs each method with Holm correction over the family."""
    tests = {m: compare_methods(results, reference, m, sequences, **kw) for m in methods}
    adj = holm([tests[m].p_value for m in methods])
    return {m: {"test": tests[m], "p_holm": float(adj[i]), "reject": bool(adj[i] <= alpha)}
            for i, m in enumerate(methods)}
