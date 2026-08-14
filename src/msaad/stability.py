"""Stability of a multiscale complexity metric across realizations (Fig. 2D/E).

A useful metric should give a consistent curve when recomputed on independent
realizations of the *same* process. This module quantifies that consistency as
the spread of z-scored curves at each scale, and compares two metrics'
stability. The paper's headline: MSAAD's dispersion grows more slowly with scale
than MSSE's for 0 ≤ β ≤ 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

__all__ = ["dispersion_curve", "dispersion_area", "compare_dispersion"]


def dispersion_curve(curves: np.ndarray) -> np.ndarray:
    """Per-scale normalized dispersion (coefficient of variation) across realizations.

    At each scale, the standard deviation of the metric across realizations is
    divided by its mean — a scale- and magnitude-independent measure of how
    reproducible the metric is. Lower ⇒ more stable. The paper's headline is that
    this grows more slowly with scale for MSAAD than for sample entropy (β ≥ 0).

    Parameters
    ----------
    curves
        Array of shape ``(n_realizations, n_scales)``.
    """
    curves = np.asarray(curves, dtype=float)
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0)
    return std / np.where(np.abs(mean) < 1e-12, np.nan, np.abs(mean))


def dispersion_area(curves: np.ndarray, scales: np.ndarray) -> float:
    """Area under the dispersion curve (Fig. 2E) — a scalar stability score."""
    disp = dispersion_curve(curves)
    scales = np.asarray(scales, dtype=float)
    ok = np.isfinite(disp)
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2 / 1 compat
    return float(trapz(disp[ok], scales[ok])) if ok.sum() > 1 else np.nan


def compare_dispersion(
    curves_a: np.ndarray,
    curves_b: np.ndarray,
    scales: np.ndarray,
    *,
    bin_size: int = 20,
) -> pd.DataFrame:
    """Compare two metrics' dispersion in non-overlapping scale bins (Fig. 2D).

    Within each bin of ``bin_size`` scales, the two metrics' per-scale dispersion
    values are compared with a Mann–Whitney U test (as in the paper).

    Returns
    -------
    pandas.DataFrame
        Rows per bin: ``scale_lo, scale_hi, disp_a, disp_b, u_stat, p_value``.
    """
    disp_a = dispersion_curve(curves_a)
    disp_b = dispersion_curve(curves_b)
    scales = np.asarray(scales)

    rows = []
    for lo in range(0, len(scales), bin_size):
        hi = min(lo + bin_size, len(scales))
        a = disp_a[lo:hi][np.isfinite(disp_a[lo:hi])]
        b = disp_b[lo:hi][np.isfinite(disp_b[lo:hi])]
        if len(a) < 2 or len(b) < 2:
            continue
        u, p = mannwhitneyu(a, b)
        rows.append(
            {
                "scale_lo": int(scales[lo]),
                "scale_hi": int(scales[hi - 1]),
                "disp_a": float(np.median(a)),
                "disp_b": float(np.median(b)),
                "u_stat": float(u),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(rows)
