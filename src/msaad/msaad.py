"""Multiscale Average Absolute Difference (MSAAD) — the core algorithm.

MSAAD is a multiscale adaptation of *line length*. At each coarse-graining scale
τ it measures the mean absolute first difference of the signal — the average
size of a minute-to-minute transition — and the MSAAD *curve* records how that
local variability changes with scale. It is fully nonparametric (its only inputs
are the scales), O(N) per scale, and robust to the isolated large perturbations
that distort fractal-dimension and entropy estimates on uncontrolled wearable
data (paper Sections 3.1, 4).

    L(τ)   = Σ |yᵢ(τ) − yᵢ₋₁(τ)|                          (line length, Eq. 2)
    AAD(τ) = L(τ) / (Nτ − 1)  =  mean |Δ y(τ)|            (Eq. 3)
    MSAAD  = [ AAD(τ) for τ in scales ]
"""

from __future__ import annotations

import numpy as np

from .coarsegrain import coarse_grain
from .config import DEFAULT_CONFIG, MSAADConfig

__all__ = ["aad", "msaad", "loglog_slope", "beta_from_slope", "slope_from_beta"]

# The paper's empirical law relating the log–log MSAAD slope to the 1/fᵝ power
# decay exponent β, fit over β ∈ [-1, 2] (p < 1e-4, R ≈ 0.997):  slope ≈ 0.48β − 0.48.
_SLOPE_VS_BETA = (0.48, -0.48)  # (gain, intercept)


def aad(x: np.ndarray) -> float:
    """Average absolute difference of a series: ``mean(|diff(x)|)`` (Eq. 3).

    This is line length normalized by the number of differences — a scale of the
    typical step size between consecutive samples.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x))))


def msaad(
    x: np.ndarray,
    scales: tuple[int, ...] | np.ndarray | None = None,
    *,
    config: MSAADConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Compute the MSAAD curve of a signal over a set of scales.

    Parameters
    ----------
    x
        1-D input series.
    scales
        Coarse-graining scales τ. Defaults to ``config.scales``.
    config
        Provides the default ``scales``.

    Returns
    -------
    numpy.ndarray
        AAD evaluated at each scale, same length as ``scales``. A scale that
        leaves fewer than two coarse-grained samples yields ``nan``.
    """
    scales = config.scales if scales is None else scales
    return np.array([aad(coarse_grain(x, int(s))) for s in scales])


def loglog_slope(
    curve: np.ndarray, scales: tuple[int, ...] | np.ndarray
) -> float:
    """Slope of ``log(MSAAD)`` vs. ``log(scale)`` by ordinary least squares.

    The slope characterizes how variability scales with resolution; for 1/fᵝ
    processes it is (approximately) an affine function of β — see
    :func:`beta_from_slope`.
    """
    curve = np.asarray(curve, dtype=float)
    scales = np.asarray(scales, dtype=float)
    ok = np.isfinite(curve) & (curve > 0)
    if ok.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(scales[ok]), np.log(curve[ok]), 1)
    return float(slope)


def slope_from_beta(beta: float | np.ndarray) -> float | np.ndarray:
    """Predicted log–log MSAAD slope for a 1/fᵝ process (paper's empirical law)."""
    gain, intercept = _SLOPE_VS_BETA
    return gain * np.asarray(beta) + intercept


def beta_from_slope(slope: float | np.ndarray) -> float | np.ndarray:
    """Estimate the 1/fᵝ exponent β from an observed log–log MSAAD slope.

    Inverts the paper's empirical law ``slope ≈ 0.48·β − 0.48`` (valid for
    β ∈ [-1, 2]). This lets MSAAD act as a cheap spectral-exponent estimator.
    """
    gain, intercept = _SLOPE_VS_BETA
    return (np.asarray(slope) - intercept) / gain
