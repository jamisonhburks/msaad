"""Multiscale coarse-graining (Costa 2002).

Coarse-graining at scale τ replaces the series with the means of consecutive,
non-overlapping windows of length τ — a low-pass filter that reveals structure
at progressively slower timescales. It is the shared first step of every
multiscale complexity metric here (MSAAD and all baselines).

    yⱼ(τ) = (1/τ) · Σ_{i=(j-1)τ+1}^{jτ} xᵢ ,   1 ≤ j ≤ ⌊N/τ⌋        (paper Eq. 1)
"""

from __future__ import annotations

import numpy as np

__all__ = ["coarse_grain"]


def coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    """Non-overlapping moving average of ``x`` at the given ``scale`` (τ).

    Parameters
    ----------
    x
        1-D input series.
    scale
        Window length τ ≥ 1. ``scale == 1`` returns the series unchanged.

    Returns
    -------
    numpy.ndarray
        The coarse-grained series of length ``len(x) // scale``. Trailing
        samples that do not fill a full window are dropped.
    """
    x = np.asarray(x, dtype=float)
    if scale < 1:
        raise ValueError("scale must be >= 1")
    if scale == 1:
        return x
    n_windows = x.size // scale
    if n_windows == 0:
        return np.empty(0)
    return x[: n_windows * scale].reshape(n_windows, scale).mean(axis=1)
