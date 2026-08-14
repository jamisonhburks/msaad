"""Reference complexity metrics MSAAD is benchmarked against (paper Table 1).

Each is implemented from scratch in NumPy so the comparison is transparent and
dependency-free: Katz Fractal Dimension (KFD), Sample Entropy (SampEn),
Permutation Entropy (PE), and the Hurst exponent. ``multiscale`` wraps any
single-scale metric into its multiscale (coarse-grained) form, so MS-KFD, MSSE,
MSPE, etc. are one-liners — and MSAAD slots into the exact same interface.

Optional: install the ``[reference]`` extra to cross-check these against
neurokit2 / antropy in the test suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .coarsegrain import coarse_grain

__all__ = [
    "katz_fractal_dimension",
    "sample_entropy",
    "permutation_entropy",
    "hurst_exponent",
    "multiscale",
]


def katz_fractal_dimension(x: np.ndarray) -> float:
    """Katz fractal dimension of a 1-D series (Katz 1988; paper Eq. A1).

    ``D = log₁₀(n) / (log₁₀(n) + log₁₀(d / L))`` where ``L`` is the line length
    (Σ|Δx|), ``d`` the greatest distance from the first sample, ``a`` the mean
    step, and ``n = L / a`` the number of steps. KFD is dominated by ``d`` — the
    single largest excursion — which is why isolated perturbations (waking,
    activity) inflate it on uncontrolled wearable data (paper Section 4).
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.nan
    steps = np.abs(np.diff(x))
    L = steps.sum()
    a = steps.mean()
    d = np.max(np.abs(x - x[0]))
    if L == 0 or d == 0 or a == 0:
        return np.nan
    n = L / a
    return float(np.log10(n) / (np.log10(n) + np.log10(d / L)))


def sample_entropy(x: np.ndarray, *, m: int = 2, r: float | None = None) -> float:
    """Sample entropy (Richman & Moorman 2000).

    ``SampEn = −ln(A / B)`` where ``B`` counts template-pair matches of length
    ``m`` and ``A`` of length ``m+1`` within Chebyshev tolerance ``r`` (default
    ``0.2 · std(x)``). Returns ``nan`` when no matches exist — the instability at
    large coarse-graining scales that the paper highlights (Figure A2).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= m + 1:
        return np.nan
    r = 0.2 * x.std() if r is None else r
    if r == 0:
        return np.nan

    def _count(emb_dim: int) -> int:
        templates = np.array([x[i : i + emb_dim] for i in range(n - emb_dim)])
        count = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += np.count_nonzero(dist <= r)
        return count

    b = _count(m)
    a = _count(m + 1)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def permutation_entropy(x: np.ndarray, *, order: int = 3, delay: int = 1) -> float:
    """Normalized permutation entropy (Bandt & Pompe 2002).

    Shannon entropy of the distribution of ordinal patterns of length ``order``,
    normalized by ``ln(order!)`` to lie in ``[0, 1]``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < order * delay + 1:
        return np.nan
    # Ordinal pattern = argsort of each embedded window; encode as a factorial-
    # base integer so patterns can be counted with np.unique.
    idx = np.arange(0, order * delay, delay)
    windows = np.array([x[i + idx] for i in range(n - (order - 1) * delay)])
    patterns = np.argsort(windows, axis=1, kind="quicksort")
    codes = patterns @ (order ** np.arange(order))
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(math.factorial(order)))


def hurst_exponent(x: np.ndarray) -> float:
    """Hurst exponent via rescaled-range (R/S) analysis.

    Estimates long-range dependence: H≈0.5 is uncorrelated, H>0.5 persistent,
    H<0.5 anti-persistent. Computed as the log–log slope of the rescaled range
    against window size.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 16:
        return np.nan
    window_sizes = np.unique(np.floor(np.logspace(2, np.log10(n), 12) / 2 * 2).astype(int))
    window_sizes = window_sizes[(window_sizes >= 8) & (window_sizes <= n)]
    rs = []
    for w in window_sizes:
        chunks = n // w
        values = []
        for c in range(chunks):
            seg = x[c * w : (c + 1) * w]
            z = np.cumsum(seg - seg.mean())
            spread = z.max() - z.min()
            sd = seg.std()
            if sd > 0:
                values.append(spread / sd)
        if values:
            rs.append(np.mean(values))
    if len(rs) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(window_sizes[: len(rs)]), np.log(rs), 1)
    return float(slope)


def multiscale(
    x: np.ndarray,
    metric: Callable[[np.ndarray], float],
    scales: tuple[int, ...] | np.ndarray,
) -> np.ndarray:
    """Apply any single-scale ``metric`` to ``x`` coarse-grained at each scale.

    Turns a scalar complexity metric into its multiscale curve — the common
    interface behind MS-KFD, MSSE, MSPE, and MSAAD.
    """
    return np.array([metric(coarse_grain(x, int(s))) for s in scales])
