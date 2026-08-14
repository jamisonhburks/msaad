"""msaad — Multiscale Average Absolute Difference.

A fast, nonparametric multiscale complexity feature for noisy, uncontrolled
wearable time series. Reference implementation of:

    Burks et al., "Multiscale Average Absolute Difference (MSAAD): A
    Computationally Efficient and Nonparametric Adaptation of Line Length for
    Noisy, Uncontrolled Wearables Time Series", Algorithms 18(9):577 (2025).
    https://doi.org/10.3390/a18090577

Modules (one concern each):

    config       hyper-parameters, constants, noise conventions
    coarsegrain  multiscale coarse-graining (Eq. 1)
    msaad        the MSAAD algorithm + β↔slope law (Eqs. 2–3) — the core
    baselines    from-scratch KFD / SampEn / permutation entropy / Hurst
    noise        1/fᵝ power-law noise generator
    stability    across-realization stability comparison (Fig. 2D/E)
    stats_utils  Cohen's d, Dunn's post-hoc, corrections
    cohorts      diabetes (Kruskal + Dunn + ANCOVA) and age (Spearman) analyses
    io           schema-validated loading
    pipeline     feature extraction over a directory of records
    plotting     paper-analogous figures
    synthetic    forward-model data generator (no real data required)

Quick start::

    from msaad import msaad
    curve = msaad(signal, scales=range(1, 31))
"""

from __future__ import annotations

from .baselines import (
    hurst_exponent,
    katz_fractal_dimension,
    multiscale,
    permutation_entropy,
    sample_entropy,
)
from .coarsegrain import coarse_grain
from .config import DEFAULT_CONFIG, MSAADConfig
from .msaad import aad, beta_from_slope, loglog_slope, msaad, slope_from_beta
from .noise import powerlaw_noise

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "MSAADConfig",
    "DEFAULT_CONFIG",
    "coarse_grain",
    "aad",
    "msaad",
    "loglog_slope",
    "beta_from_slope",
    "slope_from_beta",
    "powerlaw_noise",
    "multiscale",
    "katz_fractal_dimension",
    "sample_entropy",
    "permutation_entropy",
    "hurst_exponent",
]
