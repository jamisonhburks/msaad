"""Tests for the core MSAAD algorithm and its scaling law."""

from __future__ import annotations

import numpy as np

from msaad.coarsegrain import coarse_grain
from msaad.msaad import aad, beta_from_slope, loglog_slope, msaad
from msaad.noise import powerlaw_noise


def test_aad_equals_mean_abs_diff():
    x = np.array([1, 2, 1, 20, 20, 20, 1.0])
    assert np.isclose(aad(x), np.mean(np.abs(np.diff(x))))  # paper's identity


def test_coarse_grain_shape_and_values():
    x = np.arange(10, dtype=float)
    cg = coarse_grain(x, 2)
    assert cg.size == 5
    assert np.allclose(cg, [0.5, 2.5, 4.5, 6.5, 8.5])
    assert np.array_equal(coarse_grain(x, 1), x)  # scale 1 is identity


def test_msaad_curve_length_matches_scales():
    x = powerlaw_noise(1.0, 2048, rng=np.random.default_rng(0))
    scales = np.arange(1, 21)
    curve = msaad(x, scales)
    assert curve.shape == scales.shape
    assert np.all(np.isfinite(curve))


def test_constant_signal_has_zero_aad():
    assert aad(np.ones(50)) == 0.0


def test_beta_slope_law_recovers_exponent():
    """The log–log MSAAD slope should recover β for 1/fᵝ noise (β ∈ [-1, 2])."""
    rng = np.random.default_rng(1)
    scales = np.arange(1, 97)
    for beta in (0.0, 1.0, 2.0):
        slopes = [loglog_slope(msaad(powerlaw_noise(beta, 2**14, rng=rng), scales), scales)
                  for _ in range(8)]
        assert abs(beta_from_slope(np.mean(slopes)) - beta) < 0.35


def test_msaad_monotone_in_amplitude():
    """Scaling a signal by c scales its AAD by c (linearity of line length)."""
    rng = np.random.default_rng(2)
    x = powerlaw_noise(1.0, 4096, rng=rng)
    scales = np.arange(1, 31)
    base = msaad(x, scales)
    scaled = msaad(3.0 * x, scales)
    assert np.allclose(scaled, 3.0 * base, rtol=1e-9)
