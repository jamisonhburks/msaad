"""Tests for the across-realization stability comparison."""

from __future__ import annotations

import numpy as np

from msaad.baselines import multiscale, sample_entropy
from msaad.msaad import msaad
from msaad.noise import powerlaw_noise
from msaad.stability import compare_dispersion, dispersion_area, dispersion_curve


def test_dispersion_curve_shape_and_nonnegative():
    curves = np.abs(np.random.default_rng(0).standard_normal((15, 20))) + 1.0
    disp = dispersion_curve(curves)
    assert disp.shape == (20,)
    assert np.all(disp[np.isfinite(disp)] >= 0)


def test_msaad_more_stable_than_sample_entropy():
    """On white noise, MSAAD's coefficient of variation is lower than SampEn's."""
    rng = np.random.default_rng(0)
    scales = np.arange(1, 25)
    m = np.array([msaad(powerlaw_noise(0.0, 1024, rng=rng), scales) for _ in range(20)])
    s = np.array([multiscale(powerlaw_noise(0.0, 1024, rng=rng), sample_entropy, scales)
                  for _ in range(20)])
    assert dispersion_area(m, scales) < dispersion_area(s, scales)


def test_compare_dispersion_returns_bins():
    rng = np.random.default_rng(1)
    scales = np.arange(1, 41)
    a = np.abs(rng.standard_normal((10, 40))) + 1
    b = np.abs(rng.standard_normal((10, 40))) + 1
    out = compare_dispersion(a, b, scales, bin_size=20)
    assert set(out.columns) >= {"scale_lo", "scale_hi", "p_value"}
    assert len(out) == 2
