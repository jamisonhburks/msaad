"""Tests for the from-scratch baseline complexity metrics."""

from __future__ import annotations

import numpy as np
import pytest

from msaad.baselines import (
    hurst_exponent,
    katz_fractal_dimension,
    multiscale,
    permutation_entropy,
    sample_entropy,
)
from msaad.noise import powerlaw_noise


def test_katz_constant_and_range():
    assert np.isnan(katz_fractal_dimension(np.ones(50)))     # no variation
    x = powerlaw_noise(1.0, 1024, rng=np.random.default_rng(0))
    assert katz_fractal_dimension(x) > 1.0                   # a real curve has D > 1


def test_sample_entropy_regular_lower_than_random():
    rng = np.random.default_rng(0)
    regular = np.tile([0.0, 1.0], 200)                       # perfectly periodic
    random = rng.standard_normal(400)
    assert sample_entropy(regular) < sample_entropy(random)


def test_permutation_entropy_bounds():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(2000)
    pe = permutation_entropy(white, order=3)
    assert 0.9 < pe <= 1.0                                   # white noise ≈ maximal
    assert permutation_entropy(np.arange(100.0)) < 0.1       # monotone ≈ minimal


def test_hurst_white_noise_near_half():
    rng = np.random.default_rng(0)
    h = hurst_exponent(rng.standard_normal(4096))
    assert 0.3 < h < 0.7                                     # uncorrelated ⇒ H≈0.5


def test_multiscale_wraps_any_metric():
    x = powerlaw_noise(1.0, 2048, rng=np.random.default_rng(0))
    scales = np.arange(1, 11)
    curve = multiscale(x, katz_fractal_dimension, scales)
    assert curve.shape == scales.shape


@pytest.mark.parametrize("beta", [0.0, 1.0, 2.0])
def test_sample_entropy_finite_on_short_scales(beta):
    x = powerlaw_noise(beta, 1024, rng=np.random.default_rng(1))
    assert np.isfinite(sample_entropy(x))
