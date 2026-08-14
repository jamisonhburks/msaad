"""Optional cross-checks of the from-scratch metrics against reference libraries.

These run only when the ``[reference]`` extra (neurokit2 / antropy) is installed;
otherwise pytest skips them. They guard the hand-written implementations against
drift from the community references.
"""

from __future__ import annotations

import numpy as np
import pytest

from msaad.baselines import katz_fractal_dimension, permutation_entropy
from msaad.msaad import aad
from msaad.noise import powerlaw_noise


@pytest.fixture
def signal() -> np.ndarray:
    return powerlaw_noise(1.0, 1024, rng=np.random.default_rng(0))


def test_aad_matches_neurokit_linelength(signal):
    nk = pytest.importorskip("neurokit2")
    # neurokit2's fractal_linelength returns the mean absolute difference (= AAD).
    ref = nk.fractal_linelength(signal)[0]
    assert np.isclose(aad(signal), ref, rtol=1e-6)


def test_katz_matches_neurokit(signal):
    nk = pytest.importorskip("neurokit2")
    ref = nk.fractal_katz(signal)[0]
    assert np.isclose(katz_fractal_dimension(signal), ref, rtol=1e-3)


def test_permutation_entropy_matches_antropy(signal):
    ant = pytest.importorskip("antropy")
    ref = ant.perm_entropy(signal, order=3, delay=1, normalize=True)
    assert np.isclose(permutation_entropy(signal, order=3, delay=1), ref, rtol=1e-3)
