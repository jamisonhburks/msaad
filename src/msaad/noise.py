"""Power-law (1/fᵝ) noise generation.

Synthetic 1/fᵝ processes are the paper's controlled test bed: their spectral
exponent β is known, so we can check how faithfully each complexity metric
recovers it and how stable each is across realizations. β indexes the familiar
"colours of noise": violet (−2), blue (−1), white (0), pink (+1), brown (+2).

The generator follows the manuscript's Methods 2.1: white noise → FFT → shape
amplitudes by 1/f^(β/2) → normalize total power → inverse FFT with random phases.
"""

from __future__ import annotations

import numpy as np

__all__ = ["powerlaw_noise"]


def powerlaw_noise(
    beta: float, n: int, *, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate one real-valued 1/fᵝ noise realization of length ``n``.

    Parameters
    ----------
    beta
        Spectral exponent: ``P(f) ∝ 1/f^β``. 0 = white, 1 = pink, 2 = brown,
        negative = blue/violet.
    n
        Number of samples.
    rng
        NumPy generator for reproducibility.

    Returns
    -------
    numpy.ndarray
        Zero-mean, unit-variance 1/fᵝ series.

    Notes
    -----
    Amplitudes are scaled by ``1/f^(β/2)`` because power is amplitude squared, so
    ``|X(f)|² ∝ 1/f^β`` gives the target power spectrum. The DC term is left
    unshaped, and the output is standardized to unit variance so realizations are
    comparable across β.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Frequencies (rfft grid); avoid dividing by f=0 at DC.
    freqs = np.fft.rfftfreq(n)
    scaling = np.ones_like(freqs)
    nonzero = freqs > 0
    scaling[nonzero] = freqs[nonzero] ** (-beta / 2.0)

    # White-noise spectrum with random phases, shaped to the power law.
    white = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    spectrum = white * scaling

    y = np.fft.irfft(spectrum, n=n)
    y = y - y.mean()
    std = y.std()
    return y / std if std > 0 else y
