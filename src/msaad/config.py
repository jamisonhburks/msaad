"""Configuration, constants, and column conventions.

Defaults reproduce the analysis in:

    Burks et al., "Multiscale Average Absolute Difference (MSAAD): A
    Computationally Efficient and Nonparametric Adaptation of Line Length for
    Noisy, Uncontrolled Wearables Time Series", Algorithms 18(9):577 (2025).
    https://doi.org/10.3390/a18090577
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------- #
# Signal / data conventions (Oura Ring Gen2 skin temperature)
# --------------------------------------------------------------------------- #
COL_TEMP = "temp_skin"     # distal skin temperature (°C)
COL_MET = "met"            # metabolic equivalents (activity proxy)
COL_AWAKE = "is_awake"     # boolean sleep/wake label
COL_PID = "pid"

#: The five canonical 1/fᵝ "colours of noise" used throughout the paper.
NOISE_BETAS: tuple[int, ...] = (-2, -1, 0, 1, 2)
NOISE_LABELS: dict[int, str] = {-2: "violet", -1: "blue", 0: "white", 1: "pink", 2: "brown"}
NOISE_COLORS: dict[int, str] = {
    -2: "#8000ff", -1: "#1f77b4", 0: "#7f7f7f", 1: "#e377c2", 2: "#8c564b",
}


@dataclass(frozen=True)
class MSAADConfig:
    """Analysis hyper-parameters. Defaults reproduce the manuscript.

    Attributes
    ----------
    scales
        Coarse-graining scales τ at which AAD is evaluated. Feature-extraction
        on temperature uses 1…30 min (paper Section 3.3).
    n_realizations
        Independent 1/fᵝ realizations per β for the noise stability study.
    realization_length
        Length (samples) of each synthetic noise realization (2**13 in the paper).
    beta_fit_range
        β interval over which log(MSAAD)–log(scale) slope is linear in β
        (paper: [-1, 2]).
    age_min, age_max
        Age window for the diabetes cohort comparison (45–65).
    age_tranche_edges
        10-year age-tranche edges for the age analysis (20…80).
    age_subsample
        Per-tranche participant subsample size, to unbias the age correlation
        (117 in the paper — the smallest tranche).
    alpha
        Significance level.
    random_seed
        Global seed for reproducibility.
    """

    scales: tuple[int, ...] = tuple(range(1, 31))
    n_realizations: int = 100
    realization_length: int = 2**13
    beta_fit_range: tuple[float, float] = (-1.0, 2.0)

    age_min: int = 45
    age_max: int = 65
    age_tranche_edges: tuple[int, ...] = (20, 30, 40, 50, 60, 70, 80)
    age_subsample: int = 117

    alpha: float = 0.05
    random_seed: int = 0

    #: Scales for the age "critical slowing down" study — log-spaced to cover
    #: minutes → ~days (paper uses powers of two up to 4096 min).
    age_scales: tuple[int, ...] = field(
        default=tuple(int(s) for s in np.unique(np.logspace(0, 12, 50, base=2).astype(int))),
        repr=False,
    )


DEFAULT_CONFIG = MSAADConfig()
