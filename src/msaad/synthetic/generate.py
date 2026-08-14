"""Synthetic Oura-like skin-temperature generator.

The TemPredict/Oura data are not redistributable, so this module synthesizes
temperature records whose *fluctuation structure* carries planted diabetes, sex,
and age effects — the quantities MSAAD measures. Running the analyses on this
data recovers the paper's qualitative findings, which validates the estimators
and code path (not human physiology).

Each participant's temperature is a baseline plus a two-timescale fluctuation:
a fast (whiter) component that dominates short MSAAD scales and a slow (browner)
component that dominates long scales. Planting effects on these two amplitudes
independently lets the synthetic reproduce the paper's *scale-dependent* age
effect (men lose short-scale variability, women long-scale) and the
state-dependent diabetes effect (controls higher awake, diabetics higher asleep).

Output schema:
    <out_dir>/<pid>.parquet   columns: temp_skin, met, is_awake, day
    <out_dir>/demographics.parquet   index pid; age, sex, group
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import COL_AWAKE, COL_MET, COL_PID, COL_TEMP
from ..noise import powerlaw_noise

__all__ = ["generate_participant", "generate_dataset"]

_DIABETES_PREVALENCE = 0.18
_BETA_FAST, _BETA_SLOW = 0.4, 1.8   # spectral exponents of the two components


def _sample_demographics(rng: np.random.Generator) -> dict:
    age = float(np.clip(rng.normal(50, 16), 20, 79))
    sex = str(rng.choice(["male", "female"]))
    group = "DM" if rng.random() < _DIABETES_PREVALENCE else "No-DM"
    return {"age": age, "sex": sex, "group": group}


def _amplitudes(demo: dict, state: str, gain: float) -> tuple[float, float]:
    """Fast/slow fluctuation amplitudes (°C) for one participant and state.

    Encodes the planted effects: a per-participant idiosyncratic ``gain`` (so
    cohort effect sizes are realistic, not perfect), age-related "critical
    slowing down" that is scale- and sex-dependent, and the state-dependent
    diabetes effect. The fast component dominates short MSAAD scales, the slow
    component long scales, so per-component age effects become per-scale effects.
    """
    age_z = (demo["age"] - 50) / 16.0
    fast, slow = 0.42 * gain, 0.28 * gain

    # Critical slowing down concentrated at different scales per sex: men lose
    # short-scale (fast) variability, women long-scale (slow); the other
    # component is nearly age-invariant (matching the paper's <30 min / >80 min
    # crossover).
    if demo["sex"] == "male":
        fast *= np.exp(-0.45 * age_z)
        slow *= np.exp(-0.05 * age_z)
    else:
        fast *= np.exp(-0.05 * age_z)
        slow *= np.exp(-0.45 * age_z)

    # Diabetes effect reverses with state (Section 3.3); stronger in women.
    sex_gain = 1.5 if demo["sex"] == "female" else 1.0
    if demo["group"] == "DM":
        factor = np.exp(-0.24 * sex_gain) if state == "awake" else np.exp(0.16 * sex_gain)
        fast *= factor
        slow *= factor
    return float(fast), float(slow)


def _segment(
    length: int, demo: dict, state: str, gain: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one temperature segment (and its MET) for a state."""
    fast, slow = _amplitudes(demo, state, gain)
    baseline = 34.0 if state == "awake" else 35.6
    fluct = fast * powerlaw_noise(_BETA_FAST, length, rng=rng) + \
        slow * powerlaw_noise(_BETA_SLOW, length, rng=rng)
    temp = baseline + fluct

    # MET: higher/more variable awake; only weakly coupled to fluctuation
    # amplitude, so it is a genuine but non-collinear covariate — the diabetes
    # effect on MSAAD survives adjusting for MET (paper Table 2).
    met_base = 1.3 if state == "awake" else 0.95
    met = met_base + 0.05 * fast * rng.standard_normal(length) + rng.gamma(1.0, 0.12, length)
    return temp, np.clip(met, 0.9, None)


def generate_participant(
    pid: str, *, n_days: int = 6, rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, dict]:
    """Generate a multi-day record and demographics for one participant."""
    rng = np.random.default_rng() if rng is None else rng
    demo = _sample_demographics(rng)
    gain = float(np.exp(rng.normal(0, 0.26)))  # idiosyncratic between-person spread

    blocks = []
    for day in range(n_days):
        awake_len = int(rng.integers(660, 960))    # ~11–16 waking hours
        asleep_len = int(rng.integers(320, 460))   # ~5–7.5 h sleep
        for state, length, awake in (("awake", awake_len, True), ("asleep", asleep_len, False)):
            temp, met = _segment(length, demo, state, gain, rng)
            blocks.append(pd.DataFrame({COL_TEMP: temp, COL_MET: met,
                                        COL_AWAKE: awake, "day": day}))
    record = pd.concat(blocks, ignore_index=True)
    demo[COL_PID] = pid
    return record, demo


def generate_dataset(
    n_participants: int, out_dir: str | Path, *, n_days: int = 6, seed: int = 0
) -> pd.DataFrame:
    """Generate ``n_participants`` synthetic records + a demographics table.

    Writes ``<pid>.parquet`` per participant and ``demographics.parquet`` into
    ``out_dir`` (clearing any prior synthetic files first) and returns the
    demographics frame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out_dir.glob("P*.parquet"), out_dir / "demographics.parquet"]:
        stale.unlink(missing_ok=True)

    master = np.random.default_rng(seed)
    demographics = []
    for i in range(n_participants):
        pid = f"P{i:04d}"
        rng = np.random.default_rng(master.integers(0, 2**63))
        record, demo = generate_participant(pid, n_days=n_days, rng=rng)
        record.to_parquet(out_dir / f"{pid}.parquet")
        demographics.append(demo)

    demo_df = pd.DataFrame(demographics).set_index(COL_PID)
    demo_df.to_parquet(out_dir / "demographics.parquet")
    return demo_df
