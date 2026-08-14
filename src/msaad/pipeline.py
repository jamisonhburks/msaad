"""High-level feature extraction over a directory of participant records.

Two extractors mirror the paper's two test cases:

* :func:`extract_diabetes_features` — per participant × state (awake/asleep),
  the median summed-MSAAD curve across days plus mean MET.
* :func:`extract_age_features` — per participant, the MSAAD curve over a wide
  scale range computed on the concatenated awake series (long timescales need
  long records).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import COL_AWAKE, COL_MET, COL_TEMP, DEFAULT_CONFIG, MSAADConfig
from .io import iter_participants
from .msaad import msaad

__all__ = ["extract_diabetes_features", "extract_age_features"]

#: A day/state segment must be at least this many multiples of the largest scale.
_MIN_LEN_FACTOR = 3


def _iter(data_dir, progress, desc):
    it = iter_participants(data_dir)
    return tqdm(it, desc=desc) if progress else it


def extract_diabetes_features(
    data_dir: str | Path,
    *,
    config: MSAADConfig = DEFAULT_CONFIG,
    progress: bool = True,
) -> pd.DataFrame:
    """Median summed-MSAAD per participant × state (input to the diabetes test).

    Returns
    -------
    pandas.DataFrame
        Rows keyed by ``(pid, state)`` with ``msaad_sum``, ``met``, ``n_days``.
    """
    scales = np.asarray(config.scales)
    min_len = _MIN_LEN_FACTOR * int(scales.max())
    rows = []
    for pid, record in _iter(data_dir, progress, "diabetes features"):
        for state, awake in (("awake", True), ("asleep", False)):
            sums, mets = [], []
            for _, day in record[record[COL_AWAKE] == awake].groupby("day"):
                temp = day[COL_TEMP].to_numpy(dtype=float)
                if temp.size < min_len:
                    continue
                sums.append(np.nansum(msaad(temp, scales, config=config)))
                mets.append(day[COL_MET].mean())
            if sums:
                rows.append({"pid": pid, "state": state,
                             "msaad_sum": float(np.median(sums)),
                             "met": float(np.mean(mets)), "n_days": len(sums)})
    return pd.DataFrame(rows)


def extract_age_features(
    data_dir: str | Path,
    *,
    config: MSAADConfig = DEFAULT_CONFIG,
    progress: bool = True,
) -> pd.DataFrame:
    """MSAAD curve per participant over ``config.age_scales`` (awake series).

    The awake segments are concatenated so long coarse-graining scales have
    enough samples (the age effect lives at minute→hour scales).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``pid``; one column per scale (``s{τ}``).
    """
    scales = np.asarray(config.age_scales)
    cols = [f"s{s}" for s in scales]
    rows = {}
    for pid, record in _iter(data_dir, progress, "age features"):
        awake = record.loc[record[COL_AWAKE], COL_TEMP].to_numpy(dtype=float)
        if awake.size < _MIN_LEN_FACTOR * int(scales.max()):
            # Fall back gracefully: scales too large for this record yield nan.
            pass
        rows[pid] = msaad(awake, scales, config=config)
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)
