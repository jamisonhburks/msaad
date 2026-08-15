"""Loading and schema validation of participant temperature records.

The pipeline is source-agnostic: it needs per-participant frames with the
columns in :data:`REQUIRED_COLUMNS` plus a demographics table. Real Oura exports
and the bundled synthetic generator both satisfy this contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .config import COL_AWAKE, COL_PID, COL_TEMP

REQUIRED_COLUMNS: tuple[str, ...] = (COL_TEMP, COL_AWAKE)

__all__ = [
    "REQUIRED_COLUMNS",
    "NON_PARTICIPANT_STEMS",
    "load_participant",
    "iter_participants",
    "load_demographics",
]


def load_participant(path: str | Path) -> pd.DataFrame:
    """Load one participant's parquet record and validate its schema."""
    path = Path(path)
    df = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.stem}: missing required columns {sorted(missing)}")
    return df


#: Parquet files in a data directory that are not participant records and must
#: be skipped when iterating (e.g. the demographics table).
NON_PARTICIPANT_STEMS: frozenset[str] = frozenset({"demographics"})


def iter_participants(
    directory: str | Path, *, pattern: str = "*.parquet"
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(participant_id, frame)`` for every participant parquet in ``directory``.

    The default ``*.parquet`` glob works for real exports (hashed-id filenames)
    and synthetic files (``P0000`` …) alike; non-participant tables such as
    ``demographics.parquet`` are skipped (see :data:`NON_PARTICIPANT_STEMS`), so
    pointing this at a data directory "just works" without a custom pattern.
    """
    for path in sorted(Path(directory).glob(pattern)):
        if path.stem in NON_PARTICIPANT_STEMS:
            continue
        yield path.stem, load_participant(path)


def load_demographics(path: str | Path) -> pd.DataFrame:
    """Load the demographics table (``age``, ``sex``, ``group``) indexed by pid."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if df.index.name != COL_PID and COL_PID in df.columns:
        df = df.set_index(COL_PID)
    return df
