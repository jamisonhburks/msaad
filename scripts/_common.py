"""Shared argument parsing and paths for the stage scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from msaad.config import PROCESSED_DIR, SYNTHETIC_DIR


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-dir", type=Path, default=SYNTHETIC_DIR,
                   help="directory of per-participant parquet records")
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR,
                   help="directory for intermediate/output artifacts")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--no-progress", action="store_true", help="disable progress bars")
    return p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
