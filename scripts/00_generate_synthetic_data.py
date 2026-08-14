#!/usr/bin/env python3
"""Stage 0 — generate a synthetic Oura-like dataset.

Writes ``<pid>.parquet`` per participant and ``demographics.parquet`` so the
diabetes and age analyses run without any real data.

    python scripts/00_generate_synthetic_data.py --n-participants 500 --n-days 6
"""

from __future__ import annotations

from _common import base_parser, ensure_dir
from msaad.synthetic import generate_dataset


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--n-participants", type=int, default=500)
    parser.add_argument("--n-days", type=int, default=6)
    args = parser.parse_args()

    ensure_dir(args.data_dir)
    demo = generate_dataset(args.n_participants, args.data_dir,
                            n_days=args.n_days, seed=args.seed)
    print(f"Wrote {args.n_participants} participants to {args.data_dir}")
    print(f"Diabetes (DM) fraction: {(demo['group'] == 'DM').mean():.0%} | "
          f"age 45–65: {((demo['age'] >= 45) & (demo['age'] <= 65)).sum()}")


if __name__ == "__main__":
    main()
