#!/usr/bin/env python3
"""Stage 3 — age effect on temperature complexity (Section 3.5; Fig. 5).

Extracts per-participant MSAAD curves over a wide scale range and correlates AAD
with age at each scale, separately by sex — exposing the scale- and sex-dependent
"critical slowing down" of aging temperature dynamics. Writes the per-scale
Spearman results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dir
from msaad.cohorts import age_scale_spearman
from msaad.config import DEFAULT_CONFIG
from msaad.io import load_demographics
from msaad.pipeline import extract_age_features


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = DEFAULT_CONFIG
    scales = np.asarray(config.age_scales)

    features = extract_age_features(args.data_dir, config=config,
                                    progress=not args.no_progress)
    demo = load_demographics(args.data_dir / "demographics.parquet")
    features = features.join(demo)

    ensure_dir(args.processed_dir)
    features.to_parquet(args.processed_dir / "age_features.parquet")

    scale_cols = [f"s{s}" for s in scales]
    results = []
    for sex in ("male", "female"):
        sub = features[features["sex"] == sex]
        sp = age_scale_spearman(sub[scale_cols].to_numpy(), sub["age"].to_numpy(), scales)
        sp["sex"] = sex
        results.append(sp)
        short = sp.loc[sp["scale"] < 30, "rho"].mean()
        long = sp.loc[sp["scale"] > 80, "rho"].mean()
        print(f"{sex}: mean ρ short(<30 min)={short:+.2f}, long(>80 min)={long:+.2f}")

    out = pd.concat(results, ignore_index=True)
    out.to_parquet(args.processed_dir / "age_spearman.parquet")
    print(f"Wrote age results to {args.processed_dir}")


if __name__ == "__main__":
    main()
