#!/usr/bin/env python3
"""Stage 2 — diabetes cohort separation with MSAAD (Section 3.3; Fig. 4, Tables 1–2).

Extracts summed-MSAAD features from synthetic temperature, compares DM Dx vs
No-DM Dx (ages 45–65) by sex with Kruskal–Wallis + Dunn's post-hoc + Cohen's d,
and fits the MSAAD ~ Group + MET ANCOVA. Writes features + result tables.
"""

from __future__ import annotations

from _common import base_parser, ensure_dir
from msaad.cohorts import compare_diabetes_cohorts, msaad_met_ancova
from msaad.config import DEFAULT_CONFIG
from msaad.io import load_demographics
from msaad.pipeline import extract_diabetes_features


def main() -> None:
    args = base_parser(__doc__).parse_args()
    config = DEFAULT_CONFIG

    features = extract_diabetes_features(args.data_dir, config=config,
                                         progress=not args.no_progress)
    demo = load_demographics(args.data_dir / "demographics.parquet")
    features = features.merge(demo, left_on="pid", right_index=True)
    cohort = features[(features["age"] >= config.age_min) & (features["age"] <= config.age_max)]

    ensure_dir(args.processed_dir)
    features.to_parquet(args.processed_dir / "diabetes_features.parquet")

    print(f"{cohort['pid'].nunique()} participants in the 45–65 cohort\n")
    for state in ("awake", "asleep"):
        res = compare_diabetes_cohorts(cohort[cohort["state"] == state])
        dm = res.cohens_d["No-DM_male_vs_DM_male"]
        df_ = res.cohens_d["No-DM_female_vs_DM_female"]
        print(f"[{state}] Kruskal p={res.kruskal_p:.1e} | "
              f"Cohen's d (No-DM vs DM): male={dm:+.2f}, female={df_:+.2f}")

    ancova = msaad_met_ancova(cohort)
    ancova.to_csv(args.processed_dir / "diabetes_ancova.csv")
    grp = ancova.xs("Group", level="term")["p_value"]
    print("\nANCOVA (MSAAD ~ Group + MET) — Group p-values:")
    print(grp.to_string())


if __name__ == "__main__":
    main()
