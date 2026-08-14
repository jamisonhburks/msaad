#!/usr/bin/env python3
"""Regenerate paper-analogous figures from the processed artifacts.

Produces (as .png + .pdf in results/figures/):
    fig2_noise_curves    MSAAD curves per noise colour, linear + log–log
    fig3a_beta_slope     β ↔ log–log MSAAD slope law
    fig3b_runtime        MSAAD vs. sample-entropy runtime
    fig4_diabetes        summed MSAAD by sex/diabetes group (awake & asleep)
    fig5_age_spearman    Spearman ρ(AAD, age) vs. scale, by sex
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import base_parser
from msaad import plotting as viz
from msaad.config import FIGURES_DIR, NOISE_BETAS


def main() -> None:
    args = base_parser(__doc__).parse_args()
    viz.apply_style()
    pdir = args.processed_dir

    # --- Figs 2–3 from the noise benchmarks ---
    if (pdir / "noise_benchmarks.npz").exists():
        nb = np.load(pdir / "noise_benchmarks.npz")
        scales = nb["scales"]
        msaad_by_beta = {b: nb[f"msaad_b{b}"] for b in NOISE_BETAS}

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
        viz.plot_noise_curves(msaad_by_beta, scales, ax=axes[0], log=False)
        viz.plot_noise_curves(msaad_by_beta, scales, ax=axes[1], log=True)
        fig.suptitle("MSAAD of 1/fᵝ noise")
        viz.save_figure(fig, "fig2_noise_curves")

        fig, ax = plt.subplots(figsize=(4, 3))
        viz.plot_beta_slope_law(nb["beta_grid"], nb["slopes"], ax=ax)
        viz.save_figure(fig, "fig3a_beta_slope")

        fig, ax = plt.subplots(figsize=(4, 3))
        viz.plot_runtime(nb["sizes"], {"MSAAD": nb["t_msaad"], "SampEn": nb["t_sampen"]}, ax=ax)
        viz.save_figure(fig, "fig3b_runtime")

    # --- Fig 4 diabetes ---
    if (pdir / "diabetes_features.parquet").exists():
        feats = pd.read_parquet(pdir / "diabetes_features.parquet")
        feats = feats[(feats["age"] >= 45) & (feats["age"] <= 65)]
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False, layout="constrained")
        for ax, state in zip(axes, ("awake", "asleep")):
            viz.plot_diabetes_boxplot(feats[feats["state"] == state], ax=ax)
            ax.set_title(state)
        fig.suptitle("MSAAD separates diabetes cohorts")
        viz.save_figure(fig, "fig4_diabetes")

    # --- Fig 5 age ---
    if (pdir / "age_spearman.parquet").exists():
        sp = pd.read_parquet(pdir / "age_spearman.parquet")
        by_sex = {sex: sp[sp["sex"] == sex] for sex in ("male", "female")}
        fig, ax = plt.subplots(figsize=(5, 3))
        viz.plot_age_spearman(by_sex, ax=ax)
        viz.save_figure(fig, "fig5_age_spearman")

    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
