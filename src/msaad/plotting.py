"""Reusable plotting helpers for the paper-analogous figures.

Plotting is separate from computation: every function takes already-computed
arrays/DataFrames and draws onto an Axes. Figures save as both ``.png`` and
``.pdf``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, NOISE_COLORS, NOISE_LABELS

MSAAD_COLOR = "#e6550d"
BASELINE_COLOR = "#3182bd"

__all__ = [
    "apply_style",
    "save_figure",
    "plot_noise_curves",
    "plot_beta_slope_law",
    "plot_runtime",
    "plot_diabetes_boxplot",
    "plot_age_spearman",
]


def apply_style() -> None:
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig: plt.Figure, stem: str, *, directory: Path = FIGURES_DIR) -> None:
    """Save ``fig`` as ``<stem>.png`` and ``<stem>.pdf`` under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{ext}", bbox_inches="tight")


def plot_noise_curves(
    curves_by_beta: dict[int, np.ndarray],
    scales: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    log: bool = False,
) -> plt.Axes:
    """Median complexity curve per noise colour (cf. Fig. 2C).

    ``curves_by_beta`` maps β → array of shape ``(n_realizations, n_scales)``.
    On log axes the MSAAD curves are near-straight lines whose slope encodes β.
    """
    ax = ax or plt.gca()
    x = np.log(scales) if log else scales
    for beta, curves in curves_by_beta.items():
        med = np.nanmedian(curves, axis=0)
        sd = np.nanstd(curves, axis=0)
        y = np.log(med) if log else med
        ax.plot(x, y, color=NOISE_COLORS[beta], label=NOISE_LABELS[beta], lw=1.5)
        if not log:
            ax.fill_between(x, med - sd, med + sd, color=NOISE_COLORS[beta], alpha=0.2)
    ax.set_xlabel("log scale" if log else "scale (samples)")
    ax.set_ylabel("log AAD" if log else "AAD (a.u.)")
    ax.legend(fontsize=7, title="1/fᵝ")
    return ax


def plot_beta_slope_law(
    betas: np.ndarray, slopes: np.ndarray, *, fit_range=(-1.0, 2.0), ax: plt.Axes | None = None
) -> plt.Axes:
    """log–log MSAAD slope vs. β, with the linear fit over ``fit_range`` (Fig. 3A)."""
    from .msaad import slope_from_beta

    ax = ax or plt.gca()
    ax.scatter(betas, slopes, c="k", s=12, zorder=3)
    for beta, color in NOISE_COLORS.items():
        ax.axvline(beta, color=color, lw=1, alpha=0.6, zorder=0)
    grid = np.linspace(*fit_range, 50)
    ax.plot(grid, slope_from_beta(grid), "k--", lw=1, label="0.48β − 0.48")
    ax.set(xlabel="β power decay", ylabel="log(MSAAD)/log(scale) slope",
           title="β ↔ MSAAD slope")
    ax.legend(fontsize=7)
    return ax


def plot_runtime(
    sizes: np.ndarray, times: dict[str, np.ndarray], *, ax: plt.Axes | None = None
) -> plt.Axes:
    """Runtime vs. number of series, log–log (cf. Fig. 3B)."""
    ax = ax or plt.gca()
    for name, t in times.items():
        color = MSAAD_COLOR if name.upper().startswith("MSAAD") else BASELINE_COLOR
        ax.plot(sizes, t, marker="o", label=name, color=color)
    ax.set(xscale="log", yscale="log", xlabel="number of series",
           ylabel="time (s)", title="runtime")
    ax.legend(fontsize=7)
    return ax


def plot_diabetes_boxplot(features: pd.DataFrame, *, ax: plt.Axes | None = None) -> plt.Axes:
    """Summed-MSAAD by sex and diabetes group for one state (cf. Fig. 4 right)."""
    ax = ax or plt.gca()
    sns.boxplot(data=features, x="sex", y="msaad_sum", hue="group",
                hue_order=["No-DM", "DM"], fliersize=0, ax=ax,
                palette=[BASELINE_COLOR, MSAAD_COLOR])
    ax.set(xlabel="sex", ylabel="MSAAD sum")
    ax.legend(fontsize=7, title="")
    return ax


def plot_age_spearman(
    spearman_by_sex: dict[str, pd.DataFrame], *, alpha: float = 0.05, ax: plt.Axes | None = None
) -> plt.Axes:
    """Spearman ρ(AAD, age) vs. scale per sex (Fig. 5B) — critical slowing down.

    ``spearman_by_sex`` maps ``"male"``/``"female"`` → the frame returned by
    :func:`msaad.cohorts.age_scale_spearman`. Significant points are solid.
    """
    ax = ax or plt.gca()
    colors = {"male": "#e6842a", "female": "#3182bd"}
    for sex, df in spearman_by_sex.items():
        sig = df["p_value"] < alpha
        ax.scatter(df.loc[sig, "scale"], df.loc[sig, "rho"], color=colors[sex],
                   s=18, label=f"{sex} (p<{alpha})")
        ax.scatter(df.loc[~sig, "scale"], df.loc[~sig, "rho"], facecolor="none",
                   edgecolor=colors[sex], s=18)
    ax.axhline(0, ls="--", c="k", lw=1)
    ax.set(xscale="log", xlabel="scale (min)", ylabel=r"Spearman's $\rho$ (AAD vs age)",
           title="age effect by scale")
    ax.legend(fontsize=7)
    return ax
