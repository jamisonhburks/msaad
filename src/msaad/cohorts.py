"""Cohort analyses of MSAAD applied to distal skin temperature.

Two real-world test cases from the paper:

* **Diabetes (categorical):** compare the summed MSAAD curve between diabetic
  (DM Dx) and condition-free (No-DM Dx) cohorts, by sex, with Kruskal–Wallis +
  Dunn's post-hoc and Cohen's d (Section 3.3), and an ANCOVA that controls for
  activity/MET (Table 2).
* **Age (continuous):** Spearman correlation of AAD with age *at each scale*,
  revealing the sex- and scale-dependent "critical slowing down" of aging
  temperature dynamics (Section 3.5, Fig. 5B).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .stats_utils import cohen_d, dunn_test

__all__ = [
    "DiabetesComparison",
    "compare_diabetes_cohorts",
    "msaad_met_ancova",
    "age_scale_spearman",
]


@dataclass(frozen=True)
class DiabetesComparison:
    """Result of the four-group (sex × diabetes) MSAAD comparison."""

    kruskal_stat: float
    kruskal_p: float
    dunn_p: pd.DataFrame          # 4×4 adjusted p-values
    cohens_d: dict[str, float]    # pairwise effect sizes
    group_labels: list[str]


def compare_diabetes_cohorts(
    df: pd.DataFrame,
    *,
    value_col: str = "msaad_sum",
    p_adjust: str = "bonferroni",
) -> DiabetesComparison:
    """Kruskal–Wallis + Dunn's post-hoc across the four sex × diabetes groups.

    Parameters
    ----------
    df
        One row per participant with columns ``sex`` (``male``/``female``),
        ``group`` (``DM``/``No-DM``) and ``value_col``.
    value_col
        The scalar compared between groups (default: the summed MSAAD curve).
    p_adjust
        Multiple-comparison method for Dunn's test.

    Returns
    -------
    DiabetesComparison
    """
    labels = ["No-DM_male", "DM_male", "No-DM_female", "DM_female"]
    specs = [("male", "No-DM"), ("male", "DM"), ("female", "No-DM"), ("female", "DM")]
    groups = [
        df.loc[(df["sex"] == s) & (df["group"] == g), value_col].dropna().to_numpy()
        for s, g in specs
    ]

    h, p = stats.kruskal(*groups)
    dunn = dunn_test(groups, p_adjust=p_adjust)
    dunn.index = dunn.columns = labels

    effects = {
        f"{labels[i]}_vs_{labels[j]}": cohen_d(groups[i], groups[j])
        for i in range(len(groups))
        for j in range(i + 1, len(groups))
    }
    return DiabetesComparison(float(h), float(p), dunn, effects, labels)


def msaad_met_ancova(
    df: pd.DataFrame, *, value_col: str = "msaad_sum"
) -> pd.DataFrame:
    """ANCOVA of MSAAD on diabetes group while controlling for activity (Table 2).

    Fits ``value ~ group + met`` separately for each (sex, state) cell. A
    significant ``group`` coefficient means the diabetes effect on MSAAD survives
    adjustment for mean MET. ``group`` is coded No-DM = 0 (reference) / DM = 1.

    Parameters
    ----------
    df
        Rows with ``sex``, ``state`` (``awake``/``asleep``), ``group``, ``met``
        and ``value_col``.

    Returns
    -------
    pandas.DataFrame
        MultiIndex (``sex``, ``state``, ``term``) → ``coef``, ``p_value``.
    """
    data = df.copy()
    data["group_code"] = (data["group"] == "DM").astype(int)

    records = []
    for (sex, state), cell in data.groupby(["sex", "state"]):
        model = smf.ols(f"{value_col} ~ group_code + met", data=cell).fit()
        for term in model.params.index:
            records.append(
                {
                    "sex": sex,
                    "state": state,
                    "term": term.replace("group_code", "Group"),
                    "coef": model.params[term],
                    "p_value": model.pvalues[term],
                }
            )
    return pd.DataFrame(records).set_index(["sex", "state", "term"])


def age_scale_spearman(
    curves: np.ndarray, ages: np.ndarray, scales: np.ndarray
) -> pd.DataFrame:
    """Spearman ρ between AAD and age at every coarse-graining scale (Fig. 5B).

    A negative ρ means temperature variability at that scale *decreases* with
    age — the "critical slowing down" signature. Reporting ρ per scale exposes
    that this effect is scale-dependent (and, when split by sex, sex-dependent).

    Parameters
    ----------
    curves
        MSAAD curves, shape ``(n_participants, n_scales)``.
    ages
        Age per participant, shape ``(n_participants,)``.
    scales
        The scales, shape ``(n_scales,)``.

    Returns
    -------
    pandas.DataFrame
        Rows: ``scale, rho, p_value``.
    """
    curves = np.asarray(curves, dtype=float)
    ages = np.asarray(ages, dtype=float)
    rows = []
    for k, scale in enumerate(scales):
        col = curves[:, k]
        ok = np.isfinite(col) & np.isfinite(ages)
        if ok.sum() < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = stats.spearmanr(ages[ok], col[ok])
        rows.append({"scale": int(scale), "rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows)
