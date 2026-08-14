"""Small, dependency-light statistics helpers.

Kept separate and unit-tested so the cohort logic reads cleanly. Dunn's post-hoc
test is implemented from scratch (rank-based, with a tie correction) so the
package needs no ``scikit-posthocs`` dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["cohen_d", "dunn_test", "bonferroni"]


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between samples ``a`` and ``b`` (pooled SD).

    Positive ⇒ ``a`` has the larger mean. 0.2/0.5/0.8 are the conventional
    small/medium/large thresholds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return 0.0 if pooled == 0 else float((a.mean() - b.mean()) / np.sqrt(pooled))


def dunn_test(groups: list[np.ndarray], *, p_adjust: str = "bonferroni") -> pd.DataFrame:
    """Dunn's (1964) post-hoc test of mean-rank differences between groups.

    Run after a significant Kruskal–Wallis test to locate which pairs differ.
    Ranks are computed over the pooled data (with a tie correction on the
    variance term); each pair yields a z-statistic and a two-sided p-value,
    Bonferroni- or BH-adjusted for the number of pairwise comparisons.

    Parameters
    ----------
    groups
        List of 1-D samples, one per group.
    p_adjust
        ``"bonferroni"`` or ``"fdr_bh"``.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix of adjusted p-values (diagonal = 1).
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    sizes = np.array([g.size for g in groups])
    pooled = np.concatenate(groups)
    n = pooled.size
    ranks = stats.rankdata(pooled)

    # Tie correction to the rank variance (Dunn 1964).
    _, tie_counts = np.unique(pooled, return_counts=True)
    tie_term = (tie_counts**3 - tie_counts).sum()
    sigma2 = (n * (n + 1) - tie_term / (n - 1)) / 12.0

    # Mean rank per group.
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    mean_ranks = np.array(
        [ranks[offsets[i] : offsets[i + 1]].mean() for i in range(k)]
    )

    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    pvals = []
    for i, j in pairs:
        se = np.sqrt(sigma2 * (1.0 / sizes[i] + 1.0 / sizes[j]))
        z = abs(mean_ranks[i] - mean_ranks[j]) / se
        pvals.append(2.0 * stats.norm.sf(z))

    pvals = _adjust(np.array(pvals), p_adjust)
    out = np.ones((k, k))
    for (i, j), p in zip(pairs, pvals):
        out[i, j] = out[j, i] = p
    return pd.DataFrame(out)


def _adjust(pvals: np.ndarray, method: str) -> np.ndarray:
    """Multiple-comparison adjustment: Bonferroni or Benjamini–Hochberg."""
    m = len(pvals)
    if method == "bonferroni":
        return np.minimum(pvals * m, 1.0)
    if method == "fdr_bh":
        order = np.argsort(pvals)
        ranked = pvals[order] * m / (np.arange(1, m + 1))
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(ranked, 1.0)
        return out
    raise ValueError(f"unknown p_adjust: {method!r}")


def bonferroni(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses surviving Bonferroni correction."""
    pvalues = np.asarray(pvalues, dtype=float)
    return pvalues < (alpha / len(pvalues))
