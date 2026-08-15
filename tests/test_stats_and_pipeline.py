"""Tests for statistics helpers and the end-to-end feature pipeline."""

from __future__ import annotations

import warnings

import numpy as np

from msaad.cohorts import age_scale_spearman, compare_diabetes_cohorts
from msaad.config import MSAADConfig
from msaad.io import iter_participants
from msaad.pipeline import extract_age_features, extract_diabetes_features
from msaad.stats_utils import cohen_d, dunn_test
from msaad.synthetic import generate_dataset


def test_iter_participants_skips_demographics(tmp_path):
    """The default glob must not mis-load demographics.parquet as a participant."""
    generate_dataset(4, tmp_path / "syn", n_days=2, seed=0)
    assert (tmp_path / "syn" / "demographics.parquet").exists()
    ids = [pid for pid, _ in iter_participants(tmp_path / "syn")]  # default pattern
    assert "demographics" not in ids
    assert len(ids) == 4


def test_cohen_d_sign():
    a = np.array([2.0, 3.0, 2.5, 3.5, 2.0])
    b = np.array([1.0, 2.0, 1.5, 2.5, 1.0])
    assert cohen_d(a, b) > 0
    assert cohen_d(b, a) < 0


def test_dunn_test_separates_shifted_group():
    rng = np.random.default_rng(0)
    g = [rng.normal(0, 1, 40), rng.normal(0, 1, 40), rng.normal(5, 1, 40)]
    p = dunn_test(g, p_adjust="bonferroni")
    assert p.iloc[0, 1] > 0.05      # groups 0,1 identical → not significant
    assert p.iloc[0, 2] < 0.01      # group 2 shifted → significant
    assert np.allclose(np.diag(p), 1.0)


def test_pipeline_recovers_diabetes_and_age_effects(tmp_path):
    warnings.simplefilter("ignore")
    cfg = MSAADConfig()
    demo = generate_dataset(120, tmp_path / "syn", n_days=5, seed=3)
    assert (tmp_path / "syn" / "demographics.parquet").exists()

    # Diabetes: effect reverses between awake and asleep (paper's key result).
    feats = extract_diabetes_features(tmp_path / "syn", config=cfg, progress=False)
    feats = feats.merge(demo, left_on="pid", right_index=True)
    cohort = feats[(feats.age >= 45) & (feats.age <= 65)]
    key = "No-DM_male_vs_DM_male"
    d_awake = compare_diabetes_cohorts(cohort[cohort.state == "awake"]).cohens_d[key]
    d_asleep = compare_diabetes_cohorts(cohort[cohort.state == "asleep"]).cohens_d[key]
    assert d_awake > 0 and d_asleep < 0

    # Age: critical slowing down — negative ρ, and men strongest at short scales.
    age = extract_age_features(tmp_path / "syn", config=cfg, progress=False).join(demo)
    scales = np.asarray(cfg.age_scales)
    men = age[age.sex == "male"]
    sp = age_scale_spearman(men[[f"s{s}" for s in scales]].to_numpy(), men.age.to_numpy(), scales)
    assert sp.loc[sp.scale < 30, "rho"].mean() < 0
