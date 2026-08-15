# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-08-15

First PyPI release (`pip install msaad`): a clean, modular reference
implementation of MSAAD (Algorithms 18(9):577, 2025). The public API is small
and considered stable.

### Added
- Core `msaad` algorithm: coarse-graining, AAD, MSAAD curve, log–log slope, and
  the β↔slope law (`beta_from_slope` / `slope_from_beta`).
- From-scratch baseline metrics (Katz fractal dimension, sample entropy,
  permutation entropy, Hurst) unified by `multiscale()`.
- 1/fᵝ power-law noise generator; across-realization stability comparison.
- Diabetes (Kruskal–Wallis + Dunn's + ANCOVA) and age (Spearman-vs-scale)
  cohort analyses, plus a synthetic Oura-like data generator with planted effects.
- CLI stage scripts, figure generation, a narrated tutorial notebook, and a
  pytest suite of algorithm invariants and effect-recovery checks.
- PEP 561 `py.typed` marker so downstream type checkers use the package's hints;
  the `scales` argument is typed `Sequence[int] | np.ndarray` (accepts `range`,
  `list`, `tuple`, ndarray).
- Robust `iter_participants`: a `*.parquet` default that works for real
  (hashed-id) and synthetic files and skips non-participant tables
  (`demographics.parquet`).
