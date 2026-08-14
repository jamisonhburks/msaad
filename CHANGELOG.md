# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2025-09-12

Initial public release: a clean, modular reference implementation of MSAAD
(Algorithms 18(9):577, 2025).

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
