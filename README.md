# msaad

[![PyPI](https://img.shields.io/pypi/v/msaad.svg)](https://pypi.org/project/msaad/)
[![Python versions](https://img.shields.io/pypi/pyversions/msaad.svg)](https://pypi.org/project/msaad/)
[![CI](https://github.com/jamisonhburks/msaad/actions/workflows/ci.yml/badge.svg)](https://github.com/jamisonhburks/msaad/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fa18090577-blue.svg)](https://doi.org/10.3390/a18090577)

**Multiscale Average Absolute Difference — a fast, nonparametric multiscale complexity feature for noisy wearable time series.**

A clean, modular reference implementation of:

> Burks, Hartogensis, Dilchert, Mason & Smarr.
> *Multiscale Average Absolute Difference (MSAAD): A Computationally Efficient
> and Nonparametric Adaptation of Line Length for Noisy, Uncontrolled Wearables
> Time Series.* **Algorithms** 18(9):577 (2025). <https://doi.org/10.3390/a18090577>

---

## The idea in one paragraph

Complexity measures like sample entropy and fractal dimension extract useful
structure from physiological signals, but they assume things wearable data
violate (stationarity, low noise, a well-chosen tolerance) and are slow. **MSAAD**
is a multiscale adaptation of *line length*: at each coarse-graining scale τ it
measures the **mean absolute first difference** of the signal, and the MSAAD
*curve* records how that local variability changes with scale. It has **no
parameters** beyond the scales, runs in **O(N) per scale**, and — because it
averages over the whole distribution of fluctuations — is far more stable than
sample entropy and far less sensitive to isolated spikes than fractal dimension.

```
AAD(τ) = mean |Δ coarse_grain(x, τ)|          # line length / (N−1)
MSAAD  = [ AAD(τ) for τ in scales ]
```

That's the whole algorithm. The value is in what it buys you (below).

---

## What it does, verified on synthetic data

Running the pipeline reproduces the paper's four headline results:

**1. It reads off the spectral exponent.** For 1/fᵝ noise the log–log MSAAD slope
is linear in β over β ∈ [−1, 2]. This repo recovers **slope ≈ 0.47·β − 0.47,
R ≈ 0.999** (paper: 0.48β − 0.48, R ≈ 0.997) — so `beta_from_slope` turns MSAAD
into a cheap spectral-exponent estimator.

**2. It is more stable and ~10× faster than sample entropy** across realizations
and scales.

**3. It separates diabetes cohorts** in distal skin temperature where the
baselines cannot — significant Kruskal–Wallis + Dunn separation of DM vs. No-DM
(ages 45–65) by sex, with the effect **reversing between awake and asleep** and
**surviving adjustment for activity (MET)**.

**4. It captures "critical slowing down" with aging** — temperature variability
declines with age, and MSAAD reveals this is **scale- and sex-dependent** (men
strongest at short scales, women at long scales).

> The bundled synthetic generator plants these effects into the *fluctuation
> structure* of the data, so recovering them validates the estimators and code —
> not human physiology. The real Oura data are not redistributable.

---

## Install

```bash
pip install msaad
```

Or for development (tests, the reproduction pipeline, the tutorial):

```bash
git clone https://github.com/jamisonhburks/msaad.git && cd msaad
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # add ".[reference]" to cross-check baselines vs neurokit2/antropy
```

Requires Python ≥ 3.10. Every complexity metric (MSAAD **and** the KFD / sample
entropy / permutation entropy / Hurst baselines) is implemented from scratch in
NumPy — no `neurokit2`/`antropy` needed to run.

## Quickstart

```bash
make demo          # generate data → noise benchmarks → diabetes → age → figures
```

or step by step:

```bash
python scripts/00_generate_synthetic_data.py --n-participants 500
python scripts/01_noise_benchmarks.py      # β↔slope law, stability, runtime (Figs 2–3)
python scripts/02_diabetes_analysis.py     # DM vs No-DM (Fig 4, Tables 1–2)
python scripts/03_age_analysis.py          # critical slowing down (Fig 5)
python scripts/make_figures.py             # results/figures/*.png,*.pdf
```

Every script takes `--data-dir` (default `data/synthetic`), so the same commands
run on real records in `data/raw/`. See also `notebooks/tutorial.ipynb`.

## Use it as a library

```python
import numpy as np
from msaad import msaad, loglog_slope, beta_from_slope

x = np.load("signal.npy")
curve = msaad(x, scales=range(1, 31))          # the MSAAD curve
beta_hat = beta_from_slope(loglog_slope(curve, np.arange(1, 31)))  # spectral exponent
```

---

## Repository layout

```
msaad/
├── src/msaad/
│   ├── config.py         # constants + MSAADConfig (all hyper-parameters)
│   ├── coarsegrain.py    # multiscale coarse-graining              (Eq. 1)
│   ├── msaad.py          # the MSAAD algorithm + β↔slope law        (Eqs. 2–3)  ← core
│   ├── baselines.py      # from-scratch KFD / SampEn / PermEn / Hurst + multiscale wrapper
│   ├── noise.py          # 1/fᵝ power-law noise generator
│   ├── stability.py      # across-realization stability comparison  (Fig. 2D/E)
│   ├── cohorts.py        # diabetes (Kruskal+Dunn+ANCOVA) & age (Spearman) analyses
│   ├── stats_utils.py    # Cohen's d, Dunn's post-hoc, corrections
│   ├── io.py             # schema-validated loading
│   ├── pipeline.py       # feature extraction over a directory of records
│   ├── plotting.py       # paper-analogous figures
│   └── synthetic/generate.py   # forward-model data generator
├── scripts/              # thin CLI entry points, one per stage
├── notebooks/tutorial.ipynb    # narrated end-to-end walkthrough
├── tests/                # pytest suite (algorithm invariants + effect recovery)
├── docs/METHODS.md       # code ↔ paper equation/figure cross-reference
└── data/  results/       # git-ignored inputs/outputs (schema docs inside)
```

Design: computation is pure and config-driven; MSAAD and every baseline share one
`multiscale(x, metric, scales)` interface; plotting is separate from analysis;
zero hard third-party dependency for the algorithms themselves. Details in
[`docs/METHODS.md`](docs/METHODS.md).

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The suite asserts algorithm **invariants** (AAD = mean|Δx|; MSAAD is linear in
amplitude; the log–log slope recovers β; permutation entropy of white noise ≈ 1)
and that the pipeline **recovers the planted diabetes and age effects** on a small
synthetic dataset.

## Contributing

New collaborators: see [`CONTRIBUTING.md`](CONTRIBUTING.md) for a 5-minute setup,
the three commands you'll use (`pytest`, `ruff check`, `ruff format`), and how to
add a new complexity metric (it's one function). CI runs lint + tests on every
push and PR across Python 3.10–3.12.

## Citing

Please cite the paper (see [`CITATION.cff`](CITATION.cff)). Code is MIT-licensed;
the manuscript is CC-BY-4.0.
