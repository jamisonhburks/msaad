# Methods — code ↔ paper cross-reference

Maps each part of the analysis to the module that implements it and to the
corresponding equation/figure of:

> Burks et al., *Multiscale Average Absolute Difference (MSAAD)…*, Algorithms
> 18(9):577 (2025). https://doi.org/10.3390/a18090577

The contribution is a complexity feature that keeps the multiscale spirit of
Multiscale Sample Entropy (MSSE) and the speed of Katz Fractal Dimension (KFD)
while being nonparametric and robust to the artifacts of uncontrolled wearable
data.

---

## The algorithm

| step | code | paper |
|------|------|-------|
| coarse-grain at scale τ (non-overlapping mean) | `coarsegrain.coarse_grain` | Eq. 1 |
| line length `L = Σ|Δy|` | (inside `msaad.aad`) | Eq. 2 |
| `AAD(τ) = L / (N−1) = mean|Δy|` | `msaad.aad` | Eq. 3 |
| MSAAD curve over scales | `msaad.msaad` | Algorithm 1 |

`aad([1,2,1,20,20,20,1]) == mean(abs(diff(...)))` — the exact identity the paper
notes; it is a unit test.

## Baselines (paper Table 1)

All implemented from scratch in `baselines.py` and exposed through one interface,
`multiscale(x, metric, scales)`:

- `katz_fractal_dimension` — KFD (Eq. A1). Dominated by the single largest
  excursion `d`, so isolated perturbations (waking, activity) inflate it —
  the paper's explanation for why MS-KFD separates cohorts inconsistently.
- `sample_entropy` — SampEn. Returns `nan` when no template matches exist, the
  instability at large scales the paper highlights (Fig. A2).
- `permutation_entropy`, `hurst_exponent` — the other Table 1 comparators.

## Synthetic 1/fᵝ noise (paper Section 2.1; Figs. 2–3)

`noise.powerlaw_noise` follows the manuscript: white noise → FFT → shape
amplitudes by `1/f^(β/2)` → normalize → inverse FFT with random phases.

* **β ↔ slope law** (`msaad.loglog_slope`, `beta_from_slope`; Fig. 3A): the
  log–log MSAAD slope is affine in β for β ∈ [−1, 2] (`slope ≈ 0.48β − 0.48`).
* **Stability** (`stability.py`; Fig. 2D/E): per-scale dispersion of z-scored
  curves across realizations, compared in scale bins with Mann–Whitney U.
* **Runtime** (script 01; Fig. 3B): MSAAD vs. multiscale sample entropy.

## Real-data analyses

### Diabetes (categorical; Section 3.3, Fig. 4, Tables 1–2)
`pipeline.extract_diabetes_features` → summed MSAAD per participant × state.
`cohorts.compare_diabetes_cohorts` runs Kruskal–Wallis + Dunn's post-hoc
(`stats_utils.dunn_test`, from scratch) with Cohen's d across the four sex ×
diabetes groups. `cohorts.msaad_met_ancova` fits `MSAAD ~ Group + MET` per
(sex, state) — the diabetes effect **reverses awake↔asleep** and **survives**
MET adjustment (Table 2).

### Age (continuous; Section 3.5, Fig. 5)
`pipeline.extract_age_features` computes each participant's MSAAD curve over a
wide scale range (on the concatenated awake series, since long scales need long
records). `cohorts.age_scale_spearman` correlates AAD with age **at each scale**;
a negative ρ is "critical slowing down", and reporting it per scale exposes the
**scale- and sex-dependence** (men strongest < 30 min, women > 80 min).

---

## What the synthetic data does (and does not) show

`synthetic/generate.py` builds temperature as a baseline plus a **two-timescale
fluctuation**: a fast (whiter) component that dominates short MSAAD scales and a
slow (browner) component that dominates long scales. Planting effects on the two
amplitudes independently reproduces the paper's patterns:

- **Age** reduces the fast component in men (short-scale effect) and the slow
  component in women (long-scale effect) → the observed ρ-vs-scale crossover.
- **Diabetes** scales both amplitudes, *down* when awake and *up* when asleep,
  more strongly in women → the sign reversal and the larger female effect.
- **MET** is only weakly coupled to amplitude, so the diabetes effect on MSAAD
  survives the ANCOVA (as in the paper).

Recovering these effects validates the **estimators and code path**; it is not
independent evidence about human physiology. Effect sizes and prevalences are set
so the effects are detectable at a few hundred synthetic participants (the real
study had thousands).
