#!/usr/bin/env python3
"""Stage 1 — synthetic 1/fᵝ noise benchmarks (Figs. 2–3).

Evaluates MSAAD on controlled power-law noise:
  * the β ↔ log–log-slope law (Fig. 3A),
  * across-realization stability vs. a baseline metric (Fig. 2D/E),
  * runtime vs. multiscale sample entropy (Fig. 3B).

Writes ``noise_benchmarks.npz`` and prints the key numbers.
"""

from __future__ import annotations

import time

import numpy as np

from _common import base_parser, ensure_dir
from msaad.baselines import multiscale, sample_entropy
from msaad.config import NOISE_BETAS
from msaad.msaad import loglog_slope, msaad
from msaad.noise import powerlaw_noise
from msaad.stability import compare_dispersion, dispersion_area


def main() -> None:
    parser = base_parser(__doc__)
    # Defaults are lighter than the paper (100 realizations of 2**13) because the
    # from-scratch sample entropy is O(N²); the qualitative results are unchanged.
    # Bump --n-realizations / --length for publication-quality curves.
    parser.add_argument("--n-realizations", type=int, default=20)
    parser.add_argument("--length", type=int, default=1536)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    scales = np.arange(1, 33)

    # MSAAD and sample-entropy curves for the five noise colours.
    msaad_curves, se_curves = {}, {}
    for beta in NOISE_BETAS:
        m = np.array([msaad(powerlaw_noise(beta, args.length, rng=rng), scales)
                      for _ in range(args.n_realizations)])
        s = np.array([multiscale(powerlaw_noise(beta, args.length, rng=rng), sample_entropy, scales)
                      for _ in range(args.n_realizations)])
        msaad_curves[beta], se_curves[beta] = m, s

    # β ↔ slope law across a fine β grid (MSAAD only, so we can afford long
    # series and a wide scale range for a clean slope estimate).
    beta_grid = np.arange(-3, 3.01, 0.25)
    slope_scales = np.arange(1, 97)
    def _mean_slope(b: float) -> float:
        return np.nanmean([
            loglog_slope(msaad(powerlaw_noise(b, 2**13, rng=rng), slope_scales), slope_scales)
            for _ in range(10)
        ])

    slopes = np.array([_mean_slope(b) for b in beta_grid])
    fit = beta_grid[(beta_grid >= -1) & (beta_grid <= 2)]
    gain, intercept = np.polyfit(fit, slopes[(beta_grid >= -1) & (beta_grid <= 2)], 1)
    r = np.corrcoef(fit, slopes[(beta_grid >= -1) & (beta_grid <= 2)])[0, 1]

    # Stability (pink noise) and runtime.
    stability = compare_dispersion(msaad_curves[1], se_curves[1], scales, bin_size=20)
    sizes = (5, 20, 60)
    run_scales = np.arange(1, 73)
    times = {"MSAAD": [], "SampEn": []}
    for n in sizes:
        batch = rng.standard_normal((n, 720))
        t0 = time.perf_counter()
        [msaad(x, run_scales) for x in batch]
        times["MSAAD"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        [multiscale(x, sample_entropy, run_scales) for x in batch]
        times["SampEn"].append(time.perf_counter() - t0)

    out = ensure_dir(args.processed_dir) / "noise_benchmarks.npz"
    np.savez(out, scales=scales, beta_grid=beta_grid, slopes=slopes,
             sizes=list(sizes), t_msaad=times["MSAAD"], t_sampen=times["SampEn"],
             **{f"msaad_b{b}": msaad_curves[b] for b in NOISE_BETAS},
             **{f"se_b{b}": se_curves[b] for b in NOISE_BETAS})

    print(f"β↔slope law: slope = {gain:.3f}·β + {intercept:.3f}, R = {r:.3f} "
          f"(paper: 0.48, −0.48, 0.997)")
    print(f"MSAAD dispersion area (pink): {dispersion_area(msaad_curves[1], scales):.2f} "
          f"vs SampEn: {dispersion_area(se_curves[1], scales):.2f} "
          f"({(stability['p_value'] < 0.05).sum()}/{len(stability)} scale-bins differ)")
    print(f"Runtime speedup (SampEn/MSAAD) at n={sizes[-1]}: "
          f"{times['SampEn'][-1] / times['MSAAD'][-1]:.1f}×")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
