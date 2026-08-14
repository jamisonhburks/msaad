# Contributing to `msaad`

Thanks for your interest! This guide gets you productive in a few minutes.

## Development setup

```bash
git clone <this-repo> && cd msaad
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # editable install + pytest, ruff, jupyter
```

Requires Python ≥ 3.10.

## The three commands you'll use

```bash
pytest                              # run the test suite (should be fast)
ruff check src scripts tests        # lint (config in pyproject.toml)
ruff format src scripts tests       # auto-format (optional but encouraged)
```

CI runs `ruff check` and `pytest` on every push and pull request (see
`.github/workflows/ci.yml`), so run them locally before opening a PR.

## How the repo is organized

One module per concern under `src/msaad/` (see the table in
[`README.md`](README.md) and the paper cross-reference in
[`docs/METHODS.md`](docs/METHODS.md)). A few conventions worth knowing:

- **Computation is pure and config-driven.** Functions take a `MSAADConfig`
  rather than reading globals, so an experiment is fully described by one object.
- **Plotting is separate from analysis** (`plotting.py` only draws; it never
  computes). This keeps figures re-styleable without rerunning anything.
- **Every complexity metric shares one interface.** A metric is a function
  `f(x) -> float`; `baselines.multiscale(x, f, scales)` turns it into a
  multiscale curve. MSAAD, KFD, sample entropy, etc. all plug in the same way.

### Adding a new complexity metric

1. Write a single-scale `my_metric(x: np.ndarray) -> float` in `baselines.py`
   (or a new module) with a NumPy-style docstring.
2. That's it — it composes with `multiscale(...)`. Add it to the comparison in
   `scripts/01_noise_benchmarks.py` if you want it benchmarked.
3. Add a unit test asserting an *invariant* (e.g. bounds, monotonicity, a known
   value), not a brittle numeric snapshot — see `tests/test_baselines.py`.

## Testing philosophy

Tests assert **invariants and effect recovery**, not exact numbers:
- algorithm identities (`aad(x) == mean(|diff(x)|)`), bounds, monotonicity;
- that the pipeline recovers the *planted* diabetes/age effects on a small
  synthetic dataset (`tests/test_stats_and_pipeline.py`).

Keep tests fast (seconds): use small signals, few realizations, and fixed seeds.

## Data policy

**Never commit participant data.** `data/raw/` is git-ignored and is only a
drop-in location for real records. All shipped/generated data are synthetic. See
[`data/raw/README.md`](data/raw/README.md) for the expected schema.

## Pull requests

- Keep changes focused; update `docs/METHODS.md` if you touch the method↔paper
  mapping, and add a line to `CHANGELOG.md`.
- Make sure `ruff check` and `pytest` are green.
