# Convenience targets for the MSAAD pipeline.
# Stages read --data-dir (default data/synthetic) and write to data/processed.

PYTHON ?= python
N_PARTICIPANTS ?= 500
N_DAYS ?= 6
SEED ?= 7

.PHONY: help install data noise diabetes age figures demo test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

data: ## Generate the synthetic dataset
	$(PYTHON) scripts/00_generate_synthetic_data.py --n-participants $(N_PARTICIPANTS) --n-days $(N_DAYS) --seed $(SEED)

noise: ## Stage 1 — 1/fᵝ noise benchmarks (Figs. 2–3)
	$(PYTHON) scripts/01_noise_benchmarks.py

diabetes: ## Stage 2 — diabetes cohort separation (Fig. 4, Tables 1–2)
	$(PYTHON) scripts/02_diabetes_analysis.py

age: ## Stage 3 — age critical-slowing-down (Fig. 5)
	$(PYTHON) scripts/03_age_analysis.py

figures: ## Render all figures to results/figures
	$(PYTHON) scripts/make_figures.py

demo: data noise diabetes age figures ## Full pipeline end to end
	@echo "Done. See results/figures/ and data/processed/."

test: ## Run the test suite
	pytest

clean: ## Remove generated data, artifacts, and figures
	rm -f data/synthetic/*.parquet data/processed/* results/figures/*
