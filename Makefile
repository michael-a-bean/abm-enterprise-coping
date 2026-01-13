# Root Makefile
# ABM Enterprise Coping Model

.PHONY: help setup install test lint format type-check run run-toy run-sim validate setup-r render-report render-report-country clean

# Default target
help:
	@echo "ABM Enterprise Coping Model - Root Makefile"
	@echo ""
	@echo "Python Targets:"
	@echo "  setup              - Install Python package in editable mode with dev deps"
	@echo "  install            - Install Python package in editable mode"
	@echo "  test               - Run Python tests"
	@echo "  lint               - Run linting (ruff)"
	@echo "  format             - Format code with ruff"
	@echo "  type-check         - Run mypy type checking"
	@echo "  run                - Run simulation (use config=PATH)"
	@echo "  run-toy            - Run toy simulation"
	@echo "  run-sim            - Run simulation (use COUNTRY=name)"
	@echo "  validate           - Validate output schema"
	@echo ""
	@echo "R/Analysis Targets:"
	@echo "  setup-r            - Setup R environment (renv restore)"
	@echo "  render-report      - Render validation report (toy mode)"
	@echo "  render-report-country - Render report for country (use country=NAME)"
	@echo ""
	@echo "Other:"
	@echo "  clean              - Clean all build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make test"
	@echo "  make run-toy"
	@echo "  make render-report"
	@echo "  make render-report-country country=tanzania"

# ========================================
# Python targets
# ========================================

# Install Python package in development mode with dev dependencies
setup:
	pip install -e ".[dev]"

# Install Python package in development mode (alias for setup)
install:
	pip install -e ".[dev]"

# Run Python tests
test:
	pytest tests/ -v

# Run linting
lint:
	ruff check src/ tests/

# Format code
format:
	ruff format src/ tests/

# Type checking
type-check:
	mypy src/

# Run simulation with config file
# Usage: make run config=config/toy.yaml
run:
ifndef config
	$(error config is not set. Usage: make run config=PATH)
endif
	python -m abm_enterprise.cli run $(config)

# Run toy simulation with CLI
run-toy:
	abm run-toy --seed 42 --output-dir outputs/toy

# Run simulation with country and scenario
# Usage: make run-sim COUNTRY=tanzania SCENARIO=baseline SEED=42
COUNTRY ?= tanzania
SCENARIO ?= baseline
SEED ?= 42

run-sim:
	abm run-sim $(COUNTRY) --scenario $(SCENARIO) --seed $(SEED) --output-dir outputs

# Validate output schema
validate:
	abm validate-schema outputs/toy

# ========================================
# R/Analysis targets
# ========================================

# Setup R environment
setup-r:
	$(MAKE) -C analysis setup

# Render validation report (toy mode)
render-report:
	$(MAKE) -C analysis render

# Render validation report for specific country
# Usage: make render-report-country country=tanzania
render-report-country:
ifndef country
	$(error country is not set. Usage: make render-report-country country=NAME)
endif
	$(MAKE) -C analysis render-country country=$(country)

# ========================================
# Cleaning targets
# ========================================

# Clean all build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	$(MAKE) -C analysis clean

# Clean outputs
clean-outputs:
	rm -rf outputs/*

# ========================================
# Full pipeline
# ========================================

# Run full pipeline: toy simulation + validation report
pipeline-toy: run-toy render-report
	@echo "Pipeline complete!"
	@echo "Simulation output: outputs/toy/"
	@echo "Validation report: analysis/quarto/_output/validation_report.html"
