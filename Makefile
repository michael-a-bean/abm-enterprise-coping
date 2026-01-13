# Root Makefile
# ABM Enterprise Coping Model

.PHONY: help setup install test lint format type-check run run-toy run-sim run-sim-synthetic run-sim-llm-stub run-sim-llm-replay run-sim-llm-claude run-sim-llm-openai validate ingest-data derive-targets setup-r render-report render-report-country clean

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
	@echo "  run-toy            - Run toy simulation with synthetic data"
	@echo "  run-sim            - Run simulation with derived targets (use COUNTRY=name)"
	@echo "  run-sim-synthetic  - Run simulation with synthetic data only"
	@echo "  run-sim-llm-stub   - Run with LLM stub policy (deterministic)"
	@echo "  run-sim-llm-replay - Run with LLM replay from logs"
	@echo "  run-sim-llm-claude - Run with Claude LLM (needs ANTHROPIC_API_KEY)"
	@echo "  run-sim-llm-openai - Run with OpenAI LLM (needs OPENAI_API_KEY)"
	@echo "  validate           - Validate output schema"
	@echo "  ingest-data        - Download and process LSMS data"
	@echo "  derive-targets     - Build derived target tables"
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
	@echo "  make run-sim COUNTRY=tanzania"
	@echo "  make run-sim COUNTRY=tanzania CALIBRATE=1"
	@echo "  make run-sim-synthetic COUNTRY=ethiopia"
	@echo "  make run-sim-llm-stub COUNTRY=tanzania"
	@echo "  make run-sim-llm-replay COUNTRY=tanzania REPLAY_LOG=path/to/log.jsonl"
	@echo "  make ingest-data country=tanzania"
	@echo "  make derive-targets country=tanzania"
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
# Usage: make run-sim COUNTRY=tanzania CALIBRATE=1 (for calibrated policy)
COUNTRY ?= tanzania
SCENARIO ?= baseline
SEED ?= 42
CALIBRATE ?= 0
POLICY ?= none

# Run simulation with derived data (default for non-toy runs)
run-sim:
ifeq ($(CALIBRATE),1)
	abm run-sim $(COUNTRY) --scenario $(SCENARIO) --seed $(SEED) --data-dir data/processed --calibrate --output-dir outputs
else
	abm run-sim $(COUNTRY) --scenario $(SCENARIO) --seed $(SEED) --data-dir data/processed --output-dir outputs
endif

# Run simulation with synthetic data only (no derived targets)
run-sim-synthetic:
	abm run-sim $(COUNTRY) --scenario $(SCENARIO) --seed $(SEED) --output-dir outputs

# Run simulation with LLM stub policy (deterministic rule-based)
run-sim-llm-stub:
	abm run-sim $(COUNTRY) --scenario llm_stub --seed $(SEED) --policy llm_stub --decision-log-dir decision_logs/$(COUNTRY) --output-dir outputs

# Run simulation with LLM replay (reproduce from logs)
# Usage: make run-sim-llm-replay COUNTRY=tanzania REPLAY_LOG=decision_logs/tanzania/decisions_*.jsonl
REPLAY_LOG ?= decision_logs/$(COUNTRY)/decisions.jsonl
run-sim-llm-replay:
	abm run-sim $(COUNTRY) --scenario llm_replay --seed $(SEED) --policy llm_replay --replay-log $(REPLAY_LOG) --decision-log-dir decision_logs/$(COUNTRY)/replay --output-dir outputs

# Run simulation with Claude LLM (requires ANTHROPIC_API_KEY)
run-sim-llm-claude:
	abm run-sim $(COUNTRY) --scenario llm_claude --seed $(SEED) --policy llm_claude --decision-log-dir decision_logs/$(COUNTRY) --output-dir outputs

# Run simulation with OpenAI LLM (requires OPENAI_API_KEY)
run-sim-llm-openai:
	abm run-sim $(COUNTRY) --scenario llm_openai --seed $(SEED) --policy llm_openai --decision-log-dir decision_logs/$(COUNTRY) --output-dir outputs

# Validate output schema
validate:
	abm validate-schema outputs/toy

# ========================================
# ETL targets
# ========================================

# Download and process LSMS data
# Usage: make ingest-data country=tanzania
country ?= tanzania

ingest-data:
	abm ingest-data --country $(country) --output-dir data/processed

# Build derived target tables
# Usage: make derive-targets country=tanzania
derive-targets:
	abm derive-targets --country $(country) --data-dir data/processed

# Full ETL pipeline for a country
# Usage: make etl-pipeline country=tanzania
etl-pipeline: ingest-data derive-targets
	@echo "ETL pipeline complete for $(country)!"
	@echo "Canonical data: data/processed/$(country)/canonical/"
	@echo "Derived targets: data/processed/$(country)/derived/"

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
