# Reproducibility Guide

This document provides complete instructions for reproducing all results in the ABM Enterprise Coping paper.

## Prerequisites

### System Requirements

- Python 3.10+
- R 4.2+
- Quarto 1.4+
- Git
- 8GB RAM minimum

### Optional (for LLM Policy)

- OpenAI API key (for `llm_openai` policy)
- Anthropic API key (for `llm_claude` policy)

## Environment Setup

### 1. Python Environment

```bash
# Clone repository
git clone https://github.com/username/abm-enterprise-coping.git
cd abm-enterprise-coping

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install package with dependencies
pip install -e ".[dev]"

# Verify installation
abm --help
pytest tests/ -v --tb=short
```

### 2. R Environment

```bash
# Install R dependencies via renv
make setup-r
# or manually:
cd analysis && Rscript -e "renv::restore()"
```

Required R packages (managed by renv):
- `ggplot2` (visualization)
- `dplyr`, `tidyr` (data manipulation)
- `arrow` (Parquet I/O)
- `fixest` (fixed effects regression)
- `gt` (tables)
- `here` (path management)

### 3. Quarto

```bash
# Install Quarto (if not installed)
# See: https://quarto.org/docs/get-started/

# Verify
quarto --version  # Should be 1.4+
```

## Data Pipeline

### Step 1: Ingest LSMS Data

```bash
# Download and process LSMS-ISA data (or generate synthetic if unavailable)
make ingest-data country=tanzania
make ingest-data country=ethiopia

# Verify
ls data/processed/tanzania/canonical/  # Should have household.parquet, etc.
```

### Step 2: Derive Target Variables

```bash
make derive-targets country=tanzania
make derive-targets country=ethiopia

# Verify
ls data/processed/tanzania/derived/  # Should have household_targets.parquet
```

### Step 3: Calibration

```bash
abm calibrate --country tanzania --data-dir data/processed

# Verify
cat artifacts/calibration/tanzania/calibration.json | head -30
```

**Output:** `artifacts/calibration/tanzania/calibration.json`

## Simulation Runs

### Baseline Simulation (LSMS-derived)

```bash
# Single run with seed=42
make run-sim COUNTRY=tanzania

# Verify
ls outputs/tanzania/baseline/  # manifest.json, household_outcomes.parquet/
```

### Multi-Seed Batch (LSMS-derived)

```bash
# 10 replicate runs
python3 scripts/run_batch.py \
  --data-source lsms \
  --country tanzania \
  --seeds 10 \
  --output-dir outputs/batch

# Verify
ls outputs/batch/lsms/  # seed_1/ through seed_10/
cat outputs/batch/lsms/batch_manifest.json
```

### Parameter Sweep (Calibrated Synthetic)

```bash
python3 scripts/run_sweep.py \
  --calibration artifacts/calibration/tanzania/calibration.json \
  --households 100 \
  --seeds 2 \
  --output-dir outputs/sweeps/calibrated

# Verify
ls outputs/sweeps/calibrated/  # sweep_agg_*.parquet, sweep_full_*.parquet
```

### Behavior Search (Calibrated Synthetic)

```bash
python3 scripts/run_behavior_search.py \
  --calibration artifacts/calibration/tanzania/calibration.json \
  --targets-from-lsms \
  --n-candidates 40 \
  --seeds 2 \
  --output-dir outputs/search/calibrated

# Verify
ls outputs/search/calibrated/  # candidates_*.parquet
```

## Rendering Documents

### Technical Report

```bash
make render-report-country country=tanzania

# Output: docs/abm_report.html
```

### Paper Manuscript

```bash
# HTML (recommended for QA)
quarto render docs/paper.qmd --to html

# PDF (requires LaTeX)
quarto render docs/paper.qmd --to pdf

# Both
quarto render docs/paper.qmd

# Output: docs/paper.html, docs/paper.pdf
```

## Complete Pipeline

Run the full pipeline from scratch:

```bash
# Clean previous outputs (optional)
make clean-outputs

# Setup
make setup
make setup-r

# Data pipeline
make ingest-data country=tanzania
make derive-targets country=tanzania
abm calibrate --country tanzania --data-dir data/processed

# Simulations (run in parallel or sequence)
make run-sim COUNTRY=tanzania
python3 scripts/run_batch.py --data-source lsms --country tanzania --seeds 10
python3 scripts/run_sweep.py --calibration artifacts/calibration/tanzania/calibration.json
python3 scripts/run_behavior_search.py --calibration artifacts/calibration/tanzania/calibration.json --targets-from-lsms

# Render
quarto render docs/paper.qmd --to html
```

**Estimated time:** ~30 minutes on a modern machine.

## Output Verification

### Expected Output Structure

```
outputs/
├── tanzania/
│   └── baseline/
│       ├── manifest.json
│       └── household_outcomes.parquet/
├── ethiopia/
│   └── baseline/
│       └── ...
├── batch/
│   └── lsms/
│       ├── batch_manifest.json
│       ├── seed_1/
│       │   └── tanzania/baseline/...
│       └── ...
├── sweeps/
│   └── calibrated/
│       ├── sweep_agg_*.parquet
│       ├── sweep_full_*.parquet
│       └── sweep_config_*.json
└── search/
    └── calibrated/
        ├── candidates_*.parquet
        └── search_config_*.json

artifacts/
└── calibration/
    └── tanzania/
        └── calibration.json

docs/
├── paper.html
├── paper.qmd
└── abm_report.html
```

### Manifest Validation

Each simulation run produces a `manifest.json` with provenance:

```json
{
  "run_id": "95fdccd3",
  "git_hash": "cd43088",
  "seed": 42,
  "timestamp": "2026-01-13T17:28:36.939064+00:00",
  "parameters": {...},
  "country": "tanzania",
  "scenario": "baseline",
  "version": "0.1.0",
  "mesa_version": "3.0.3"
}
```

Verify git hash matches your checkout: `git rev-parse --short HEAD`

## Troubleshooting

### Common Issues

**1. Missing LSMS data**
```
Error: Derived targets not found: data/processed/tanzania/derived/household_targets.parquet
```
Solution: Run `make derive-targets country=tanzania`

**2. R package errors**
```
Error in library(arrow): there is no package called 'arrow'
```
Solution: Run `make setup-r` or `Rscript -e "renv::restore()"`

**3. Quarto render fails**
```
Error: path '/home/.../outputs/...' does not exist
```
Solution: Run simulations before rendering. Check paths in paper.qmd.

**4. Calibration artifact missing**
```
FileNotFoundError: artifacts/calibration/tanzania/calibration.json
```
Solution: Run `abm calibrate --country tanzania --data-dir data/processed`

### Getting Help

- Check `docs/DATA_AUDIT.md` for data provenance issues
- Check `docs/DATA_CONTRACT.md` for schema definitions
- Run `make test` to verify codebase integrity
- Open an issue on GitHub

## Citation

If you use this code, please cite:

```bibtex
@software{abm_enterprise_coping,
  title={ABM Enterprise Coping: Agent-Based Model of Household Enterprise Strategies},
  author={Bean, Michael},
  year={2026},
  url={https://github.com/username/abm-enterprise-coping}
}
```

## Version Information

- Python package version: `0.1.0`
- Mesa version: `3.0+`
- Last updated: 2026-01-14
