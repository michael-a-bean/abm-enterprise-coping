# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent-based model (ABM) simulating household enterprise entry as a coping mechanism in response to agricultural price shocks in Sub-Saharan Africa. Based on empirical findings from "Booms, Busts, and Household Enterprise" paper, validated against LSMS-ISA harmonized panel data for Tanzania and Ethiopia.

**Core hypothesis:** Negative cash-crop price shocks induce enterprise entry as coping behavior, with heterogeneous responses by assets and credit access. Households classified as "stayers" (persistent entrepreneurs, >50% waves) or "copers" (intermittent responders).

## Commands

```bash
# Setup
make setup              # Install Python package with dev dependencies (pip install -e ".[dev]")
make setup-r            # Restore R/renv environment

# Quality
make test               # Run pytest (101 tests)
make lint               # ruff check src/ tests/
make format             # ruff format src/ tests/
make type-check         # mypy src/

# Run single test
pytest tests/test_model.py::test_function_name -v

# Data pipeline
make ingest-data country=tanzania    # Download LSMS or generate synthetic data
make derive-targets country=tanzania # Build derived target tables

# Simulation
make run-toy                         # Synthetic data, quick test
make run-sim COUNTRY=tanzania        # With derived targets
make run-sim COUNTRY=tanzania CALIBRATE=1  # Auto-calibrated thresholds
make run-sim-llm-stub COUNTRY=tanzania     # LLM stub policy with logging

# Validation reports (requires Quarto + R)
make render-report                   # Toy mode report
make render-report-country country=tanzania
```

## Architecture

### Layer Structure

```
src/
├── abm_enterprise/           # Main ABM package
│   ├── model.py              # EnterpriseCopingModel (Mesa 3 model)
│   ├── cli.py                # Typer CLI (abm command)
│   ├── agents/household.py   # HouseholdAgent with state machine
│   ├── policies/             # Decision policies
│   │   ├── base.py           # BasePolicy, Action enum
│   │   ├── rule.py           # RulePolicy, CalibratedRulePolicy
│   │   ├── llm.py            # LLMPolicy with proposal→constraints→commit pattern
│   │   ├── providers.py      # StubProvider, ReplayProvider, ClaudeProvider, OpenAIProvider
│   │   └── constraints.py    # Feasibility constraints for LLM proposals
│   ├── data/
│   │   ├── schemas.py        # Pydantic schemas (HouseholdState, SimulationConfig, etc.)
│   │   └── synthetic.py      # Synthetic data generation
│   └── utils/                # Logging, RNG, manifest
└── etl/                      # Data ingestion pipeline
    ├── ingest.py             # LSMS download or synthetic fallback
    ├── canonical.py          # Create canonical Parquet tables
    └── derive.py             # Build derived targets for ABM
```

### Key Patterns

**Policy System:** All decision logic follows `BasePolicy.decide(HouseholdState) -> Action` interface. RulePolicy uses deterministic thresholds; LLMPolicy uses LLM + constraint validation + logging for reproducibility.

**LLM Decision Pipeline:** proposal→constraints→commit pattern. LLM proposes action, constraints validate feasibility, fallback to NO_CHANGE on failure. All decisions logged to JSONL with state hashes.

**Data Flow:**
1. `ingest-data` → raw LSMS or synthetic data
2. `derive-targets` → canonical Parquet tables (household, plot, plot_crop)
3. ABM loads `household_targets.parquet` with enterprise_persistence, asset_index, price_exposure
4. Simulation outputs `household_outcomes.parquet` + `manifest.json`
5. R/Quarto validation reports compare simulated vs observed distributions

**Reproducibility:** Centralized RNG via `utils/rng.py`, all outputs include manifest.json with git commit, seed, parameters. LLM logs enable deterministic replay.

### Country Configuration

YAML configs in `config/` define wave mappings, crop codes, thresholds. Adding a country requires only new config file, no code changes:
- `config/tanzania.yaml` - 4 waves (2008-2014)
- `config/ethiopia.yaml` - 3 waves (2011-2015)

## Data Contracts

**Measurement Mapping:** `docs/measurement_mapping.csv` is the authoritative contract between ABM variables and LSMS sources. All validation must reference this mapping.

**Validation Contract:** `docs/VALIDATION_CONTRACT.md` specifies estimands, acceptance criteria, and schema contracts. Primary estimand: enterprise_entry ~ price_exposure with household + time FE.

**Schema Contracts:**
- Simulation output: household_id, wave, enterprise_status, enterprise_entry, price_exposure, assets_index, classification
- Derived targets: enterprise_indicator, enterprise_persistence, asset_quintile, credit_access

## Testing

Tests organized by component: `test_model.py`, `test_etl.py`, `test_llm_policy.py`, `test_integration.py`. Run specific test file with:
```bash
pytest tests/test_llm_policy.py -v
pytest tests/test_model.py -k "test_step" -v  # pattern match
```

## Key Files

- `docs/CONOPS.md` - Concept of operations, system architecture
- `docs/DECISIONS.md` - Architectural decision records with external review feedback
- `docs/PROJECT_STATE.md` - Phase status, agent assignments, commands reference
- `docs/VALIDATION_CONTRACT.md` - Estimands, acceptance criteria, schemas
