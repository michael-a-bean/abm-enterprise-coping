# Data Contract for ABM Enterprise Coping Model

**Version:** 1.1
**Date:** 2026-01-14
**Status:** ACTIVE - FLAG 3+4 remediation complete

## Definitions

### Run
A **run** is a single execution of the ABM simulation with:
- Fixed random seed
- Fixed parameter configuration
- Fixed household data source
- Fixed policy type

A run produces exactly one output directory containing all required artifacts.

### Seed
A **seed** is an integer that initializes the centralized RNG (`utils/rng.py`). Seeds enable:
- Reproducibility: Same seed + same parameters = identical outputs
- Stochasticity analysis: Different seeds measure variance from random components

**Seed sources of stochasticity:**
1. Price shock draws
2. Asset initialization (synthetic mode)
3. Credit access assignment (synthetic mode)
4. Agent activation order (Mesa RandomActivation)
5. LLM sampling (when temperature > 0)

### Scenario
A **scenario** is a named configuration specifying:
- Country: Which LSMS country configuration to use
- Data source: LSMS-derived targets vs synthetic generation
- Policy type: Rule-based, LLM, or none
- Parameter overrides: Thresholds, sample sizes, etc.

**Canonical scenarios:**
| Scenario Name | Country | Data Source | Policy | N | Waves | Notes |
|---------------|---------|-------------|--------|---|-------|-------|
| `tanzania/baseline` | Tanzania | LSMS-derived | None | 500 | 4 | Primary validated baseline |
| `ethiopia/baseline` | Ethiopia | LSMS-derived | None | 500 | 3 | Secondary validation |
| `batch/lsms` | Tanzania | LSMS-derived | None | 500 | 4 | Multi-seed robustness (10 seeds) |
| `batch/calibrated` | Tanzania | Calibrated synthetic | None | 500 | 4 | Calibrated sensitivity |
| `toy` | Tanzania | Uncalibrated synthetic | None | 100 | 4 | Quick testing only |
| `sweep` | Tanzania | Uncalibrated synthetic* | RulePolicy | 100 | 4 | Exploratory parameter sweep |
| `search` | Tanzania | Uncalibrated synthetic* | RulePolicy | 100 | 4 | Exploratory behavior search |

*Note: Sweep/Search currently use uncalibrated synthetic data. FLAG 1 remediation will add calibrated versions.

**Canonical N values:**
- **LSMS-derived:** N = actual household count from LSMS (500 for Tanzania, varies by country)
- **Calibrated synthetic:** N = 500 (matches LSMS scale)
- **Exploratory synthetic:** N = 100 (fast iteration)

### Sweep Cell
A **sweep cell** is a single point in parameter space defined by:
- `price_threshold`: Negative shock threshold for entry
- `asset_threshold`: Asset threshold for entry
- Additional parameters as defined by sweep config

Each cell is evaluated with multiple seeds to estimate variance.

**Sweep output structure:**
- `sweep_full_*.parquet`: Per-cell, per-seed results
- `sweep_agg_*.parquet`: Aggregated across seeds (mean, std)

## Required Outputs

### Per-Run Outputs

Every run MUST emit the following artifacts:

```
outputs/{country}/{scenario}/
├── manifest.json                    # REQUIRED
├── household_outcomes.parquet/      # REQUIRED (partitioned by wave)
│   ├── wave=1/*.parquet
│   ├── wave=2/*.parquet
│   └── ...
└── simulation.log                   # OPTIONAL
```

### manifest.json Schema

```json
{
  "run_id": "string (8-char hex)",
  "git_hash": "string",
  "seed": "integer",
  "timestamp": "ISO8601 datetime",
  "parameters": {
    "num_waves": "integer",
    "policy_type": "string (none|rule|llm)",
    "price_exposure_threshold": "float",
    "asset_threshold_percentile": "float",
    "stayer_threshold": "float",
    "num_households": "integer"
  },
  "country": "string",
  "scenario": "string",
  "version": "semver string",
  "mesa_version": "semver string"
}
```

**Required fields:** All fields above are REQUIRED.

### household_outcomes.parquet Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `household_id` | string | YES | Unique household identifier |
| `wave` | int | YES | Survey wave (1-indexed) |
| `enterprise_status` | int (0/1) | YES | Current enterprise status |
| `enterprise_entry` | int (0/1) | YES | Entry transition indicator |
| `price_exposure` | float | YES | Price shock exposure |
| `assets_index` | float | YES | Standardized asset index |
| `credit_access` | int (0/1) | YES | Credit access indicator |
| `classification` | string | YES | stayer/coper/none |
| `action_taken` | string | NO | Policy action (if policy active) |

### Sweep Outputs

```
outputs/sweeps/
├── sweep_config_{timestamp}.json    # REQUIRED
├── sweep_full_{timestamp}.parquet   # REQUIRED
├── sweep_agg_{timestamp}.parquet    # REQUIRED
├── sweep_agg_{timestamp}.csv        # OPTIONAL (human-readable)
└── *_latest.* symlinks              # REQUIRED
```

**sweep_config Schema:**
```json
{
  "price_thresholds": [float],
  "asset_thresholds": [float],
  "seeds": [int],
  "num_households": "int",
  "num_waves": "int",
  "country": "string",
  "data_source": "string (synthetic|calibrated|lsms_derived)"  // NEW - REQUIRED
}
```

**sweep_agg Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `price_threshold` | float | Parameter value |
| `asset_threshold` | float | Parameter value |
| `enterprise_rate_mean` | float | Mean across seeds |
| `enterprise_rate_std` | float | Std across seeds |
| `entry_rate_mean` | float | Mean entry rate |
| `exit_rate_mean` | float | Mean exit rate |
| `n_stayers_mean` | float | Mean stayer count |
| `n_copers_mean` | float | Mean coper count |
| `n_none_mean` | float | Mean none count |

### Search Outputs

```
outputs/search/
├── search_config_{timestamp}.json   # REQUIRED
├── candidates_{timestamp}.parquet   # REQUIRED
├── candidates_{timestamp}.csv       # OPTIONAL
└── *_latest.* symlinks              # REQUIRED
```

**search_config Schema:**
```json
{
  "target_enterprise_rates": {"wave": float},
  "price_threshold_range": [min, max],
  "asset_threshold_range": [min, max],
  "exit_threshold_range": [min, max],
  "n_candidates": "int",
  "seeds_per_candidate": "int",
  "num_households": "int",
  "num_waves": "int",
  "data_source": "string",  // NEW - REQUIRED
  "target_source": "string"  // NEW - REQUIRED (hardcoded|lsms_derived)
}
```

### Batch Outputs

```
outputs/batch/
├── {data_source}/                    # lsms | calibrated | synthetic_uncalibrated
│   ├── batch_manifest.json           # REQUIRED - aggregate batch metadata
│   ├── seed_{N}/
│   │   ├── {country}/{scenario}/
│   │   │   ├── manifest.json
│   │   │   └── household_outcomes.parquet/
│   └── ...
```

**batch_manifest.json Schema:**
```json
{
  "batch_id": "string",
  "timestamp": "YYYYMMDD_HHMMSS",
  "git_commit": "string",
  "config": {
    "data_source": "string (lsms|calibrated|synthetic_uncalibrated)",
    "calibration_path": "string|null",
    "country": "string",
    "scenario": "string",
    "num_waves": "int",
    "seeds": [int],
    "num_households": "int"
  },
  "seeds_completed": [int],
  "seeds_failed": [int],
  "total_runs": "int",
  "run_paths": ["string"]
}
```

**Batch contract:**
- All batch runs MUST use identical parameters except seed
- All batch runs MUST use the same data source (all LSMS OR all calibrated OR all uncalibrated)
- `batch_manifest.json` MUST be present at batch root
- Minimum 5 seeds for robustness analysis
- Recommended 10+ seeds for CV estimation
- Current batch: 10 seeds (seeds 1-10), LSMS-derived

**Batch generation command:**
```bash
# LSMS-derived batch (current)
python scripts/run_batch.py --data-source lsms --seeds 10

# Calibrated synthetic batch (after FLAG 1)
python scripts/run_batch.py --data-source calibrated \
  --calibration artifacts/calibration/tanzania/calibration.json --seeds 10

# Verify consistency
python scripts/run_batch.py --verify outputs/batch/lsms
```

## Data Source Classification

Every data artifact MUST be classified as one of:

| Classification | Code | Description |
|----------------|------|-------------|
| LSMS-derived | `lsms` | Uses `load_derived_targets()` from processed LSMS |
| Calibrated synthetic | `calibrated` | Uses `SyntheticPanelGenerator` with `CalibrationArtifact` |
| Uncalibrated synthetic | `synthetic` | Uses `generate_synthetic_households()` directly |

**Report sections MUST NOT mix data from different classifications without explicit disclosure.**

## Validation Rules

### Rule V1: Manifest Completeness
Every run output directory MUST contain a valid `manifest.json`.

### Rule V2: Schema Compliance
All parquet files MUST contain the required columns per their schema.

### Rule V3: Provenance Chain
Sweep/Search outputs MUST specify their `data_source` in config.

### Rule V4: Seed Tracking
All runs MUST record the seed used. No anonymous seeds.

### Rule V5: Git Hash Recording
All runs MUST record the git commit hash (dirty flag acceptable).

### Rule V6: No Data Mixing
Report sections presenting empirical results MUST NOT combine:
- LSMS-derived and synthetic runs
- Runs with different N values (without explicit pooling justification)
- Runs with different policy types (unless comparative analysis)

## Implementation Checklist

**FLAG 3+4 (Baseline Standardization) - COMPLETE:**
- [x] Create `scripts/run_batch.py` with data source classification
- [x] Generate LSMS-derived batch outputs (10 seeds)
- [x] Add `batch_manifest.json` schema to contract
- [x] Document canonical N values (500 for LSMS, 100 for exploratory)

**FLAG 1 (Calibrated Synthetic) - COMPLETE:**
- [x] Run calibration: `abm calibrate --country tanzania`
- [x] Update `scripts/run_sweep.py` to add `--calibration` option
- [x] Update `scripts/run_behavior_search.py` to add `--calibration` option
- [x] Generate calibrated sweep outputs (`outputs/sweeps/calibrated/`)
- [ ] Add calibration fit plots to report (pending report integration)

**FLAG 6 (Search Targets) - COMPLETE:**
- [x] Refactor search to load targets from LSMS derived data (`--targets-from-lsms`)
- [x] Generate calibrated search outputs (`outputs/search/calibrated/`)
- [ ] Add validation metrics (optional enhancement)

**FLAG 2 (LLM Policy) - PENDING:**
- [ ] Generate LLM policy runs OR formalize execution checklist

**Report Integration - PENDING:**
- [ ] Update report to use `outputs/batch/lsms` for robustness analysis
- [ ] Add manifest validation to report rendering
- [ ] Add data source badges to figure captions
