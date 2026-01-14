# Data Contract for ABM Enterprise Coping Model

**Version:** 1.0
**Date:** 2026-01-14
**Status:** DRAFT - Pending implementation

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
| Scenario Name | Country | Data Source | Policy | N | Waves |
|---------------|---------|-------------|--------|---|-------|
| `tanzania/baseline` | Tanzania | LSMS-derived | RulePolicy | 500 | 4 |
| `ethiopia/baseline` | Ethiopia | LSMS-derived | RulePolicy | 500 | 3 |
| `toy` | Tanzania | Synthetic | None | 100 | 4 |
| `sweep` | Tanzania | Synthetic* | RulePolicy | 100 | 4 |
| `search` | Tanzania | Synthetic* | RulePolicy | 100 | 4 |

*Note: Sweep/Search currently use uncalibrated synthetic data. See FLAG 1 in DATA_AUDIT.md.

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
├── seed_{N}/
│   ├── manifest.json
│   └── household_outcomes.parquet/
└── ...
```

**Batch contract:**
- All batch runs MUST use identical parameters except seed
- All batch runs MUST use the same data source (all synthetic OR all LSMS-derived)
- Minimum 5 seeds for robustness analysis
- Recommended 10+ seeds for CV estimation

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

- [ ] Update `scripts/run_sweep.py` to add `data_source` field
- [ ] Update `scripts/run_behavior_search.py` to add `data_source` and `target_source` fields
- [ ] Create calibration-aware sweep runner option
- [ ] Add manifest validation to report rendering
- [ ] Add data source badges to figure captions
