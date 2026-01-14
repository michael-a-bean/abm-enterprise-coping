# Simulation Outputs

This directory contains simulation results from the ABM Enterprise Coping Model.

## Directory Structure

```
outputs/
├── toy/                      # Quick test simulation (synthetic data)
├── tanzania/
│   ├── baseline/             # Baseline simulation with rule-based policy
│   └── llm_stub/             # LLM policy with stub provider
├── ethiopia/
│   └── baseline/             # Ethiopia baseline simulation
├── batch/                    # Multi-seed batch runs for robustness
│   ├── seed_1/
│   ├── seed_2/
│   └── ...seed_10/
├── sweeps/                   # Parameter sweep outputs (for heatmaps)
│   ├── sweep_agg_latest.parquet  # Aggregated results
│   ├── sweep_full_latest.parquet # Full per-seed results
│   └── sweep_config_latest.json  # Sweep configuration
└── search/                   # Behavior search outputs (optimization)
    ├── candidates_latest.parquet # Candidate evaluations
    └── search_config_latest.json # Search configuration
```

## Output Files

Each simulation run produces:

| File | Description |
|------|-------------|
| `household_outcomes.parquet/` | Partitioned Parquet dataset with wave-level household outcomes |
| `manifest.json` | Metadata: git commit, RNG seed, parameters, timestamps |
| `simulation.log` | Structured log of simulation execution |
| `decision_logs/` | (Optional) JSONL logs of LLM decisions when using LLM policy |

## Regeneration Commands

### Quick Test (Toy Mode)
```bash
abm run-toy
```

### Country Baseline (Tanzania)
```bash
# First ensure derived targets exist
abm derive-targets --country tanzania

# Run baseline simulation
abm run-sim tanzania --scenario baseline --seed 42
```

### Country Baseline (Ethiopia)
```bash
abm derive-targets --country ethiopia
abm run-sim ethiopia --scenario baseline --seed 42
```

### LLM Policy Simulation
```bash
# Stub provider (deterministic, for testing)
abm run-sim tanzania --policy llm_stub --seed 42

# OpenAI provider (requires OPENAI_API_KEY)
abm run-sim tanzania --policy llm_openai --seed 42

# Claude provider (requires ANTHROPIC_API_KEY)
abm run-sim tanzania --policy llm_claude --seed 42
```

### Batch Simulations (Robustness Analysis)
```bash
# Run 10 seeds in parallel
for seed in {1..10}; do
  abm run-sim tanzania --scenario baseline --seed $seed \
    --output-dir outputs/batch/seed_$seed &
done
wait
```

Or using the Python API:
```python
from abm_enterprise.model import EnterpriseCopingModel
from abm_enterprise.policies.rule import RulePolicy

for seed in range(1, 11):
    model = EnterpriseCopingModel(
        country="tanzania",
        policy=RulePolicy(),
        seed=seed,
    )
    model.run()
    model.save_outputs(f"outputs/batch/seed_{seed}")
```

### Parameter Sweep (Sensitivity Analysis)
```bash
# Automated sweep script (6x6 grid, 2 seeds each = 72 runs)
python3 scripts/run_sweep.py --grid-size 6 --seeds 2 --households 100
```

### Behavior Search (Optimization)
```bash
# Random search for optimal parameters (40 candidates, 2 seeds each)
python3 scripts/run_behavior_search.py --n-candidates 40 --seeds 2
```

## Data Schema

The `household_outcomes.parquet` dataset contains:

| Column | Type | Description |
|--------|------|-------------|
| `household_id` | string | Unique household identifier |
| `wave` | int | Simulation wave (1-4 for Tanzania) |
| `assets_index` | float | Standardized asset index |
| `credit_access` | int | Credit access indicator (0/1) |
| `enterprise_status` | int | Current enterprise status (0/1) |
| `enterprise_entry` | int | Entry this wave (0/1) |
| `price_exposure` | float | Commodity price shock exposure |
| `crop_count` | int | Number of crops in portfolio |
| `land_area_ha` | float | Land area in hectares |
| `action_taken` | string | Action: ENTER_ENTERPRISE, EXIT_ENTERPRISE, NO_CHANGE |
| `policy_applied` | int | Whether policy was applied (0/1) |
| `asset_quintile` | int | Asset quintile (1-5) |
| `classification` | string | Household type: stayer, coper, none |
| `enterprise_persistence` | float | Fraction of waves with enterprise |
| `welfare_proxy` | float | Welfare proxy value |

## Loading Outputs in R

```r
library(arrow)
library(dplyr)

# Single simulation
outcomes <- open_dataset("outputs/tanzania/baseline/household_outcomes.parquet") |>
  collect()

# Batch simulations
source("R/analysis_helpers.R")
batch_df <- read_batch_simulations("outputs/batch")
robustness <- calculate_robustness_metrics(batch_df)
```

## Loading Outputs in Python

```python
import pyarrow.parquet as pq
import pandas as pd

# Load partitioned dataset
outcomes = pq.read_table("outputs/tanzania/baseline/household_outcomes.parquet").to_pandas()

# Or with arrow
import pyarrow.dataset as ds
dataset = ds.dataset("outputs/batch", partitioning="hive")
batch_df = dataset.to_table().to_pandas()
```

## Manifest Schema

Each `manifest.json` contains:

```json
{
  "simulation_id": "uuid",
  "git_commit": "abc1234",
  "timestamp": "2026-01-13T12:00:00Z",
  "config": {
    "country": "tanzania",
    "scenario": "baseline",
    "seed": 42,
    "n_households": 1000,
    "n_waves": 4,
    "policy": "rule"
  },
  "metrics": {
    "runtime_seconds": 12.5,
    "n_records": 4000
  }
}
```

## Notes

- All timestamps are UTC
- Parquet files use Snappy compression
- Wave partitioning enables efficient filtering by wave
- Manifests enable full reproducibility from git commit + seed
