# Phase 0: Ground Truth Audit
**Date:** 2026-01-14
**Commit:** ec8c6ec (branch: docs/abm-report-fixes)

## Repo Map

### Core ABM Implementation (Mesa 3)
```
src/abm_enterprise/
├── model.py              # EnterpriseCopingModel (lines 28-201)
├── cli.py                # Typer CLI with commands: run-toy, run-sim, calibrate, eval-direct
├── agents/household.py   # HouseholdAgent (lines 74-309)
├── policies/
│   ├── llm.py            # MultiSampleLLMPolicy (lines 429-703)
│   ├── voting.py         # majority_vote() (lines 76-130)
│   ├── cache.py          # DecisionCache (lines 83-212)
│   └── constraints.py    # Feasibility validation
├── calibration/fit.py    # Distribution fitting (lines 80-200)
├── data/synthetic.py     # Panel generation (lines 45-180)
└── eval/                 # Direct prediction + baselines
```

### R Analysis Infrastructure
```
R/
├── plot_theme.R          # theme_abm_minimal(), colorblind palettes
└── analysis_helpers.R    # read_batch_simulations(), phase portraits, heatmaps

analysis/R/
├── read_simulation.R     # Parquet loading
└── validation_helpers.R  # FE regressions, threshold sensitivity
```

### Report
- `docs/abm_report.qmd` - Main Quarto report (1527 lines)
- `docs/abm_report.html` - Rendered output (5MB, too large to read directly)

---

## Data Map

### Existing Outputs (Real Data)
| Path | Status | Description |
|------|--------|-------------|
| `outputs/toy/` | EXISTS | Toy simulation (N=100, 4 waves) |
| `outputs/tanzania/baseline/` | EXISTS | Tanzania baseline (RulePolicy) |
| `outputs/tanzania/llm_stub/` | EXISTS | Tanzania with StubProvider |
| `outputs/ethiopia/baseline/` | EXISTS | Ethiopia baseline |
| `outputs/batch/seed_1..10/` | EXISTS | 10 seed batch runs for robustness |

### Missing Outputs (CRITICAL GAPS)
| Path | Status | Impact |
|------|--------|--------|
| `outputs/sweeps/` | MISSING | **Heatmap in report uses synthetic data!** (lines 1079-1118) |
| `outputs/search/` | MISSING | Behavior search section describes methods only, no results |

### Data Pipeline Status
- `data/processed/tanzania/` - Assumed present (no validation)
- `data/processed/ethiopia/` - Assumed present
- `artifacts/calibration/` - Calibration artifacts (not verified)

---

## Placeholder Audit

### CRITICAL: Synthetic Heatmap (Lines 1079-1118)
```r
# This code generates FAKE sweep data:
sweep_df <- expand.grid(
  price_threshold = seq(-0.3, 0.0, by = 0.05),
  asset_threshold = seq(-1.0, 1.0, by = 0.25)
) |>
  mutate(
    enterprise_rate = 0.15 + 0.25 * pnorm(price_threshold + 0.15, sd = 0.1) +
                     0.15 * pnorm(-asset_threshold, sd = 0.5)
  )
```
**FIX REQUIRED:** Replace with actual parameter sweep outputs.

### Other Sections Using Real Data
- `fig-baseline-enterprise` - Uses real outputs (conditional on existence)
- `fig-robustness` - Uses batch data (EXISTS)
- `fig-phase-portrait` - Uses Tanzania baseline (EXISTS)
- `fig-multi-seed-phase` - Uses batch data (EXISTS)
- `tbl-fe-regression` - Uses real simulation data

---

## Evidence Map

| Feature | Implemented? | Evidence (path::lines) | Notes |
|---------|--------------|----------------------|-------|
| Mesa 3 Model | YES | `model.py::28-201` | EnterpriseCopingModel |
| HouseholdAgent | YES | `agents/household.py::74-309` | Full state machine |
| MultiSampleLLMPolicy | YES | `policies/llm.py::429-703` | K=5 voting |
| Decision Caching | YES | `policies/cache.py::83-212` | SHA-256 state hash |
| Constraint Validation | YES | `policies/constraints.py` | Feasibility checks |
| Majority Voting | YES | `policies/voting.py::76-130` | Conservative tie-break |
| Early Stopping | YES | `policies/llm.py::571-587` | 3-sample agreement |
| Calibration | YES | `calibration/fit.py::80-200` | Distribution fitting |
| Synthetic Panel Gen | YES | `data/synthetic.py::45-180` | From CalibrationArtifact |
| Direct Prediction Eval | YES | `eval/direct_prediction.py::80-195` | Transition dataset |
| ML Baselines | YES | `eval/baselines.py` | Logistic, RF, GBM |
| Batch Run Analysis | YES | `R/analysis_helpers.R::24-59` | read_batch_simulations() |
| Phase Portraits | YES | `R/analysis_helpers.R::146-192` | create_phase_portrait() |
| Sensitivity Heatmap | PARTIAL | `R/analysis_helpers.R::285-322` | Function exists, no data |
| Parameter Sweep CLI | NO | --- | Manual loop in README only |
| Behavior Search | NO | --- | Described but not implemented |
| Bayesian Optimization | NO | --- | Proposed in report, not coded |

---

## CLI Commands Available

```bash
abm run-toy              # Toy mode with synthetic data
abm run-sim              # Country simulation with policies
abm run-sim-synthetic    # Synthetic ABM from calibration
abm calibrate            # Fit distributions from LSMS
abm eval-direct          # Direct prediction evaluation
abm ingest-data          # Download/process LSMS
abm derive-targets       # Build derived tables
abm validate-schema      # Validate output schemas
```

### Missing Commands
- `abm sweep` or `abm batch-sweep` - Would automate parameter grid search
- `abm behavior-search` - Would implement optimization

---

## Required Actions

### Phase 3: Generate Real Outputs
1. **Parameter Sweep** - Run grid over (price_threshold, asset_threshold)
   ```bash
   # Proposed 6x6 grid with 2 seeds each = 72 runs
   for pt in -0.3 -0.2 -0.15 -0.1 -0.05 0.0; do
     for at in -1.0 -0.5 0.0 0.5 1.0; do
       for seed in 42 43; do
         abm run-toy --seed $seed \
           --output-dir outputs/sweeps/pt_${pt}_at_${at}_s${seed}
       done
     done
   done
   ```

2. **Behavior Search** - Random search over parameter space
   - Define objective: Minimize (enterprise_rate - target)^2
   - Run 40-80 candidates with 2 seeds each
   - Store in `outputs/search/`

### Phase 4: Update Report
- Replace synthetic heatmap with real sweep data
- Add behavior search results section
- Update "Known Limitations" to reflect actual state
