# RUN_LOG.md - Pipeline Validation Run

**Date:** 2026-01-13
**Branch:** runbook/20260113-pipeline-validation
**Initial Commit:** e1df44f

---

## 1. Preflight

### 1.1 Tests

**Command:** `make test`

**Result:** PASS (101/101 tests)

```
============================= 101 passed in 16.40s =============================
```

**Test Coverage:**
- test_etl.py: 22 tests (schemas, prices, ingest, canonical, derive, integration)
- test_integration.py: 10 tests (derived targets, calibration, classification)
- test_llm_policy.py: 26 tests (providers, logging, constraints, prompts, integration)
- test_manifest.py: 11 tests (creation, save/load, git hash)
- test_model.py: 10 tests (initialization, step, run, reproducibility)
- test_rng.py: 8 tests (seeding, reproducibility)
- test_schemas.py: 14 tests (validation, config)

### 1.2 Lint

**Command:** `make lint`

**Initial Result:** 44 errors

**Fixes Applied:**
- Added B008 to ruff ignore list (typer false positive)
- Added Action type hint import to household.py
- Added `from None` to exception re-raises
- Converted set generators to set comprehensions
- Added `strict=True` to zip calls
- Removed unused variables
- Renamed unused loop variables with underscore prefix

**Final Result:** PASS

```
All checks passed!
```

**Commit:** e1df44f - "fix: resolve all lint errors for clean preflight"

---

## 2. Toy Pipeline

**Command:** `make run-toy`

**Result:** PASS

```
Running toy simulation with seed=42
Simulation complete. Outputs written to outputs/toy
  - Outcomes: outputs/toy/household_outcomes.parquet
  - Manifest: outputs/toy/manifest.json

Enterprise participation rate by wave:
  Wave 1: 57.0%
  Wave 2: 74.0%
  Wave 3: 75.0%
  Wave 4: 76.0%
```

**Output Validation:**
- `outputs/toy/manifest.json`: EXISTS
  - seed: 42
  - git_hash: e1df44f-dirty
  - country: tanzania
  - scenario: toy
- `outputs/toy/household_outcomes.parquet`: EXISTS
  - Rows: 400 (100 households × 4 waves)
  - Columns: 18 (all required columns present)
  - Waves: [1, 2, 3, 4]

---

## 3. Tanzania Baseline

**Command:** `make run-sim COUNTRY=tanzania`

**Result:** PASS

```
Running simulation for tanzania - baseline
Loaded derived targets from data/processed
  Households: 500
  Observations: 2000

Enterprise participation rate by wave:
  Wave 1: 19.6%
  Wave 2: 19.6%
  Wave 3: 19.6%
  Wave 4: 19.6%

Classification distribution:
  coper: 308 (15.4%)
  none: 1256 (62.8%)
  stayer: 436 (21.8%)
```

**Output Validation:**
- `outputs/tanzania/baseline/manifest.json`: EXISTS
  - seed: 42
  - git_hash: e1df44f-dirty
  - country: tanzania
  - scenario: baseline
- `outputs/tanzania/baseline/household_outcomes.parquet`: EXISTS
  - Rows: 2000 (500 households × 4 waves)
  - Unique households: 500
  - Waves: [1, 2, 3, 4]

---

## 4. Tanzania Report

**Command:** `make render-report-country country=tanzania`

**Result:** PASS

**Output:** `analysis/quarto/_output/validation_report.html`

**R fixes applied:**
- Handle partitioned parquet directories with `arrow::open_dataset()`
- Normalize `assets_index` → `assets` column name

---

## 5. Ethiopia Baseline

**Command:** `make run-sim COUNTRY=ethiopia`

**Result:** PASS

**Note:** Initial run had stale wave=4 partition from previous test. Required clean + re-run.
This indicates a potential issue: parquet writer doesn't clean old partitions when wave count changes.

```
Running simulation for ethiopia - baseline
Loaded derived targets from data/processed
  Households: 500
  Observations: 1500

Enterprise participation rate by wave:
  Wave 1: 19.6%
  Wave 2: 19.6%
  Wave 3: 19.6%

Classification distribution:
  coper: 117 (7.8%)
  none: 1002 (66.8%)
  stayer: 381 (25.4%)
```

**Output Validation:**
- `outputs/ethiopia/baseline/manifest.json`: EXISTS
  - seed: 42
  - git_hash: e1df44f-dirty
  - country: ethiopia
  - scenario: baseline
  - num_waves: 3
- `outputs/ethiopia/baseline/household_outcomes.parquet`: EXISTS
  - Rows: 1500 (500 households × 3 waves)
  - Unique households: 500
  - Waves: [1, 2, 3]

---

## 6. Ethiopia Report

**Command:** `make render-report-country country=ethiopia`

**Result:** PASS

**Output:** `analysis/quarto/_output/validation_report.html`

---

## 7. LLM Stub Run

**Command:** `make run-sim-llm-stub COUNTRY=tanzania`

**Result:** PASS

```
Using LLMPolicy with StubProvider, logs: decision_logs/tanzania

LLM Decision Summary:
  Total decisions: 400
  Constraint failure rate: 7.0%
  NO_CHANGE: 386
  ENTER_ENTERPRISE: 14

Enterprise participation rate by wave:
  Wave 1: 54.0%
  Wave 2: 58.0%
  Wave 3: 63.0%
  Wave 4: 67.0%
```

**Output Validation:**
- `outputs/tanzania/llm_stub/household_outcomes.parquet`: EXISTS (400 rows)
- `outputs/tanzania/llm_stub/manifest.json`: EXISTS
- `decision_logs/tanzania/decisions_20260113_163755.jsonl`: EXISTS (568KB)

**Decision Log Validation:**
- Total constraint failures: 28
- All failures properly fell back to NO_CHANGE (no infeasible actions committed)
- Failed constraints observed: CreditRequiredConstraint, MinimumAssetsConstraint
- All records include: state_hash, prompt, response, parsed_action, final_action, provider, latency_ms

**Constraint System Verification:** PASS
- Proposal → Constraints → Commit pattern working correctly
- No infeasible action was ever committed as final_action

---

## Summary (runbook/20260113-pipeline-validation)

| Step | Status | Notes |
|------|--------|-------|
| Preflight Tests | PASS | 101/101 |
| Preflight Lint | PASS | After fixes |
| Toy Pipeline | PASS | 400 records, manifest valid |
| Tanzania Baseline | PASS | 2000 records, 500 households |
| Tanzania Report | PASS | HTML generated |
| Ethiopia Baseline | PASS | 1500 records, 500 households, 3 waves |
| Ethiopia Report | PASS | HTML generated |
| LLM Stub Run | PASS | Constraint system verified |

---

# Threshold Robustness Validation Run

**Date:** 2026-01-13
**Branch:** release/20260113-threshold-robustness
**Base Commit:** f746b17

---

## 8. Threshold Robustness Preflight

### 8.1 Tests

**Command:** `make test`

**Result:** PASS (108/108 tests)

```
============================= 108 passed in 16.87s =============================
```

**New Tests Added:**
- TestStalePartitionGuard: 7 tests for CLI stale partition detection

### 8.2 Toy Simulation

**Command:** `make run-toy`

**Result:** PASS

```
Running toy simulation with seed=42
Simulation complete. Outputs written to outputs/toy
  - Outcomes: outputs/toy/household_outcomes.parquet (400 rows)
  - Manifest: outputs/toy/manifest.json

Enterprise participation rate by wave:
  Wave 1: 57.0%
  Wave 2: 74.0%
  Wave 3: 75.0%
  Wave 4: 76.0%
```

### 8.3 Tanzania Baseline (with --clean-output)

**Command:** `make run-sim COUNTRY=tanzania CLEAN_OUTPUT=1`

**Result:** PASS

```
Running simulation for tanzania - baseline
Loaded derived targets from data/processed
  Households: 500
  Observations: 2000

Simulation complete. Outputs written to outputs/tanzania/baseline
  - Outcomes: 2000 rows
  - Manifest: includes stayer_threshold parameter

Classification distribution:
  coper: 308 (15.4%)
  none: 1256 (62.8%)
  stayer: 436 (21.8%)
```

### 8.4 Report Rendering

**Commands:**
- `make render-report` (toy mode)
- `make render-report-country country=tanzania`

**Result:** BOTH PASS

- R warnings (14) present but reports render successfully
- Sensitivity analysis section renders with threshold comparison
- Quarto check-quarto target working as expected

---

## 9. Changes Validated

| Feature | Status | Notes |
|---------|--------|-------|
| Threshold documentation | COMPLETE | THRESHOLD_JUSTIFICATION.md |
| Sensitivity analysis (R) | COMPLETE | run_threshold_sensitivity() |
| Threshold in config | COMPLETE | stayer_threshold: 0.5 |
| Threshold in manifest | COMPLETE | Parameters include threshold |
| Stale partition guard | COMPLETE | 7 tests, CLI flags work |
| Quarto check | COMPLETE | clear error message |
| Report sensitivity section | COMPLETE | Table + plot render |

---

## Summary (release/20260113-threshold-robustness)

| Step | Status | Notes |
|------|--------|-------|
| Preflight Tests | PASS | 108/108 (+7 new) |
| Toy Pipeline | PASS | 400 records |
| Tanzania Baseline | PASS | 2000 records, --clean-output tested |
| Toy Report | PASS | With sensitivity section |
| Tanzania Report | PASS | With sensitivity section |
| Stale Partition Guard | PASS | Tested via unit tests + integration |
| Quarto Check | PASS | Quarto 1.8.26 detected |

**Final Status:** READY FOR MERGE
