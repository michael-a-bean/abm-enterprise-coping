# Data Audit Remediation Plan

**Created:** 2026-01-14
**Branch:** docs/abm-report-fixes
**Checkpoint:** audit-checkpoint-20260114

## Executive Summary

This document outlines the remediation plan for the 5 open flags identified in `docs/DATA_AUDIT.md`. The fix order minimizes rework by addressing foundational issues first.

---

## FLAG 3 + FLAG 4: Baseline Configuration Standardization

### Issue
- **FLAG 3:** Inconsistent N values (500 vs 100) across runs
- **FLAG 4:** Batch runs use toy mode, not Tanzania baseline configuration

### What to Change

1. **Define canonical baseline configuration:**
   - N = 100 for exploratory analysis (sweeps, search, batch)
   - N = 500 for LSMS-derived baseline (Tanzania, Ethiopia)
   - Document this split explicitly in DATA_CONTRACT.md

2. **Create baseline batch runner:**
   - Update batch generation to use LSMS-derived data (Tanzania baseline config)
   - All batch seeds use identical config except seed value

### Where (Paths/Modules)

- `scripts/run_batch.py` (NEW) - Batch runner for baseline config
- `docs/DATA_CONTRACT.md` - Update canonical baseline definition
- `outputs/batch/` - Regenerated batch outputs

### How to Validate

- [ ] All batch manifests show `scenario: "baseline"`, `data_source: "lsms_derived"`
- [ ] All batch runs have identical N (500), horizon (4 waves), country (tanzania)
- [ ] Quick check script confirms consistency

### New Outputs

- `outputs/batch/baseline/seed_{1..10}/` - LSMS-derived baseline batch
- `outputs/batch/baseline/batch_manifest.json` - Aggregate metadata

---

## FLAG 1: Calibrate Synthetic Generation to LSMS Patterns

### Issue
Sweeps and search use `generate_synthetic_households()` which samples from hardcoded distributions, NOT the calibration artifact. Results don't match LSMS patterns.

### What to Change

1. **Run calibration to create artifact:**
   ```bash
   abm calibrate --country tanzania --data-dir data/processed
   ```
   Creates: `artifacts/calibration/tanzania/calibration.json`

2. **Update sweep runner to use calibrated synthetic:**
   - Add `--calibrated` flag to use `SyntheticPanelGenerator`
   - Load calibration artifact, generate panels from fitted distributions

3. **Regenerate sweeps with calibrated data:**
   - Clear label: "calibrated synthetic" vs "uncalibrated synthetic"
   - Store in separate directories

### Where (Paths/Modules)

- `scripts/run_sweep.py` - Add `--calibration-path` argument
- `scripts/run_behavior_search.py` - Add `--calibration-path` argument
- `artifacts/calibration/tanzania/calibration.json` - Calibration artifact
- `outputs/sweeps/calibrated/` - New calibrated sweep outputs

### How to Validate

- [ ] Calibration artifact exists with goodness-of-fit metrics
- [ ] At least 2 calibration fit plots (assets, shocks) comparing LSMS vs fitted
- [ ] Sweep config files include `data_source: "calibrated"`
- [ ] Table showing pattern error (RMSE) for key distributions

### New Outputs

- `artifacts/calibration/tanzania/calibration.json`
- `outputs/sweeps/calibrated/sweep_*.parquet`
- `docs/figures/calibration_fit_*.png` (fit comparison plots)

---

## FLAG 6: Remove Hardcoded Target Enterprise Rates

### Issue
Search script uses hardcoded target rates: `{1: 0.25, 2: 0.28, 3: 0.32, 4: 0.35}`
These are not dynamically loaded from LSMS derived targets.

### What to Change

1. **Load targets from LSMS data:**
   - Read `data/processed/tanzania/derived/enterprise_targets.parquet`
   - Extract enterprise rates by wave

2. **Refactor search objective:**
   - Multi-objective: minimize distance to LSMS patterns + regularization
   - Add holdout validation (fit on waves 1-3, validate on wave 4)
   - Report in-sample vs out-of-sample scores

3. **Add overfitting safeguards:**
   - Cross-validation by wave
   - Early stopping if validation loss diverges

### Where (Paths/Modules)

- `scripts/run_behavior_search.py` - Refactor objective and data loading
- `src/abm_enterprise/data/targets.py` (NEW) - Target loading utilities
- `outputs/search/calibrated/` - Search outputs with loaded targets

### How to Validate

- [ ] Search config shows `target_source: "lsms_derived"`
- [ ] Objective function documented in report
- [ ] In-sample vs out-of-sample scores reported
- [ ] Pareto frontier plot if multi-objective

### New Outputs

- `outputs/search/calibrated/candidates_*.parquet`
- `outputs/search/calibrated/validation_metrics.json`

---

## FLAG 2: LLM Policy Runs

### Issue
All outputs show `policy_type: "none"` or use RulePolicy. No LLM decisions in empirical sections.

### What to Change

**Option A: Generate LLM runs (if API keys available)**

1. Configure LLM provider (OpenAI or Claude)
2. Enable decision caching for reproducibility
3. Generate baseline + LLM comparison runs
4. Store decision logs with prompt hash, action, confidence

**Option B: Formalize boundary (if API keys unavailable)**

1. Keep LLM sections as "Design Documentation"
2. Add explicit "Required to Execute" checklist
3. Create stub pipeline that fails gracefully
4. Document exact commands + env vars needed

### Where (Paths/Modules)

- `src/abm_enterprise/policies/llm.py` - LLM policy implementation
- `src/abm_enterprise/policies/cache.py` - Decision caching
- `outputs/tanzania/llm_*/` - LLM policy outputs
- `docs/abm_report.qmd` - Update LLM sections

### How to Validate

**If executed:**
- [ ] At least one figure comparing rule-based vs LLM outcomes
- [ ] Decision logs with prompt hash, action, confidence, timestamp
- [ ] Manifest shows `policy_type: "llm_openai"` or similar

**If not executed:**
- [ ] LLM sections clearly labeled "Design Documentation - Not Yet Executed"
- [ ] Execution checklist with commands and env vars
- [ ] Stub pipeline that fails gracefully with helpful message

### New Outputs

**If executed:**
- `outputs/tanzania/llm_openai/` - LLM policy outputs
- `outputs/tanzania/llm_openai/decision_log.parquet` - Decision audit trail

**If not executed:**
- Updated callouts in `docs/abm_report.qmd`

---

## Execution Order

```
Phase 2A: FLAG 3 + FLAG 4 (baseline standardization)
    ↓
Phase 2B: FLAG 1 (calibration + calibrated sweeps)
    ↓
Phase 2C: FLAG 6 (search refactor with LSMS targets)
    ↓
Phase 2D: FLAG 2 (LLM policy or boundary formalization)
    ↓
Phase 3: Report alignment + render
```

---

## Commit Strategy

After each phase, commit with message:
1. `fix: standardize baseline config + regenerate batch (FLAG 3+4)`
2. `feat: calibrate synthetic baseline + regenerate sweeps (FLAG 1)`
3. `refactor: search objective with LSMS targets (FLAG 6)`
4. `feat: LLM policy runs` OR `docs: LLM pipeline stub + checklist (FLAG 2)`
5. `docs: report alignment + render validation`

---

## Success Criteria

| Flag | Success Metric |
|------|----------------|
| FLAG 3 | All batch manifests show consistent N=500 |
| FLAG 4 | All batch manifests show scenario="baseline" |
| FLAG 1 | Sweep data labeled "calibrated", fit plots in report |
| FLAG 6 | Search loads targets from LSMS, validation metrics reported |
| FLAG 2 | LLM figure in report OR execution checklist complete |

---

*Document created: 2026-01-14*
*Branch: docs/abm-report-fixes*
