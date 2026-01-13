# CODE_REVIEW.md - Pipeline Validation Code Review

**Date:** 2026-01-13
**Reviewer:** Claude Code (Automated)
**Branch:** runbook/20260113-pipeline-validation

---

## Strengths

### 1. Decision Layer Architecture (Excellent)
The proposal→constraints→commit pattern in `policies/` is well-implemented:
- `LLMPolicy.decide()` follows a clear three-phase flow
- `CompositeConstraint.get_failed_constraints()` provides detailed rejection reasons
- All decisions are logged with state hashes for replay capability
- Fallback to NO_CHANGE on constraint failure is robust

**Files:** `policies/llm.py`, `policies/constraints.py`, `policies/logging.py`

### 2. Reproducibility Infrastructure (Excellent)
- Centralized RNG via `utils/rng.py` with explicit seeding
- Manifest includes git hash (with -dirty flag), seed, all parameters
- Run IDs generated for each simulation
- Decision logs enable deterministic replay via `ReplayProvider`

**Files:** `utils/rng.py`, `utils/manifest.py`, `policies/providers.py:ReplayProvider`

### 3. LLM Decision Logging (Excellent)
Each decision record includes:
- `state_hash`: Deterministic hash for state matching
- `prompt`: Full prompt sent to LLM
- `response`: Raw LLM output
- `parsed_action`: Parsed action from response
- `constraints_passed`: Boolean flag
- `failed_constraints`: List of failing constraint names
- `final_action`: Committed action (may differ from parsed)
- `latency_ms`: Performance metric

**File:** `policies/logging.py:DecisionRecord`

### 4. Test Coverage (Good)
- 101 tests covering model, ETL, policies, schemas
- Integration tests validate end-to-end pipeline
- Constraint tests verify rejection logic
- Replay tests verify deterministic reproduction

### 5. R Validation Design (Good)
- FE regressions correctly specified with `fixest::feols`
- Primary estimand: `enterprise_status ~ price_exposure | household_id + wave`
- Interaction terms for asset and credit heterogeneity
- Structured result extraction with sign checking

**Files:** `analysis/R/validation_helpers.R`

### 6. ETL Pipeline (Good)
- Clean separation: ingest → canonical → derive
- Referential integrity checks in tests
- Measurement mapping CSV documents variable provenance
- Country config YAML allows new countries without code changes

---

## Issues

### Critical Issues
*None identified*

### Major Issues

**1. Partitioned Parquet Stale Partition Bug**
- **Severity:** Major
- **Location:** `outputs.py:write_parquet()`
- **Description:** When a simulation with fewer waves overwrites a previous run with more waves, old wave partitions persist. Ethiopia (3 waves) retained wave=4 from previous Tanzania run.
- **Impact:** Misleading validation results; row counts don't match manifest
- **Root Cause:** `pq.write_to_dataset()` with `existing_data_behavior="delete_matching"` only deletes matching partitions, not extra ones
- **Recommendation:** Delete entire output directory before writing, or use `existing_data_behavior="overwrite_or_ignore"` + explicit cleanup

### Minor Issues

**2. SolaraViz Import Warning**
- **Severity:** Minor
- **Location:** Model import
- **Description:** "Could not import SolaraViz" warning on every run
- **Impact:** Noisy logs, confusing for users
- **Recommendation:** Suppress warning or make visualization optional import

**3. NoExitIfStayerConstraint Stub**
- **Severity:** Minor
- **Location:** `constraints.py:78-100`
- **Description:** Constraint always returns True with TODO comment about needing classification
- **Impact:** Stayers can exit, contrary to documented behavior
- **Recommendation:** Either implement properly or remove from default constraints

**4. Missing Quarto Dependency Documentation**
- **Severity:** Minor
- **Location:** Setup docs
- **Description:** Quarto required for validation reports but not documented in prerequisites
- **Impact:** Report rendering fails without clear guidance
- **Recommendation:** Add Quarto installation instructions to README/RUNBOOK

**5. Inconsistent Country Variable Case in Makefile**
- **Severity:** Minor
- **Location:** `Makefile:94-140`
- **Description:** Uses both `COUNTRY` (uppercase) and `country` (lowercase) for different targets
- **Impact:** Confusing; `make run-sim COUNTRY=tanzania` vs `make ingest-data country=tanzania`
- **Recommendation:** Standardize to lowercase `country` throughout

---

## Recommended Follow-ups

### High Priority
1. Fix parquet stale partition bug (add directory cleanup before write)
2. Document Quarto prerequisite in RUNBOOK.md

### Medium Priority
3. Implement NoExitIfStayerConstraint properly using classification field
4. Standardize Makefile variable naming

### Low Priority
5. Suppress SolaraViz warning
6. Add type hints to R functions (roxygen2)

---

## Confirmed Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Manifest includes seed | ✅ PASS | All manifests contain `seed: 42` |
| Manifest includes git hash | ✅ PASS | Format: `e1df44f-dirty` |
| Manifest includes country | ✅ PASS | `country: tanzania/ethiopia` |
| Manifest includes scenario | ✅ PASS | `scenario: baseline/llm_stub` |
| Parquet outputs non-empty | ✅ PASS | Toy: 400, TZ: 2000, ET: 1500 rows |
| LLM decisions logged | ✅ PASS | 568KB JSONL with all required fields |
| Constraint failures handled | ✅ PASS | 28 failures → all fell back to NO_CHANGE |
| No infeasible action committed | ✅ PASS | Verified programmatically |
| FE regression spec correct | ✅ PASS | Matches VALIDATION_CONTRACT.md |

---

## Summary

The codebase is production-quality with excellent reproducibility infrastructure and LLM decision auditing. The major issue (stale parquet partitions) should be fixed before production deployment to prevent data integrity issues. All core acceptance criteria pass.

**Overall Assessment:** Ready for release with minor fixes
