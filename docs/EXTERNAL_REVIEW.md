# EXTERNAL_REVIEW.md - Multi-Model External Review

**Date:** 2026-01-13
**Models Consulted:** GPT-4o, Gemini 2.0 Flash
**Review Type:** Scientific Coherence + Reproducibility Red-Team

---

## 1. Gemini Review: Scientific Coherence

### Overfitting Risk Assessment

**CRITICAL CONCERN:** Using the same LSMS data for both calibration AND validation creates circular reasoning.

> "The primary trade-off lies in the potential for overfitting and circular reasoning. Using the same data for calibration and validation makes it difficult to assess the model's true predictive power."

**Recommendations:**
1. Use a separate dataset (different time period or region) for validation
2. Implement out-of-sample validation techniques
3. Conduct sensitivity analysis on parameter values

### Stayer/Coper Classification Threshold

**CONCERN:** The >50% persistence threshold may be arbitrary.

> "Using a >50% persistence threshold might oversimplify the classification of enterprises as stayers or copers, possibly misrepresenting nuanced behaviors."

**Recommendations:**
1. Provide scientific justification for the 50% threshold
2. Explore sensitivity to different threshold values
3. Consider data-driven clustering as alternative

### Fixed Effects Specification

**ASSESSMENT:** Generally appropriate but with caveats.

**Strengths:**
- Controls for unobserved time-invariant heterogeneity (α_i)
- Wave fixed effects control for macro shocks (γ_t)

**Weaknesses:**
- Cannot control for time-varying confounders
- Does not address serial correlation in errors
- Potential endogeneity of price_exposure

### 10pp Tolerance

**CONCERN:** May be too lenient.

> "A 10pp tolerance for distribution matching might be too lenient, potentially masking significant mismatches."

**Recommendation:** Consider tightening to 5pp or using statistical tests (KS, χ²)

---

## 2. GPT Review: Reproducibility/Logging Red-Team

### State Hash Collision Risk

**HIGH RISK:** 16 hex chars = 64 bits is insufficient.

> "SHA-256 is a strong hash function, but using only 16 hex characters (64 bits) of the hash reduces collision resistance significantly."

**Probability Analysis:**
- With 400 decisions per simulation, collision probability is low but non-zero
- For production runs with millions of decisions, risk becomes significant

**Recommendation:** Use full SHA-256 (64 hex chars) or minimum 32 hex chars (128 bits)

### Replay Failure Modes

**CRITICAL:**
1. **Floating-point arithmetic variations** - Different hardware/libraries produce slightly different FP results
2. **External data dependencies** - Unversioned external data breaks replay
3. **OS thread scheduling** - Multi-threaded operations may execute in different order

**HIGH:**
1. **Uncaptured randomness** - Library functions with internal RNG
2. **Library version discrepancies** - Even minor version differences can affect behavior

### R Fixed Effects Critique

**CRITICAL ISSUE: Endogeneity**

> "The `price_exposure` variable is likely endogenous. Enterprise_status may *affect* price_exposure, and/or unobserved variables affect both."

**CRITICAL ISSUE: Serial Correlation**

> "Does not address serial correlation in the error term. Serial correlation will invalidate standard errors."

**Recommendations:**
1. Use clustered standard errors: `feols(..., cluster = ~household_id)`
2. Consider instrumental variable approach
3. Test for endogeneity with Hausman test

### Unlogged Failure Modes

**Identified Gaps:**
1. Silent errors that don't raise exceptions
2. Resource exhaustion (memory, disk)
3. External service failures
4. Implicit constraint violations
5. Logging system failures themselves

---

## 3. Consensus Points

Both models agree on:

| Issue | Severity | Status |
|-------|----------|--------|
| Overfitting risk from shared data | High | Acknowledged |
| 50% threshold needs justification | Medium | Documentation needed |
| State hash too short | High | Fix recommended |
| FE specification needs clustered SEs | High | Code change needed |
| 10pp tolerance may be lenient | Medium | Review recommended |

---

## 4. Divergence Points

| Topic | GPT | Gemini |
|-------|-----|--------|
| Overall assessment | Cautiously positive | More critical |
| ABM complexity | Appropriate | Potentially excessive |
| Alternative models | Suggests ML | Suggests simpler econometrics |

---

## 5. Action Items

### Critical (Fix Before Production)

1. **Increase state_hash length** to minimum 32 hex chars
   - File: `policies/logging.py:95`
   - Change: `hexdigest()[:16]` → `hexdigest()[:32]`

2. **Add clustered standard errors to R FE regressions**
   - File: `analysis/R/validation_helpers.R:181`
   - Change: `feols(formula, data = df)` → `feols(formula, data = df, cluster = ~household_id)`

### High Priority

3. **Document scientific justification** for 50% stayer/coper threshold
4. **Capture full dependency versions** in manifest (pip freeze, renv snapshot)
5. **Address endogeneity concern** - document limitation or implement IV approach

### Medium Priority

6. **Tighten distribution tolerance** from 10pp to 5pp
7. **Add floating-point determinism tests**
8. **Implement out-of-sample validation** using held-out data or cross-validation

---

## 6. Implementation Status

| Action | Status | Commit |
|--------|--------|--------|
| Increase state_hash | DONE | 7628e27 |
| Add clustered SEs | DONE | 7628e27 |
| Document 50% threshold | DONE | See THRESHOLD_JUSTIFICATION.md |
| Sensitivity analysis | DONE | Implemented in validation report |
| Dependency versioning | EXISTS (renv.lock) | - |

---

## 7. Reviewer Agreement

**Gemini Assessment:** Model methodology is sound but overfitting risk needs mitigation
**GPT Assessment:** Infrastructure is well-designed but state_hash and FE specification need fixes

**Overall Verdict:** CONDITIONALLY APPROVED pending critical fixes
