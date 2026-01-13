# Documentation Changelog

**Date**: 2026-01-13
**Target File**: ABM_Enterprise_Coping_Model.qmd
**Branch**: docs/20260113-qmd-audit-regeneration

## Issues Addressed

This changelog maps each fix to the original issue identifiers (A-K).

---

### Issue A: Wrong External Model Used

**Original Problem**: Previous review used "Gemini 2.0 Flash" instead of "Gemini Deep Research" for methodology framing.

**Fix Applied**:
- Attempted to call Gemini Deep Research; API returned error (documented)
- Used alternative models: o1, Gemini 2.0 Flash, GPT-4o-mini
- Updated External Model Consultations section to accurately reflect models used
- Added explicit note about Gemini Deep Research unavailability
- Created `docs/EXTERNAL_MODEL_CALL_LOGS.md` with full call logs

**Lines Changed**: 955-994

---

### Issue B: Missing/Incomplete Equations

**Original Problem**: Secondary estimands showed only coefficient targets without full specifications.

**Fix Applied**:
- Added explicit LaTeX equations for asset and credit heterogeneity estimands
- Added Variable Definitions section with all terms defined
- Specified clustered standard errors at household level
- Referenced actual R implementation (`fixest::feols()`)

**Lines Changed**: 66-86

---

### Issue C: "Acceptance: p > 0.05" Framing

**Original Problem**: Using p > 0.05 as "acceptance" criterion implies proving equivalence.

**Fix Applied**:
- Replaced "Acceptance: p > 0.05" with "Interpretation: If p > 0.05, we fail to reject..."
- Added explicit caveat: "This does **not** prove distributions are identical"
- Added Critical Statistical Caveats section explaining non-equivalence, power limitations
- Mentioned alternative approaches (TOST procedures for equivalence testing)

**Lines Changed**: 784-807

---

### Issue D: CAS/Emergence Language Drift

**Original Problem**: CAS framing needed to be more conservative and consistent.

**Fix Applied**:
- Added "Important Caveat" paragraph upfront in CAS section
- Changed section subheadings to "Partial CAS Features" and "Missing CAS Features"
- Added explicit statement: "Results should not be interpreted as capturing emergent system dynamics"
- Expanded explanations of what each missing feature means

**Lines Changed**: 113-129

---

### Issue E: AdaptiveRulePolicy Naming

**Original Problem**: Concern that AdaptiveRulePolicy was mentioned but not implemented.

**Finding**: AdaptiveRulePolicy EXISTS in code at `src/abm_enterprise/policies/rule.py:117`

**Fix Applied**: No change needed; documentation correctly describes implemented class.

---

### Issue F: Over-Specific Empirical Claims

**Original Problem**: Claim "wave 2 for Tanzania" needed citation or softening.

**Fix Applied**:
- Changed "wave 2 for Tanzania" to "specific wave varies by country and price configuration; see validation reports"
- Removed unsupported specificity while preserving substantive claim

**Lines Changed**: 735

---

### Issue G: Sample Size Claims

**Original Problem**: QMD stated "500-1000 households per country" but manifests show exactly 500.

**Fix Applied**:
- Changed to "Sample size varies by run configuration; see `manifest.json` for exact N (default: 500 households per country in synthetic mode)"
- Provides accurate information while noting variability

**Lines Changed**: 245

---

### Issue H: Asset Index PCA Description

**Original Problem**: QMD described asset index as PCA, but implementation uses welfare_proxy directly.

**Fix Applied**:
- Corrected description to: "Uses `welfare_proxy` variable from LSMS data as simplified asset proxy"
- Added note: "A full implementation would compute the first principal component of household durables, livestock, and land holdings"
- Clarified quintile computation method

**Lines Changed**: 359-361

---

### Issue I: Makefile CLI Casing

**Original Problem**: Inconsistent COUNTRY= vs country= in commands.

**Finding**: Intentional design - uppercase for Make variables, lowercase for shell variables.

**Fix Applied**: Documented both conventions as correct; no change to Makefile examples needed.

---

### Issue J: Environment-Specific Claims

**Original Problem**: Version numbers stated as facts rather than variable.

**Fix Applied**:
- Added Appendix C: Build Environment section
- Added "Version varies by installation" notes
- Added environment variability disclaimer
- Moved specific version claims to appendix with appropriate caveats

**Lines Changed**: 1135-1153

---

### Issue K: Threshold Sensitivity Table

**Original Problem**: Table only showed Tanzania; discrete operationalization note missing.

**Fix Applied**:
- Added Ethiopia column to threshold table
- Added "Note on Discrete Operationalization" explaining why 50% and 67% yield same requirement for Tanzania
- Referenced `docs/THRESHOLD_JUSTIFICATION.md` for detailed rationale

**Lines Changed**: 809-821

---

## Additional Improvements

### Documentation Revision Notes Section

Added near front of document (lines 34-52) with table summarizing all fixes.

### Appendix B: External Model Call Logs

Added summary table of model consultations (lines 1114-1133) with reference to full logs.

### Generalizability Note

Added explicit note about cross-context transferability limitations (line 28).

### Causal Language Softening

Per external model recommendations:
- "mediate" → "potentially influence" (line 26)
- "induce" → "are associated with" (line 38)
- Added "explores this association rather than establishing definitive causation"

---

## Verification

All fixes verified against:
- Source code in `src/abm_enterprise/`
- R validation code in `analysis/R/validation_helpers.R`
- Manifest files in `data/processed/*/derived/`
- Documentation in `docs/`

---

## Files Modified

| File | Type |
|------|------|
| `ABM_Enterprise_Coping_Model.qmd` | Updated |
| `docs/EXTERNAL_MODEL_CALL_LOGS.md` | Created |
| `docs/CODE_REVIEW_DOC_AUDIT.md` | Created |
| `docs/DOC_CHANGELOG.md` | Created |
