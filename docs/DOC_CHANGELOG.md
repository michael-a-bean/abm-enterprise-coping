# Documentation Changelog

**Date**: 2026-01-13
**Target File**: ABM_Enterprise_Coping_Model.qmd
**Branch**: docs/20260113-qmd-audit-regeneration

## Issues Addressed

This changelog maps each fix to the original issue identifiers (A-K).

---

### Issue A: External Model Consultation

**Original Problem**: Previous review used "Gemini 2.0 Flash" instead of required "Gemini Deep Research" for methodology framing.

**Fix Applied**:
- Successfully called **Gemini Deep Research** for Sections 1-3 methodology audit
- Successfully called **OpenAI o1** for Sections 8-10 statistical review
- Updated External Model Consultations section with correct models and comprehensive feedback
- Updated Appendix B summary table with correct model information
- Created comprehensive `docs/EXTERNAL_MODEL_CALL_LOGS.md` with full prompts and verbatim responses

**Lines Changed**: External Model Consultations section (lines 982-1024), Appendix B (lines 1124-1142)

---

### Issue B: Missing/Incomplete Equations

**Original Problem**: Secondary estimands showed only coefficient targets without full specifications.

**Status**: VERIFIED (resolved in prior revision)
- Full LaTeX equations present for asset and credit heterogeneity estimands
- Variable definitions section with all terms defined
- Explicit clustered SE specification at household level
- Reference to R implementation (fixest::feols())

**Lines Changed**: Preserved from prior revision (lines 88-106)

---

### Issue C: "Acceptance: p > 0.05" Framing

**Original Problem**: Using p > 0.05 as "acceptance" criterion implies proving equivalence.

**Fix Applied**:
- Added TOST (Two One-Sided Tests) as explicit alternative methodology
- Added multiple testing correction note (Benjamini-Hochberg, Bonferroni)
- Added power considerations note with N=500 sample size context
- Added measurement error caveat
- **External Review Confirmation**: OpenAI o1 confirmed interpretation is "appropriately conservative"

**Lines Changed**: Critical Statistical Caveats section (lines 827-834)

---

### Issue D: CAS/Emergence Language Drift

**Original Problem**: CAS framing needed to be more conservative and consistent.

**Fix Applied** (based on Gemini Deep Research recommendations):
- Changed "simple ABM" to **"agent-based microsimulation"** throughout document
- Added "cognitive realism" framing for threshold-based rules
- Added **"bounded rationality"** as explicit partial CAS feature
- Added "tipping point" dynamics explanation for non-linearity
- Introduction now says "investigates... coping dynamics" instead of "simulates... as coping mechanism"

**Lines Changed**: Introduction (line 24), CAS Framing section (lines 137-152)

---

### Issue E: AdaptiveRulePolicy Naming

**Original Problem**: Concern that AdaptiveRulePolicy was mentioned but not implemented.

**Finding**: AdaptiveRulePolicy **EXISTS** in code at `src/abm_enterprise/policies/rule.py:117`

**Fix Applied**: No change needed; documentation correctly describes implemented class.

---

### Issue F: Over-Specific Empirical Claims

**Original Problem**: Claim "wave 2 for Tanzania" needed citation or softening.

**Status**: VERIFIED (resolved in prior revision)
- Changed to "specific wave varies by country and price configuration; see validation reports"

**Lines Changed**: Preserved from prior revision (line 758)

---

### Issue G: Sample Size Claims

**Original Problem**: QMD stated "500-1000 households per country" but manifests show exactly 500.

**Status**: VERIFIED (resolved in prior revision)
- Changed to "see manifest.json for exact N (default: 500 households per country in synthetic mode)"

**Lines Changed**: Preserved from prior revision (line 268)

---

### Issue H: Asset Index PCA Description

**Original Problem**: QMD described asset index as PCA, but implementation uses welfare_proxy directly.

**Status**: VERIFIED (resolved in prior revision)
- Corrected to describe actual welfare_proxy implementation
- Added note about production implementation using actual PCA

**Lines Changed**: Preserved from prior revision (lines 379-381)

---

### Issue I: Makefile CLI Casing

**Original Problem**: Inconsistent COUNTRY= vs country= in commands.

**Finding**: Intentional design - uppercase for Make variables, lowercase for shell variables.

**Fix Applied**: Documented both conventions as correct; no change to Makefile examples needed.

---

### Issue J: Environment-Specific Claims

**Original Problem**: Version numbers stated as facts rather than variable.

**Status**: VERIFIED (resolved in prior revision)
- Appendix C includes version variability notes
- Environment variability disclaimer present

**Lines Changed**: Preserved from prior revision (lines 1144-1161)

---

### Issue K: Threshold Sensitivity Table

**Original Problem**: Table only showed Tanzania; discrete operationalization note missing.

**Status**: VERIFIED (resolved in prior revision)
- Ethiopia column added to threshold table
- Discrete operationalization note present
- Reference to THRESHOLD_JUSTIFICATION.md

**Lines Changed**: Preserved from prior revision (lines 836-848)

---

## Additional Improvements (From External Model Reviews)

### From Gemini Deep Research:

1. **Lucas Critique Argument** (NEW)
   - Added to "Why ABM" section as point 5
   - Structural invariance argument for threshold-based rules
   - Line 122

2. **Crowding-Out Caveat** (NEW)
   - Added to "What ABM Does NOT Provide" section
   - Explicit market saturation/crowding-out limitation
   - Lines 131-132

3. **Panel Attrition Bias** (NEW)
   - Added new "Data Limitations" paragraph after Generalizability Note
   - LSMS-ISA tracking and split-off limitations
   - Line 30

4. **Bounded Rationality Feature** (NEW)
   - Added as explicit partial CAS feature
   - Heuristic vs. optimization framing
   - Line 144

### From OpenAI o1:

1. **Multiple Testing Note** (NEW)
   - Added Benjamini-Hochberg/Bonferroni consideration
   - Line 832

2. **Power Considerations** (NEW)
   - Added power caveat for N=500 sample size
   - Line 833

3. **Measurement Error** (NEW)
   - Added endogeneity/measurement error caveat
   - Line 834

---

## Files Modified

| File | Type | Description |
|------|------|-------------|
| `ABM_Enterprise_Coping_Model.qmd` | Updated | All fixes applied |
| `docs/EXTERNAL_MODEL_CALL_LOGS.md` | Rewritten | New model calls with correct tools |
| `docs/CODE_REVIEW_DOC_AUDIT.md` | Rewritten | Complete audit status |
| `docs/DOC_CHANGELOG.md` | Rewritten | This file |

---

## External Model Consultations Summary

| Model | Sections | Key Contributions |
|-------|----------|-------------------|
| **Gemini Deep Research** | 1-3 | Agent-based microsimulation terminology, Lucas Critique, crowding-out caveat, attrition bias |
| **OpenAI o1** | 8-10 | Multiple testing notes, power considerations, confirmed p-value interpretation is conservative |

---

## Verification

All fixes verified against:
- Source code in `src/abm_enterprise/`
- R validation code in `analysis/R/validation_helpers.R`
- Manifest files in `data/processed/*/derived/`
- Documentation in `docs/`

**External Model Calls**: Logged in `docs/EXTERNAL_MODEL_CALL_LOGS.md`

**Render Status**: PASSED (`quarto render ABM_Enterprise_Coping_Model.qmd` → `ABM_Enterprise_Coping_Model.html`)
