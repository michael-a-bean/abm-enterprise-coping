# Code Review and Documentation Audit

**Date**: 2026-01-13
**Branch**: docs/20260113-qmd-audit-regeneration
**Auditor**: Gen (Claude Code via PAI)

## Audit Scope

Comprehensive review of:
1. Source code (src/abm_enterprise/**, src/etl/**)
2. Analysis code (analysis/**)
3. Documentation (docs/**)
4. Configuration and Makefile
5. QMD documentation alignment with implementation

## Severity Levels

- **CRITICAL**: Claims that are factually wrong or could mislead readers
- **HIGH**: Significant misalignments between docs and implementation
- **MEDIUM**: Minor inconsistencies or ambiguous language
- **LOW**: Style or formatting issues

---

## Findings

### Issue A: External Model Consultation (RESOLVED)

**Location**: QMD External Model Consultations section

**Original Problem**: Previous documentation used Gemini 2.0 Flash; requirement specified Gemini Deep Research.

**Fix Applied**:
- Successfully called **Gemini Deep Research** for Sections 1-3 methodology audit
- Successfully called **OpenAI o1** (reasoning tier) for Sections 8-10 statistical review
- Updated External Model Consultations section with correct models
- Created comprehensive `docs/EXTERNAL_MODEL_CALL_LOGS.md` with full prompts and responses

**Status**: RESOLVED

---

### Issue B: Missing/Incomplete Equations (RESOLVED)

**Location**: QMD sections 2.3, 8.2-8.3

**Original Problem**: Secondary estimands showed only coefficient targets without full specifications.

**Evidence**: QMD now has complete LaTeX equations (lines 89-94) with:
- Full asset heterogeneity equation with interaction term
- Full credit heterogeneity equation with interaction term
- Variable definitions section (lines 96-106)
- Explicit household + wave FE specification
- Note on household-level clustering

**Status**: RESOLVED (in prior revision; verified)

---

### Issue C: "Acceptance: p > 0.05" Framing (RESOLVED)

**Location**: QMD Distribution Tests section

**Original Problem**: Using p > 0.05 as "acceptance" criterion implies proving equivalence.

**Fix Applied**:
- Replaced "Acceptance" with "Interpretation: If p > 0.05, we fail to reject..."
- Added explicit non-equivalence caveat
- Added TOST (Two One-Sided Tests) as explicit alternative
- Added multiple testing correction note (Benjamini-Hochberg, Bonferroni)
- Added power considerations note with sample size context
- Added measurement error caveat

**External Review Confirmation**: OpenAI o1 confirmed current p-value interpretation is "appropriately conservative"

**Status**: RESOLVED

---

### Issue D: CAS/Emergence Language (RESOLVED)

**Location**: QMD CAS Framing section

**Original Problem**: CAS framing needed to be more conservative and consistent.

**Fix Applied** (based on Gemini Deep Research recommendations):
- Changed "simple ABM" to "agent-based microsimulation" throughout
- Added "cognitive realism" framing for threshold-based rules
- Added "bounded rationality" as explicit partial CAS feature
- Added "tipping point" dynamics explanation
- Retained explicit "Not Implemented" sections for emergence, learning, interaction

**External Review Confirmation**: Gemini Deep Research recommended "agent-based microsimulation" terminology as methodologically precise.

**Status**: RESOLVED

---

### Issue E: AdaptiveRulePolicy Naming (NO ISSUE)

**Location**: QMD line 314, rule.py line 117

**Assessment**: AdaptiveRulePolicy **EXISTS** in the codebase at `src/abm_enterprise/policies/rule.py:117`.

**Finding**: No issue - naming matches implementation.

**Status**: CONFIRMED - NO ACTION NEEDED

---

### Issue F: Over-Specific Empirical Claims (RESOLVED)

**Location**: QMD line 755

**Original Problem**: Claim "wave 2 for Tanzania" needed citation or softening.

**Evidence**: Current text (line 755): "Copers show higher enterprise entry rates in shock waves (specific wave varies by country and price configuration; see validation reports)"

**Status**: RESOLVED (in prior revision; verified)

---

### Issue G: Sample Size Claims (RESOLVED)

**Location**: QMD line 265

**Original Problem**: QMD stated "500-1000 households per country" but manifests show exactly 500.

**Evidence**: Current text (line 268): "Sample size varies by run configuration; see `manifest.json` for exact N (default: 500 households per country in synthetic mode)"

**Verified Manifests**:
- Tanzania: 500 households, 4 waves = 2,000 observations
- Ethiopia: 500 households, 3 waves = 1,500 observations

**Status**: RESOLVED (in prior revision; verified)

---

### Issue H: Asset Index PCA Description (RESOLVED)

**Location**: QMD lines 379-381

**Original Problem**: QMD described asset index as PCA, but implementation uses welfare_proxy directly.

**Evidence**: Current text correctly states: "Uses `welfare_proxy` variable from LSMS data as simplified asset proxy. In the current implementation, this is a direct proxy rather than a computed PCA."

**Code Verification**: `src/etl/derive.py:106` confirms `pl.col("welfare_proxy").alias("asset_index")`

**Status**: RESOLVED (in prior revision; verified)

---

### Issue I: Makefile CLI Casing (LOW - Intentional Design)

**Location**: Makefile lines 94-150

**Assessment**: The casing difference is intentional:
- `COUNTRY=` (uppercase) for Make variable defaults in run-sim targets
- `country=` (lowercase) for targets using shell variable substitution

**Documentation**: QMD correctly shows both conventions in appropriate contexts.

**Status**: DOCUMENTED - NO CHANGE NEEDED

---

### Issue J: Environment-Specific Claims (RESOLVED)

**Location**: QMD Appendix C

**Original Problem**: Version numbers stated as facts rather than variable.

**Evidence**: Appendix C now includes:
- Version variability notes ("Version varies by installation")
- Reproducibility guidance (check manifest.json)
- Environment variability disclaimer

**Status**: RESOLVED (in prior revision; verified)

---

### Issue K: Threshold Sensitivity Table (RESOLVED)

**Location**: QMD lines 833-847

**Original Problem**: Table only showed Tanzania; discrete operationalization note missing.

**Evidence**: Current table includes:
- Both Tanzania and Ethiopia columns
- Note on discrete operationalization (50% and 67% yield same requirement for Tanzania)
- Reference to THRESHOLD_JUSTIFICATION.md

**Status**: RESOLVED (in prior revision; verified)

---

## New Recommendations (From External Model Reviews)

### From Gemini Deep Research:

1. **Lucas Critique Argument**: Added to "Why ABM" section - structural invariance argument for threshold-based rules
2. **Crowding-Out Caveat**: Added explicit market saturation/crowding-out limitation
3. **Panel Attrition Bias**: Added LSMS-ISA tracking/split-off limitations
4. **Coping vs. Adaptation**: Document notes exploratory framing; could distinguish distress-push vs opportunity-pull entry in future

### From OpenAI o1:

1. **Multiple Testing**: Added Benjamini-Hochberg/Bonferroni note (not implemented in code)
2. **Power Analysis**: Added power considerations caveat for N=500
3. **Equivalence Testing**: Added TOST as alternative methodology

---

## Analysis Features Status

### Implemented
- Fixed effects regression with household + wave FE
- Clustered standard errors at household level
- Threshold sensitivity analysis (33%, 50%, 67%)
- KS and chi-squared distribution tests
- Stayer/coper classification comparison

### NOT Implemented (Document Clearly)
- Heat maps / response surfaces
- Phase portraits / equilibrium analysis
- Behavior search / genetic algorithm optimization
- Quasi-global sensitivity analysis (only threshold sensitivity)
- Parameter uncertainty quantification
- Agent learning / adaptation
- Agent-agent interaction / network effects
- Multiple testing correction (noted as limitation)
- Equivalence testing (TOST)

---

## Code Quality Observations

### Strengths
1. Well-structured policy pattern (BasePolicy → RulePolicy/LLMPolicy)
2. Comprehensive logging infrastructure
3. Centralized RNG for reproducibility
4. Clear data contracts with Pydantic schemas
5. Manifest tracking for provenance
6. AdaptiveRulePolicy EXISTS at rule.py:117 (includes wave-based threshold adaptation)

### Areas for Improvement
1. Asset index should use actual PCA if claiming PCA methodology (documented as limitation)
2. Multiple testing correction should be implemented in R validation
3. Sample size documentation should reference manifests (now fixed)

---

## Final Status

| Phase | Status |
|-------|--------|
| Phase 0: Branch/files setup | COMPLETE |
| Phase 1: Repository audit | COMPLETE |
| Phase 2: External model calls | COMPLETE (Gemini Deep Research + o1) |
| Phase 3: QMD regeneration | COMPLETE |
| Phase 4: Validation | PENDING |
| Phase 5: Final output | PENDING |

---

## Render Verification

**PASSED** - `quarto render ABM_Enterprise_Coping_Model.qmd` completed successfully.

```
Output created: ABM_Enterprise_Coping_Model.html
```

No rendering errors. All LaTeX equations rendered correctly.

---

## Remaining Known Limitations

1. **Multiple testing not formally corrected**: Documentation now notes this as limitation; code does not implement Bonferroni/FDR correction
2. **Power analysis not formally conducted**: Documentation notes power considerations; no formal power calculation implemented
3. **Asset index is proxy, not PCA**: Documentation correctly describes this; production implementation would compute actual PCA
4. **Equivalence testing not implemented**: TOST procedures mentioned as alternative but not implemented
5. **Crowding-out effects excluded**: Explicitly documented as limitation of exogenous returns assumption

---

## Recommended Next Steps

1. **Consider implementing multiple testing correction** in R validation code
2. **Consider formal power analysis** for distribution tests
3. **Consider TOST procedures** if equivalence claims become important
4. **Consider actual PCA** for asset index in production implementation
5. **Document crowding-out limitation** prominently in any policy recommendations
