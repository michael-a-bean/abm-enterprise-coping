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

### Issue A: Wrong External Model Used (CRITICAL)

**Location**: QMD lines 937-948, docs/EXTERNAL_REVIEW.md line 4

**Problem**: Documentation states "Gemini 2.0 Flash" was used for methodology review, but the user requirement specifies "Gemini Deep Research" must be used for Sections 1-3.

**Evidence**: EXTERNAL_REVIEW.md line 4: "Models Consulted: GPT-4o, Gemini 2.0 Flash"

**Fix Required**: Re-run external consultation with Gemini Deep Research and update documentation accordingly.

---

### Issue B: Missing/Incomplete Equations (HIGH)

**Location**: QMD sections 2.3, 8.2-8.3

**Problem**: While primary estimand equation exists (line 59), the FE specification lacks explicit variable definitions. The R code shows:
```r
formula <- enterprise_status ~ price_exposure | household_id + wave
```
With clustered SEs at household level (validation_helpers.R:182).

**Evidence**: QMD has equations but secondary estimands (lines 66-67) only show `β₂ < 0` without full specification.

**Fix Required**: Add explicit LaTeX for all FE specifications with variable definitions matching the R implementation.

---

### Issue C: "Acceptance: p > 0.05" Framing (CRITICAL)

**Location**: QMD lines 771-789, VALIDATION_CONTRACT.md lines 121-122

**Problem**: Using p > 0.05 as "acceptance" criterion for KS/chi-square tests is statistically indefensible. Failing to reject the null hypothesis does not prove distributions are identical - it only means insufficient evidence of difference was found.

**Evidence**:
- validation_helpers.R:452-453: `pass <- !is.na(ks_pval) && ks_pval > 0.05`
- VALIDATION_CONTRACT.md:121: "KS statistic | p > 0.05"
- QMD:771: "Acceptance: p > 0.05"

**Fix Required**: Replace "Acceptance" language with conservative framing like "Cannot reject similarity" or "Insufficient evidence of difference." Add explicit caveat that these tests cannot prove equivalence.

---

### Issue D: CAS/Emergence Language (LOW - Already Conservative)

**Location**: QMD lines 98-112, 266-269

**Assessment**: The QMD already has conservative, explicit CAS framing:
- Lines 106-111 explicitly list what is "Not Implemented" (emergence, agent learning, agent-agent interaction)
- Line 112 states "This is a 'simple ABM' rather than a full CAS model"

**Finding**: Current framing is appropriately conservative. No major changes needed, but ensure consistency throughout document.

---

### Issue E: AdaptiveRulePolicy Naming (NO ISSUE)

**Location**: QMD line 274, rule.py lines 117-188

**Assessment**: AdaptiveRulePolicy **does exist** in the codebase at `src/abm_enterprise/policies/rule.py:117`. The QMD correctly describes it at line 274.

**Finding**: No issue - naming matches implementation.

---

### Issue F: Over-Specific Empirical Claims (MEDIUM)

**Location**: QMD lines 717-719

**Problem**: The claim "wave 2 for Tanzania" as shock wave needs citation to report output.

**Evidence**: QMD:717-718: "Copers show higher enterprise entry rates in shock waves (wave 2 for Tanzania)"

**Fix Required**: Either cite specific figure/table from validation report or soften to "shock waves" without specifying which wave.

---

### Issue G: Sample Size Claims (MEDIUM)

**Location**: QMD line 229

**Problem**: QMD states "500-1000 households per country" but manifests show exactly 500 for both countries.

**Evidence**:
- data/processed/tanzania/derived/derive_manifest.json: `"num_households": 500`
- data/processed/ethiopia/derived/derive_manifest.json: `"num_households": 500`
- outputs/tanzania/baseline/manifest.json: `"num_households": 500`

**Fix Required**: State exact number (500) or clarify "varies by run; see manifest.json for exact N" if configurable.

---

### Issue H: Asset Index PCA Description (CRITICAL)

**Location**: QMD lines 343-344, derive.py lines 101-107

**Problem**: QMD states asset index is "First principal component of household durables, livestock, and land" but implementation uses welfare_proxy directly as proxy.

**Evidence**:
- QMD:343-344: "First principal component of household durables, livestock, and land. Standardized to mean=0, SD=1."
- derive.py:101-107: `# Use welfare_proxy as asset proxy (simplified PCA)` followed by `pl.col("welfare_proxy").alias("asset_index")`

**Fix Required**: Correct description to match implementation: "Uses welfare_proxy as simplified asset proxy. In production implementation, would compute first principal component of household durables, livestock, and land."

---

### Issue I: Makefile CLI Casing (LOW - Intentional Design)

**Location**: QMD line 658, Makefile lines 94-150

**Assessment**: The casing difference is intentional:
- `COUNTRY=` (uppercase) for Make variable defaults in run-sim targets
- `country=` (lowercase) for targets using shell variable substitution in ingest-data/derive-targets

**Evidence**: Makefile line 94: `COUNTRY ?= tanzania` vs line 150: `country ?= tanzania`

**Fix Required**: Document both conventions accurately in QMD. Current QMD examples at line 658 show lowercase for `render-report-country`, which is correct.

---

### Issue J: Environment-Specific Claims (LOW)

**Location**: QMD line 931

**Problem**: Version numbers for software components should be in appendix with note about variability.

**Evidence**: QMD:924-934 lists specific versions without noting they may vary.

**Fix Required**: Move to "Build Environment" appendix section or add caveat about version variability.

---

### Issue K: Threshold Sensitivity Table (MEDIUM - Verify Math)

**Location**: QMD lines 793-799, THRESHOLD_JUSTIFICATION.md lines 94-98

**Problem**: Need to verify mathematical consistency of threshold → waves mapping.

**Verification**:
- Tanzania (4 waves):
  - >33%: need >1.32 waves → 2+ waves ✓
  - >50%: need >2.0 waves → 3+ waves ✓
  - >67%: need >2.68 waves → 3+ waves ✓
- Ethiopia (3 waves):
  - >33%: need >0.99 waves → 1+ waves ✓
  - >50%: need >1.5 waves → 2+ waves ✓
  - >67%: need >2.01 waves → 3+ waves (all 3 waves)

**Issue Found**: QMD line 796-799 shows:
- 33%: ≥2 of 4 waves - should be "≥2 of 4" for >33%
- 50%: ≥3 of 4 waves - CORRECT
- 67%: ≥3 of 4 waves - same as 50% for 4 waves, technically >67% would be ≥3 but it's the same threshold operationally

**Fix Required**: Clarify that for Tanzania, both 50% and 67% thresholds operationally require ≥3 of 4 waves due to discrete wave counts.

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
- Quasi-global sensitivity analysis
- Parameter uncertainty quantification
- Agent learning / adaptation
- Agent-agent interaction / network effects

---

## Code Quality Observations

### Strengths
1. Well-structured policy pattern (BasePolicy → RulePolicy/LLMPolicy)
2. Comprehensive logging infrastructure
3. Centralized RNG for reproducibility
4. Clear data contracts with Pydantic schemas
5. Manifest tracking for provenance

### Areas for Improvement
1. Asset index should use actual PCA if claiming PCA methodology
2. Distribution test interpretation needs statistical rigor
3. Sample size documentation should reference manifests

---

## Status

**Phase 1**: COMPLETE
**Phase 2**: COMPLETE (Gemini Deep Research unavailable; used o1, Gemini 2.0 Flash, GPT-4o-mini)
**Phase 3**: COMPLETE (QMD regenerated with all fixes)
**Phase 4**: COMPLETE (QMD rendered successfully)

## Render Verification

```
Output created: ABM_Enterprise_Coping_Model.html
```

No rendering errors. All LaTeX equations rendered correctly.

## Remaining Known Limitations

1. **Gemini Deep Research unavailable**: API returned error during audit. Alternative models used but may not have equivalent deep research capabilities.

2. **Asset index is proxy, not PCA**: The documentation now correctly describes this, but a production implementation should compute actual PCA.

3. **Distribution tests cannot prove equivalence**: Documentation now correctly caveats this, but equivalence testing methodology (TOST) is not implemented.

4. **Robustness protocol not standardized**: Number of runs and comparison metrics are not formally specified.

## Recommended Next Steps

1. **Retry Gemini Deep Research** when API becomes available for deeper methodology review
2. **Implement actual PCA** for asset index if claiming PCA methodology
3. **Add equivalence testing** (TOST procedures) if equivalence claims are desired
4. **Standardize robustness protocol** with minimum run count and formal metrics
5. **Add model diagnostics** (multicollinearity, homoscedasticity) to R validation
