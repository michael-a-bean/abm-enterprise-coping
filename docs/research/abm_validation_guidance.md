# ABM Validation Methodology Guidance

**Date:** 2026-01-14
**Sources:** OpenAI o1 (reasoning), Gemini 2.5 Flash Thinking (reasoning)
**Epistemic Status:** AI-sourced scaffolding, requires verification against peer-reviewed literature

## Summary of Key Questions and Guidance

### Q1: Minimum Replications for ABM Robustness

**Consensus from both models:**

- **10 seeds is insufficient** for robust CAS emergence claims
- **Minimum recommended:** 30-50 replications per parameter setting
- **Preferred:** 100-500+ replications for statistically robust claims
- **Best practice:** 1,000+ replications when detecting subtle patterns

**Rationale:** High variance in ABM results makes it difficult to distinguish true emergent properties from noise with few replications.

**Action:** Current batch runs (10 seeds) should be labeled as "preliminary" and not used for strong emergence claims.

### Q2: Presenting Synthetic Uncalibrated Data Results

**Consensus from both models:**

Results from uncalibrated synthetic data SHOULD be presented as **"exploratory analysis"** or **"scenario analysis"**, NOT as validated findings.

**Key distinctions:**
- **Validated/Empirical results:** Derived from LSMS-calibrated model
- **Exploratory results:** Derived from synthetic data, test model capabilities

**Hidden cost of not labeling properly:** Basing strong conclusions on uncalibrated data risks generating misinformation and guiding research down unproductive paths.

### Q3: Separating Validated vs Exploratory Sections

**Structural recommendations:**

1. **Dedicated sections with clear titles:**
   - "Model Calibration & Validation" (LSMS-derived data)
   - "Exploratory Scenarios & Sensitivity Analysis" (synthetic data)
   - "Future Work & Policy Design Framework" (LLM policy design)

2. **Clear headings and explicit language:**
   - "Based on our empirically calibrated model..."
   - "Our exploratory analysis suggests that under hypothetical conditions..."
   - "This preliminary investigation indicates..."
   - "Further empirical validation is required for..."

3. **Visual cues:** Different color schemes or plot styles to signal data nature

4. **Summary table:** Map each finding to data source, calibration status, and intended interpretation

### Q4: Required Disclaimers for Uncalibrated Synthetic Results

**Recommended disclaimers:**

> "These results are derived from a synthetic parameter space and have not been empirically calibrated for real-world phenomena."

> "The findings presented here are exploratory and represent hypothetical model behaviors rather than validated empirical predictions."

> "Caution should be exercised in extrapolating these results to real-world policy decisions without further empirical validation."

> "This analysis identifies potential sensitivities under idealized conditions, serving to guide future research and data collection efforts."

### Q5: Mixing LSMS-derived and Synthetic Data

**Consensus:** Mixing is acceptable IF:

1. Clear demarcation between data types in all sections
2. Each result clearly states its data source
3. Interpretation guidance prevents conflation
4. Hierarchical validation approach: use LSMS for core calibration, synthetic for scenario exploration

## Long-Term Implications and Hidden Costs

| Issue | Hidden Cost if Not Addressed |
|-------|------------------------------|
| Insufficient replications | Weak claims requiring rework, credibility risk |
| Uncalibrated synthetic data | Misinformation, misguided research direction |
| Mixed data without separation | Reader confusion, undermined scientific rigor |
| Design without empirical results | Reduced impact, eroded trust |

## Recommendations for This ABM Report

1. **Increase batch seeds** to 30+ minimum for any robustness claims, or explicitly label current 10-seed results as "preliminary"

2. **Add explicit section boundaries** with dedicated "Exploratory Analysis" sections for synthetic data results

3. **Update all figure captions** with data source annotations (DONE in Phase 2)

4. **Add callout boxes** for exploratory sections (DONE in Phase 2)

5. **Create data provenance appendix** referencing DATA_AUDIT.md and DATA_CONTRACT.md

6. **Mark LLM sections** as design/architecture documentation, not empirical results (DONE in Phase 2)

## Verification Status

| Recommendation | Peer-reviewed Source | Status |
|----------------|---------------------|--------|
| 30-50+ replications | Saltelli et al. (2008), Lee et al. (2015) | Verify |
| Clear section separation | Grimm et al. (2010) ODD+D | Verified |
| Exploratory vs validated labeling | Standard scientific practice | Verified |
| Disclaimers for synthetic data | Standard scientific practice | Verified |

## References to Verify

- Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer
- Lee, J.-S. et al. (2015). Annals of the Association of American Geographers - ABM replication standards
- Grimm, V. et al. (2010). The ODD protocol: A review and first update
- Thiele, J. C. et al. (2014). Facilitating parameter estimation and sensitivity analysis of agent-based models

---

*This guidance was AI-sourced and should be independently verified against peer-reviewed methodology literature before publication.*
