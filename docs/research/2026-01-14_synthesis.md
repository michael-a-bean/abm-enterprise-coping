# Research Synthesis: ABM Validation Methodology
**Date:** 2026-01-14
**Sources:** OpenAI o3 deep research, Google Gemini deep research

## Summary of Consensus Recommendations

### 1. ODD+D Documentation

**Consensus:**
- Document LLM prompt templates as model parameters in the "Details" section
- Include temperature settings and constraint configurations
- Provide pseudocode that maps directly to source code (with line numbers)
- Specify context window contents as agent state variables

**Adopted for this project:**
- Already implemented: Detailed code references with file paths and line numbers
- Already implemented: Prompt templates in `policies/prompts.py`
- Already implemented: Constraint documentation in ODD+D section

### 2. Empirical Validation with LSMS-ISA

**Consensus:**
- Use Pattern-Oriented Modeling (POM) approach
- Compare transition matrices using Frobenius norm or KL divergence
- Stratify validation by asset quintile and credit access
- Run "history-friendly" validation with real household initialization

**Adopted for this project:**
- Transition matrix analysis in `validation_helpers.R::run_fe_regression()`
- Stratified validation by threshold sensitivity (`run_threshold_sensitivity()`)
- Already comparing simulated vs observed enterprise rates

### 3. LLM-Mediated Decisions

**Consensus: "Intrinsicality, Instruction, Imitation" Framework**
1. Test zero-shot behavior
2. Provide risk-framed instructions
3. Use few-shot examples from real data

**Consensus on Error Mitigation:**
- Enforce budget constraints programmatically (not just in prompts)
- Monitor decision entropy for mode collapse
- Use ensemble voting (K samples) for robustness

**Adopted for this project:**
- Already implemented: K=5 voting in `MultiSampleLLMPolicy`
- Already implemented: Constraint validation in `policies/constraints.py`
- Already implemented: Decision caching for reproducibility

### 4. CAS Diagnostics

**Consensus:**
- Use Recurrence Quantification Analysis (RQA) for regime shift detection
- Generate hysteresis loops (price down -> up recovery curves)
- Plot phase portraits: state variable vs lagged value or two variables

**Adopted for this project:**
- Phase portraits implemented in `R/analysis_helpers.R::create_phase_portrait()`
- Multi-seed phase portraits show trajectory variability
- **GAP:** RQA not implemented (add to extensions)
- **GAP:** Hysteresis loops not implemented

### 5. Sensitivity Analysis

**Consensus:**
- Morris Method for screening (computationally efficient)
- Sobol indices for critical parameters (S1 = direct, ST = with interactions)
- FANOVA graphs for visualizing parameter interactions

**Adopted for this project:**
- Parameter sweep script created: `scripts/run_sweep.py`
- 6x6 grid with 2 seeds = 72 runs completed
- Heatmap visualization using `create_sensitivity_heatmap()`
- **GAP:** Sobol indices not computed (consider `sensobol` R package)
- **GAP:** FANOVA graphs not implemented

---

## Divergences Between Models

### Minor Differences

1. **RQA vs simpler methods:**
   - Gemini emphasized RQA strongly
   - OpenAI mentioned it but also suggested simpler transition matrix analysis
   - **Resolution:** Use transition matrices now (implemented), add RQA as extension

2. **Validation metrics:**
   - Gemini: Frobenius norm + KL divergence
   - OpenAI: Pattern-oriented modeling with multiple targets
   - **Resolution:** Both valid; our FE regression approach is consistent with both

3. **LLM calibration:**
   - Gemini: Temperature tuning to match decision variance
   - OpenAI: Few-shot learning from real cases
   - **Resolution:** Adopt both - configurable temperature (implemented) and option for few-shot (extension)

---

## Action Items from Research

### Immediate (This Session)

1. [x] Replace synthetic heatmap with real sweep data
2. [ ] Update report to load `outputs/sweeps/sweep_agg_latest.parquet`
3. [ ] Add behavior search section with real results
4. [ ] Update "Known Limitations" to reflect new capabilities

### Near-term Extensions

1. Implement RQA using `ruptures` (Python) or `nonlinearTseries` (R)
2. Add Sobol indices using `sensobol` R package
3. Add hysteresis loop visualization
4. Implement few-shot LLM calibration option

### Documentation Updates

1. Add AI-sourced guidance appendix (already exists, verify completeness)
2. Update Evidence Map with new sweep outputs
3. Document sweep parameters and seeds in reproducibility section

---

## Key Quotes from Research

### On ODD+D for LLMs (Gemini)
> "Standard ODD+D protocols must be expanded to document the 'Cognitive Engine' of LLM agents, specifically treating prompts as model parameters and context windows as state variables."

### On Validation (OpenAI)
> "To quantify the total error between the simulated matrix and the empirical matrix, calculate the Frobenius norm of the difference matrix."

### On CAS Diagnostics (Gemini)
> "Recurrence Quantification Analysis (RQA) and hysteresis loop visualization are the most robust methods for detecting regime shifts and path dependence in discrete-time economic simulations."

### On Sensitivity (OpenAI)
> "Morris Method is the most practical screening technique for computationally expensive models... Sobol indices should be used for the critical parameters identified by Morris."
