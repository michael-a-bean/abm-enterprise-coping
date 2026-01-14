# Deep Research Output: Gemini
**Date:** 2026-01-14
**Query:** ABM manuscript methodology guidance
**Source:** Gemini (gemini-2.5-flash-thinking) via Crowdsource skill

## Summary

Comprehensive methodological guidance for structuring an ABM manuscript, with emphasis on current best practices, emerging trends, and actionable recommendations.

---

## (A) Paper Structure: Main Text vs Appendix

### Main Text Structure:
1. **Introduction:** Research question, motivation, why ABM is suitable, main contributions
2. **Literature Review:** Position work within household coping strategies, ABM in development economics
3. **Model Overview (High-Level):**
   - Conceptual Framework
   - Agents, Environment, Interactions, Decision Rules
   - Key Outcomes
4. **Data and Calibration:**
   - Data Sources (LSMS-ISA)
   - Calibration Strategy (POM approach)
   - Key Calibration Results
5. **Experimental Design & Counterfactuals:**
   - Baseline Scenario
   - Shock Implementation
   - Counterfactual Scenarios
6. **Results:**
   - Validation of Baseline
   - Main Counterfactual Findings
   - Sensitivity Analysis (Key Findings)
7. **Discussion:** Interpret results, policy implications, limitations
8. **Conclusion**

### Appendix:
- Detailed ODD Protocol description
- Full equations and algorithms
- Complete parameter list
- Detailed calibration procedures (all distribution fits)
- Full sensitivity analysis
- LLM-augmented policy design details (prompts, architecture, ethics)
- Extended figures/tables

---

## (B) Epistemic Boundaries and Emergence

### Key Recommendations:

1. **Define "Emergence" Precisely:**
   - Acknowledge agents don't directly interact
   - Focus on "macro-level patterns arising from micro-level rules and heterogeneity"
   - Example wording: "While our model's agents do not engage in direct strategic interaction, the observed macro-level dynamics emerge from the aggregation of heterogeneous household coping strategies in response to exogenous price shocks."

2. **Focus on Heterogeneity:**
   - Emphasize diversity of agent characteristics (calibrated from LSMS-ISA)
   - Highlight how model explores distribution of outcomes, not just averages

3. **Transparency About Assumptions:**
   - State shocks are exogenous explicitly
   - State agents don't interact directly if that's the case

4. **Avoid Buzzwords:**
   - Refrain from "self-organization," "co-evolution," "adaptive learning" unless explicitly modeled
   - Use: "aggregate behavior," "system-level responses," "heterogeneous outcomes"

---

## (C) POM Calibration with Failed K-S Tests

### Recommendations:

1. **Emphasize POM Philosophy:**
   - Aim to reproduce multiple, diverse patterns simultaneously
   - List specific patterns targeted and why they're important

2. **Report Beyond P-values:**
   - **Visual Inspection:** Histograms, CDFs, QQ plots
   - **Descriptive Statistics:** Mean, median, SD, IQR, skewness, kurtosis, specific quantiles
   - **Inequality Measures:** Gini coefficient, Theil index (for wealth/income)

3. **Address Failed K-S Tests Explicitly:**
   - Acknowledge: "While the K-S test indicated statistically significant differences (p < 0.05)..."
   - Contextualize K-S limitations:
     - Sensitivity to large N
     - Heavy tails are notoriously difficult to fit
     - Focus on key features relevant to research question
   - Consider alternative metrics: Wasserstein distance, Cramer-von Mises

4. **Emphasize Multi-Pattern Fit:**
   - No single pattern is sole arbiter of calibration success
   - Robustness comes from fitting multiple, diverse patterns

---

## (D) Sensitivity Analysis and Parameter Sweeps

### Recommendations:

1. **State Purpose and Method Clearly:**
   - What parameters varied, how varied, how many runs per parameter set
   - What outputs measured

2. **Reporting:**
   - **Key Findings in Main Text**
   - **Visualizations:** Response surfaces/heatmaps, tornado plots, scatter plots
   - **Statistical Analysis:** Fit regression models, report R², significant coefficients, interactions
   - **Uncertainty Quantification:** Confidence intervals, acknowledge stochasticity

3. **Advanced Methods (Emerging Trends):**
   - Global Sensitivity Analysis (Sobol indices, Morris method)
   - ML approaches (Random Forests, Gradient Boosting) for complex interactions

---

## (E) Behavior Search: Avoiding Overfitting

### Recommendations:

1. **Frame as "Behavior Search" or "Exploration," NOT "Optimization":**
   - Use language like "identifying candidate strategies"
   - Example: "We conducted a behavior search over a defined parameter space..."

2. **Define Objective Function Clearly:**
   - What patterns/outcomes trying to achieve?
   - How was "performance" measured?

3. **Address Limited Seeds (2):**
   - State limitation directly
   - Explain implications for robustness
   - Indicate promising candidates would require more extensive runs (30-100 seeds)

4. **Report Landscape of Behaviors:**
   - Don't just present the "best"
   - Show range of behaviors across candidates
   - Pareto fronts for multi-objective

5. **Avoid Overfitting Claims:**
   - Cannot claim "optimal" without out-of-sample validation
   - Emphasize identifying interesting regions of parameter space

---

## (F) Separating Design from Empirical Results

### Main Text (Methods):
- Introduce LLM policy as "designed" or "planned" alternative
- Focus on architecture and rationale
- Use conditional/future tense: "We designed..." "This paper proposes..."
- Keep description concise

### Results Section:
- **Strictly empirical results from executed policies only**
- No speculative results for LLM policy
- Clear subsection titles

### Discussion Section:
- Elaborate on LLM policy potential and challenges
- Frame as next step in research agenda
- Example: "While the current study employs rule-based policies, a key methodological innovation designed for future iterations involves augmenting household decision-making with Large Language Models..."

### Appendix:
- Detailed technical specification
- Prompt engineering examples
- Expected LLM output structures
- Architectural diagrams
- Ethical considerations

---

## References

- Grimm, V., et al. (2010). The ODD protocol: A review and first update. Ecological Modelling, 221(23), 2760-2768.
- Grimm, V., & Railsback, S. F. (2005). Population Ecology for Researchers.
- Grimm, V., et al. (2017). The ODD protocol for describing agent-based models: Updates and practical guidelines. JASSS, 20(4).
- Saltelli, A., et al. (2008). Global Sensitivity Analysis: The Primer.
- Pianosi, F., et al. (2016). Sensitivity analysis of environmental models. Environmental Modelling & Software, 79: 214-232.
- Macal, C. M., & North, M. J. (2010). Tutorial on agent-based modeling and simulation.
- Railsback, S. F., & Grimm, V. (2019). Agent-Based and Individual-Based Modeling: A Practical Guide.
