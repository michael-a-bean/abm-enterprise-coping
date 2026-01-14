# Research Synthesis: ABM Manuscript Methodology
**Date:** 2026-01-14
**Sources:** OpenAI o1, Gemini (gemini-2.5-flash-thinking)

## Consensus Points (Strong Agreement)

Both sources strongly agree on:

### 1. Paper Structure
- **Adopt:** Follow ODD protocol for model description
- **Adopt:** Main text for narrative, appendices for detailed calibration/validation
- **Adopt:** Separate conceptual framework from technical implementation

### 2. Epistemic Boundaries
- **Adopt:** Be explicit that model lacks agent-agent interactions
- **Adopt:** Avoid CAS/emergence language when not warranted
- **Adopt:** Use "heterogeneous responses" and "aggregate outcomes" terminology
- **Adopt:** Frame results as "model outputs under specified assumptions"

### 3. POM Calibration Reporting
- **Adopt:** Report multiple goodness-of-fit measures (not just K-S p-values)
- **Adopt:** Use QQ plots, CDFs, and descriptive statistics
- **Adopt:** Explicitly acknowledge and contextualize K-S test failures
- **Adopt:** Report Gini coefficients and inequality measures for economic data

### 4. Sensitivity Analysis
- **Adopt:** Document parameter ranges and experimental design
- **Adopt:** Use response surfaces/heatmaps for visualization
- **Adopt:** Report regression coefficients for parameter effects
- **Adopt:** Include confidence intervals and uncertainty quantification

### 5. Behavior Search Framing
- **Adopt:** Use "behavior search" or "exploration" NOT "optimization"
- **Adopt:** Acknowledge limited seeds (2) as exploratory limitation
- **Adopt:** Report landscape of behaviors, not just "best" candidate
- **Adopt:** Recommend 30-100 seeds for robust validation (future work)

### 6. Design vs. Results Separation
- **Adopt:** LLM policy goes in Methods as "designed" or "planned"
- **Adopt:** Results section contains ONLY executed policy outputs
- **Adopt:** Discussion section elaborates on LLM potential
- **Adopt:** Appendix contains full technical specification

---

## Divergences and Resolutions

### D1: Emergence Language
- **o1:** Emphasizes avoiding all emergence language
- **Gemini:** Allows "macro-level patterns from micro-level rules" with caveats
- **Resolution:** Use Gemini's nuanced framing - acknowledge limited emergence while being precise about what IS captured (heterogeneity aggregation)

### D2: K-S Test Alternative Metrics
- **o1:** Suggests Anderson-Darling, AIC/BIC
- **Gemini:** Suggests Wasserstein distance, Cramer-von Mises
- **Resolution:** Report what's implemented (K-S, descriptive stats, QQ plots); mention alternatives as future improvement in limitations

### D3: Sensitivity Method Depth
- **o1:** Focus on standard regression surfaces
- **Gemini:** Recommends Global SA (Sobol) and ML approaches
- **Resolution:** Keep standard approach in main text; note Sobol as extension in future work (not currently implemented)

---

## What to Adopt in Manuscript

### Main Text Structure (following consensus):
1. Abstract
2. Introduction (problem + why ABM + contributions)
3. Related Work
4. Data (LSMS sources + provenance + limitations)
5. Model (ODD+D summary; architecture diagram)
6. Experimental Design (baseline/scenarios; seeds; N; sweep grid)
7. Results
   - 7A: Pattern-matching validation (POM framing)
   - 7B: Parameter sweeps (calibrated synthetic, exploratory)
8. Robustness & Diagnostics
9. Limitations and Future Work
10. Conclusion
11. References
12. Appendices

### Key Wording Decisions:
- Use "aggregate enterprise dynamics" not "emergent enterprise behavior"
- Use "heterogeneous household responses" not "adaptive agents"
- Use "behavior exploration" not "optimization"
- Use "design-only" for LLM policy throughout
- Use "exploratory" for 10-seed robustness analysis

### Figure/Table Requirements:
- All captions include data source paths
- Distinguish LSMS-derived vs. calibrated synthetic vs. uncalibrated
- No plot titles (captions carry narrative per minimalist style)
- Viridis scale for heatmaps (colorblind-safe)

---

## What to Reject

### R1: Extensive Global SA
- **Reason:** Not implemented in current codebase
- **Note:** Mentioned as future extension

### R2: ML-based Sensitivity Analysis
- **Reason:** Not implemented; over-engineering for current scope
- **Note:** Not mentioned in limitations

### R3: Out-of-sample Validation for Behavior Search
- **Reason:** Not feasible with 40 candidates, 2 seeds
- **Action:** Explicitly acknowledge as limitation

### R4: Strong Claims About Emergence
- **Reason:** Model lacks agent interactions, feedback loops
- **Action:** Use careful language throughout

---

## Action Items for Paper

1. Create docs/research/ directory with raw outputs
2. Write docs/paper.qmd following synthesized structure
3. Import validated figures from docs/abm_report.qmd
4. Add explicit data source annotations to all figures
5. Create REPRODUCIBILITY.md with commands
6. Add new references to references.bib
7. Render and QA

---

## New References to Add

From o1:
- Bankes, S. (1993). Exploratory modeling for policy analysis
- Clauset, A., et al. (2009). Power-law distributions in empirical data
- Epstein, J. (2008). Why model?
- Gilbert, N. (2008). Agent-Based Models
- Luke, S. (2013). Essentials of Metaheuristics

From Gemini:
- Pianosi, F., et al. (2016). Sensitivity analysis of environmental models
- Macal, C. M., & North, M. J. (2010). Tutorial on agent-based modeling

Already in references.bib:
- Grimm et al. (2006, 2010, 2020) - ODD protocol
- Dercon (2002) - Income risk
- Mesa (2024)
- LSMS-ISA
