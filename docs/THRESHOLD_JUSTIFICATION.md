# Threshold Justification: Stayer/Coper Classification

**Status:** Addresses conditional approval from external review (EXTERNAL_REVIEW.md, Section 2.2)
**Date:** 2026-01-13
**Version:** 1.0

---

## 1. Purpose

This document provides the scientific justification for the stayer/coper classification threshold used in the ABM Enterprise Coping Model. The classification distinguishes households by enterprise persistence:

- **Stayer**: Operates enterprise in >50% of observed waves (persistent entrepreneurs)
- **Coper**: Operates enterprise in >0% and <=50% of waves (intermittent, shock-responsive)
- **None**: Never operates enterprise (0% of waves)

---

## 2. Conceptual Framework

### 2.1 Enterprise Participation as Coping Behavior

The "Booms, Busts, and Household Enterprise" literature establishes that household enterprise participation responds heterogeneously to agricultural price shocks:

1. **Persistent entrepreneurs ("stayers")** maintain enterprises across economic cycles, suggesting structural commitment to non-farm income
2. **Intermittent participants ("copers")** enter enterprises temporarily in response to shocks, using non-farm income as a coping mechanism

This behavioral distinction is theoretically grounded in the portfolio diversification and risk management literature (Ellis 2000; Barrett et al. 2001).

### 2.2 Active Share of Periods

The classification metric—proportion of observed periods with enterprise operation—captures the **intensity** of enterprise commitment across the panel. This approach:

- Accounts for varying panel lengths across households
- Provides a continuous underlying measure (enterprise_persistence)
- Enables threshold-based classification while preserving information

Alternative approaches considered:
- **Duration-based**: Number of consecutive waves (loses intermittency information)
- **Entry/exit counts**: Number of transitions (conflates churning with persistence)
- **Data-driven clustering**: K-means on persistence (introduces additional assumptions)

The active-share approach was selected for interpretability and alignment with the empirical literature.

---

## 3. Threshold Selection: 50% Rationale

### 3.1 Majority-of-Periods Interpretation

The 50% threshold operationalizes a natural interpretation: **stayers operate enterprises in a majority of observed periods**. This creates:

- Clear semantic meaning ("more often than not")
- Symmetric classification around the midpoint
- Consistency with binary-outcome classification conventions

### 3.2 Panel Length Considerations

For the LSMS-ISA panels used:
- **Tanzania**: 4 waves (2008-2014) → 50% = 2+ waves
- **Ethiopia**: 3 waves (2011-2015) → 50% = 2+ waves

The 50% threshold maps to a minimum of 2 waves for both countries, providing consistent operationalization.

### 3.3 Literature Precedent

While no universal standard exists, the majority-of-periods approach aligns with:
- Labor economics (full-time vs. part-time classification)
- Firm survival literature (persistent vs. transient entrants)
- The original "Booms, Busts, and Household Enterprise" empirical specification

---

## 4. Threshold as Operationalization

**Important clarification:** The 50% threshold is an **operationalization choice**, not a fundamental parameter of the underlying economic model.

### 4.1 What the Threshold Is

- A discretization of continuous persistence measure
- A convenient binary for regression interaction terms
- A communication device for policy interpretation

### 4.2 What the Threshold Is Not

- A claim about true behavioral discontinuity at 50%
- A structural parameter to be estimated
- A target for policy optimization

### 4.3 Sensitivity Testing Requirement

Given the operationalization nature, the validation framework **requires sensitivity analysis** across alternative thresholds:

| Threshold | Interpretation | Tanzania (4 waves) | Ethiopia (3 waves) |
|-----------|----------------|-------------------|-------------------|
| 33% | "At least one-third" | 2+ waves | 1+ waves |
| 50% | "Majority" (baseline) | 2+ waves | 2+ waves |
| 67% | "Supermajority" | 3+ waves | 2+ waves |

Results must demonstrate that core conclusions (coefficient signs, heterogeneity patterns) are **robust** to threshold choice.

---

## 5. Validation Criteria

### 5.1 Primary Criterion: Sign Stability

The primary estimand β₁ (enterprise ~ price_exposure) must maintain:
- Negative sign across all thresholds
- Statistical significance (p < 0.05 or p < 0.10)

### 5.2 Secondary Criterion: Heterogeneity Pattern

The interaction terms (price_exposure × low_assets, price_exposure × no_credit) must:
- Preserve direction across thresholds
- Show copers responding more strongly than stayers

### 5.3 Acceptable Variation

Magnitude variation across thresholds is expected and acceptable. The validation focuses on **qualitative** stability of conclusions, not exact coefficient replication.

---

## 6. Implementation

### 6.1 Code Locations

The 50% threshold is implemented in:
- `src/etl/derive.py:57` - Classification derivation
- `src/abm_enterprise/agents/household.py:25` - Agent classification
- `analysis/R/validation_helpers.R:588` - R analysis classification

### 6.2 Configuration

The threshold is parameterized as `stayer_threshold` in:
- `config/tanzania.yaml` (default: 0.5)
- `config/ethiopia.yaml` (default: 0.5)
- Manifest parameters (recorded for reproducibility)

### 6.3 Sensitivity Analysis

Sensitivity analysis is implemented in:
- `analysis/quarto/validation_report.qmd` - Sensitivity section
- Output includes threshold in manifest for each run

---

## 7. Response to External Review

**Gemini Review Comment (EXTERNAL_REVIEW.md, Section 2.2):**
> "Using a >50% persistence threshold might oversimplify the classification of enterprises as stayers or copers, possibly misrepresenting nuanced behaviors."

**Response:**

1. **Acknowledged**: The threshold is indeed a simplification. This is inherent in any discretization of continuous behavior.

2. **Justified**: The 50% threshold has clear semantic meaning and aligns with precedent. It is not arbitrary but rather a principled operationalization.

3. **Sensitivity-Tested**: The validation framework requires demonstrating robustness across thresholds {0.33, 0.50, 0.67}. If conclusions depend critically on threshold choice, this will be documented as a limitation.

4. **Alternatives Considered**: Data-driven clustering was considered but rejected for interpretability. The continuous persistence measure is preserved in outputs for researchers preferring alternative specifications.

---

## 8. References

- Barrett, C. B., Reardon, T., & Webb, P. (2001). Nonfarm income diversification and household livelihood strategies in rural Africa. Food Policy, 26(4), 315-331.
- Ellis, F. (2000). Rural Livelihoods and Diversity in Developing Countries. Oxford University Press.
- Nagler, P., & Naudé, W. (2017). Non-farm entrepreneurship in rural sub-Saharan Africa. World Development, 96, 296-313.

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-13 | Initial documentation addressing external review |

---

**Document Status:** COMPLETE
**Addresses:** EXTERNAL_REVIEW.md conditional approval (threshold documentation)
