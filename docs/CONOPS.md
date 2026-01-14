# CONCEPT OF OPERATIONS (ConOps)
## ABM Enterprise Coping Model

---

## 1. Purpose

This system implements and validates an agent-based model (ABM) that simulates household enterprise entry as a coping mechanism in response to agricultural price shocks. The model is grounded in the empirical findings of "Booms, Busts, and Household Enterprise" and validated against LSMS-ISA harmonized panel data.

**Research Question:** "Do LLM-based household coping decisions (given survey-like state inputs) replicate observed enterprise entry/exit patterns across contexts (countries), and how do they compare to standard statistical/ML baselines?"

---

## 2. Scope

### In Scope
- Household-level decision modeling for enterprise entry/exit
- Price shock exposure based on crop portfolios
- Heterogeneity by asset holdings and credit access
- Classification of households as "stayers" (persistent entrepreneurs) vs "copers" (intermittent responders)
- Validation against Tanzania and Ethiopia LSMS-ISA panels
- Rule-based and LLM-augmented decision policies

### Out of Scope
- General equilibrium effects (prices exogenous)
- Intra-household dynamics
- Spatial spillovers between households
- Migration decisions
- Detailed enterprise type modeling

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                             │
│                   (Control + Coordination)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│  Agent-   │   │  Agent-   │   │  Agent-   │
│  Scaffold │   │    ETL    │   │  ABM-Core │
└───────────┘   └───────────┘   └───────────┘
                      │                 │
                      ▼                 ▼
              ┌───────────────────────────┐
              │      Parquet Data Lake    │
              │  (Canonical + Simulation) │
              └───────────────────────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │   R/Quarto Validation     │
              │   (Agent-Validation-R)    │
              └───────────────────────────┘
```

---

## 4. Operational Modes

### 4.1 Toy Mode
- Runs without real LSMS data
- Uses synthetic household/plot data matching expected schemas
- Purpose: Development, testing, CI validation
- Command: `abm run-toy`

### 4.2 Calibration Mode
- Fits distributional parameters from LSMS data
- Creates calibration artifacts for synthetic generation
- Command: `abm calibrate --country tanzania --data-dir data/processed`

### 4.3 Synthetic Simulation Mode
- Generates synthetic panel from calibration artifacts
- Runs ABM with LLM-driven decisions (o4-mini)
- Multi-sample voting (K samples at temperature T)
- Command: `abm run-sim-synthetic calibration.json --policy llm_o4mini`

### 4.4 Direct Prediction Evaluation Mode
- Tests LLM on real LSMS states predicting observed transitions
- Compares to ML baselines (logistic, RF, GBM)
- Supports cross-country generalization (train Tanzania, test Ethiopia)
- Command: `abm eval-direct --train-country tanzania --test-country ethiopia`

### 4.5 Full Simulation Mode
- Requires ingested LSMS data
- Runs complete ABM with specified country and scenario
- Produces outputs for validation
- Command: `abm run-sim tanzania --scenario baseline`

### 4.6 LLM Policy Mode
- Uses LLM-augmented decision policy with multi-sample voting
- Caching for reproducibility and cost efficiency
- Logs all prompts/outputs for replay
- Command: `abm run-sim tanzania --policy llm_openai`

---

## 5. Data Flow

```
LSMS-ISA Release (v2.0)
         │
         ▼
    ┌─────────┐
    │  ETL    │ ← Country config (tanzania.yaml)
    └────┬────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Canonical Parquet Panels        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │plot_crop │ │   plot   │ │household ││
│  └──────────┘ └──────────┘ └──────────┘│
│  ┌──────────┐                          │
│  │individual│                          │
│  └──────────┘                          │
└─────────────────────────────────────────┘
         │
         ▼
    ┌──────────┐
    │  Derive  │ ← measurement_mapping.csv
    │  Targets │
    └────┬─────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Derived Target Tables           │
│  • enterprise_indicator + persistence   │
│  • asset_proxy                          │
│  • crop_summaries                       │
│  • welfare_proxy                        │
└─────────────────────────────────────────┘
         │
         ▼
    ┌──────────┐
    │   ABM    │ ← Country config + Price series
    │  (Mesa)  │
    └────┬─────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Simulation Outputs              │
│  • household_outcomes.parquet           │
│  • manifest.json (provenance)           │
│  • decision_logs/ (if LLM policy)       │
└─────────────────────────────────────────┘
         │
         ▼
    ┌──────────┐
    │ R/Quarto │
    │Validation│
    └────┬─────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Validation Reports              │
│  • Distribution comparisons             │
│  • FE regression results                │
│  • Portability analysis                 │
└─────────────────────────────────────────┘
```

---

## 6. Agent Descriptions

### 6.1 Household Agent
- **State**: household_id, wave, assets, credit_access, enterprise_status, crop_portfolio
- **Behavior**: Observes price exposure, evaluates coping options, decides enterprise entry/exit
- **Classification**: Labeled as "stayer" or "coper" based on enterprise persistence

### 6.2 Plot Agent (Passive)
- **State**: plot_id, household_id, crops, area
- **Purpose**: Provides production context for price exposure calculation

---

## 7. Decision Policies

### 7.1 RulePolicy (Baseline)
- Deterministic rules based on empirical findings
- Enterprise entry triggered by negative price shock + low assets
- Feasibility constraints: minimum assets, labor availability

### 7.2 CalibratedRulePolicy
- Data-driven thresholds from LSMS distributions
- Adapts to country-specific patterns
- Command: `abm run-sim tanzania --calibrate`

### 7.3 MultiSampleLLMPolicy (Primary Experimental)
- LLM generates K action proposals at temperature T
- Each proposal validated against feasibility constraints
- Majority vote aggregation with configurable tie-break
- Decision caching by state hash for reproducibility and cost efficiency

**LLM Decision Pipeline:**
```
State → Prompt → K samples at T → Constraint validation → Majority vote → Cache → Action
```

**Configuration:**
- Model: o4-mini (OpenAI)
- Temperature: 0.6 (default)
- K samples: 5 (default)
- Tie-break: Conservative (prefer NO_CHANGE)
- Caching: Enabled (LRU with state+config hash)

---

## 8. Validation Strategy

### 8.1 Two Evaluation Tracks

**Track A: Direct Prediction**
- Feed real LSMS household states to LLM
- Predict observed transitions (ENTER/EXIT/STAY)
- Compare to ML baselines (logistic, RF, GBM)
- Cross-country generalization: train Tanzania → test Ethiopia

**Track B: Full Simulation**
- Generate synthetic panels from calibration artifacts
- Run ABM with LLM decisions
- Compare aggregate patterns to LSMS stylized facts
- Focus on matching enterprise prevalence, entry/exit rates

### 8.2 Estimands
1. **Enterprise Entry Rate**: Proportion entering enterprise after price bust
2. **Heterogeneity**: Differential response by asset quintile and credit access
3. **Persistence**: Classification into stayers (>50% waves) vs copers

### 8.3 Direct Prediction Metrics
- Accuracy, balanced accuracy, macro F1
- Per-class precision/recall (ENTER, EXIT, STAY)
- Confusion matrices
- Subgroup analysis by assets and credit access

### 8.4 Simulation Comparison Metrics
- Enterprise prevalence by wave (simulated vs observed)
- Entry/exit transition rates
- Asset-stratified enterprise rates
- FE regression coefficient comparison

---

## 9. Reproducibility Requirements

1. **Seeding**: All random operations use centralized RNG with recorded seed
2. **Manifests**: Every output bundle includes manifest.json with:
   - Git commit hash
   - All seeds used
   - Parameter values
   - Country/scenario identifiers
   - Timestamps
3. **LLM Logging**: All prompts and outputs recorded with hashes
4. **Environment**: Pinned Python (pyproject.toml) and R (renv.lock) dependencies

---

## 10. Extension Pathway

### Adding a New Country
1. Create `config/{country}.yaml` with wave mappings, variable mappings
2. Run `make ingest-data country={country}`
3. Run `make derive-targets country={country}`
4. Run `make run-sim country={country}`
5. Run `make render-report country={country}`

No code changes should be required for country extension.
