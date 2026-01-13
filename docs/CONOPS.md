# CONCEPT OF OPERATIONS (ConOps)
## ABM Enterprise Coping Model

---

## 1. Purpose

This system implements and validates an agent-based model (ABM) that simulates household enterprise entry as a coping mechanism in response to agricultural price shocks. The model is grounded in the empirical findings of "Booms, Busts, and Household Enterprise" and validated against LSMS-ISA harmonized panel data.

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
- Command: `make run-toy`

### 4.2 Full Simulation Mode
- Requires ingested LSMS data
- Runs complete ABM with specified country and scenario
- Produces outputs for validation
- Command: `make run-sim country=tanzania scenario=baseline`

### 4.3 LLM Policy Mode
- Uses LLM-augmented decision policy
- Requires API configuration (or uses stub)
- Logs all prompts/outputs for reproducibility
- Command: `make run-sim policy=llm`

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

### 7.2 LLMPolicy (Experimental)
- LLM generates action proposals given household state
- Constraint validator checks feasibility
- Commit stage records final decision
- Full logging for reproducibility

---

## 8. Validation Strategy

### 8.1 Estimands
1. **Enterprise Entry Rate**: Proportion entering enterprise after price bust
2. **Heterogeneity**: Differential response by asset quintile and credit access
3. **Persistence**: Classification into stayers (>50% waves) vs copers

### 8.2 Methods
- Household fixed effects regression: `enterprise ~ price_exposure + HH_FE + time_FE`
- Distribution comparisons: simulated vs observed enterprise rates by group
- Portability test: Model trained/validated on Tanzania, tested on Ethiopia

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
