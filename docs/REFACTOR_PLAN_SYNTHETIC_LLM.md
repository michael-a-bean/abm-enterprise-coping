# Refactoring Plan: Synthetic ABM with LLM Decision-Making

**Branch:** `refactor/synthetic-llm-policy`
**Date:** 2026-01-13
**Status:** In Progress

---

## Executive Summary

This document outlines the plan to refactor the ABM Enterprise Coping project to:

1. Transform the ABM from using real LSMS data as agent states to a **generative microsimulation** with synthetic states drawn from calibrated distributions
2. Use real LSMS-ISA data only for **(a) calibration** and **(b) validation/evaluation**
3. Replace deterministic rule-based decisions with **LLM-driven decisions** (OpenAI o4-mini) with controlled stochasticity
4. Enable **cross-country generalization testing**: calibrate on Tanzania, validate on Ethiopia
5. Implement **two evaluation modes**:
   - **Direct prediction**: Test LLM on real LSMS states predicting observed transitions
   - **Full simulation**: Run synthetic ABM with LLM decisions, compare aggregate patterns to LSMS

---

## Research Question

> "Do LLM-based household coping decisions (given survey-like state inputs) replicate observed enterprise entry/exit patterns across contexts (countries), and how do they compare to standard statistical/ML baselines?"

---

## Current State Inventory

### Files to Modify

| File | Current Role | Changes Needed |
|------|-------------|----------------|
| `src/abm_enterprise/model.py` | Loads real data as agent states | Refactor to use synthetic panel from calibration artifact |
| `src/abm_enterprise/data/synthetic.py` | Simple synthetic generator | Major rewrite: distribution-based generation with transition dynamics |
| `src/abm_enterprise/policies/llm.py` | Single-sample LLM decisions | Add multi-sample voting, caching, temperature config |
| `src/abm_enterprise/policies/providers.py` | OpenAI provider exists | Add O4MiniProvider with proper config, caching |
| `src/abm_enterprise/cli.py` | CLI commands | Add calibrate, run-sim-synthetic, eval-direct commands |
| `src/abm_enterprise/data/schemas.py` | Pydantic schemas | Add CalibrationArtifact, TransitionLabel schemas |
| `config/tanzania.yaml` | Country config | Add calibration parameters section |
| `analysis/quarto/validation_report.qmd` | Current report | Major rewrite for new architecture |

### New Modules to Add

| Module | Purpose |
|--------|---------|
| `src/abm_enterprise/calibration/__init__.py` | Calibration package |
| `src/abm_enterprise/calibration/fit.py` | Distribution fitting from LSMS data |
| `src/abm_enterprise/calibration/schemas.py` | Calibration artifact Pydantic models |
| `src/abm_enterprise/eval/__init__.py` | Evaluation package |
| `src/abm_enterprise/eval/direct_prediction.py` | Direct prediction on LSMS states |
| `src/abm_enterprise/eval/baselines.py` | ML baseline models (logit, RF, GBM) |
| `src/abm_enterprise/eval/metrics.py` | Evaluation metrics computation |
| `src/abm_enterprise/policies/voting.py` | Multi-sample voting aggregation |
| `src/abm_enterprise/policies/cache.py` | Decision caching by state hash |

### Existing Tests to Update

| Test File | Updates Needed |
|-----------|---------------|
| `tests/test_model.py` | Update for synthetic panel mode |
| `tests/test_llm_policy.py` | Add voting, caching tests |
| `tests/test_integration.py` | Update end-to-end flow |

### New Tests to Add

| Test File | Purpose |
|-----------|---------|
| `tests/test_calibration.py` | Calibration artifact schema, parameter sanity |
| `tests/test_synthetic_panel.py` | Synthetic generation validation |
| `tests/test_voting.py` | Multi-sample voting logic |
| `tests/test_direct_prediction.py` | Dataset construction, label generation |

---

## Phase 1: Calibration Subsystem

### Goals
- Create a calibration module that estimates distribution parameters from LSMS data
- Output reusable calibration artifacts (JSON)

### Implementation

#### 1.1 Calibration Schemas (`src/abm_enterprise/calibration/schemas.py`)

```python
class DistributionSpec(BaseModel):
    family: str  # "lognormal", "normal", "t", "skew_normal"
    params: dict[str, float]
    standardization: str | None = None  # "zscore", "minmax", None

class CreditModelSpec(BaseModel):
    type: str = "logistic"
    coefficients: dict[str, float]
    intercept: float
    feature_names: list[str]

class CalibrationArtifact(BaseModel):
    country_source: str
    created_at: datetime
    git_commit: str
    waves: list[int]

    assets_distribution: DistributionSpec
    shock_distribution: DistributionSpec | dict[int, DistributionSpec]  # by wave or pooled
    credit_model: CreditModelSpec

    enterprise_baseline: dict[str, float]  # prevalence, entry_rate, exit_rate
    transition_rates: dict[str, float]  # observed ENTER, EXIT, STAY rates

    # Optional heterogeneity
    household_intercept_distribution: DistributionSpec | None = None
    region_effects: dict[str, float] | None = None
```

#### 1.2 Fitting Functions (`src/abm_enterprise/calibration/fit.py`)

```python
def fit_calibration(
    country: str,
    data_dir: Path,
    out_dir: Path,
    config: dict | None = None
) -> CalibrationArtifact:
    """
    Main entry point for calibration.

    1. Load household_targets.parquet
    2. Fit asset distribution (log-normal on raw, or t-distribution)
    3. Fit shock distribution (per-wave or pooled normal/t)
    4. Fit credit model (logistic on assets)
    5. Compute enterprise baseline rates
    6. Compute observed transition rates
    7. Write calibration.json and calibration_manifest.json
    """

def fit_asset_distribution(assets: pd.Series) -> DistributionSpec:
    """Fit log-normal or t-distribution to asset data."""

def fit_shock_distribution(
    shocks: pd.DataFrame,  # wave, price_exposure
    by_wave: bool = False
) -> DistributionSpec | dict[int, DistributionSpec]:
    """Fit shock distribution, optionally per-wave."""

def fit_credit_model(
    df: pd.DataFrame,  # must have assets_index, credit_access
    additional_features: list[str] | None = None
) -> CreditModelSpec:
    """Fit logistic regression for credit access ~ assets."""
```

#### 1.3 CLI Command

```bash
abm calibrate --country tanzania --data-dir data/processed --out-dir artifacts/calibration
```

### Acceptance Criteria

- [ ] `calibration.json` written with all required fields
- [ ] `calibration_manifest.json` includes git commit, timestamp, input paths
- [ ] Distribution parameters are non-degenerate (positive variance, etc.)
- [ ] Credit model coefficients are reasonable (assets should predict credit positively)
- [ ] Unit tests verify schema compliance and parameter sanity

---

## Phase 2: Synthetic Panel Generation

### Goals
- Replace current synthetic generator with calibration-artifact-driven generation
- Implement explicit transition dynamics across waves

### Transition Dynamics Model

**Assets Evolution:**
```
A_{t+1} = ρ × A_t + λ × I(E_t=1) + δ × P_t + ε_t
where:
  ρ = asset persistence (default: 0.85)
  λ = enterprise effect on assets (default: 0.05)
  δ = shock effect on assets (default: 0.1)
  ε_t ~ N(0, σ_ε)
```

**Credit Stickiness:**
```
C_{t+1} ~ Bernoulli(p)
where p = α × C_t + (1-α) × logit^{-1}(β × A_{t+1})
  α = credit stickiness (default: 0.7)
```

**Price Shock:**
```
P_{t} = μ_t + ν_{it}
where:
  μ_t = wave-level systematic component (from calibration)
  ν_{it} ~ N(0, σ_ν) idiosyncratic component
```

### Implementation

#### 2.1 New Synthetic Generator (`src/abm_enterprise/data/synthetic.py`)

```python
@dataclass
class TransitionConfig:
    rho_assets: float = 0.85
    lambda_enterprise_assets: float = 0.05
    delta_shock_assets: float = 0.1
    credit_stickiness: float = 0.7
    shock_sd_idiosyncratic: float = 0.10

def generate_synthetic_panel(
    calibration: CalibrationArtifact,
    n_households: int,
    config: TransitionConfig | None = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic panel from calibration artifact.

    1. Draw initial assets from calibrated distribution
    2. Initialize credit from calibrated model
    3. Initialize enterprise status from calibrated baseline
    4. For each subsequent wave:
       a. Draw price shock (wave component + idiosyncratic)
       b. Evolve assets via transition equation
       c. Update credit with stickiness
       d. Enterprise status updated by policy (placeholder for LLM)
    """
```

#### 2.2 CLI Command

```bash
abm run-sim-synthetic \
  --calibration artifacts/calibration/tanzania/calibration.json \
  --n-households 1000 \
  --policy llm_o4mini \
  --output-dir outputs/synthetic
```

### Acceptance Criteria

- [ ] Synthetic panel has correct shape: n_households × n_waves rows
- [ ] All required columns present (household_id, wave, assets_index, credit_access, enterprise_status, price_exposure)
- [ ] Asset distribution approximates calibration (KS test p > 0.05)
- [ ] Credit access correlated with assets
- [ ] Wave-to-wave correlation in assets > 0.7 (persistence)
- [ ] Reproducible with same seed

---

## Phase 3: o4-mini LLM Policy with Controlled Stochasticity

### Goals
- Replace single-sample deterministic LLM with multi-sample voting
- Add caching, temperature control, and structured logging

### Implementation

#### 3.1 Voting Module (`src/abm_enterprise/policies/voting.py`)

```python
@dataclass
class VoteResult:
    final_action: Action
    vote_counts: dict[Action, int]
    vote_share: dict[Action, float]
    samples: list[Action]
    tie_broken: bool = False

def majority_vote(
    samples: list[Action],
    tie_break: Action = Action.NO_CHANGE
) -> VoteResult:
    """Aggregate K samples via majority vote."""
```

#### 3.2 Caching Module (`src/abm_enterprise/policies/cache.py`)

```python
class DecisionCache:
    def __init__(self, max_size: int = 10000):
        self._cache: dict[str, VoteResult] = {}

    def get(self, state_hash: str, policy_config_hash: str) -> VoteResult | None:
        """Retrieve cached decision if exists."""

    def put(self, state_hash: str, policy_config_hash: str, result: VoteResult):
        """Cache decision result."""

    def save(self, path: Path):
        """Persist cache to disk."""

    def load(self, path: Path):
        """Load cache from disk."""
```

#### 3.3 Enhanced LLM Policy (`src/abm_enterprise/policies/llm.py`)

```python
class LLMPolicyConfig(BaseModel):
    model: str = "o4-mini"  # or "gpt-4o-mini"
    temperature: float = 0.6
    k_samples: int = 5
    max_tokens: int = 150
    timeout_seconds: float = 30.0
    max_retries: int = 2
    fallback_action: Action = Action.NO_CHANGE
    cache_enabled: bool = True
    tie_break: Action = Action.NO_CHANGE

class MultiSampleLLMPolicy(BasePolicy):
    def __init__(
        self,
        provider: LLMProvider,
        config: LLMPolicyConfig,
        constraints: list[Constraint] | None = None,
        logger: DecisionLogger | None = None,
        cache: DecisionCache | None = None,
    ):
        ...

    def decide(self, state: HouseholdState) -> Action:
        """
        1. Check cache
        2. If not cached:
           a. Generate K samples at temperature
           b. Parse each response
           c. Validate constraints on each
           d. Majority vote
           e. Cache result
        3. Log decision
        4. Return action
        """
```

#### 3.4 O4MiniProvider (`src/abm_enterprise/policies/providers.py`)

```python
class O4MiniProvider(LLMProvider):
    """OpenAI o4-mini specialized provider."""

    def __init__(
        self,
        api_key: str | None = None,
        temperature: float = 0.6,
        max_tokens: int = 150,
    ):
        self.model_id = "o4-mini"  # or gpt-4o-mini if o4-mini not yet available
        ...
```

#### 3.5 Logging Enhancement

Add to decision logs:
- `k_samples`: Number of samples taken
- `sample_responses`: List of all K responses
- `sample_actions`: List of parsed actions
- `vote_counts`: Action → count mapping
- `tie_broken`: Whether tie-break was applied
- `cache_hit`: Whether result was from cache

### CLI Flags

```bash
abm run-sim-synthetic \
  --policy llm_o4mini \
  --llm-model o4-mini \
  --llm-temperature 0.6 \
  --llm-k-samples 5 \
  --cache-decisions \
  --output-dir outputs/synthetic
```

### Acceptance Criteria

- [ ] Multi-sample voting produces consistent results
- [ ] Temperature > 0 produces variation across samples
- [ ] Caching prevents redundant API calls
- [ ] ReplayProvider still works for deterministic replay
- [ ] Decision logs capture all K samples
- [ ] Constraint failures handled gracefully with fallback

---

## Phase 4: Direct Prediction Evaluation Pipeline

### Goals
- Build evaluation mode that tests LLM on real LSMS states
- Compare to classical ML baselines
- Enable cross-country generalization testing

### Transition Dataset Schema

```python
class TransitionRow(BaseModel):
    household_id: str
    wave_t: int
    wave_t1: int

    # Features at t
    assets_index: float
    credit_access: int
    enterprise_status: int
    price_exposure: float

    # Target
    transition: str  # "ENTER", "EXIT", "STAY"

    # Metadata
    country: str
```

### Implementation

#### 4.1 Dataset Builder (`src/abm_enterprise/eval/direct_prediction.py`)

```python
def build_transition_dataset(
    country: str,
    data_dir: Path,
) -> pd.DataFrame:
    """
    Build transition dataset from LSMS derived targets.

    1. Load household_targets.parquet
    2. For each household, for each adjacent wave pair (t, t+1):
       a. Get state at t
       b. Compute transition label from enterprise_status change
       c. Create row
    3. Return dataset with train/test split indicators
    """

def compute_transition_label(
    enterprise_t: int,
    enterprise_t1: int
) -> str:
    if enterprise_t == 0 and enterprise_t1 == 1:
        return "ENTER"
    elif enterprise_t == 1 and enterprise_t1 == 0:
        return "EXIT"
    else:
        return "STAY"
```

#### 4.2 Baselines (`src/abm_enterprise/eval/baselines.py`)

```python
def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LogisticRegression:
    """Multinomial logistic regression baseline."""

def train_random_forest_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """Random forest baseline."""

def train_gbm_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> GradientBoostingClassifier:
    """Gradient boosting baseline (sklearn HistGB or XGBoost)."""
```

#### 4.3 LLM Prediction Mode

```python
def predict_with_llm(
    dataset: pd.DataFrame,
    policy: MultiSampleLLMPolicy,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Run LLM predictions on transition dataset.

    For each row:
    1. Convert row to HouseholdState
    2. Call policy.decide()
    3. Map action to transition label
    4. Store prediction and vote shares
    """
```

#### 4.4 Metrics (`src/abm_enterprise/eval/metrics.py`)

```python
def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.DataFrame | None = None
) -> dict:
    """
    Compute metrics:
    - accuracy
    - balanced_accuracy
    - macro_f1
    - per_class_precision
    - per_class_recall
    - confusion_matrix
    - brier_score (if probabilities available)
    """

def compute_subgroup_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    subgroup_cols: list[str]
) -> pd.DataFrame:
    """Compute metrics by subgroup (low/high assets, credit, country)."""
```

### CLI Command

```bash
abm eval-direct \
  --train-country tanzania \
  --test-country ethiopia \
  --model llm_o4mini \
  --baselines logit,rf,gbm \
  --output-dir outputs/eval
```

### Outputs

```
outputs/eval/direct_prediction/tanzania_to_ethiopia/
├── metrics.json
├── confusion_matrix_llm.csv
├── confusion_matrix_logit.csv
├── confusion_matrix_rf.csv
├── confusion_matrix_gbm.csv
├── predictions.parquet
└── subgroup_metrics.csv
```

### Acceptance Criteria

- [ ] Transition dataset correctly labels ENTER/EXIT/STAY
- [ ] Train/test split respects country boundaries
- [ ] All baselines train and predict without errors
- [ ] LLM predictions return valid actions
- [ ] Metrics computed for all models
- [ ] Subgroup analysis by assets, credit, country

---

## Phase 5: Full Simulation Validation Mode

### Goals
- Run synthetic ABM with LLM decisions
- Compare simulated outcomes to LSMS stylized facts

### Implementation

Updates to existing validation pipeline:

#### 5.1 Simulation Runner

```python
def run_synthetic_simulation(
    calibration: CalibrationArtifact,
    policy: MultiSampleLLMPolicy,
    n_households: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run synthetic ABM with LLM policy.

    1. Generate initial synthetic panel (wave 1)
    2. For each subsequent wave:
       a. For each household, get current state
       b. Call policy.decide(state)
       c. Apply action to update enterprise_status
       d. Evolve assets, credit, shocks
    3. Return full panel with outcomes
    """
```

#### 5.2 Validation Comparisons

Keep existing R/Quarto pipeline but update to compare:
- Simulated synthetic panel (from ABM)
- Real LSMS derived targets (observed)

Metrics:
- FE regression coefficient comparison
- Enterprise rate by wave comparison
- Stayer/coper distribution comparison
- Response heterogeneity by assets/credit

### Acceptance Criteria

- [ ] Synthetic simulation runs to completion
- [ ] Output matches expected schema
- [ ] R validation pipeline accepts synthetic outputs
- [ ] FE regression runs without errors
- [ ] Results clearly labeled as "synthetic" vs "observed"

---

## Phase 6: Documentation Updates

### Files to Update

| File | Updates |
|------|---------|
| `docs/CONOPS.md` | Add new architecture diagram, update data flow |
| `docs/VALIDATION_CONTRACT.md` | Add direct prediction metrics, cross-country expectations |
| `docs/PROJECT_STATE.md` | Update phase status for refactoring |
| `docs/DECISIONS.md` | Add decisions for distribution families, transition dynamics |
| `analysis/quarto/validation_report.qmd` | Major rewrite for new architecture |
| `CLAUDE.md` | Update commands reference |

### New Documentation

| File | Purpose |
|------|---------|
| `docs/CALIBRATION_DESIGN.md` | Document distribution choices, calibration methodology |
| `docs/EVALUATION_DESIGN.md` | Document direct prediction vs simulation evaluation |
| `docs/RUNBOOK_REFACTORED.md` | Step-by-step commands for new pipeline |

### Quarto Report Structure

```
1. Introduction
   - Research question
   - Two evaluation tracks

2. Calibration
   - Tanzania data description
   - Distribution fitting results
   - Calibration artifact summary

3. Direct Prediction Evaluation
   - Dataset description
   - LLM vs baseline comparison
   - Cross-country generalization (Tanzania → Ethiopia)
   - Subgroup analysis

4. Synthetic ABM Simulation
   - Transition dynamics model
   - LLM decision patterns
   - Stylized fact comparison to LSMS

5. Discussion
   - LLM realism assessment
   - Limitations
   - Future work
```

---

## Testing Plan

### Unit Tests

| Test | Location | Description |
|------|----------|-------------|
| `test_calibration_schema` | `tests/test_calibration.py` | CalibrationArtifact validation |
| `test_calibration_fit` | `tests/test_calibration.py` | Distribution fitting produces valid params |
| `test_synthetic_panel_shape` | `tests/test_synthetic_panel.py` | Correct rows, columns |
| `test_synthetic_panel_schema` | `tests/test_synthetic_panel.py` | All required columns present |
| `test_synthetic_panel_dynamics` | `tests/test_synthetic_panel.py` | Asset persistence > 0.5 |
| `test_voting_majority` | `tests/test_voting.py` | Majority vote logic |
| `test_voting_tie_break` | `tests/test_voting.py` | Tie-break behavior |
| `test_cache_hit_miss` | `tests/test_voting.py` | Cache retrieval |
| `test_transition_labels` | `tests/test_direct_prediction.py` | ENTER/EXIT/STAY labeling |
| `test_dataset_split` | `tests/test_direct_prediction.py` | Train/test by country |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_calibration_to_synthetic` | Calibration → synthetic panel end-to-end |
| `test_synthetic_simulation` | Synthetic panel + LLM policy → outcomes |
| `test_direct_prediction_pipeline` | LSMS → dataset → LLM predictions |
| `test_full_pipeline` | Calibrate → simulate → evaluate |

---

## End-to-End Commands (Done Criteria)

When complete, these commands must all work:

```bash
# 1. Calibration
abm calibrate --country tanzania --data-dir data/processed --out-dir artifacts/calibration

# 2. Synthetic ABM with o4-mini decisions
abm run-sim-synthetic \
  --calibration artifacts/calibration/tanzania/calibration.json \
  --scenario baseline \
  --policy llm_o4mini \
  --llm-temperature 0.6 \
  --llm-k-samples 5 \
  --output-dir outputs/synthetic

# 3. Direct prediction evaluation
abm eval-direct \
  --train-country tanzania \
  --test-country ethiopia \
  --model llm_o4mini \
  --baselines logit,rf,gbm \
  --output-dir outputs/eval

# 4. Render report
make render-report
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| o4-mini API costs | Aggressive caching, K=3-5 samples |
| LLM rate limits | Implement backoff, batch processing |
| Distribution fitting failures | Fallback to robust families (t-distribution) |
| Cross-country generalization poor | Document as finding, not implementation failure |
| Quarto/R environment issues | Clear renv lockfile, documented setup |

---

## Timeline Notes

*No specific timeline provided - implementation should proceed phase by phase with testing at each stage.*

---

## Appendix: Distribution Family Choices

### Assets Distribution
- **Primary choice**: Log-normal
- **Rationale**: Asset data is typically positive, right-skewed
- **Fallback**: Student-t on log scale for heavy tails

### Shock Distribution
- **Primary choice**: Normal
- **Rationale**: Price shocks centered around 0, symmetric
- **Configuration**: Allow per-wave means with pooled variance, or fully pooled

### Credit Model
- **Choice**: Logistic regression on assets
- **Rationale**: Standard approach, interpretable coefficients
- **Features**: assets_index (required), optional region/hhsize

---

*Document version: 1.0*
*Author: Gen (AI Assistant)*
