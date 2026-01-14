"""Mesa 3 model for enterprise coping simulation.

This module defines the EnterpriseCopingModel which orchestrates
household agents making enterprise participation decisions.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import mesa
import pandas as pd
import polars as pl

from abm_enterprise.agents.household import HouseholdAgent
from abm_enterprise.data.schemas import PolicyType, SimulationConfig
from abm_enterprise.data.synthetic import generate_synthetic_households
from abm_enterprise.policies.base import BasePolicy
from abm_enterprise.policies.rule import RulePolicy
from abm_enterprise.utils.logging import get_logger
from abm_enterprise.utils.rng import get_rng, set_seed

logger = get_logger(__name__)


class EnterpriseCopingModel(mesa.Model):
    """Mesa 3 model for enterprise coping simulation.

    Manages household agents and simulates their enterprise
    participation decisions over multiple survey waves.

    Attributes:
        config: Simulation configuration.
        current_wave: Current simulation wave number.
        household_data: Panel data for households.
        agents_by_id: Dictionary mapping household IDs to agents.
        outcomes: List of outcome records collected during simulation.
    """

    def __init__(
        self,
        config: SimulationConfig,
        household_data: pd.DataFrame | None = None,
        policy: BasePolicy | None = None,
    ) -> None:
        """Initialize the enterprise coping model.

        Args:
            config: Simulation configuration.
            household_data: Optional pre-loaded household data.
            policy: Optional policy for agent decisions.
        """
        super().__init__()

        self.config = config
        self.run_id = str(uuid.uuid4())[:8]

        # Initialize RNG
        set_seed(config.seed)

        # Set model's random state for Mesa compatibility
        self.random = get_rng()

        self.current_wave = 1
        self.outcomes: list[dict[str, Any]] = []

        # Load or generate household data
        if household_data is not None:
            self.household_data = household_data
        else:
            logger.info("Generating synthetic household data")
            self.household_data = generate_synthetic_households(
                n=100,
                waves=config.num_waves,
                country=config.country,
            )

        # Set up policy
        if policy is not None:
            self.policy = policy
        else:
            self.policy = self._create_policy_from_config()

        # Create agents from wave 1 data
        self._initialize_agents()

        logger.info(
            "Model initialized",
            run_id=self.run_id,
            country=config.country,
            scenario=config.scenario,
            num_households=len(self.agents_by_id),
            num_waves=config.num_waves,
        )

    def _create_policy_from_config(self) -> BasePolicy:
        """Create policy based on configuration.

        Returns:
            Policy instance based on config.policy_type.
        """
        if self.config.policy_type == PolicyType.NONE:
            return RulePolicy(
                price_threshold=self.config.price_exposure_threshold,
                asset_threshold=0.0,
            )
        elif self.config.policy_type == PolicyType.CREDIT_ACCESS:
            # Credit access policy: more lenient asset thresholds
            return RulePolicy(
                price_threshold=self.config.price_exposure_threshold,
                asset_threshold=0.5,  # Higher threshold = more likely to enter
                credit_required_for_stability=True,
            )
        elif self.config.policy_type == PolicyType.PRICE_SUPPORT:
            # Price support: less sensitive to price shocks
            return RulePolicy(
                price_threshold=-0.2,  # More negative = less sensitive
                asset_threshold=0.0,
            )
        elif self.config.policy_type == PolicyType.ASSET_TRANSFER:
            # Asset transfer: focus on very low asset households
            return RulePolicy(
                price_threshold=self.config.price_exposure_threshold,
                asset_threshold=-0.5,  # Target lowest asset households
                exit_asset_threshold=-2.0,  # Prevent exit
            )
        else:
            return RulePolicy()

    def _initialize_agents(self) -> None:
        """Initialize agents from wave 1 data."""
        self.agents_by_id: dict[str, HouseholdAgent] = {}

        wave_1_data = self.household_data[self.household_data["wave"] == 1]

        for _, row in wave_1_data.iterrows():
            household_id = str(row["household_id"])
            agent = HouseholdAgent(
                model=self,
                household_id=household_id,
                initial_data=row.to_dict(),
            )
            agent.policy = self.policy
            self.agents_by_id[household_id] = agent

    def step(self) -> None:
        """Execute one simulation step (wave).

        Updates agent states from data, runs agent decisions,
        and collects outcomes.
        """
        logger.debug("Starting wave", wave=self.current_wave)

        # Update agents with current wave data
        wave_data = self.household_data[self.household_data["wave"] == self.current_wave]

        for _, row in wave_data.iterrows():
            household_id = str(row["household_id"])
            if household_id in self.agents_by_id:
                self.agents_by_id[household_id].update_state(row.to_dict())

        # Run all agents
        for agent in self.agents_by_id.values():
            agent.step()

        # Collect outcomes
        self._collect_outcomes()

        logger.debug(
            "Completed wave",
            wave=self.current_wave,
            outcomes_collected=len(self.outcomes),
        )

        # Advance wave
        self.current_wave += 1

    def _collect_outcomes(self) -> None:
        """Collect outcome records from all agents."""
        for agent in self.agents_by_id.values():
            record = agent.get_output_record()
            record["country"] = self.config.country
            record["scenario"] = self.config.scenario
            record["run_id"] = self.run_id
            self.outcomes.append(record)

    def run(self) -> None:
        """Run the simulation for all waves."""
        logger.info("Starting simulation run", num_waves=self.config.num_waves)

        for wave in range(1, self.config.num_waves + 1):
            self.current_wave = wave
            self.step()

        logger.info(
            "Simulation complete",
            total_outcomes=len(self.outcomes),
        )

    def get_outcomes_dataframe(self) -> pd.DataFrame:
        """Get outcomes as a pandas DataFrame.

        Returns:
            DataFrame with all outcome records.
        """
        return pd.DataFrame(self.outcomes)


def run_toy_simulation(
    seed: int = 42,
    num_households: int = 100,
    num_waves: int = 4,
    country: str = "tanzania",
) -> tuple[EnterpriseCopingModel, pd.DataFrame]:
    """Run a toy simulation with synthetic data.

    Convenience function for quick testing and development.

    Args:
        seed: Random seed.
        num_households: Number of synthetic households.
        num_waves: Number of waves to simulate.
        country: Country code.

    Returns:
        Tuple of (model, outcomes_dataframe).

    Example:
        >>> model, outcomes = run_toy_simulation(seed=42)
        >>> outcomes.shape
        (400, 12)
    """
    config = SimulationConfig(
        country=country,
        scenario="toy",
        seed=seed,
        num_waves=num_waves,
    )

    set_seed(seed)
    household_data = generate_synthetic_households(
        n=num_households,
        waves=num_waves,
        country=country,
    )

    model = EnterpriseCopingModel(config=config, household_data=household_data)
    model.run()

    return model, model.get_outcomes_dataframe()


def load_real_data(country: str, data_dir: Path | str) -> pd.DataFrame:
    """Load real household data from parquet files.

    Args:
        country: Country code.
        data_dir: Directory containing processed data.

    Returns:
        DataFrame with household panel data.

    Raises:
        FileNotFoundError: If data file doesn't exist.
    """
    data_dir = Path(data_dir)
    data_file = data_dir / f"{country}_household_panel.parquet"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    return pd.read_parquet(data_file)


def load_derived_targets(country: str, data_dir: Path | str) -> pd.DataFrame:
    """Load derived target data from Phase 2 parquet files.

    Loads the household_targets.parquet file which contains:
    - enterprise_indicator, enterprise_persistence, classification
    - asset_index, asset_quintile, credit_access
    - price_exposure, welfare_proxy

    Maps columns to ABM-expected names:
    - enterprise_indicator -> enterprise_status
    - asset_index -> assets_index (for consistency with synthetic data)

    Args:
        country: Country code (tanzania or ethiopia).
        data_dir: Directory containing processed data (e.g., data/processed).

    Returns:
        DataFrame with household panel data ready for ABM initialization.

    Raises:
        FileNotFoundError: If derived targets file doesn't exist.
    """
    data_dir = Path(data_dir)
    targets_file = data_dir / country / "derived" / "household_targets.parquet"

    if not targets_file.exists():
        raise FileNotFoundError(
            f"Derived targets not found: {targets_file}\n"
            f"Run 'abm derive-targets --country {country}' first."
        )

    # Load with polars for efficiency, convert to pandas
    df_pl = pl.read_parquet(targets_file)

    # Rename columns to match ABM expected schema
    df_pl = df_pl.rename({
        "enterprise_indicator": "enterprise_status",
        "asset_index": "assets_index",
    })

    # Add synthetic-data-style columns that may be missing
    if "crop_count" not in df_pl.columns:
        df_pl = df_pl.with_columns(pl.lit(1).alias("crop_count"))
    if "land_area_ha" not in df_pl.columns:
        df_pl = df_pl.with_columns(pl.lit(1.0).alias("land_area_ha"))

    logger.info(
        "Loaded derived targets",
        country=country,
        num_households=df_pl["household_id"].n_unique(),
        num_observations=len(df_pl),
        waves=sorted(df_pl["wave"].unique().to_list()),
    )

    return df_pl.to_pandas()


def compute_calibration_thresholds(
    targets_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute calibrated thresholds from derived targets.

    Calculates thresholds based on actual data distributions:
    - price_threshold: Median of negative price exposures
    - asset_threshold: Median asset index
    - exit_asset_threshold: 10th percentile of asset index

    Args:
        targets_df: DataFrame with derived target data.

    Returns:
        Dictionary with calibrated thresholds.
    """
    # Price threshold: use median of negative price exposures
    negative_prices = targets_df[targets_df["price_exposure"] < 0]["price_exposure"]
    if len(negative_prices) > 0:
        price_threshold = float(negative_prices.median())
    else:
        price_threshold = -0.1  # Default

    # Asset threshold: median asset index
    asset_threshold = float(targets_df["assets_index"].median())

    # Exit threshold: 10th percentile (very low assets)
    exit_asset_threshold = float(targets_df["assets_index"].quantile(0.10))

    return {
        "price_threshold": price_threshold,
        "asset_threshold": asset_threshold,
        "exit_asset_threshold": exit_asset_threshold,
    }


def run_synthetic_simulation(
    calibration_path: Path | str,
    policy: BasePolicy,
    n_households: int = 1000,
    seed: int = 42,
    scenario: str = "synthetic",
) -> tuple["EnterpriseCopingModel", pd.DataFrame]:
    """Run synthetic ABM with calibrated distributions and LLM policy.

    Generates synthetic household panel from calibration artifact,
    then runs the ABM with the provided policy to produce outcomes.

    Args:
        calibration_path: Path to calibration.json artifact.
        policy: Policy for agent decisions (typically MultiSampleLLMPolicy).
        n_households: Number of synthetic households to generate.
        seed: Random seed for reproducibility.
        scenario: Scenario name for outputs.

    Returns:
        Tuple of (model, outcomes_dataframe).

    Example:
        >>> from abm_enterprise.policies.llm import MultiSampleLLMPolicyFactory
        >>> policy = MultiSampleLLMPolicyFactory.create_stub_policy(k_samples=5)
        >>> model, outcomes = run_synthetic_simulation(
        ...     "artifacts/calibration/tanzania/calibration.json",
        ...     policy=policy,
        ...     n_households=500,
        ... )
    """
    from abm_enterprise.calibration import CalibrationArtifact
    from abm_enterprise.data.synthetic import (
        SyntheticPanelConfig,
        SyntheticPanelGenerator,
    )

    calibration_path = Path(calibration_path)

    # Load calibration artifact
    calibration = CalibrationArtifact.load(calibration_path)

    logger.info(
        "Running synthetic simulation",
        calibration_source=calibration.country_source,
        n_households=n_households,
        seed=seed,
    )

    # Generate synthetic panel
    set_seed(seed)
    config = SyntheticPanelConfig(
        n_households=n_households,
        waves=calibration.waves,
        seed=seed,
    )
    generator = SyntheticPanelGenerator(calibration, config)
    synthetic_panel = generator.generate()

    # Map column names to ABM expectations
    synthetic_panel = synthetic_panel.rename(columns={
        "enterprise_status": "enterprise_status",  # Already correct
    })

    # Create simulation config
    sim_config = SimulationConfig(
        country=calibration.country_source,
        scenario=scenario,
        seed=seed,
        num_waves=len(calibration.waves),
    )

    # Run model with synthetic panel and provided policy
    model = EnterpriseCopingModel(
        config=sim_config,
        household_data=synthetic_panel,
        policy=policy,
    )
    model.run()

    outcomes = model.get_outcomes_dataframe()

    logger.info(
        "Synthetic simulation complete",
        total_outcomes=len(outcomes),
        enterprise_rate=outcomes["enterprise_status"].mean(),
    )

    return model, outcomes


def compare_simulation_to_lsms(
    outcomes: pd.DataFrame,
    lsms_targets: pd.DataFrame,
) -> dict[str, Any]:
    """Compare simulation outcomes to LSMS stylized facts.

    Computes key comparisons between simulated and observed data:
    - Enterprise prevalence by wave
    - Entry/exit rates
    - Response heterogeneity by assets

    Args:
        outcomes: Simulation outcomes from ABM.
        lsms_targets: Observed LSMS derived targets.

    Returns:
        Dictionary with comparison metrics.
    """
    comparisons = {}

    # Enterprise prevalence by wave
    sim_prev = outcomes.groupby("wave")["enterprise_status"].mean()
    obs_prev = lsms_targets.groupby("wave")["enterprise_indicator"].mean()

    comparisons["enterprise_prevalence"] = {
        "simulated": sim_prev.to_dict(),
        "observed": obs_prev.to_dict(),
    }

    # Calculate transition rates for simulation
    sim_transitions = _compute_transition_rates(outcomes, "enterprise_status")
    obs_transitions = _compute_transition_rates(lsms_targets, "enterprise_indicator")

    comparisons["transition_rates"] = {
        "simulated": sim_transitions,
        "observed": obs_transitions,
    }

    # Asset-stratified enterprise rates
    sim_by_assets = _stratify_by_assets(outcomes, "enterprise_status")
    obs_by_assets = _stratify_by_assets(lsms_targets, "enterprise_indicator")

    comparisons["enterprise_by_assets"] = {
        "simulated": sim_by_assets,
        "observed": obs_by_assets,
    }

    return comparisons


def _compute_transition_rates(
    df: pd.DataFrame,
    enterprise_col: str,
) -> dict[str, float]:
    """Compute entry/exit/stay rates from panel data."""
    # Sort by household and wave
    df = df.sort_values(["household_id", "wave"])

    transitions = {"enter": 0, "exit": 0, "stay_in": 0, "stay_out": 0, "total": 0}

    for hh_id, group in df.groupby("household_id"):
        statuses = group[enterprise_col].values
        for i in range(len(statuses) - 1):
            t0, t1 = statuses[i], statuses[i + 1]
            transitions["total"] += 1
            if t0 == 0 and t1 == 1:
                transitions["enter"] += 1
            elif t0 == 1 and t1 == 0:
                transitions["exit"] += 1
            elif t0 == 1 and t1 == 1:
                transitions["stay_in"] += 1
            else:
                transitions["stay_out"] += 1

    total = transitions["total"]
    if total > 0:
        return {
            "enter_rate": transitions["enter"] / total,
            "exit_rate": transitions["exit"] / total,
            "stay_in_rate": transitions["stay_in"] / total,
            "stay_out_rate": transitions["stay_out"] / total,
            "n_transitions": total,
        }
    return {"enter_rate": 0, "exit_rate": 0, "stay_in_rate": 0, "stay_out_rate": 0, "n_transitions": 0}


def _stratify_by_assets(
    df: pd.DataFrame,
    enterprise_col: str,
    n_quantiles: int = 4,
) -> dict[str, float]:
    """Compute enterprise rate by asset quantile."""
    df = df.copy()

    # Handle different column names
    asset_col = "assets_index" if "assets_index" in df.columns else "asset_index"

    df["asset_quantile"] = pd.qcut(
        df[asset_col],
        q=n_quantiles,
        labels=[f"Q{i+1}" for i in range(n_quantiles)],
        duplicates="drop",
    )

    return df.groupby("asset_quantile")[enterprise_col].mean().to_dict()
