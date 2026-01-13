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
