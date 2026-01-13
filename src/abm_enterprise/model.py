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
