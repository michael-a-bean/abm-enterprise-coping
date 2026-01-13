"""Tests for the Mesa 3 model."""

import pandas as pd

from abm_enterprise.data.schemas import SimulationConfig
from abm_enterprise.data.synthetic import generate_synthetic_households
from abm_enterprise.model import EnterpriseCopingModel, run_toy_simulation
from abm_enterprise.policies.rule import RulePolicy
from abm_enterprise.utils.rng import GlobalRNG, set_seed


class TestEnterpriseCopingModel:
    """Tests for EnterpriseCopingModel."""

    def setup_method(self) -> None:
        """Reset RNG state before each test."""
        GlobalRNG.reset()

    def teardown_method(self) -> None:
        """Reset RNG state after each test."""
        GlobalRNG.reset()

    def test_model_initialization(self) -> None:
        """Test model initializes correctly."""
        config = SimulationConfig(country="tanzania", seed=42, num_waves=4)
        model = EnterpriseCopingModel(config=config)

        assert model.config == config
        assert len(model.agents_by_id) == 100  # Default synthetic data
        assert model.current_wave == 1

    def test_model_with_custom_data(self) -> None:
        """Test model with custom household data."""
        set_seed(42)
        config = SimulationConfig(country="tanzania", seed=42, num_waves=2)
        data = generate_synthetic_households(n=50, waves=2)
        model = EnterpriseCopingModel(config=config, household_data=data)

        assert len(model.agents_by_id) == 50

    def test_model_step(self) -> None:
        """Test single step execution."""
        config = SimulationConfig(country="tanzania", seed=42, num_waves=4)
        model = EnterpriseCopingModel(config=config)

        model.step()
        assert len(model.outcomes) == 100  # One record per household

    def test_model_run(self) -> None:
        """Test full simulation run."""
        config = SimulationConfig(country="tanzania", seed=42, num_waves=4)
        model = EnterpriseCopingModel(config=config)

        model.run()
        assert len(model.outcomes) == 400  # 100 households * 4 waves

    def test_outcomes_dataframe(self) -> None:
        """Test outcomes DataFrame structure."""
        config = SimulationConfig(country="tanzania", seed=42, num_waves=2)
        model = EnterpriseCopingModel(config=config)
        model.run()

        df = model.get_outcomes_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200  # 100 * 2 waves

        # Check required columns
        required_cols = [
            "household_id",
            "wave",
            "assets_index",
            "credit_access",
            "enterprise_status",
            "price_exposure",
            "action_taken",
            "policy_applied",
        ]
        for col in required_cols:
            assert col in df.columns

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        config1 = SimulationConfig(country="tanzania", seed=42, num_waves=2)
        model1 = EnterpriseCopingModel(config=config1)
        model1.run()
        df1 = model1.get_outcomes_dataframe()

        GlobalRNG.reset()
        config2 = SimulationConfig(country="tanzania", seed=42, num_waves=2)
        model2 = EnterpriseCopingModel(config=config2)
        model2.run()
        df2 = model2.get_outcomes_dataframe()

        # Compare numeric columns
        pd.testing.assert_frame_equal(
            df1[["assets_index", "enterprise_status", "price_exposure"]],
            df2[["assets_index", "enterprise_status", "price_exposure"]],
        )

    def test_policy_integration(self) -> None:
        """Test model with custom policy."""
        config = SimulationConfig(country="tanzania", seed=42, num_waves=2)
        policy = RulePolicy(price_threshold=-0.05, asset_threshold=0.5)
        model = EnterpriseCopingModel(config=config, policy=policy)
        model.run()

        df = model.get_outcomes_dataframe()
        # Verify policy was applied
        assert df["policy_applied"].sum() > 0


class TestRunToySimulation:
    """Tests for run_toy_simulation helper."""

    def setup_method(self) -> None:
        """Reset RNG state before each test."""
        GlobalRNG.reset()

    def teardown_method(self) -> None:
        """Reset RNG state after each test."""
        GlobalRNG.reset()

    def test_run_toy_returns_model_and_df(self) -> None:
        """Test that run_toy returns model and DataFrame."""
        model, df = run_toy_simulation(seed=42, num_households=50, num_waves=2)

        assert isinstance(model, EnterpriseCopingModel)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100  # 50 * 2

    def test_run_toy_reproducibility(self) -> None:
        """Test run_toy reproducibility (excluding run_id which is unique per run)."""
        _, df1 = run_toy_simulation(seed=123, num_households=30, num_waves=2)

        GlobalRNG.reset()
        _, df2 = run_toy_simulation(seed=123, num_households=30, num_waves=2)

        # Exclude run_id from comparison as it's meant to be unique per run
        cols_to_compare = [c for c in df1.columns if c != "run_id"]
        pd.testing.assert_frame_equal(df1[cols_to_compare], df2[cols_to_compare])

    def test_run_toy_different_countries(self) -> None:
        """Test run_toy with different countries."""
        _, df_tz = run_toy_simulation(seed=42, country="tanzania")

        GlobalRNG.reset()
        _, df_et = run_toy_simulation(seed=42, country="ethiopia")

        assert df_tz["country"].iloc[0] == "tanzania"
        assert df_et["country"].iloc[0] == "ethiopia"
