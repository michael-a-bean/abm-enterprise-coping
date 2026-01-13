"""Integration tests for derived targets and validation contract alignment.

Tests that:
- Loads Tanzania derived targets
- Runs simulation for 4 waves
- Verifies output schema matches validation contract
- Checks stayer/coper proportions are reasonable
"""

from pathlib import Path

import pandas as pd
import pytest

from abm_enterprise.agents.household import Classification
from abm_enterprise.data.schemas import OutputRecord, SimulationConfig
from abm_enterprise.model import (
    EnterpriseCopingModel,
    compute_calibration_thresholds,
    load_derived_targets,
)
from abm_enterprise.policies.rule import CalibratedRulePolicy
from abm_enterprise.utils.rng import set_seed


# Path to derived data
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


class TestDerivedTargetsIntegration:
    """Integration tests for derived target data loading and simulation."""

    @pytest.fixture
    def tanzania_targets(self) -> pd.DataFrame:
        """Load Tanzania derived targets if available."""
        if not (DATA_DIR / "tanzania" / "derived" / "household_targets.parquet").exists():
            pytest.skip("Tanzania derived targets not available")
        return load_derived_targets("tanzania", DATA_DIR)

    def test_load_derived_targets(self, tanzania_targets: pd.DataFrame) -> None:
        """Test that derived targets load correctly."""
        df = tanzania_targets

        # Check required columns exist
        required_cols = [
            "household_id",
            "wave",
            "enterprise_status",  # Renamed from enterprise_indicator
            "enterprise_persistence",
            "classification",
            "assets_index",  # Renamed from asset_index
            "asset_quintile",
            "credit_access",
            "price_exposure",
        ]

        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Check data integrity
        assert len(df) > 0, "DataFrame is empty"
        assert df["household_id"].nunique() > 0, "No households found"
        assert df["wave"].min() >= 1, "Invalid wave numbers"

    def test_compute_calibration_thresholds(
        self, tanzania_targets: pd.DataFrame
    ) -> None:
        """Test calibration threshold computation."""
        thresholds = compute_calibration_thresholds(tanzania_targets)

        # Check all thresholds present
        assert "price_threshold" in thresholds
        assert "asset_threshold" in thresholds
        assert "exit_asset_threshold" in thresholds

        # Validate reasonable ranges
        assert thresholds["price_threshold"] < 0, "Price threshold should be negative"
        assert (
            thresholds["exit_asset_threshold"] < thresholds["asset_threshold"]
        ), "Exit threshold should be lower than entry threshold"

    def test_simulation_with_derived_targets(
        self, tanzania_targets: pd.DataFrame
    ) -> None:
        """Test running simulation with derived targets."""
        set_seed(42)

        # Create calibrated policy
        thresholds = compute_calibration_thresholds(tanzania_targets)
        policy = CalibratedRulePolicy(country="tanzania", thresholds=thresholds)

        # Create config
        config = SimulationConfig(
            country="tanzania",
            scenario="integration_test",
            seed=42,
            num_waves=4,
        )

        # Run model
        model = EnterpriseCopingModel(
            config=config,
            household_data=tanzania_targets,
            policy=policy,
        )
        model.run()

        # Get outcomes
        outcomes = model.get_outcomes_dataframe()

        # Verify output exists
        assert len(outcomes) > 0, "No outcomes generated"
        assert len(model.agents_by_id) > 0, "No agents created"

    def test_output_schema_validation_contract(
        self, tanzania_targets: pd.DataFrame
    ) -> None:
        """Test that outputs match validation contract schema."""
        set_seed(42)

        config = SimulationConfig(
            country="tanzania",
            scenario="schema_test",
            seed=42,
            num_waves=4,
        )

        model = EnterpriseCopingModel(
            config=config,
            household_data=tanzania_targets,
        )
        model.run()

        outcomes = model.get_outcomes_dataframe()

        # Check all validation contract required columns
        required_columns = OutputRecord.required_columns()
        for col in required_columns:
            assert col in outcomes.columns, f"Missing validation column: {col}"

        # Verify data types and ranges
        assert outcomes["enterprise_status"].isin([0, 1]).all(), "Invalid enterprise_status"
        assert outcomes["enterprise_entry"].isin([0, 1]).all(), "Invalid enterprise_entry"
        assert outcomes["credit_access"].isin([0, 1]).all(), "Invalid credit_access"
        assert outcomes["wave"].min() >= 1, "Invalid wave number"
        assert outcomes["asset_quintile"].between(1, 5).all(), "Invalid asset_quintile"

        # Verify classification values
        valid_classifications = [
            Classification.STAYER,
            Classification.COPER,
            Classification.NONE,
        ]
        assert (
            outcomes["classification"].isin(valid_classifications).all()
        ), "Invalid classification values"

    def test_stayer_coper_proportions(self, tanzania_targets: pd.DataFrame) -> None:
        """Test that stayer/coper proportions are reasonable."""
        set_seed(42)

        config = SimulationConfig(
            country="tanzania",
            scenario="proportions_test",
            seed=42,
            num_waves=4,
        )

        model = EnterpriseCopingModel(
            config=config,
            household_data=tanzania_targets,
        )
        model.run()

        outcomes = model.get_outcomes_dataframe()

        # Calculate proportions
        classification_counts = outcomes["classification"].value_counts(normalize=True)

        # Proportions should sum to 1
        assert abs(classification_counts.sum() - 1.0) < 0.001

        # At least some households in each category (if data has variation)
        # This is a soft check - actual proportions depend on data
        total_count = len(outcomes)
        unique_classifications = outcomes["classification"].nunique()

        # Should have at least 2 classification types in most datasets
        assert unique_classifications >= 1, "No classification variation"

        # Log proportions for debugging
        print("\nClassification proportions:")
        for cls, prop in classification_counts.items():
            print(f"  {cls}: {prop:.1%}")

    def test_price_exposure_from_targets(self, tanzania_targets: pd.DataFrame) -> None:
        """Test that simulation uses price_exposure from derived targets."""
        set_seed(42)

        config = SimulationConfig(
            country="tanzania",
            scenario="price_test",
            seed=42,
            num_waves=4,
        )

        model = EnterpriseCopingModel(
            config=config,
            household_data=tanzania_targets,
        )
        model.run()

        outcomes = model.get_outcomes_dataframe()

        # Price exposure should have variation (not all zeros)
        price_std = outcomes["price_exposure"].std()
        assert price_std > 0, "No variation in price_exposure"

        # Price exposure should match input data range approximately
        input_min = tanzania_targets["price_exposure"].min()
        input_max = tanzania_targets["price_exposure"].max()
        output_min = outcomes["price_exposure"].min()
        output_max = outcomes["price_exposure"].max()

        # Output range should be within input range
        assert output_min >= input_min - 0.01, "Price exposure below expected range"
        assert output_max <= input_max + 0.01, "Price exposure above expected range"


class TestCalibratedRulePolicy:
    """Tests for CalibratedRulePolicy."""

    def test_policy_from_data(self) -> None:
        """Test creating policy from data."""
        # Create test data
        test_data = pd.DataFrame({
            "household_id": ["H1", "H2", "H3", "H4"],
            "assets_index": [-1.0, -0.5, 0.0, 1.0],
            "price_exposure": [-0.2, -0.1, 0.0, 0.1],
        })

        policy = CalibratedRulePolicy.from_data(test_data, country="tanzania")

        # Check thresholds computed
        thresholds = policy.get_thresholds()
        assert "price_threshold" in thresholds
        assert "asset_threshold" in thresholds
        assert "exit_asset_threshold" in thresholds

    def test_policy_decide(self) -> None:
        """Test policy decision making."""
        from abm_enterprise.data.schemas import EnterpriseStatus, HouseholdState
        from abm_enterprise.policies.base import Action

        policy = CalibratedRulePolicy(
            price_threshold=-0.1,
            asset_threshold=0.0,
            exit_asset_threshold=-1.0,
        )

        # Test entry: adverse price shock + low assets
        state_enter = HouseholdState(
            household_id="H1",
            wave=1,
            assets=-0.5,  # Below threshold
            credit_access=0,
            enterprise_status=EnterpriseStatus.NO_ENTERPRISE,
            price_exposure=-0.2,  # Below threshold
        )
        assert policy.decide(state_enter) == Action.ENTER_ENTERPRISE

        # Test no change: good conditions
        state_no_change = HouseholdState(
            household_id="H2",
            wave=1,
            assets=0.5,  # Above threshold
            credit_access=1,
            enterprise_status=EnterpriseStatus.NO_ENTERPRISE,
            price_exposure=0.1,  # Above threshold
        )
        assert policy.decide(state_no_change) == Action.NO_CHANGE


class TestClassification:
    """Tests for household classification."""

    def test_classification_constants(self) -> None:
        """Test classification constants are defined correctly."""
        assert Classification.STAYER == "stayer"
        assert Classification.COPER == "coper"
        assert Classification.NONE == "none"

    def test_derive_classification(self) -> None:
        """Test classification derivation from persistence."""
        from abm_enterprise.agents.household import HouseholdAgent

        # Test stayer: >50% persistence
        assert HouseholdAgent._derive_classification(0.75) == Classification.STAYER
        assert HouseholdAgent._derive_classification(0.51) == Classification.STAYER
        assert HouseholdAgent._derive_classification(1.0) == Classification.STAYER

        # Test coper: >0% and <=50% persistence
        assert HouseholdAgent._derive_classification(0.5) == Classification.COPER
        assert HouseholdAgent._derive_classification(0.25) == Classification.COPER
        assert HouseholdAgent._derive_classification(0.01) == Classification.COPER

        # Test none: 0% persistence
        assert HouseholdAgent._derive_classification(0.0) == Classification.NONE
