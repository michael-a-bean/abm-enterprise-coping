"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from abm_enterprise.data.schemas import (
    CountryConfig,
    EnterpriseStatus,
    HouseholdState,
    OutputRecord,
    PolicyType,
    SimulationConfig,
)


class TestHouseholdState:
    """Tests for HouseholdState schema."""

    def test_valid_household_state(self) -> None:
        """Test creating a valid HouseholdState."""
        state = HouseholdState(
            household_id="HH_00001",
            wave=1,
            assets=0.5,
            credit_access=1,
            enterprise_status=EnterpriseStatus.NO_ENTERPRISE,
            price_exposure=-0.1,
        )
        assert state.household_id == "HH_00001"
        assert state.wave == 1
        assert state.assets == 0.5

    def test_invalid_wave(self) -> None:
        """Test that wave must be positive."""
        with pytest.raises(ValidationError):
            HouseholdState(
                household_id="HH_00001",
                wave=0,  # Invalid
                assets=0.5,
                credit_access=1,
                enterprise_status=EnterpriseStatus.NO_ENTERPRISE,
                price_exposure=-0.1,
            )

    def test_invalid_credit_access(self) -> None:
        """Test that credit_access must be 0 or 1."""
        with pytest.raises(ValidationError):
            HouseholdState(
                household_id="HH_00001",
                wave=1,
                assets=0.5,
                credit_access=2,  # Invalid
                enterprise_status=EnterpriseStatus.NO_ENTERPRISE,
                price_exposure=-0.1,
            )

    def test_household_state_is_frozen(self) -> None:
        """Test that HouseholdState is immutable."""
        state = HouseholdState(
            household_id="HH_00001",
            wave=1,
            assets=0.5,
            credit_access=1,
            enterprise_status=EnterpriseStatus.NO_ENTERPRISE,
            price_exposure=-0.1,
        )
        with pytest.raises(ValidationError):
            state.assets = 1.0  # type: ignore


class TestSimulationConfig:
    """Tests for SimulationConfig schema."""

    def test_valid_config(self) -> None:
        """Test creating a valid SimulationConfig."""
        config = SimulationConfig(
            country="tanzania",
            scenario="baseline",
            seed=42,
            num_waves=4,
        )
        assert config.country == "tanzania"
        assert config.seed == 42

    def test_country_validation(self) -> None:
        """Test that country must be valid."""
        with pytest.raises(ValidationError):
            SimulationConfig(
                country="invalid",
                seed=42,
            )

    def test_country_case_insensitive(self) -> None:
        """Test that country validation is case-insensitive."""
        config = SimulationConfig(
            country="TANZANIA",
            seed=42,
        )
        assert config.country == "tanzania"

    def test_default_values(self) -> None:
        """Test default values."""
        config = SimulationConfig(country="ethiopia", seed=42)
        assert config.scenario == "baseline"
        assert config.num_waves == 4
        assert config.policy_type == PolicyType.NONE

    def test_seed_must_be_non_negative(self) -> None:
        """Test that seed must be >= 0."""
        with pytest.raises(ValidationError):
            SimulationConfig(country="tanzania", seed=-1)


class TestOutputRecord:
    """Tests for OutputRecord schema."""

    def test_valid_output_record(self) -> None:
        """Test creating a valid OutputRecord."""
        record = OutputRecord(
            household_id="HH_00001",
            wave=1,
            assets_index=0.5,
            credit_access=1,
            enterprise_status=1,
            price_exposure=-0.1,
            crop_count=3,
            land_area_ha=2.5,
            action_taken="NO_CHANGE",
            policy_applied=0,
        )
        assert record.household_id == "HH_00001"
        assert record.to_dict()["wave"] == 1

    def test_to_dict(self) -> None:
        """Test to_dict method."""
        record = OutputRecord(
            household_id="HH_00001",
            wave=1,
            assets_index=0.5,
            credit_access=1,
            enterprise_status=0,
            price_exposure=-0.1,
            crop_count=2,
            land_area_ha=1.0,
            action_taken="ENTER_ENTERPRISE",
            policy_applied=1,
        )
        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["household_id"] == "HH_00001"
        assert d["action_taken"] == "ENTER_ENTERPRISE"


class TestPolicyType:
    """Tests for PolicyType enum."""

    def test_policy_types(self) -> None:
        """Test policy type values."""
        assert PolicyType.NONE.value == "none"
        assert PolicyType.CREDIT_ACCESS.value == "credit_access"
        assert PolicyType.PRICE_SUPPORT.value == "price_support"
        assert PolicyType.ASSET_TRANSFER.value == "asset_transfer"


class TestEnterpriseStatus:
    """Tests for EnterpriseStatus enum."""

    def test_enterprise_status_values(self) -> None:
        """Test enterprise status values."""
        assert EnterpriseStatus.NO_ENTERPRISE.value == "no_enterprise"
        assert EnterpriseStatus.HAS_ENTERPRISE.value == "has_enterprise"


class TestCountryConfig:
    """Tests for CountryConfig schema."""

    def test_valid_config(self) -> None:
        """Test creating a valid CountryConfig."""
        config = CountryConfig(
            country="tanzania",
            waves=[1, 2, 3, 4],
            wave_years={1: 2008, 2: 2010, 3: 2012, 4: 2014},
        )
        assert config.country == "tanzania"
        assert len(config.waves) == 4
        assert config.wave_years[1] == 2008
