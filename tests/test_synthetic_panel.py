"""Tests for calibration-based synthetic panel generation.

Tests the new synthetic panel generator that uses calibration artifacts
with explicit transition dynamics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abm_enterprise.calibration.schemas import (
    CalibrationArtifact,
    CreditModelSpec,
    DistributionFamily,
    DistributionSpec,
    EnterpriseBaseline,
    TransitionRates,
)
from abm_enterprise.data.synthetic import (
    SyntheticPanelConfig,
    SyntheticPanelGenerator,
    TransitionConfig,
    generate_synthetic_panel,
    validate_synthetic_data,
    validate_synthetic_panel,
)
from abm_enterprise.utils.rng import set_seed


@pytest.fixture
def sample_calibration() -> CalibrationArtifact:
    """Create a sample calibration artifact for testing."""
    return CalibrationArtifact(
        country_source="tanzania",
        git_commit="test123",
        waves=[1, 2, 3, 4],
        n_households=500,
        n_observations=2000,
        assets_distribution=DistributionSpec(
            family=DistributionFamily.NORMAL,
            params={"mean": 0.0, "std": 1.0},
        ),
        shock_distribution=DistributionSpec(
            family=DistributionFamily.NORMAL,
            params={"mean": -0.05, "std": 0.15},
        ),
        credit_model=CreditModelSpec(
            coefficients={"assets_index": 0.5},
            intercept=-0.5,
            feature_names=["assets_index"],
        ),
        enterprise_baseline=EnterpriseBaseline(
            prevalence=0.30,
            prevalence_by_wave={1: 0.28, 2: 0.30, 3: 0.31, 4: 0.31},
            entry_rate=0.08,
            exit_rate=0.10,
        ),
        transition_rates=TransitionRates(
            enter_rate=0.08,
            exit_rate=0.10,
            stay_rate=0.82,
            enter_count=80,
            exit_count=100,
            stay_count=820,
        ),
    )


class TestTransitionConfig:
    """Tests for TransitionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TransitionConfig()

        assert config.rho_assets == 0.85
        assert config.lambda_enterprise_assets == 0.05
        assert config.credit_stickiness == 0.70

    def test_validation_passes(self):
        """Test validation with valid parameters."""
        config = TransitionConfig(
            rho_assets=0.9,
            credit_stickiness=0.8,
            asset_noise_sd=0.1,
        )
        config.validate()  # Should not raise

    def test_validation_fails_rho_out_of_range(self):
        """Test validation fails for rho out of [0,1]."""
        config = TransitionConfig(rho_assets=1.5)
        with pytest.raises(ValueError, match="rho_assets"):
            config.validate()

    def test_validation_fails_negative_noise(self):
        """Test validation fails for negative noise std."""
        config = TransitionConfig(asset_noise_sd=-0.1)
        with pytest.raises(ValueError, match="asset_noise_sd"):
            config.validate()


class TestSyntheticPanelGenerator:
    """Tests for SyntheticPanelGenerator."""

    def test_generate_correct_shape(self, sample_calibration):
        """Test generated panel has correct shape."""
        config = SyntheticPanelConfig(
            n_households=100,
            waves=[1, 2, 3, 4],
            seed=42,
        )
        generator = SyntheticPanelGenerator(sample_calibration, config)
        df = generator.generate()

        assert len(df) == 400  # 100 households × 4 waves
        assert df["household_id"].nunique() == 100
        assert df["wave"].nunique() == 4

    def test_generate_correct_columns(self, sample_calibration):
        """Test generated panel has required columns."""
        generator = SyntheticPanelGenerator(
            sample_calibration,
            SyntheticPanelConfig(n_households=50, seed=42),
        )
        df = generator.generate()

        required_cols = [
            "household_id",
            "wave",
            "assets_index",
            "credit_access",
            "enterprise_status",
            "price_exposure",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_generate_correct_dtypes(self, sample_calibration):
        """Test generated panel has correct data types."""
        generator = SyntheticPanelGenerator(
            sample_calibration,
            SyntheticPanelConfig(n_households=50, seed=42),
        )
        df = generator.generate()

        assert df["household_id"].dtype == object  # string
        assert df["wave"].dtype in [int, np.int64, np.int32]
        assert df["credit_access"].dtype in [int, np.int64, np.int32]
        assert df["enterprise_status"].dtype in [int, np.int64, np.int32]
        assert df["assets_index"].dtype == np.float64
        assert df["price_exposure"].dtype == np.float64

    def test_generate_binary_columns(self, sample_calibration):
        """Test binary columns are actually binary."""
        generator = SyntheticPanelGenerator(
            sample_calibration,
            SyntheticPanelConfig(n_households=100, seed=42),
        )
        df = generator.generate()

        assert df["credit_access"].isin([0, 1]).all()
        assert df["enterprise_status"].isin([0, 1]).all()

    def test_generate_balanced_panel(self, sample_calibration):
        """Test panel is balanced (same waves for all households)."""
        generator = SyntheticPanelGenerator(
            sample_calibration,
            SyntheticPanelConfig(n_households=100, seed=42),
        )
        df = generator.generate()

        # Each household should have exactly 4 observations
        obs_per_hh = df.groupby("household_id").size()
        assert (obs_per_hh == 4).all()

    def test_generate_reproducible(self, sample_calibration):
        """Test generation is reproducible with same seed."""
        config = SyntheticPanelConfig(n_households=50, seed=123)

        gen1 = SyntheticPanelGenerator(sample_calibration, config)
        df1 = gen1.generate()

        gen2 = SyntheticPanelGenerator(sample_calibration, config)
        df2 = gen2.generate()

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_results(self, sample_calibration):
        """Test different seeds produce different results."""
        config1 = SyntheticPanelConfig(n_households=50, seed=123)
        config2 = SyntheticPanelConfig(n_households=50, seed=456)

        gen1 = SyntheticPanelGenerator(sample_calibration, config1)
        df1 = gen1.generate()

        gen2 = SyntheticPanelGenerator(sample_calibration, config2)
        df2 = gen2.generate()

        # Assets should differ
        assert not np.allclose(
            df1["assets_index"].values,
            df2["assets_index"].values,
        )

    def test_asset_persistence(self, sample_calibration):
        """Test assets show persistence across waves."""
        config = SyntheticPanelConfig(
            n_households=500,
            seed=42,
            transition=TransitionConfig(rho_assets=0.9),
        )
        generator = SyntheticPanelGenerator(sample_calibration, config)
        df = generator.generate()

        # Compute within-household correlation between wave 1 and wave 4
        wave1_assets = df[df["wave"] == 1].set_index("household_id")["assets_index"]
        wave4_assets = df[df["wave"] == 4].set_index("household_id")["assets_index"]

        correlation = wave1_assets.corr(wave4_assets)
        assert correlation > 0.5, f"Asset persistence too low: {correlation}"

    def test_credit_correlated_with_assets(self, sample_calibration):
        """Test credit access is positively correlated with assets."""
        config = SyntheticPanelConfig(n_households=1000, seed=42)
        generator = SyntheticPanelGenerator(sample_calibration, config)
        df = generator.generate()

        # Higher assets should have more credit access
        high_assets = df[df["assets_index"] > 0]["credit_access"].mean()
        low_assets = df[df["assets_index"] < 0]["credit_access"].mean()

        assert high_assets > low_assets, (
            f"Credit not correlated with assets: "
            f"high={high_assets:.2f}, low={low_assets:.2f}"
        )

    def test_enterprise_prevalence_reasonable(self, sample_calibration):
        """Test enterprise prevalence matches calibration baseline."""
        config = SyntheticPanelConfig(n_households=1000, seed=42)
        generator = SyntheticPanelGenerator(sample_calibration, config)
        df = generator.generate()

        # Overall prevalence should be within reasonable range of baseline (0.30)
        actual_prevalence = df["enterprise_status"].mean()
        assert 0.10 < actual_prevalence < 0.50, (
            f"Enterprise prevalence out of range: {actual_prevalence}"
        )


class TestGenerateSyntheticPanel:
    """Tests for the convenience function."""

    def test_generate_synthetic_panel_basic(self, sample_calibration):
        """Test basic synthetic panel generation."""
        df = generate_synthetic_panel(
            sample_calibration,
            n_households=50,
            seed=42,
        )

        assert len(df) == 200  # 50 × 4
        assert df["household_id"].nunique() == 50

    def test_generate_with_custom_transition(self, sample_calibration):
        """Test generation with custom transition config."""
        custom_config = TransitionConfig(
            rho_assets=0.95,
            credit_stickiness=0.9,
        )

        df = generate_synthetic_panel(
            sample_calibration,
            n_households=100,
            config=custom_config,
            seed=42,
        )

        # With higher persistence, should see stronger correlation
        wave1 = df[df["wave"] == 1].set_index("household_id")["assets_index"]
        wave4 = df[df["wave"] == 4].set_index("household_id")["assets_index"]

        corr = wave1.corr(wave4)
        assert corr > 0.6, f"Expected high correlation with rho=0.95, got {corr}"


class TestValidateSyntheticData:
    """Tests for validation functions."""

    def test_validate_good_data(self, sample_calibration):
        """Test validation passes for good data."""
        df = generate_synthetic_panel(sample_calibration, n_households=50, seed=42)

        validations = validate_synthetic_data(df)

        assert all(validations.values()), f"Validation failed: {validations}"

    def test_validate_missing_column(self, sample_calibration):
        """Test validation fails for missing column."""
        df = generate_synthetic_panel(sample_calibration, n_households=50, seed=42)
        df = df.drop(columns=["assets_index"])

        validations = validate_synthetic_data(df)

        assert not validations["has_required_columns"]

    def test_validate_unbalanced_panel(self, sample_calibration):
        """Test validation fails for unbalanced panel."""
        df = generate_synthetic_panel(sample_calibration, n_households=50, seed=42)
        # Drop some rows to make unbalanced
        df = df[~((df["household_id"] == "SH_00000") & (df["wave"] == 4))]

        validations = validate_synthetic_data(df)

        assert not validations["balanced_panel"]


class TestValidateSyntheticPanel:
    """Tests for enhanced validation function."""

    def test_validate_with_expected_counts(self, sample_calibration):
        """Test validation with expected counts."""
        df = generate_synthetic_panel(sample_calibration, n_households=100, seed=42)

        validations = validate_synthetic_panel(df, n_households=100, n_waves=4)

        assert validations["correct_n_households"]
        assert validations["correct_n_waves"]

    def test_validate_wrong_household_count(self, sample_calibration):
        """Test validation fails with wrong household count."""
        df = generate_synthetic_panel(sample_calibration, n_households=50, seed=42)

        validations = validate_synthetic_panel(df, n_households=100)

        assert not validations["correct_n_households"]

    def test_validate_asset_persistence_check(self, sample_calibration):
        """Test asset persistence validation."""
        config = SyntheticPanelConfig(
            n_households=500,
            seed=42,
            transition=TransitionConfig(rho_assets=0.85),
        )
        generator = SyntheticPanelGenerator(sample_calibration, config)
        df = generator.generate()

        validations = validate_synthetic_panel(df)

        assert validations.get("asset_persistence", True), "Asset persistence check failed"


class TestSyntheticPanelHouseholdIds:
    """Tests for household ID generation."""

    def test_household_ids_unique(self, sample_calibration):
        """Test household IDs are unique."""
        df = generate_synthetic_panel(sample_calibration, n_households=100, seed=42)

        # Get unique households per wave
        for wave in df["wave"].unique():
            wave_df = df[df["wave"] == wave]
            assert wave_df["household_id"].nunique() == len(wave_df)

    def test_household_ids_prefix(self, sample_calibration):
        """Test household IDs have correct prefix."""
        df = generate_synthetic_panel(sample_calibration, n_households=50, seed=42)

        # All IDs should start with SH_
        assert all(hh.startswith("SH_") for hh in df["household_id"].unique())


class TestShockDistribution:
    """Tests for shock distribution handling."""

    def test_shocks_have_negative_mean(self, sample_calibration):
        """Test shocks follow calibrated distribution (negative mean)."""
        df = generate_synthetic_panel(sample_calibration, n_households=500, seed=42)

        # Mean shock should be close to calibrated mean (-0.05)
        mean_shock = df["price_exposure"].mean()
        assert abs(mean_shock + 0.05) < 0.05, f"Mean shock: {mean_shock}"

    def test_per_wave_shock_distribution(self):
        """Test per-wave shock distributions are used when available."""
        calibration = CalibrationArtifact(
            country_source="tanzania",
            git_commit="test",
            waves=[1, 2, 3],
            n_households=100,
            n_observations=300,
            assets_distribution=DistributionSpec(
                family=DistributionFamily.NORMAL,
                params={"mean": 0.0, "std": 1.0},
            ),
            shock_distribution=DistributionSpec(
                family=DistributionFamily.NORMAL,
                params={"mean": 0.0, "std": 0.15},
            ),
            shock_distribution_by_wave={
                1: DistributionSpec(
                    family=DistributionFamily.NORMAL,
                    params={"mean": 0.0, "std": 0.10},
                ),
                2: DistributionSpec(
                    family=DistributionFamily.NORMAL,
                    params={"mean": -0.20, "std": 0.10},  # Large negative shock
                ),
                3: DistributionSpec(
                    family=DistributionFamily.NORMAL,
                    params={"mean": 0.05, "std": 0.10},  # Positive
                ),
            },
            credit_model=CreditModelSpec(
                coefficients={"assets_index": 0.5},
                intercept=0.0,
                feature_names=["assets_index"],
            ),
            enterprise_baseline=EnterpriseBaseline(
                prevalence=0.30,
                entry_rate=0.10,
                exit_rate=0.10,
            ),
            transition_rates=TransitionRates(
                enter_rate=0.10,
                exit_rate=0.10,
                stay_rate=0.80,
                enter_count=100,
                exit_count=100,
                stay_count=800,
            ),
        )

        df = generate_synthetic_panel(calibration, n_households=500, seed=42)

        # Wave 2 should have most negative shocks
        wave1_mean = df[df["wave"] == 1]["price_exposure"].mean()
        wave2_mean = df[df["wave"] == 2]["price_exposure"].mean()
        wave3_mean = df[df["wave"] == 3]["price_exposure"].mean()

        assert wave2_mean < wave1_mean, "Wave 2 should have more negative shocks"
        assert wave2_mean < wave3_mean, "Wave 2 should have most negative shocks"
