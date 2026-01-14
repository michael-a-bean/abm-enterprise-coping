"""Tests for calibration subsystem.

Tests distribution fitting, credit model fitting, and calibration artifact creation.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from abm_enterprise.calibration.schemas import (
    CalibrationArtifact,
    CalibrationManifest,
    CreditModelSpec,
    DistributionFamily,
    DistributionSpec,
    EnterpriseBaseline,
    StandardizationMethod,
    TransitionRates,
)
from abm_enterprise.calibration.fit import (
    compute_enterprise_baseline,
    compute_transition_rates,
    fit_asset_distribution,
    fit_credit_model,
    fit_shock_distribution,
)


class TestDistributionSpec:
    """Tests for DistributionSpec schema."""

    def test_normal_distribution_spec(self):
        """Test creating normal distribution spec."""
        spec = DistributionSpec(
            family=DistributionFamily.NORMAL,
            params={"mean": 0.0, "std": 1.0},
            standardization=StandardizationMethod.ZSCORE,
            raw_stats={"mean": 0.0, "std": 1.0, "n": 100},
        )

        assert spec.family == DistributionFamily.NORMAL
        assert spec.params["mean"] == 0.0
        assert spec.params["std"] == 1.0

    def test_get_scipy_params_normal(self):
        """Test conversion to scipy params for normal."""
        spec = DistributionSpec(
            family=DistributionFamily.NORMAL,
            params={"mean": 1.5, "std": 0.8},
        )

        scipy_params = spec.get_scipy_params()
        assert scipy_params["loc"] == 1.5
        assert scipy_params["scale"] == 0.8

    def test_get_scipy_params_t(self):
        """Test conversion to scipy params for t-distribution."""
        spec = DistributionSpec(
            family=DistributionFamily.T,
            params={"df": 5.0, "loc": 0.0, "scale": 1.0},
        )

        scipy_params = spec.get_scipy_params()
        assert scipy_params["df"] == 5.0
        assert scipy_params["loc"] == 0.0
        assert scipy_params["scale"] == 1.0

    def test_empty_params_rejected(self):
        """Test that empty params are rejected."""
        with pytest.raises(ValueError):
            DistributionSpec(
                family=DistributionFamily.NORMAL,
                params={},
            )


class TestCreditModelSpec:
    """Tests for CreditModelSpec schema."""

    def test_credit_model_spec_creation(self):
        """Test creating credit model spec."""
        spec = CreditModelSpec(
            type="logistic",
            coefficients={"assets_index": 0.5},
            intercept=-1.0,
            feature_names=["assets_index"],
            model_metrics={"accuracy": 0.75, "auc": 0.80},
        )

        assert spec.type == "logistic"
        assert spec.coefficients["assets_index"] == 0.5
        assert spec.intercept == -1.0

    def test_predict_proba(self):
        """Test probability prediction."""
        spec = CreditModelSpec(
            type="logistic",
            coefficients={"assets_index": 1.0},
            intercept=0.0,
            feature_names=["assets_index"],
        )

        # At assets=0, should be 0.5
        prob = spec.predict_proba({"assets_index": 0.0})
        assert abs(prob - 0.5) < 0.001

        # High assets should have high prob
        prob_high = spec.predict_proba({"assets_index": 3.0})
        assert prob_high > 0.9

        # Low assets should have low prob
        prob_low = spec.predict_proba({"assets_index": -3.0})
        assert prob_low < 0.1


class TestTransitionRates:
    """Tests for TransitionRates schema."""

    def test_transition_rates_creation(self):
        """Test creating transition rates."""
        rates = TransitionRates(
            enter_rate=0.10,
            exit_rate=0.15,
            stay_rate=0.75,
            enter_count=100,
            exit_count=150,
            stay_count=750,
        )

        assert rates.enter_rate == 0.10
        assert rates.exit_rate == 0.15
        assert rates.stay_rate == 0.75

    def test_rate_bounds(self):
        """Test that rates are bounded 0-1."""
        with pytest.raises(ValueError):
            TransitionRates(
                enter_rate=1.5,  # Invalid
                exit_rate=0.15,
                stay_rate=0.75,
                enter_count=100,
                exit_count=150,
                stay_count=750,
            )


class TestFitAssetDistribution:
    """Tests for asset distribution fitting."""

    def test_fit_normal_distribution(self):
        """Test fitting normal distribution to asset data."""
        np.random.seed(42)
        assets = pd.Series(np.random.normal(0, 1, 1000))

        spec = fit_asset_distribution(assets, family=DistributionFamily.NORMAL)

        assert spec.family == DistributionFamily.NORMAL
        assert "mean" in spec.params
        assert "std" in spec.params
        assert abs(spec.params["mean"]) < 0.1  # Close to 0
        assert abs(spec.params["std"] - 1.0) < 0.1  # Close to 1

    def test_fit_t_distribution(self):
        """Test fitting t-distribution to asset data."""
        np.random.seed(42)
        # Generate data with heavier tails
        assets = pd.Series(np.random.standard_t(df=5, size=1000))

        spec = fit_asset_distribution(assets, family=DistributionFamily.T)

        assert spec.family == DistributionFamily.T
        assert "df" in spec.params
        assert "loc" in spec.params
        assert "scale" in spec.params
        assert spec.params["df"] > 1  # Valid degrees of freedom

    def test_raw_stats_captured(self):
        """Test that raw statistics are captured."""
        np.random.seed(42)
        assets = pd.Series(np.random.normal(2.0, 0.5, 500))

        spec = fit_asset_distribution(assets)

        assert "mean" in spec.raw_stats
        assert "std" in spec.raw_stats
        assert "min" in spec.raw_stats
        assert "max" in spec.raw_stats
        assert "n" in spec.raw_stats
        assert spec.raw_stats["n"] == 500


class TestFitShockDistribution:
    """Tests for shock distribution fitting."""

    def test_fit_pooled_shock_distribution(self):
        """Test fitting pooled shock distribution."""
        np.random.seed(42)
        df = pd.DataFrame({
            "wave": [1] * 250 + [2] * 250 + [3] * 250 + [4] * 250,
            "price_exposure": np.random.normal(-0.05, 0.15, 1000),
        })

        pooled, by_wave = fit_shock_distribution(df, by_wave=False)

        assert pooled.family == DistributionFamily.NORMAL
        assert abs(pooled.params["mean"] + 0.05) < 0.05  # Close to -0.05
        assert abs(pooled.params["std"] - 0.15) < 0.05  # Close to 0.15
        assert by_wave is None

    def test_fit_by_wave_shock_distribution(self):
        """Test fitting per-wave shock distributions."""
        np.random.seed(42)
        df = pd.DataFrame({
            "wave": [1] * 250 + [2] * 250 + [3] * 250 + [4] * 250,
            "price_exposure": np.concatenate([
                np.random.normal(0.0, 0.10, 250),   # Wave 1
                np.random.normal(-0.15, 0.12, 250), # Wave 2 (shock)
                np.random.normal(-0.05, 0.11, 250), # Wave 3
                np.random.normal(0.05, 0.10, 250),  # Wave 4
            ]),
        })

        pooled, by_wave = fit_shock_distribution(df, by_wave=True)

        assert pooled is not None
        assert by_wave is not None
        assert len(by_wave) == 4
        assert all(wave in by_wave for wave in [1, 2, 3, 4])

        # Wave 2 should have most negative mean
        assert by_wave[2].params["mean"] < by_wave[1].params["mean"]


class TestFitCreditModel:
    """Tests for credit model fitting."""

    def test_fit_logistic_credit_model(self):
        """Test fitting logistic credit model."""
        np.random.seed(42)
        n = 500
        assets = np.random.normal(0, 1, n)
        # Credit positively correlated with assets
        credit_prob = 1 / (1 + np.exp(-0.5 * assets))
        credit = (np.random.random(n) < credit_prob).astype(int)

        df = pd.DataFrame({
            "assets_index": assets,
            "credit_access": credit,
        })

        spec = fit_credit_model(df)

        assert spec.type == "logistic"
        assert "assets_index" in spec.coefficients
        assert spec.coefficients["assets_index"] > 0  # Positive relationship
        assert "accuracy" in spec.model_metrics

    def test_credit_model_with_multiple_features(self):
        """Test credit model with multiple features."""
        np.random.seed(42)
        n = 500
        assets = np.random.normal(0, 1, n)
        region = np.random.choice([0, 1], n)
        credit = ((assets + 0.5 * region) > 0).astype(int)

        df = pd.DataFrame({
            "assets_index": assets,
            "region": region,
            "credit_access": credit,
        })

        spec = fit_credit_model(
            df,
            feature_cols=["assets_index", "region"],
        )

        assert len(spec.feature_names) == 2
        assert "assets_index" in spec.coefficients
        assert "region" in spec.coefficients


class TestComputeTransitionRates:
    """Tests for transition rate computation."""

    def test_compute_transition_rates(self):
        """Test computing transition rates from panel data."""
        df = pd.DataFrame({
            "household_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "wave": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "enterprise_status": [0, 1, 1, 1, 0, 0, 0, 0, 1],
        })

        rates = compute_transition_rates(df)

        # A: 0->1 (enter), 1->1 (stay)
        # B: 1->0 (exit), 0->0 (stay)
        # C: 0->0 (stay), 0->1 (enter)
        assert rates.enter_count == 2  # A wave 2, C wave 3
        assert rates.exit_count == 1  # B wave 2
        assert rates.stay_count == 3  # A wave 3, B wave 3, C wave 2

    def test_empty_transitions(self):
        """Test with only first wave (no transitions)."""
        df = pd.DataFrame({
            "household_id": ["A", "B", "C"],
            "wave": [1, 1, 1],
            "enterprise_status": [0, 1, 0],
        })

        rates = compute_transition_rates(df)

        assert rates.enter_count == 0
        assert rates.exit_count == 0
        assert rates.stay_count == 0


class TestComputeEnterpriseBaseline:
    """Tests for enterprise baseline computation."""

    def test_compute_enterprise_baseline(self):
        """Test computing enterprise baseline statistics."""
        df = pd.DataFrame({
            "household_id": ["A", "A", "B", "B", "C", "C"],
            "wave": [1, 2, 1, 2, 1, 2],
            "enterprise_status": [0, 1, 1, 1, 0, 0],
        })

        baseline = compute_enterprise_baseline(df)

        assert baseline.prevalence == 3 / 6  # 3 out of 6 observations
        assert baseline.prevalence_by_wave[1] == 1 / 3
        assert baseline.prevalence_by_wave[2] == 2 / 3


class TestCalibrationArtifact:
    """Tests for CalibrationArtifact schema."""

    def test_artifact_creation(self):
        """Test creating calibration artifact."""
        artifact = CalibrationArtifact(
            country_source="tanzania",
            git_commit="abc123",
            waves=[1, 2, 3, 4],
            n_households=100,
            n_observations=400,
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
                intercept=-1.0,
                feature_names=["assets_index"],
            ),
            enterprise_baseline=EnterpriseBaseline(
                prevalence=0.30,
                prevalence_by_wave={1: 0.28, 2: 0.30, 3: 0.32, 4: 0.30},
                entry_rate=0.08,
                exit_rate=0.12,
            ),
            transition_rates=TransitionRates(
                enter_rate=0.08,
                exit_rate=0.12,
                stay_rate=0.80,
                enter_count=80,
                exit_count=120,
                stay_count=800,
            ),
        )

        assert artifact.country_source == "tanzania"
        assert len(artifact.waves) == 4

    def test_artifact_save_load(self):
        """Test saving and loading artifact."""
        artifact = CalibrationArtifact(
            country_source="tanzania",
            git_commit="abc123",
            waves=[1, 2, 3, 4],
            n_households=100,
            n_observations=400,
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
                intercept=-1.0,
                feature_names=["assets_index"],
            ),
            enterprise_baseline=EnterpriseBaseline(
                prevalence=0.30,
                entry_rate=0.08,
                exit_rate=0.12,
            ),
            transition_rates=TransitionRates(
                enter_rate=0.08,
                exit_rate=0.12,
                stay_rate=0.80,
                enter_count=80,
                exit_count=120,
                stay_count=800,
            ),
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            artifact.save(str(path))

            assert path.exists()

            loaded = CalibrationArtifact.load(str(path))

            assert loaded.country_source == artifact.country_source
            assert loaded.waves == artifact.waves
            assert loaded.assets_distribution.params == artifact.assets_distribution.params


class TestCalibrationManifest:
    """Tests for CalibrationManifest schema."""

    def test_manifest_creation_and_save(self):
        """Test creating and saving manifest."""
        manifest = CalibrationManifest(
            calibration_id="test123",
            git_commit="abc123",
            country="tanzania",
            input_data_path="/data/processed/tanzania/derived/household_targets.parquet",
            input_data_hash="d41d8cd98f00b204e9800998ecf8427e",
            output_artifact_path="/artifacts/calibration/tanzania/calibration.json",
            parameters_used={"asset_family": "normal"},
        )

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            manifest.save(str(path))

            assert path.exists()

            import json
            with open(path) as f:
                data = json.load(f)

            assert data["calibration_id"] == "test123"
            assert data["country"] == "tanzania"
