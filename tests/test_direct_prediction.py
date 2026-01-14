"""Tests for direct prediction evaluation module.

Tests transition dataset construction, baseline training,
and metric computation.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from abm_enterprise.eval.direct_prediction import (
    TransitionLabel,
    compute_transition_label,
    build_transition_dataset,
    state_from_row,
    action_to_transition,
)
from abm_enterprise.eval.baselines import (
    LogisticBaseline,
    RandomForestBaseline,
    GradientBoostingBaseline,
    train_all_baselines,
    prepare_features,
)
from abm_enterprise.eval.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_subgroup_metrics,
    compare_models,
)
from abm_enterprise.policies.base import Action
from abm_enterprise.data.schemas import HouseholdState


class TestTransitionLabel:
    """Tests for transition label computation."""

    def test_enter_transition(self):
        """Test ENTER label when going from 0 to 1."""
        label = compute_transition_label(0, 1)
        assert label == TransitionLabel.ENTER

    def test_exit_transition(self):
        """Test EXIT label when going from 1 to 0."""
        label = compute_transition_label(1, 0)
        assert label == TransitionLabel.EXIT

    def test_stay_in_enterprise(self):
        """Test STAY label when staying in enterprise."""
        label = compute_transition_label(1, 1)
        assert label == TransitionLabel.STAY

    def test_stay_out_of_enterprise(self):
        """Test STAY label when staying out of enterprise."""
        label = compute_transition_label(0, 0)
        assert label == TransitionLabel.STAY


class TestActionToTransition:
    """Tests for action to transition label mapping."""

    def test_enter_action_when_not_in_enterprise(self):
        """Test ENTER_ENTERPRISE action from non-enterprise state."""
        label = action_to_transition(Action.ENTER_ENTERPRISE, False)
        assert label == TransitionLabel.ENTER

    def test_enter_action_when_already_in_enterprise(self):
        """Test ENTER_ENTERPRISE action when already in enterprise."""
        label = action_to_transition(Action.ENTER_ENTERPRISE, True)
        assert label == TransitionLabel.STAY

    def test_exit_action_when_in_enterprise(self):
        """Test EXIT_ENTERPRISE action from enterprise state."""
        label = action_to_transition(Action.EXIT_ENTERPRISE, True)
        assert label == TransitionLabel.EXIT

    def test_exit_action_when_not_in_enterprise(self):
        """Test EXIT_ENTERPRISE action when not in enterprise."""
        label = action_to_transition(Action.EXIT_ENTERPRISE, False)
        assert label == TransitionLabel.STAY

    def test_no_change_action(self):
        """Test NO_CHANGE action."""
        label = action_to_transition(Action.NO_CHANGE, True)
        assert label == TransitionLabel.STAY
        label = action_to_transition(Action.NO_CHANGE, False)
        assert label == TransitionLabel.STAY


class TestStateFromRow:
    """Tests for row to HouseholdState conversion."""

    def test_state_conversion(self):
        """Test basic state conversion."""
        from abm_enterprise.data.schemas import EnterpriseStatus

        row = pd.Series({
            "household_id": "HH_001",
            "wave_t": 1,
            "assets_index": 0.5,
            "credit_access": 1,
            "enterprise_status": 0,
            "price_exposure": -0.1,
        })

        state = state_from_row(row)

        assert state.household_id == "HH_001"
        assert state.wave == 1
        assert state.assets == 0.5
        assert state.credit_access == 1
        assert state.enterprise_status == EnterpriseStatus.NO_ENTERPRISE
        assert state.price_exposure == -0.1

    def test_state_conversion_with_enterprise(self):
        """Test state conversion when in enterprise."""
        from abm_enterprise.data.schemas import EnterpriseStatus

        row = pd.Series({
            "household_id": "HH_002",
            "wave_t": 2,
            "assets_index": 1.5,
            "credit_access": 1,
            "enterprise_status": 1,
            "price_exposure": 0.05,
        })

        state = state_from_row(row)

        assert state.enterprise_status == EnterpriseStatus.HAS_ENTERPRISE


class TestBuildTransitionDataset:
    """Tests for transition dataset construction."""

    @pytest.fixture
    def mock_household_targets(self):
        """Create mock household targets data."""
        return pd.DataFrame({
            "household_id": ["HH_001"] * 4 + ["HH_002"] * 4,
            "wave": [1, 2, 3, 4] * 2,
            "assets_index": [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            "credit_access": [0, 0, 1, 1, 1, 1, 0, 0],
            "enterprise_indicator": [0, 1, 1, 1, 1, 0, 0, 1],  # HH1: enter once, HH2: exit then enter
            "price_exposure": [-0.05, -0.10, 0.05, 0.00, -0.15, -0.05, 0.00, -0.10],
        })

    def test_build_from_parquet(self, mock_household_targets):
        """Test building dataset from parquet file."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            country_dir = tmpdir / "tanzania"
            country_dir.mkdir()
            mock_household_targets.to_parquet(country_dir / "household_targets.parquet")

            df = build_transition_dataset("tanzania", tmpdir)

            # 2 households × 3 transitions (wave pairs) = 6 rows
            assert len(df) == 6
            assert "transition" in df.columns
            assert all(df["transition"].isin(["ENTER", "EXIT", "STAY"]))

    def test_transition_labels_correct(self, mock_household_targets):
        """Test that transition labels are correctly assigned."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            country_dir = tmpdir / "tanzania"
            country_dir.mkdir()
            mock_household_targets.to_parquet(country_dir / "household_targets.parquet")

            df = build_transition_dataset("tanzania", tmpdir)

            # HH_001: 0→1 (ENTER), 1→1 (STAY), 1→1 (STAY)
            hh1 = df[df["household_id"] == "HH_001"].sort_values("wave_t")
            assert hh1["transition"].tolist() == ["ENTER", "STAY", "STAY"]

            # HH_002: 1→0 (EXIT), 0→0 (STAY), 0→1 (ENTER)
            hh2 = df[df["household_id"] == "HH_002"].sort_values("wave_t")
            assert hh2["transition"].tolist() == ["EXIT", "STAY", "ENTER"]

    def test_file_not_found(self):
        """Test error when file not found."""
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                build_transition_dataset("nonexistent", tmpdir)

    def test_insufficient_waves(self, mock_household_targets):
        """Test error with insufficient waves."""
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            country_dir = tmpdir / "tanzania"
            country_dir.mkdir()
            # Only one wave
            single_wave = mock_household_targets[mock_household_targets["wave"] == 1]
            single_wave.to_parquet(country_dir / "household_targets.parquet")

            with pytest.raises(ValueError, match="at least 2 waves"):
                build_transition_dataset("tanzania", tmpdir)


class TestLogisticBaseline:
    """Tests for logistic regression baseline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 4)
        # Generate labels with some structure
        probs = np.exp(0.5 * X[:, 0]) / (1 + np.exp(0.5 * X[:, 0]))
        y = np.where(probs > 0.7, "ENTER", np.where(probs < 0.3, "EXIT", "STAY"))
        return X, y

    def test_fit_predict(self, sample_data):
        """Test basic fit and predict."""
        X, y = sample_data
        model = LogisticBaseline()
        model.fit(X, y)

        preds = model.predict(X)

        assert len(preds) == len(y)
        assert all(p in ["ENTER", "EXIT", "STAY"] for p in preds)

    def test_predict_proba(self, sample_data):
        """Test probability predictions."""
        X, y = sample_data
        model = LogisticBaseline()
        model.fit(X, y)

        probs = model.predict_proba(X)

        assert probs.shape == (len(y), 3)  # 3 classes
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_coefficients(self, sample_data):
        """Test coefficient extraction."""
        X, y = sample_data
        model = LogisticBaseline()
        model.fit(X, y)

        coefs = model.get_coefficients()

        assert "coef" in coefs
        assert "intercept" in coefs
        assert coefs["coef"].shape[1] == 4  # 4 features


class TestRandomForestBaseline:
    """Tests for random forest baseline."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 4)
        y = np.array(["ENTER", "EXIT", "STAY"] * (n // 3 + 1))[:n]
        return X, y

    def test_fit_predict(self, sample_data):
        """Test basic fit and predict."""
        X, y = sample_data
        model = RandomForestBaseline(n_estimators=10)
        model.fit(X, y)

        preds = model.predict(X)

        assert len(preds) == len(y)

    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        model = RandomForestBaseline(n_estimators=10)
        model.fit(X, y)

        importance = model.get_feature_importance(["f1", "f2", "f3", "f4"])

        assert len(importance) == 4
        assert "importance" in importance.columns


class TestGradientBoostingBaseline:
    """Tests for gradient boosting baseline."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 4)
        y = np.array(["ENTER", "EXIT", "STAY"] * (n // 3 + 1))[:n]
        return X, y

    def test_fit_predict(self, sample_data):
        """Test basic fit and predict."""
        X, y = sample_data
        model = GradientBoostingBaseline(n_estimators=10)
        model.fit(X, y)

        preds = model.predict(X)

        assert len(preds) == len(y)


class TestTrainAllBaselines:
    """Tests for training multiple baselines."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 4)
        y = np.array(["ENTER", "EXIT", "STAY"] * (n // 3 + 1))[:n]
        return X, y

    def test_train_all_default(self, sample_data):
        """Test training all default baselines."""
        X, y = sample_data
        models = train_all_baselines(X, y)

        assert "logit" in models
        assert "rf" in models
        assert "gbm" in models

    def test_train_specific_baselines(self, sample_data):
        """Test training specific baselines."""
        X, y = sample_data
        models = train_all_baselines(X, y, baselines=["logit", "rf"])

        assert "logit" in models
        assert "rf" in models
        assert "gbm" not in models


class TestPrepareFeatures:
    """Tests for feature preparation."""

    def test_prepare_features(self):
        """Test feature matrix extraction."""
        df = pd.DataFrame({
            "assets_index": [0.1, 0.2, 0.3],
            "credit_access": [0, 1, 1],
            "enterprise_status": [0, 1, 0],
            "price_exposure": [-0.1, 0.0, 0.1],
            "other_col": [1, 2, 3],
        })

        X, cols = prepare_features(df)

        assert X.shape == (3, 4)
        assert len(cols) == 4
        assert "other_col" not in cols

    def test_prepare_features_missing_col(self):
        """Test with missing column."""
        df = pd.DataFrame({
            "assets_index": [0.1, 0.2, 0.3],
            "credit_access": [0, 1, 1],
        })

        X, cols = prepare_features(df)

        # Should use available columns
        assert X.shape == (3, 2)


class TestClassificationMetrics:
    """Tests for classification metrics computation."""

    @pytest.fixture
    def predictions(self):
        """Create sample predictions."""
        y_true = np.array(["ENTER", "ENTER", "EXIT", "EXIT", "STAY", "STAY", "STAY", "STAY"])
        y_pred = np.array(["ENTER", "STAY", "EXIT", "STAY", "STAY", "STAY", "ENTER", "EXIT"])
        return y_true, y_pred

    def test_compute_metrics(self, predictions):
        """Test basic metric computation."""
        y_true, y_pred = predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert isinstance(metrics, ClassificationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.balanced_accuracy <= 1
        assert 0 <= metrics.macro_f1 <= 1
        assert metrics.n_samples == len(y_true)

    def test_per_class_metrics(self, predictions):
        """Test per-class metric computation."""
        y_true, y_pred = predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert "ENTER" in metrics.per_class_f1
        assert "EXIT" in metrics.per_class_f1
        assert "STAY" in metrics.per_class_f1

    def test_confusion_matrix_shape(self, predictions):
        """Test confusion matrix has correct shape."""
        y_true, y_pred = predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        assert metrics.confusion_matrix.shape == (3, 3)

    def test_to_dict(self, predictions):
        """Test serialization to dict."""
        y_true, y_pred = predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        d = metrics.to_dict()

        assert "accuracy" in d
        assert "balanced_accuracy" in d
        assert "macro_f1" in d
        assert "confusion_matrix" in d

    def test_summary(self, predictions):
        """Test summary string generation."""
        y_true, y_pred = predictions
        metrics = compute_classification_metrics(y_true, y_pred)

        summary = metrics.summary()

        assert "Accuracy" in summary
        assert "F1" in summary


class TestConfusionMatrix:
    """Tests for confusion matrix computation."""

    def test_confusion_matrix_dataframe(self):
        """Test confusion matrix as DataFrame."""
        y_true = ["ENTER", "EXIT", "STAY", "ENTER"]
        y_pred = ["ENTER", "EXIT", "EXIT", "STAY"]

        cm = compute_confusion_matrix(y_true, y_pred)

        assert isinstance(cm, pd.DataFrame)
        assert cm.shape == (3, 3)
        assert cm.index.name == "True"
        assert cm.columns.name == "Predicted"

    def test_normalized_confusion_matrix(self):
        """Test normalized confusion matrix."""
        y_true = ["ENTER", "ENTER", "EXIT", "EXIT", "STAY", "STAY"]
        y_pred = ["ENTER", "EXIT", "EXIT", "EXIT", "STAY", "ENTER"]

        cm = compute_confusion_matrix(y_true, y_pred, normalize="true")

        # Each row should sum to 1 (when there are samples for that class)
        row_sums = cm.sum(axis=1)
        # All classes have samples, so all rows should sum to 1
        assert np.allclose(row_sums, 1.0)


class TestSubgroupMetrics:
    """Tests for subgroup metric computation."""

    @pytest.fixture
    def subgroup_data(self):
        """Create sample data with subgroups."""
        return pd.DataFrame({
            "true_label": ["ENTER", "ENTER", "EXIT", "EXIT", "STAY", "STAY"],
            "pred_label": ["ENTER", "STAY", "EXIT", "EXIT", "STAY", "ENTER"],
            "group": ["A", "A", "A", "B", "B", "B"],
        })

    def test_compute_subgroup_metrics(self, subgroup_data):
        """Test subgroup metric computation."""
        metrics = compute_subgroup_metrics(
            subgroup_data,
            y_true_col="true_label",
            y_pred_col="pred_label",
            subgroup_cols=["group"],
        )

        assert len(metrics) == 2  # Two groups
        assert "accuracy" in metrics.columns
        assert "n_samples" in metrics.columns


class TestCompareModels:
    """Tests for model comparison."""

    @pytest.fixture
    def comparison_data(self):
        """Create sample multi-model predictions."""
        return pd.DataFrame({
            "true_label": ["ENTER", "EXIT", "STAY", "ENTER", "EXIT", "STAY"],
            "model1_transition": ["ENTER", "EXIT", "STAY", "ENTER", "STAY", "STAY"],
            "model2_transition": ["STAY", "EXIT", "STAY", "ENTER", "EXIT", "ENTER"],
        })

    def test_compare_models(self, comparison_data):
        """Test model comparison."""
        comparison = compare_models(
            comparison_data,
            y_true_col="true_label",
            model_cols=["model1_transition", "model2_transition"],
        )

        assert len(comparison) == 2
        assert "model" in comparison.columns
        assert "accuracy" in comparison.columns
        assert "macro_f1" in comparison.columns

    def test_comparison_sorted_by_f1(self, comparison_data):
        """Test comparison is sorted by F1."""
        comparison = compare_models(
            comparison_data,
            y_true_col="true_label",
            model_cols=["model1_transition", "model2_transition"],
        )

        # Should be sorted by macro_f1 descending
        assert comparison.iloc[0]["macro_f1"] >= comparison.iloc[1]["macro_f1"]
