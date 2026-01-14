"""ML baseline classifiers for transition prediction.

Provides logistic regression, random forest, and gradient boosting
baselines for comparison with LLM predictions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from abm_enterprise.utils.logging import get_logger

logger = get_logger(__name__)


# Default feature columns for training
DEFAULT_FEATURE_COLS = [
    "assets_index",
    "credit_access",
    "enterprise_status",
    "price_exposure",
]


class BaselineModel(ABC):
    """Abstract base class for baseline models."""

    name: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        """Fit the model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        pass

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """Return the class labels."""
        pass

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Args:
            path: Output path.
        """
        joblib.dump(self, path)
        logger.info(f"Model saved", path=str(path), model=self.name)

    @classmethod
    def load(cls, path: Path | str) -> "BaselineModel":
        """Load model from disk.

        Args:
            path: Model path.

        Returns:
            Loaded model instance.
        """
        return joblib.load(path)


@dataclass
class LogisticBaseline(BaselineModel):
    """Multinomial logistic regression baseline.

    Attributes:
        C: Regularization strength (inverse).
        max_iter: Maximum iterations.
        scale_features: Whether to standardize features.
    """

    name: str = "logit"
    C: float = 1.0
    max_iter: int = 1000
    scale_features: bool = True
    _model: LogisticRegression | None = field(default=None, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        """Fit logistic regression model."""
        if self.scale_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
        self._model.fit(X, y)

        logger.info(
            "Logistic baseline trained",
            n_samples=len(y),
            classes=list(self._model.classes_),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.classes_

    def get_coefficients(self) -> dict[str, np.ndarray]:
        """Get model coefficients for interpretability.

        Returns:
            Dictionary with coefficients and intercept.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return {
            "coef": self._model.coef_,
            "intercept": self._model.intercept_,
            "classes": self._model.classes_,
        }


@dataclass
class RandomForestBaseline(BaselineModel):
    """Random forest baseline.

    Attributes:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
    """

    name: str = "rf"
    n_estimators: int = 100
    max_depth: int | None = 10
    min_samples_leaf: int = 5
    _model: RandomForestClassifier | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        """Fit random forest model."""
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)

        logger.info(
            "Random forest baseline trained",
            n_samples=len(y),
            n_estimators=self.n_estimators,
            classes=list(self._model.classes_),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.classes_

    def get_feature_importance(
        self,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Get feature importances.

        Args:
            feature_names: Optional feature names.

        Returns:
            DataFrame with feature importances.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted.")

        importances = self._model.feature_importances_

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        return pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)


@dataclass
class GradientBoostingBaseline(BaselineModel):
    """Gradient boosting baseline (sklearn HistGradientBoosting).

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
    """

    name: str = "gbm"
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    _model: GradientBoostingClassifier | None = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingBaseline":
        """Fit gradient boosting model."""
        self._model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        self._model.fit(X, y)

        logger.info(
            "Gradient boosting baseline trained",
            n_samples=len(y),
            n_estimators=self.n_estimators,
            classes=list(self._model.classes_),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.classes_


def train_baseline(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    **kwargs,
) -> BaselineModel:
    """Train a single baseline model.

    Args:
        name: Model name ('logit', 'rf', 'gbm').
        X_train: Training features.
        y_train: Training labels.
        **kwargs: Model-specific parameters.

    Returns:
        Trained baseline model.

    Raises:
        ValueError: If unknown model name.
    """
    if name == "logit":
        model = LogisticBaseline(**kwargs)
    elif name == "rf":
        model = RandomForestBaseline(**kwargs)
    elif name == "gbm":
        model = GradientBoostingBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown baseline model: {name}")

    return model.fit(X_train, y_train)


def train_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    baselines: list[str] | None = None,
    configs: dict[str, dict] | None = None,
) -> dict[str, BaselineModel]:
    """Train all specified baseline models.

    Args:
        X_train: Training features.
        y_train: Training labels.
        baselines: List of baseline names (default: ['logit', 'rf', 'gbm']).
        configs: Model-specific configurations.

    Returns:
        Dictionary of {name: trained_model}.
    """
    if baselines is None:
        baselines = ["logit", "rf", "gbm"]

    if configs is None:
        configs = {}

    models = {}
    for name in baselines:
        config = configs.get(name, {})
        models[name] = train_baseline(name, X_train, y_train, **config)

    logger.info(
        "All baselines trained",
        models=list(models.keys()),
        n_train=len(y_train),
    )

    return models


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Prepare feature matrix from dataframe.

    Args:
        df: DataFrame with feature columns.
        feature_cols: Columns to use (default: DEFAULT_FEATURE_COLS).

    Returns:
        Tuple of (feature_matrix, feature_names).
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    # Use only columns that exist
    available_cols = [c for c in feature_cols if c in df.columns]

    if not available_cols:
        raise ValueError(f"No feature columns found. Expected: {feature_cols}")

    return df[available_cols].values, available_cols


def save_baselines(
    baselines: dict[str, BaselineModel],
    output_dir: Path | str,
) -> dict[str, Path]:
    """Save all baseline models to disk.

    Args:
        baselines: Dictionary of trained models.
        output_dir: Output directory.

    Returns:
        Dictionary of {name: path}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, model in baselines.items():
        path = output_dir / f"baseline_{name}.joblib"
        model.save(path)
        paths[name] = path

    return paths


def load_baselines(
    model_dir: Path | str,
    baselines: list[str] | None = None,
) -> dict[str, BaselineModel]:
    """Load baseline models from disk.

    Args:
        model_dir: Directory with saved models.
        baselines: Specific baselines to load (default: all found).

    Returns:
        Dictionary of {name: model}.
    """
    model_dir = Path(model_dir)

    if baselines is None:
        baselines = ["logit", "rf", "gbm"]

    models = {}
    for name in baselines:
        path = model_dir / f"baseline_{name}.joblib"
        if path.exists():
            models[name] = BaselineModel.load(path)
            logger.info(f"Loaded baseline", name=name, path=str(path))

    return models
