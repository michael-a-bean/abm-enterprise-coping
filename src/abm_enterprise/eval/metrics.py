"""Evaluation metrics for transition prediction.

Provides classification metrics, confusion matrix computation,
and subgroup analysis for comparing LLM vs baseline predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

from abm_enterprise.eval.direct_prediction import TransitionLabel
from abm_enterprise.utils.logging import get_logger

logger = get_logger(__name__)


# Standard class order for consistent metrics
CLASS_ORDER = [TransitionLabel.ENTER.value, TransitionLabel.EXIT.value, TransitionLabel.STAY.value]


@dataclass
class ClassificationMetrics:
    """Container for classification metrics.

    Attributes:
        accuracy: Overall accuracy.
        balanced_accuracy: Class-balanced accuracy.
        macro_f1: Macro-averaged F1 score.
        weighted_f1: Weighted F1 score.
        per_class_precision: Precision by class.
        per_class_recall: Recall by class.
        per_class_f1: F1 by class.
        confusion_matrix: Confusion matrix array.
        n_samples: Number of samples.
        brier_score: Brier score (if probabilities available).
        log_loss_value: Log loss (if probabilities available).
    """

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    confusion_matrix: np.ndarray
    n_samples: int
    brier_score: float | None = None
    log_loss_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "n_samples": self.n_samples,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss_value,
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Classification Metrics (n={self.n_samples})",
            f"  Accuracy:          {self.accuracy:.3f}",
            f"  Balanced Accuracy: {self.balanced_accuracy:.3f}",
            f"  Macro F1:          {self.macro_f1:.3f}",
            f"  Weighted F1:       {self.weighted_f1:.3f}",
            "",
            "  Per-class F1:",
        ]
        for cls, f1 in self.per_class_f1.items():
            lines.append(f"    {cls}: {f1:.3f}")

        if self.brier_score is not None:
            lines.append(f"  Brier Score: {self.brier_score:.4f}")
        if self.log_loss_value is not None:
            lines.append(f"  Log Loss:    {self.log_loss_value:.4f}")

        return "\n".join(lines)


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.DataFrame | None = None,
    labels: list[str] | None = None,
) -> ClassificationMetrics:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional).
        labels: Class labels in order.

    Returns:
        ClassificationMetrics instance.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = CLASS_ORDER

    # Filter to known labels
    known_mask = np.isin(y_true, labels) & np.isin(y_pred, labels)
    y_true = y_true[known_mask]
    y_pred = y_pred[known_mask]

    n_samples = len(y_true)

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # F1 scores
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # Per-class metrics
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_precision = dict(zip(labels, precisions))
    per_class_recall = dict(zip(labels, recalls))
    per_class_f1 = dict(zip(labels, f1s))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Probabilistic metrics
    brier = None
    logloss = None

    if y_prob is not None:
        try:
            y_prob = np.asarray(y_prob)
            if known_mask.any():
                y_prob = y_prob[known_mask]

            # Brier score (multi-class extension)
            # One-hot encode true labels
            y_true_onehot = np.zeros((len(y_true), len(labels)))
            for i, label in enumerate(labels):
                y_true_onehot[:, i] = (y_true == label).astype(float)

            brier = np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))

            # Log loss
            logloss = log_loss(y_true, y_prob, labels=labels)
        except Exception as e:
            logger.warning(f"Could not compute probabilistic metrics: {e}")

    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
        n_samples=n_samples,
        brier_score=brier,
        log_loss_value=logloss,
    )


def compute_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[str] | None = None,
    normalize: str | None = None,
) -> pd.DataFrame:
    """Compute confusion matrix as DataFrame.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Class labels.
        normalize: Normalization mode ('true', 'pred', 'all', or None).

    Returns:
        Confusion matrix DataFrame.
    """
    if labels is None:
        labels = CLASS_ORDER

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    return pd.DataFrame(
        cm,
        index=pd.Index(labels, name="True"),
        columns=pd.Index(labels, name="Predicted"),
    )


def compute_subgroup_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    subgroup_cols: list[str],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute metrics by subgroup.

    Args:
        df: DataFrame with true labels, predictions, and subgroup columns.
        y_true_col: Column name for true labels.
        y_pred_col: Column name for predictions.
        subgroup_cols: Columns to group by.
        labels: Class labels.

    Returns:
        DataFrame with metrics by subgroup.
    """
    if labels is None:
        labels = CLASS_ORDER

    results = []

    for group_values, group_df in df.groupby(subgroup_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        y_true = group_df[y_true_col].values
        y_pred = group_df[y_pred_col].values

        metrics = compute_classification_metrics(y_true, y_pred, labels=labels)

        row = dict(zip(subgroup_cols, group_values))
        row.update(
            {
                "n_samples": metrics.n_samples,
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "macro_f1": metrics.macro_f1,
            }
        )
        results.append(row)

    return pd.DataFrame(results)


def compute_asset_subgroup_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    n_quantiles: int = 4,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute metrics by asset quantile.

    Args:
        df: DataFrame with predictions and assets.
        y_true_col: Column for true labels.
        y_pred_col: Column for predictions.
        n_quantiles: Number of asset quantiles.
        labels: Class labels.

    Returns:
        DataFrame with metrics by asset quantile.
    """
    # Create asset quantile column
    df = df.copy()
    df["asset_quantile"] = pd.qcut(
        df["assets_index"],
        q=n_quantiles,
        labels=[f"Q{i+1}" for i in range(n_quantiles)],
        duplicates="drop",
    )

    return compute_subgroup_metrics(
        df,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        subgroup_cols=["asset_quantile"],
        labels=labels,
    )


def compute_credit_subgroup_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compute metrics by credit access.

    Args:
        df: DataFrame with predictions and credit access.
        y_true_col: Column for true labels.
        y_pred_col: Column for predictions.
        labels: Class labels.

    Returns:
        DataFrame with metrics by credit access.
    """
    df = df.copy()
    df["credit_group"] = df["credit_access"].map({0: "No Credit", 1: "Has Credit"})

    return compute_subgroup_metrics(
        df,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        subgroup_cols=["credit_group"],
        labels=labels,
    )


def compare_models(
    df: pd.DataFrame,
    y_true_col: str,
    model_cols: list[str],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compare multiple models' performance.

    Args:
        df: DataFrame with true labels and predictions from multiple models.
        y_true_col: Column for true labels.
        model_cols: List of prediction column names.
        labels: Class labels.

    Returns:
        DataFrame comparing model metrics.
    """
    y_true = df[y_true_col].values

    comparisons = []
    for col in model_cols:
        y_pred = df[col].values
        metrics = compute_classification_metrics(y_true, y_pred, labels=labels)

        comparisons.append(
            {
                "model": col.replace("_transition", ""),
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "macro_f1": metrics.macro_f1,
                "weighted_f1": metrics.weighted_f1,
                "f1_ENTER": metrics.per_class_f1.get("ENTER", 0.0),
                "f1_EXIT": metrics.per_class_f1.get("EXIT", 0.0),
                "f1_STAY": metrics.per_class_f1.get("STAY", 0.0),
                "n_samples": metrics.n_samples,
            }
        )

    return pd.DataFrame(comparisons).sort_values("macro_f1", ascending=False)


def save_metrics(
    metrics: ClassificationMetrics | dict,
    output_path: Path | str,
) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Metrics to save.
        output_path: Output file path.
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(metrics, ClassificationMetrics):
        data = metrics.to_dict()
    else:
        data = metrics

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Metrics saved", path=str(output_path))


def save_confusion_matrices(
    cm_dict: dict[str, pd.DataFrame],
    output_dir: Path | str,
) -> None:
    """Save confusion matrices to CSV files.

    Args:
        cm_dict: Dictionary of {model_name: confusion_matrix}.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, cm in cm_dict.items():
        path = output_dir / f"confusion_matrix_{name}.csv"
        cm.to_csv(path)
        logger.info(f"Confusion matrix saved", model=name, path=str(path))


def print_comparison_table(comparison_df: pd.DataFrame) -> str:
    """Format comparison DataFrame as printable table.

    Args:
        comparison_df: DataFrame from compare_models.

    Returns:
        Formatted table string.
    """
    # Round numeric columns
    numeric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "f1_ENTER", "f1_EXIT", "f1_STAY"]

    df = comparison_df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(3)

    return df.to_string(index=False)
