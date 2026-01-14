"""Evaluation module for direct prediction and cross-country validation.

This module provides functionality for:
- Building transition datasets from LSMS panel data
- Training ML baseline classifiers
- Running LLM predictions on real states
- Computing classification and subgroup metrics
"""

from __future__ import annotations

from abm_enterprise.eval.direct_prediction import (
    TransitionLabel,
    build_transition_dataset,
    compute_transition_label,
    predict_with_llm,
)
from abm_enterprise.eval.baselines import (
    BaselineModel,
    LogisticBaseline,
    RandomForestBaseline,
    GradientBoostingBaseline,
    train_all_baselines,
)
from abm_enterprise.eval.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_subgroup_metrics,
    compute_confusion_matrix,
)

__all__ = [
    # Direct prediction
    "TransitionLabel",
    "build_transition_dataset",
    "compute_transition_label",
    "predict_with_llm",
    # Baselines
    "BaselineModel",
    "LogisticBaseline",
    "RandomForestBaseline",
    "GradientBoostingBaseline",
    "train_all_baselines",
    # Metrics
    "ClassificationMetrics",
    "compute_classification_metrics",
    "compute_subgroup_metrics",
    "compute_confusion_matrix",
]
