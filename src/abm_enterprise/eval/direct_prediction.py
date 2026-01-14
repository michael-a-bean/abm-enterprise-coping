"""Direct prediction evaluation on LSMS transition data.

This module builds transition datasets from LSMS panel data and enables
evaluation of LLM predictions against observed outcomes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel

from abm_enterprise.policies.base import Action
from abm_enterprise.data.schemas import HouseholdState
from abm_enterprise.utils.logging import get_logger

if TYPE_CHECKING:
    from abm_enterprise.policies.llm import MultiSampleLLMPolicy

logger = get_logger(__name__)


class TransitionLabel(str, Enum):
    """Labels for household enterprise transitions."""

    ENTER = "ENTER"
    EXIT = "EXIT"
    STAY = "STAY"


class TransitionRow(BaseModel):
    """Schema for a single transition observation.

    Attributes:
        household_id: Unique household identifier.
        wave_t: Starting wave.
        wave_t1: Ending wave.
        assets_index: Standardized asset index at wave t.
        credit_access: Credit access indicator at wave t.
        enterprise_status: Enterprise status at wave t.
        price_exposure: Price shock exposure at wave t.
        transition: Observed transition label.
        country: Country code.
    """

    household_id: str
    wave_t: int
    wave_t1: int
    assets_index: float
    credit_access: int
    enterprise_status: int
    price_exposure: float
    transition: TransitionLabel
    country: str


def compute_transition_label(
    enterprise_t: int,
    enterprise_t1: int,
) -> TransitionLabel:
    """Compute transition label from enterprise status change.

    Args:
        enterprise_t: Enterprise status at time t (0 or 1).
        enterprise_t1: Enterprise status at time t+1 (0 or 1).

    Returns:
        TransitionLabel indicating ENTER, EXIT, or STAY.
    """
    if enterprise_t == 0 and enterprise_t1 == 1:
        return TransitionLabel.ENTER
    elif enterprise_t == 1 and enterprise_t1 == 0:
        return TransitionLabel.EXIT
    else:
        return TransitionLabel.STAY


def build_transition_dataset(
    country: str,
    data_dir: Path | str,
    waves: list[int] | None = None,
) -> pd.DataFrame:
    """Build transition dataset from LSMS derived targets.

    Constructs a dataset of household state → transition observations
    for evaluating prediction models.

    Args:
        country: Country code (e.g., 'tanzania', 'ethiopia').
        data_dir: Path to processed data directory.
        waves: Specific waves to include (default: all available).

    Returns:
        DataFrame with transition observations.

    Raises:
        FileNotFoundError: If household targets file not found.
        ValueError: If insufficient waves for transitions.
    """
    data_dir = Path(data_dir)
    targets_path = data_dir / country / "household_targets.parquet"

    if not targets_path.exists():
        raise FileNotFoundError(
            f"Household targets not found: {targets_path}. "
            f"Run 'make derive-targets country={country}' first."
        )

    df = pd.read_parquet(targets_path)

    # Validate required columns
    required_cols = [
        "household_id",
        "wave",
        "assets_index",
        "enterprise_indicator",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to requested waves
    if waves is not None:
        df = df[df["wave"].isin(waves)]

    available_waves = sorted(df["wave"].unique())
    if len(available_waves) < 2:
        raise ValueError(
            f"Need at least 2 waves for transitions, found: {available_waves}"
        )

    logger.info(
        "Building transition dataset",
        country=country,
        waves=available_waves,
        n_households=df["household_id"].nunique(),
    )

    # Handle column name variations
    enterprise_col = (
        "enterprise_indicator"
        if "enterprise_indicator" in df.columns
        else "enterprise_status"
    )
    credit_col = "credit_access" if "credit_access" in df.columns else None
    shock_col = "price_exposure" if "price_exposure" in df.columns else None

    # Build transitions for adjacent wave pairs
    transitions = []

    for hh_id, hh_df in df.groupby("household_id"):
        hh_df = hh_df.sort_values("wave")
        hh_waves = hh_df["wave"].values

        for i in range(len(hh_waves) - 1):
            wave_t = hh_waves[i]
            wave_t1 = hh_waves[i + 1]

            row_t = hh_df[hh_df["wave"] == wave_t].iloc[0]
            row_t1 = hh_df[hh_df["wave"] == wave_t1].iloc[0]

            enterprise_t = int(row_t[enterprise_col])
            enterprise_t1 = int(row_t1[enterprise_col])
            transition = compute_transition_label(enterprise_t, enterprise_t1)

            transitions.append(
                {
                    "household_id": hh_id,
                    "wave_t": int(wave_t),
                    "wave_t1": int(wave_t1),
                    "assets_index": float(row_t["assets_index"]),
                    "credit_access": int(row_t[credit_col]) if credit_col else 0,
                    "enterprise_status": enterprise_t,
                    "price_exposure": float(row_t[shock_col]) if shock_col else 0.0,
                    "transition": transition.value,
                    "country": country,
                }
            )

    result = pd.DataFrame(transitions)

    # Log transition distribution
    transition_counts = result["transition"].value_counts()
    logger.info(
        "Transition dataset built",
        n_transitions=len(result),
        distribution=transition_counts.to_dict(),
    )

    return result


def build_cross_country_dataset(
    train_country: str,
    test_country: str,
    data_dir: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test datasets from different countries.

    Args:
        train_country: Country for training data.
        test_country: Country for test data.
        data_dir: Path to processed data directory.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df = build_transition_dataset(train_country, data_dir)
    test_df = build_transition_dataset(test_country, data_dir)

    logger.info(
        "Cross-country dataset built",
        train_country=train_country,
        train_n=len(train_df),
        test_country=test_country,
        test_n=len(test_df),
    )

    return train_df, test_df


def state_from_row(row: pd.Series) -> HouseholdState:
    """Convert dataset row to HouseholdState.

    Args:
        row: Row from transition dataset.

    Returns:
        HouseholdState for LLM policy input.
    """
    from abm_enterprise.data.schemas import EnterpriseStatus

    enterprise_status = (
        EnterpriseStatus.HAS_ENTERPRISE
        if row["enterprise_status"]
        else EnterpriseStatus.NO_ENTERPRISE
    )

    return HouseholdState(
        household_id=str(row["household_id"]),
        wave=int(row["wave_t"]),
        assets=float(row["assets_index"]),
        credit_access=int(row["credit_access"]),
        enterprise_status=enterprise_status,
        price_exposure=float(row["price_exposure"]),
    )


def action_to_transition(
    action: Action,
    current_enterprise: bool,
) -> TransitionLabel:
    """Convert policy action to transition label.

    Args:
        action: Action from policy decision.
        current_enterprise: Current enterprise status.

    Returns:
        Corresponding transition label.
    """
    if action == Action.ENTER_ENTERPRISE:
        if not current_enterprise:
            return TransitionLabel.ENTER
        else:
            return TransitionLabel.STAY  # Already in enterprise
    elif action == Action.EXIT_ENTERPRISE:
        if current_enterprise:
            return TransitionLabel.EXIT
        else:
            return TransitionLabel.STAY  # Already out
    else:  # NO_CHANGE
        return TransitionLabel.STAY


def predict_with_llm(
    dataset: pd.DataFrame,
    policy: "MultiSampleLLMPolicy",
    batch_size: int = 100,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run LLM predictions on transition dataset.

    Args:
        dataset: Transition dataset from build_transition_dataset.
        policy: Multi-sample LLM policy for predictions.
        batch_size: Number of predictions per batch (for logging).
        show_progress: Whether to show progress updates.

    Returns:
        Dataset with added prediction columns:
        - llm_action: Predicted action
        - llm_transition: Predicted transition label
        - llm_confidence: Vote confidence
        - llm_cache_hit: Whether result was cached
    """
    predictions = []
    n_total = len(dataset)

    for idx, row in dataset.iterrows():
        state = state_from_row(row)

        # Get decision from policy
        action = policy.decide(state)

        # Convert to transition
        predicted_transition = action_to_transition(
            action,
            bool(row["enterprise_status"]),
        )

        # Get additional metadata if available
        confidence = 0.0
        cache_hit = False

        # Try to get vote result from cache for metadata
        if hasattr(policy, "_cache") and policy._cache is not None:
            from abm_enterprise.policies.cache import compute_state_hash, compute_config_hash

            state_hash = compute_state_hash(state.to_dict())
            config_hash = compute_config_hash(policy._config.model_dump())
            cached = policy._cache.get(state_hash, config_hash)
            if cached:
                confidence = cached.confidence
                cache_hit = True

        predictions.append(
            {
                "llm_action": action.value,
                "llm_transition": predicted_transition.value,
                "llm_confidence": confidence,
                "llm_cache_hit": cache_hit,
            }
        )

        # Progress logging
        if show_progress and (idx + 1) % batch_size == 0:
            logger.info(
                "LLM prediction progress",
                completed=idx + 1,
                total=n_total,
                pct=round(100 * (idx + 1) / n_total, 1),
            )

    # Add predictions to dataset
    result = dataset.copy()
    pred_df = pd.DataFrame(predictions, index=dataset.index)
    for col in pred_df.columns:
        result[col] = pred_df[col]

    # Log prediction distribution
    pred_counts = result["llm_transition"].value_counts()
    logger.info(
        "LLM predictions complete",
        n_predictions=len(result),
        distribution=pred_counts.to_dict(),
        cache_hit_rate=result["llm_cache_hit"].mean(),
    )

    return result


def predict_with_baselines(
    dataset: pd.DataFrame,
    baselines: dict,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run baseline model predictions on transition dataset.

    Args:
        dataset: Transition dataset.
        baselines: Dictionary of {name: trained_model}.
        feature_cols: Feature columns for prediction.

    Returns:
        Dataset with added prediction columns for each baseline.
    """
    if feature_cols is None:
        feature_cols = ["assets_index", "credit_access", "enterprise_status", "price_exposure"]

    X = dataset[feature_cols].values
    result = dataset.copy()

    for name, model in baselines.items():
        preds = model.predict(X)
        probs = model.predict_proba(X)

        result[f"{name}_transition"] = preds

        # Add probability columns
        classes = model.classes_
        for i, cls in enumerate(classes):
            result[f"{name}_prob_{cls}"] = probs[:, i]

        logger.info(
            f"Baseline {name} predictions complete",
            distribution=pd.Series(preds).value_counts().to_dict(),
        )

    return result


def save_predictions(
    predictions: pd.DataFrame,
    output_dir: Path | str,
    prefix: str = "predictions",
) -> Path:
    """Save prediction results to parquet.

    Args:
        predictions: DataFrame with predictions.
        output_dir: Output directory.
        prefix: Filename prefix.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{prefix}.parquet"
    predictions.to_parquet(path, index=False)

    logger.info("Predictions saved", path=str(path), n_rows=len(predictions))
    return path
