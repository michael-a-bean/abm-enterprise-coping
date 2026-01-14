"""Distribution fitting functions for calibration.

This module provides functions to fit distributions from LSMS-derived data
and create calibration artifacts for synthetic panel generation.
"""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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
from abm_enterprise.utils.logging import get_logger

logger = get_logger(__name__)


def get_git_commit() -> str:
    """Get current git commit hash.

    Returns:
        Short git commit hash or 'unknown' if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file.

    Args:
        path: File path.

    Returns:
        MD5 hash string.
    """
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fit_asset_distribution(
    assets: pd.Series,
    family: DistributionFamily = DistributionFamily.NORMAL,
) -> DistributionSpec:
    """Fit distribution to asset data.

    For standardized asset indices (already z-scored), fits a normal or t-distribution.
    For raw asset data, can fit lognormal on positive values.

    Args:
        assets: Asset values (Series).
        family: Distribution family to fit.

    Returns:
        Fitted DistributionSpec.
    """
    assets_clean = assets.dropna()

    raw_stats = {
        "mean": float(assets_clean.mean()),
        "std": float(assets_clean.std()),
        "min": float(assets_clean.min()),
        "max": float(assets_clean.max()),
        "median": float(assets_clean.median()),
        "n": len(assets_clean),
    }

    if family == DistributionFamily.NORMAL:
        mean = float(assets_clean.mean())
        std = float(assets_clean.std())
        params = {"mean": mean, "std": std}

    elif family == DistributionFamily.LOGNORMAL:
        # Shift to positive if needed
        shift = 0.0
        if assets_clean.min() <= 0:
            shift = abs(assets_clean.min()) + 1.0
            assets_shifted = assets_clean + shift
        else:
            assets_shifted = assets_clean

        log_assets = np.log(assets_shifted)
        mu = float(log_assets.mean())
        sigma = float(log_assets.std())
        params = {"mu": mu, "sigma": sigma, "loc": -shift, "scale": np.exp(mu)}

    elif family == DistributionFamily.T:
        # Fit t-distribution for heavier tails
        df, loc, scale = stats.t.fit(assets_clean)
        params = {"df": float(df), "loc": float(loc), "scale": float(scale)}

    elif family == DistributionFamily.SKEW_NORMAL:
        # Fit skew-normal
        a, loc, scale = stats.skewnorm.fit(assets_clean)
        params = {"skew": float(a), "loc": float(loc), "scale": float(scale)}

    else:
        raise ValueError(f"Unsupported distribution family: {family}")

    logger.debug(
        "Fitted asset distribution",
        family=family.value,
        params=params,
        n=len(assets_clean),
    )

    return DistributionSpec(
        family=family,
        params=params,
        standardization=StandardizationMethod.NONE,
        raw_stats=raw_stats,
    )


def fit_shock_distribution(
    shocks: pd.DataFrame,
    wave_col: str = "wave",
    shock_col: str = "price_exposure",
    by_wave: bool = False,
) -> tuple[DistributionSpec, dict[int, DistributionSpec] | None]:
    """Fit distribution to price shock data.

    Args:
        shocks: DataFrame with wave and shock columns.
        wave_col: Column name for wave.
        shock_col: Column name for shock values.
        by_wave: If True, fit separate distribution per wave.

    Returns:
        Tuple of (pooled_distribution, per_wave_distributions or None).
    """
    shocks_clean = shocks[[wave_col, shock_col]].dropna()

    # Pooled distribution
    all_shocks = shocks_clean[shock_col]
    raw_stats = {
        "mean": float(all_shocks.mean()),
        "std": float(all_shocks.std()),
        "min": float(all_shocks.min()),
        "max": float(all_shocks.max()),
        "n": len(all_shocks),
    }

    pooled_spec = DistributionSpec(
        family=DistributionFamily.NORMAL,
        params={"mean": raw_stats["mean"], "std": raw_stats["std"]},
        standardization=StandardizationMethod.NONE,
        raw_stats=raw_stats,
    )

    # Per-wave distributions
    per_wave_specs = None
    if by_wave:
        per_wave_specs = {}
        for wave in sorted(shocks_clean[wave_col].unique()):
            wave_shocks = shocks_clean[shocks_clean[wave_col] == wave][shock_col]
            if len(wave_shocks) > 10:  # Require minimum sample
                wave_raw = {
                    "mean": float(wave_shocks.mean()),
                    "std": float(wave_shocks.std()),
                    "n": len(wave_shocks),
                }
                per_wave_specs[int(wave)] = DistributionSpec(
                    family=DistributionFamily.NORMAL,
                    params={"mean": wave_raw["mean"], "std": wave_raw["std"]},
                    standardization=StandardizationMethod.NONE,
                    raw_stats=wave_raw,
                )

    logger.debug(
        "Fitted shock distribution",
        pooled_mean=pooled_spec.params["mean"],
        pooled_std=pooled_spec.params["std"],
        n_waves=len(per_wave_specs) if per_wave_specs else 0,
    )

    return pooled_spec, per_wave_specs


def fit_credit_model(
    df: pd.DataFrame,
    target_col: str = "credit_access",
    feature_cols: list[str] | None = None,
) -> CreditModelSpec:
    """Fit logistic regression model for credit access.

    Args:
        df: DataFrame with target and feature columns.
        target_col: Column name for credit access (0/1).
        feature_cols: Feature column names. Defaults to ['assets_index'].

    Returns:
        Fitted CreditModelSpec.
    """
    if feature_cols is None:
        feature_cols = ["assets_index"]

    # Prepare data
    df_clean = df[[target_col] + feature_cols].dropna()

    if len(df_clean) < 50:
        logger.warning(
            "Small sample for credit model",
            n=len(df_clean),
        )

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values.astype(int)

    # Fit logistic regression
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    # Extract coefficients
    coefficients = {
        name: float(coef) for name, coef in zip(feature_cols, model.coef_[0])
    }
    intercept = float(model.intercept_[0])

    # Compute metrics
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "n_observations": len(y),
        "n_credit_positive": int(y.sum()),
        "credit_rate": float(y.mean()),
    }

    # AUC if both classes present
    if len(np.unique(y)) > 1:
        metrics["auc"] = float(roc_auc_score(y, y_prob))

    logger.debug(
        "Fitted credit model",
        coefficients=coefficients,
        intercept=intercept,
        accuracy=metrics["accuracy"],
    )

    return CreditModelSpec(
        type="logistic",
        coefficients=coefficients,
        intercept=intercept,
        feature_names=feature_cols,
        model_metrics=metrics,
    )


def compute_transition_rates(
    df: pd.DataFrame,
    household_col: str = "household_id",
    wave_col: str = "wave",
    enterprise_col: str = "enterprise_status",
) -> TransitionRates:
    """Compute observed enterprise transition rates.

    Args:
        df: Panel DataFrame with household, wave, and enterprise columns.
        household_col: Household ID column.
        wave_col: Wave column.
        enterprise_col: Enterprise status column (0/1).

    Returns:
        TransitionRates with observed rates.
    """
    df_sorted = df.sort_values([household_col, wave_col])

    # Create lagged enterprise status
    df_sorted["enterprise_lag"] = df_sorted.groupby(household_col)[
        enterprise_col
    ].shift(1)

    # Filter to rows with valid lag (not first wave for each HH)
    transitions = df_sorted.dropna(subset=["enterprise_lag"])

    # Count transitions
    enter_mask = (transitions["enterprise_lag"] == 0) & (
        transitions[enterprise_col] == 1
    )
    exit_mask = (transitions["enterprise_lag"] == 1) & (
        transitions[enterprise_col] == 0
    )
    stay_mask = transitions["enterprise_lag"] == transitions[enterprise_col]

    enter_count = int(enter_mask.sum())
    exit_count = int(exit_mask.sum())
    stay_count = int(stay_mask.sum())
    total = enter_count + exit_count + stay_count

    # Compute rates
    if total > 0:
        enter_rate = enter_count / total
        exit_rate = exit_count / total
        stay_rate = stay_count / total
    else:
        enter_rate = exit_rate = stay_rate = 0.0

    # Also compute conditional rates
    n_not_enterprise = int((transitions["enterprise_lag"] == 0).sum())
    n_enterprise = int((transitions["enterprise_lag"] == 1).sum())

    if n_not_enterprise > 0:
        cond_enter_rate = enter_count / n_not_enterprise
    else:
        cond_enter_rate = 0.0

    if n_enterprise > 0:
        cond_exit_rate = exit_count / n_enterprise
    else:
        cond_exit_rate = 0.0

    logger.debug(
        "Computed transition rates",
        enter_rate=enter_rate,
        exit_rate=exit_rate,
        stay_rate=stay_rate,
        total_transitions=total,
    )

    return TransitionRates(
        enter_rate=cond_enter_rate,  # Use conditional rates
        exit_rate=cond_exit_rate,
        stay_rate=stay_rate,
        enter_count=enter_count,
        exit_count=exit_count,
        stay_count=stay_count,
    )


def compute_enterprise_baseline(
    df: pd.DataFrame,
    household_col: str = "household_id",
    wave_col: str = "wave",
    enterprise_col: str = "enterprise_status",
) -> EnterpriseBaseline:
    """Compute baseline enterprise statistics.

    Args:
        df: Panel DataFrame.
        household_col: Household ID column.
        wave_col: Wave column.
        enterprise_col: Enterprise status column (0/1).

    Returns:
        EnterpriseBaseline statistics.
    """
    # Overall prevalence
    prevalence = float(df[enterprise_col].mean())

    # Prevalence by wave
    prevalence_by_wave = {
        int(wave): float(grp[enterprise_col].mean())
        for wave, grp in df.groupby(wave_col)
    }

    # Transition rates for entry/exit
    transition_rates = compute_transition_rates(
        df, household_col, wave_col, enterprise_col
    )

    return EnterpriseBaseline(
        prevalence=prevalence,
        prevalence_by_wave=prevalence_by_wave,
        entry_rate=transition_rates.enter_rate,
        exit_rate=transition_rates.exit_rate,
    )


def fit_calibration(
    country: str,
    data_dir: Path,
    out_dir: Path,
    config: dict[str, Any] | None = None,
) -> CalibrationArtifact:
    """Main entry point for calibration.

    Fits all distributions and models from LSMS-derived targets.

    Args:
        country: Country code (e.g., 'tanzania').
        data_dir: Directory containing processed data.
        out_dir: Output directory for calibration artifacts.
        config: Optional configuration overrides.

    Returns:
        Fitted CalibrationArtifact.

    Raises:
        FileNotFoundError: If required data files not found.
    """
    import polars as pl

    config = config or {}
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    # Load derived targets
    targets_path = data_dir / country / "derived" / "household_targets.parquet"
    if not targets_path.exists():
        raise FileNotFoundError(
            f"Derived targets not found: {targets_path}\n"
            f"Run 'abm derive-targets --country {country}' first."
        )

    logger.info("Loading derived targets", path=str(targets_path))
    df_pl = pl.read_parquet(targets_path)

    # Convert to pandas for fitting
    df = df_pl.to_pandas()

    # Handle column name variations
    enterprise_col = "enterprise_indicator"
    if "enterprise_status" in df.columns:
        enterprise_col = "enterprise_status"

    assets_col = "asset_index"
    if "assets_index" in df.columns:
        assets_col = "assets_index"

    logger.info(
        "Calibration data loaded",
        n_households=df["household_id"].nunique(),
        n_observations=len(df),
        waves=sorted(df["wave"].unique()),
    )

    # 1. Fit asset distribution
    logger.info("Fitting asset distribution")
    asset_family = DistributionFamily(config.get("asset_family", "normal"))
    assets_dist = fit_asset_distribution(df[assets_col], family=asset_family)

    # 2. Fit shock distribution
    logger.info("Fitting shock distribution")
    by_wave = config.get("shock_by_wave", False)
    shock_pooled, shock_by_wave = fit_shock_distribution(
        df, wave_col="wave", shock_col="price_exposure", by_wave=by_wave
    )

    # 3. Fit credit model
    logger.info("Fitting credit model")
    credit_features = config.get("credit_features", [assets_col])
    credit_model = fit_credit_model(
        df, target_col="credit_access", feature_cols=credit_features
    )

    # 4. Compute enterprise baseline
    logger.info("Computing enterprise baseline")
    enterprise_baseline = compute_enterprise_baseline(
        df,
        household_col="household_id",
        wave_col="wave",
        enterprise_col=enterprise_col,
    )

    # 5. Compute transition rates
    logger.info("Computing transition rates")
    transition_rates = compute_transition_rates(
        df,
        household_col="household_id",
        wave_col="wave",
        enterprise_col=enterprise_col,
    )

    # Create calibration artifact
    artifact = CalibrationArtifact(
        country_source=country,
        git_commit=get_git_commit(),
        waves=sorted(df["wave"].unique().tolist()),
        n_households=int(df["household_id"].nunique()),
        n_observations=len(df),
        assets_distribution=assets_dist,
        shock_distribution=shock_pooled,
        shock_distribution_by_wave=shock_by_wave,
        credit_model=credit_model,
        enterprise_baseline=enterprise_baseline,
        transition_rates=transition_rates,
        additional_metadata={
            "config": config,
            "enterprise_col": enterprise_col,
            "assets_col": assets_col,
        },
    )

    # Save artifacts
    out_dir = out_dir / country
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = out_dir / "calibration.json"
    artifact.save(str(artifact_path))
    logger.info("Saved calibration artifact", path=str(artifact_path))

    # Create and save manifest
    manifest = CalibrationManifest(
        calibration_id=str(uuid.uuid4())[:8],
        git_commit=get_git_commit(),
        country=country,
        input_data_path=str(targets_path),
        input_data_hash=compute_file_hash(targets_path),
        output_artifact_path=str(artifact_path),
        parameters_used=config,
    )
    manifest_path = out_dir / "calibration_manifest.json"
    manifest.save(str(manifest_path))
    logger.info("Saved calibration manifest", path=str(manifest_path))

    return artifact
