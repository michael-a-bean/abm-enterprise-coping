"""Derived target tables for ABM validation.

Computes derived variables from canonical tables for use in ABM calibration
and validation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import structlog

from etl.prices import compute_household_price_exposure, load_crop_prices
from etl.schemas import COUNTRY_CONFIG, Country

logger = structlog.get_logger(__name__)


def derive_enterprise_targets(
    household_df: pl.DataFrame,
) -> pl.DataFrame:
    """Derive enterprise target variables.

    Computes:
    - enterprise_indicator: Ever had enterprise (any wave)
    - enterprise_persistence: Proportion of waves with enterprise
    - classification: stayer (>50%), coper (>0% and <=50%), none (0%)

    Args:
        household_df: Canonical household data.

    Returns:
        DataFrame with enterprise targets per household.
    """
    # Count waves and enterprise waves per household
    enterprise_stats = (
        household_df
        .group_by("household_id")
        .agg([
            pl.col("wave").n_unique().alias("total_waves"),
            pl.col("enterprise_status").sum().alias("enterprise_waves"),
            pl.col("enterprise_status").max().alias("ever_enterprise"),
        ])
    )

    # Compute persistence and classification
    enterprise_targets = (
        enterprise_stats
        .with_columns([
            (pl.col("enterprise_waves") / pl.col("total_waves"))
            .alias("enterprise_persistence"),
        ])
        .with_columns([
            pl.when(pl.col("enterprise_persistence") > 0.5)
            .then(pl.lit("stayer"))
            .when(pl.col("enterprise_persistence") > 0)
            .then(pl.lit("coper"))
            .otherwise(pl.lit("none"))
            .alias("classification"),
        ])
        .select([
            "household_id",
            pl.col("ever_enterprise").cast(pl.Int32).alias("enterprise_indicator"),
            "enterprise_persistence",
            "classification",
        ])
    )

    logger.info(
        "Derived enterprise targets",
        num_households=len(enterprise_targets),
        stayers=enterprise_targets.filter(pl.col("classification") == "stayer").height,
        copers=enterprise_targets.filter(pl.col("classification") == "coper").height,
    )

    return enterprise_targets


def derive_asset_targets(
    household_df: pl.DataFrame,
) -> pl.DataFrame:
    """Derive asset target variables.

    Computes:
    - asset_index: PCA proxy (using welfare_proxy as proxy for now)
    - asset_quintile: Quintile of asset distribution (1-5)

    Args:
        household_df: Canonical household data.

    Returns:
        DataFrame with asset targets per household-wave.
    """
    # Use welfare_proxy as asset proxy (simplified PCA)
    # In real implementation, would use multiple asset indicators
    asset_data = household_df.select([
        "household_id",
        "wave",
        pl.col("welfare_proxy").alias("asset_index"),
    ])

    # Compute quintiles within each wave
    asset_targets = (
        asset_data
        .with_columns([
            # Compute quintile (1-5) within each wave
            (
                pl.col("asset_index")
                .rank("ordinal")
                .over("wave")
                / pl.col("asset_index").len().over("wave")
                * 5
            )
            .ceil()
            .cast(pl.Int32)
            .clip(1, 5)
            .alias("asset_quintile"),
        ])
    )

    logger.info(
        "Derived asset targets",
        num_records=len(asset_targets),
    )

    return asset_targets


def derive_price_exposure(
    plot_crop_df: pl.DataFrame,
    household_df: pl.DataFrame,
    country: Country | str,
    prices_dir: Path | None = None,
) -> pl.DataFrame:
    """Derive price exposure for each household-wave.

    Computes weighted average price change based on crop portfolio.

    Args:
        plot_crop_df: Canonical plot-crop data.
        household_df: Canonical household data.
        country: Country code.
        prices_dir: Directory containing price data.

    Returns:
        DataFrame with price exposure per household-wave.
    """
    if isinstance(country, str):
        country = Country(country.lower())

    # Load prices
    prices_df = load_crop_prices(country, prices_dir)

    # Compute price exposure
    exposure_df = compute_household_price_exposure(plot_crop_df, prices_df, country)

    # Ensure all household-waves have a value (fill missing with 0)
    hh_waves = household_df.select(["household_id", "wave"]).unique()

    price_exposure = (
        hh_waves
        .join(exposure_df, on=["household_id", "wave"], how="left")
        .with_columns([
            pl.col("price_exposure").fill_null(0.0),
        ])
    )

    logger.info(
        "Derived price exposure",
        num_records=len(price_exposure),
        mean_exposure=price_exposure["price_exposure"].mean(),
    )

    return price_exposure


def derive_household_targets(
    household_df: pl.DataFrame,
    enterprise_targets: pl.DataFrame,
    asset_targets: pl.DataFrame,
    price_exposure_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge all derived variables into household targets.

    Creates the final dataset for ABM validation with all derived
    variables at the household-wave level.

    Args:
        household_df: Canonical household data.
        enterprise_targets: Enterprise target data.
        asset_targets: Asset target data.
        price_exposure_df: Price exposure data.

    Returns:
        DataFrame with merged household targets.
    """
    # Start with household base data
    base = household_df.select([
        "household_id",
        "wave",
        pl.col("enterprise_status").alias("enterprise_indicator"),
        "credit_access",
        "welfare_proxy",
    ])

    # Join enterprise targets (household-level)
    merged = base.join(
        enterprise_targets.select([
            "household_id",
            "enterprise_persistence",
            "classification",
        ]),
        on="household_id",
        how="left",
    )

    # Join asset targets (household-wave level)
    merged = merged.join(
        asset_targets.select([
            "household_id",
            "wave",
            "asset_index",
            "asset_quintile",
        ]),
        on=["household_id", "wave"],
        how="left",
    )

    # Join price exposure (household-wave level)
    merged = merged.join(
        price_exposure_df.select([
            "household_id",
            "wave",
            "price_exposure",
        ]),
        on=["household_id", "wave"],
        how="left",
    )

    # Fill any nulls
    merged = merged.with_columns([
        pl.col("price_exposure").fill_null(0.0),
        pl.col("enterprise_persistence").fill_null(0.0),
        pl.col("classification").fill_null("none"),
    ])

    # Ensure proper types
    household_targets = merged.select([
        pl.col("household_id").cast(pl.Utf8),
        pl.col("wave").cast(pl.Int32),
        pl.col("enterprise_indicator").cast(pl.Int32),
        pl.col("enterprise_persistence").cast(pl.Float64),
        pl.col("classification").cast(pl.Utf8),
        pl.col("asset_index").cast(pl.Float64),
        pl.col("asset_quintile").cast(pl.Int32),
        pl.col("credit_access").cast(pl.Int32),
        pl.col("price_exposure").cast(pl.Float64),
        pl.col("welfare_proxy").cast(pl.Float64),
    ])

    logger.info(
        "Created household targets",
        num_records=len(household_targets),
        columns=household_targets.columns,
    )

    return household_targets


def build_derived_targets(
    data_dir: Path,
    output_dir: Path,
    country: Country | str,
    prices_dir: Path | None = None,
) -> dict[str, Path]:
    """Build all derived target tables.

    Main entry point for deriving target tables from canonical data.

    Args:
        data_dir: Directory containing canonical parquet files.
        output_dir: Directory to write derived tables.
        country: Country code.
        prices_dir: Optional directory containing price data.

    Returns:
        Dictionary mapping table names to output paths.
    """
    if isinstance(country, str):
        country = Country(country.lower())

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical tables
    household_df = pl.read_parquet(data_dir / "household.parquet")
    plot_crop_df = pl.read_parquet(data_dir / "plot_crop.parquet")

    output_paths = {}

    # Derive enterprise targets
    enterprise_targets = derive_enterprise_targets(household_df)
    enterprise_path = output_dir / "enterprise_targets.parquet"
    enterprise_targets.write_parquet(enterprise_path)
    output_paths["enterprise_targets"] = enterprise_path

    # Derive asset targets
    asset_targets = derive_asset_targets(household_df)
    asset_path = output_dir / "asset_targets.parquet"
    asset_targets.write_parquet(asset_path)
    output_paths["asset_targets"] = asset_path

    # Derive price exposure
    price_exposure = derive_price_exposure(
        plot_crop_df, household_df, country, prices_dir
    )
    exposure_path = output_dir / "price_exposure.parquet"
    price_exposure.write_parquet(exposure_path)
    output_paths["price_exposure"] = exposure_path

    # Build merged household targets
    household_targets = derive_household_targets(
        household_df, enterprise_targets, asset_targets, price_exposure
    )
    targets_path = output_dir / "household_targets.parquet"
    household_targets.write_parquet(targets_path)
    output_paths["household_targets"] = targets_path

    # Write manifest
    _write_derive_manifest(output_dir, country, output_paths, household_targets)

    return output_paths


def _write_derive_manifest(
    output_dir: Path,
    country: Country,
    output_paths: dict[str, Path],
    household_targets: pl.DataFrame,
) -> None:
    """Write manifest for derive processing."""
    import subprocess

    # Try to get git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        git_commit = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        git_commit = None

    # Compute summary statistics
    classification_counts = (
        household_targets
        .select(["household_id", "classification"])
        .unique()
        .group_by("classification")
        .len()
        .to_dicts()
    )

    manifest = {
        "country": country.value,
        "processing_timestamp": datetime.now().isoformat(),
        "processing_stage": "derive",
        "num_households": household_targets["household_id"].n_unique(),
        "num_waves": household_targets["wave"].n_unique(),
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "classification_summary": classification_counts,
        "git_commit": git_commit,
    }

    manifest_path = output_dir / "derive_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote derive manifest", path=str(manifest_path))


def validate_derived_targets(
    data_dir: Path,
    country: Country | str,
) -> dict[str, bool]:
    """Validate derived target tables.

    Checks:
    - All required files exist
    - Key constraints
    - Value ranges
    - Consistency across tables

    Args:
        data_dir: Directory containing derived parquet files.
        country: Country code.

    Returns:
        Dictionary of validation results.
    """
    if isinstance(country, str):
        country = Country(country.lower())

    data_dir = Path(data_dir)
    results = {}

    # Check files exist
    required_files = [
        "enterprise_targets.parquet",
        "asset_targets.parquet",
        "price_exposure.parquet",
        "household_targets.parquet",
    ]

    for filename in required_files:
        results[f"{filename}_exists"] = (data_dir / filename).exists()

    if not all(results.values()):
        logger.warning("Some required files missing", results=results)
        return results

    # Load tables
    enterprise_df = pl.read_parquet(data_dir / "enterprise_targets.parquet")
    asset_df = pl.read_parquet(data_dir / "asset_targets.parquet")
    exposure_df = pl.read_parquet(data_dir / "price_exposure.parquet")
    targets_df = pl.read_parquet(data_dir / "household_targets.parquet")

    # Enterprise targets validation
    results["enterprise_unique_households"] = (
        enterprise_df["household_id"].n_unique() == len(enterprise_df)
    )
    results["enterprise_persistence_range"] = (
        (enterprise_df["enterprise_persistence"] >= 0).all() and
        (enterprise_df["enterprise_persistence"] <= 1).all()
    )
    results["enterprise_valid_classification"] = (
        enterprise_df["classification"].is_in(["stayer", "coper", "none"]).all()
    )

    # Asset targets validation
    results["asset_quintile_range"] = (
        (asset_df["asset_quintile"] >= 1).all() and
        (asset_df["asset_quintile"] <= 5).all()
    )

    # Household targets validation
    results["targets_key_unique"] = (
        targets_df
        .group_by(["household_id", "wave"])
        .len()
        .filter(pl.col("len") > 1)
        .height == 0
    )
    results["targets_has_required_columns"] = all(
        col in targets_df.columns
        for col in [
            "household_id", "wave", "enterprise_indicator",
            "enterprise_persistence", "classification", "asset_index",
            "asset_quintile", "credit_access", "price_exposure", "welfare_proxy"
        ]
    )

    logger.info("Validated derived targets", results=results)

    return results
