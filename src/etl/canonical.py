"""Canonical Parquet file generation.

Transforms raw LSMS data into canonical panel tables with consistent schemas.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import structlog

from etl.schemas import (
    COUNTRY_CONFIG,
    Country,
)

logger = structlog.get_logger(__name__)


def create_canonical_tables(
    raw_dir: Path,
    output_dir: Path,
    country: Country | str,
) -> dict[str, Path]:
    """Create canonical Parquet tables from raw data.

    Reads raw parquet files and ensures they conform to the canonical
    schemas, then writes to the output directory.

    Args:
        raw_dir: Directory containing raw parquet files.
        output_dir: Directory to write canonical tables.
        country: Country code.

    Returns:
        Dictionary mapping table names to output paths.
    """
    if isinstance(country, str):
        country = Country(country.lower())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {}

    # Process household table
    household_path = raw_dir / "household.parquet"
    if household_path.exists():
        household_df = _process_household_table(household_path, country)
        out_path = output_dir / "household.parquet"
        household_df.write_parquet(out_path)
        output_paths["household"] = out_path
        logger.info("Created canonical household table", rows=len(household_df))
    else:
        logger.warning("No household data found", path=str(household_path))

    # Process individual table
    individual_path = raw_dir / "individual.parquet"
    if individual_path.exists():
        individual_df = _process_individual_table(individual_path, country)
        out_path = output_dir / "individual.parquet"
        individual_df.write_parquet(out_path)
        output_paths["individual"] = out_path
        logger.info("Created canonical individual table", rows=len(individual_df))
    else:
        logger.warning("No individual data found", path=str(individual_path))

    # Process plot table
    plot_path = raw_dir / "plot.parquet"
    if plot_path.exists():
        plot_df = _process_plot_table(plot_path, country)
        out_path = output_dir / "plot.parquet"
        plot_df.write_parquet(out_path)
        output_paths["plot"] = out_path
        logger.info("Created canonical plot table", rows=len(plot_df))
    else:
        logger.warning("No plot data found", path=str(plot_path))

    # Process plot_crop table
    plot_crop_path = raw_dir / "plot_crop.parquet"
    if plot_crop_path.exists():
        plot_crop_df = _process_plot_crop_table(plot_crop_path, country)
        out_path = output_dir / "plot_crop.parquet"
        plot_crop_df.write_parquet(out_path)
        output_paths["plot_crop"] = out_path
        logger.info("Created canonical plot_crop table", rows=len(plot_crop_df))
    else:
        logger.warning("No plot_crop data found", path=str(plot_crop_path))

    # Write processing manifest
    _write_canonical_manifest(output_dir, country, output_paths)

    return output_paths


def _process_household_table(
    input_path: Path,
    country: Country,
) -> pl.DataFrame:
    """Process and validate household table."""
    df = pl.read_parquet(input_path)

    # Check and cast required columns
    df = df.select([
        pl.col("household_id").cast(pl.Utf8),
        pl.col("wave").cast(pl.Int32),
        pl.col("enterprise_status").cast(pl.Int32),
        pl.col("credit_access").cast(pl.Int32),
        pl.col("welfare_proxy").cast(pl.Float64),
        pl.col("region").cast(pl.Utf8),
        pl.col("urban").cast(pl.Int32),
        pl.col("household_size").cast(pl.Int32),
    ])

    # Validate constraints
    config = COUNTRY_CONFIG[country]
    valid_waves = config["waves"]

    df = df.filter(pl.col("wave").is_in(valid_waves))
    df = df.filter(pl.col("enterprise_status").is_in([0, 1]))
    df = df.filter(pl.col("credit_access").is_in([0, 1]))
    df = df.filter(pl.col("urban").is_in([0, 1]))
    df = df.filter(pl.col("household_size") >= 1)

    # Sort for consistency
    df = df.sort(["household_id", "wave"])

    return df


def _process_individual_table(
    input_path: Path,
    country: Country,
) -> pl.DataFrame:
    """Process and validate individual table."""
    df = pl.read_parquet(input_path)

    df = df.select([
        pl.col("household_id").cast(pl.Utf8),
        pl.col("wave").cast(pl.Int32),
        pl.col("individual_id").cast(pl.Utf8),
        pl.col("relationship_to_head").cast(pl.Int32),
        pl.col("age").cast(pl.Int32),
        pl.col("sex").cast(pl.Int32),
        pl.col("education").cast(pl.Int32),
    ])

    # Validate
    config = COUNTRY_CONFIG[country]
    valid_waves = config["waves"]

    df = df.filter(pl.col("wave").is_in(valid_waves))
    df = df.filter(pl.col("age") >= 0)
    df = df.filter(pl.col("age") <= 120)
    df = df.filter(pl.col("sex").is_in([0, 1]))
    df = df.filter(pl.col("education") >= 0)

    df = df.sort(["household_id", "wave", "individual_id"])

    return df


def _process_plot_table(
    input_path: Path,
    country: Country,
) -> pl.DataFrame:
    """Process and validate plot table."""
    df = pl.read_parquet(input_path)

    df = df.select([
        pl.col("household_id").cast(pl.Utf8),
        pl.col("wave").cast(pl.Int32),
        pl.col("plot_id").cast(pl.Utf8),
        pl.col("area_hectares").cast(pl.Float64),
        pl.col("irrigated").cast(pl.Int32),
    ])

    # Validate
    config = COUNTRY_CONFIG[country]
    valid_waves = config["waves"]

    df = df.filter(pl.col("wave").is_in(valid_waves))
    df = df.filter(pl.col("area_hectares") >= 0)
    df = df.filter(pl.col("irrigated").is_in([0, 1]))

    df = df.sort(["household_id", "wave", "plot_id"])

    return df


def _process_plot_crop_table(
    input_path: Path,
    country: Country,
) -> pl.DataFrame:
    """Process and validate plot_crop table."""
    df = pl.read_parquet(input_path)

    df = df.select([
        pl.col("household_id").cast(pl.Utf8),
        pl.col("wave").cast(pl.Int32),
        pl.col("plot_id").cast(pl.Utf8),
        pl.col("crop_code").cast(pl.Int32),
        pl.col("crop_name").cast(pl.Utf8),
        pl.col("area_planted").cast(pl.Float64),
        pl.col("harvest_quantity").cast(pl.Float64),
    ])

    # Validate
    config = COUNTRY_CONFIG[country]
    valid_waves = config["waves"]

    df = df.filter(pl.col("wave").is_in(valid_waves))
    df = df.filter(pl.col("area_planted") >= 0)
    df = df.filter(pl.col("harvest_quantity") >= 0)

    df = df.sort(["household_id", "wave", "plot_id", "crop_code"])

    return df


def _write_canonical_manifest(
    output_dir: Path,
    country: Country,
    output_paths: dict[str, Path],
) -> None:
    """Write manifest for canonical processing."""
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

    # Count records
    num_households = 0
    num_waves = 0
    if "household" in output_paths:
        hh_df = pl.read_parquet(output_paths["household"])
        num_households = hh_df["household_id"].n_unique()
        num_waves = hh_df["wave"].n_unique()

    manifest = {
        "country": country.value,
        "processing_timestamp": datetime.now().isoformat(),
        "processing_stage": "canonical",
        "num_households": num_households,
        "num_waves": num_waves,
        "output_files": {k: str(v) for k, v in output_paths.items()},
        "git_commit": git_commit,
    }

    manifest_path = output_dir / "canonical_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote canonical manifest", path=str(manifest_path))


def validate_canonical_tables(
    data_dir: Path,
    country: Country | str,
) -> dict[str, bool]:
    """Validate canonical tables for integrity.

    Checks:
    - Key uniqueness
    - Referential integrity
    - Required columns
    - Data constraints

    Args:
        data_dir: Directory containing canonical parquet files.
        country: Country code.

    Returns:
        Dictionary of validation results.
    """
    if isinstance(country, str):
        country = Country(country.lower())

    data_dir = Path(data_dir)
    results = {}

    # Load tables
    household_df = None
    plot_df = None
    plot_crop_df = None
    individual_df = None

    if (data_dir / "household.parquet").exists():
        household_df = pl.read_parquet(data_dir / "household.parquet")
    if (data_dir / "plot.parquet").exists():
        plot_df = pl.read_parquet(data_dir / "plot.parquet")
    if (data_dir / "plot_crop.parquet").exists():
        plot_crop_df = pl.read_parquet(data_dir / "plot_crop.parquet")
    if (data_dir / "individual.parquet").exists():
        individual_df = pl.read_parquet(data_dir / "individual.parquet")

    # Check household key uniqueness
    if household_df is not None:
        hh_key_unique = (
            household_df
            .group_by(["household_id", "wave"])
            .len()
            .filter(pl.col("len") > 1)
            .height == 0
        )
        results["household_key_unique"] = hh_key_unique

        # Check required columns
        required_cols = [
            "household_id", "wave", "enterprise_status", "credit_access",
            "welfare_proxy", "region", "urban", "household_size"
        ]
        results["household_has_required_columns"] = all(
            col in household_df.columns for col in required_cols
        )

        # Check valid waves
        config = COUNTRY_CONFIG[country]
        valid_waves = set(config["waves"])
        actual_waves = set(household_df["wave"].unique().to_list())
        results["household_valid_waves"] = actual_waves.issubset(valid_waves)

    # Check plot referential integrity
    if plot_df is not None and household_df is not None:
        hh_keys = set(
            household_df
            .select([pl.col("household_id"), pl.col("wave")])
            .unique()
            .iter_rows()
        )
        plot_hh_keys = set(
            plot_df
            .select([pl.col("household_id"), pl.col("wave")])
            .unique()
            .iter_rows()
        )
        results["plot_referential_integrity"] = plot_hh_keys.issubset(hh_keys)

        # Check no negative areas
        results["plot_no_negative_areas"] = (
            plot_df.filter(pl.col("area_hectares") < 0).height == 0
        )

    # Check plot_crop referential integrity
    if plot_crop_df is not None and plot_df is not None:
        plot_keys = set(
            plot_df
            .select([pl.col("household_id"), pl.col("wave"), pl.col("plot_id")])
            .unique()
            .iter_rows()
        )
        crop_plot_keys = set(
            plot_crop_df
            .select([pl.col("household_id"), pl.col("wave"), pl.col("plot_id")])
            .unique()
            .iter_rows()
        )
        results["plot_crop_referential_integrity"] = crop_plot_keys.issubset(plot_keys)

        # Check no negative areas
        results["plot_crop_no_negative_areas"] = (
            plot_crop_df.filter(pl.col("area_planted") < 0).height == 0
        )

    # Check individual referential integrity
    if individual_df is not None and household_df is not None:
        hh_keys = set(
            household_df
            .select([pl.col("household_id"), pl.col("wave")])
            .unique()
            .iter_rows()
        )
        ind_hh_keys = set(
            individual_df
            .select([pl.col("household_id"), pl.col("wave")])
            .unique()
            .iter_rows()
        )
        results["individual_referential_integrity"] = ind_hh_keys.issubset(hh_keys)

    logger.info("Validated canonical tables", results=results)

    return results
