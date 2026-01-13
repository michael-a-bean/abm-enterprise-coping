"""Crop price data loading and processing.

Handles loading of crop price data for computing price exposure measures.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

from etl.schemas import COUNTRY_CONFIG, Country

logger = structlog.get_logger(__name__)

# Default path to price data files
DEFAULT_PRICES_DIR = Path(__file__).parent.parent.parent / "data" / "prices"


def load_crop_prices(
    country: Country | str,
    prices_dir: Path | str | None = None,
) -> pl.DataFrame:
    """Load crop price data for a country.

    Args:
        country: Country code.
        prices_dir: Directory containing price CSV files.
            Defaults to data/prices/.

    Returns:
        DataFrame with columns: crop_code, crop_name, year, price_per_kg
    """
    if isinstance(country, str):
        country = Country(country.lower())

    if prices_dir is None:
        prices_dir = DEFAULT_PRICES_DIR
    else:
        prices_dir = Path(prices_dir)

    price_file = prices_dir / f"{country.value}_prices.csv"

    if not price_file.exists():
        raise FileNotFoundError(f"Price file not found: {price_file}")

    df = pl.read_csv(price_file)

    # Validate schema
    required_cols = ["crop_code", "crop_name", "year", "price_per_kg"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in price file: {missing}")

    logger.info(
        "Loaded crop prices",
        country=country.value,
        num_crops=df["crop_code"].n_unique(),
        years=sorted(df["year"].unique().to_list()),
    )

    return df


def compute_price_changes(
    prices_df: pl.DataFrame,
    country: Country | str,
) -> pl.DataFrame:
    """Compute year-over-year price changes for each crop.

    Args:
        prices_df: Price data from load_crop_prices.
        country: Country code for wave-year mapping.

    Returns:
        DataFrame with columns: crop_code, wave, price_change
        where price_change is the proportional change from previous wave.
    """
    if isinstance(country, str):
        country = Country(country.lower())

    config = COUNTRY_CONFIG[country]
    wave_years = config["wave_years"]

    # Sort by crop and year
    prices_sorted = prices_df.sort(["crop_code", "year"])

    # Compute lagged prices
    prices_with_lag = prices_sorted.with_columns([
        pl.col("price_per_kg").shift(1).over("crop_code").alias("prev_price"),
    ])

    # Compute price change
    prices_with_change = prices_with_lag.with_columns([
        ((pl.col("price_per_kg") - pl.col("prev_price")) / pl.col("prev_price"))
        .alias("price_change"),
    ])

    # Map years to waves
    year_to_wave = {v: k for k, v in wave_years.items()}
    wave_mapping = pl.DataFrame({
        "year": list(year_to_wave.keys()),
        "wave": list(year_to_wave.values()),
    })

    # Join to get wave numbers
    result = (
        prices_with_change
        .join(wave_mapping, on="year", how="inner")
        .select(["crop_code", "crop_name", "wave", "price_change", "price_per_kg"])
        .filter(pl.col("price_change").is_not_null())
    )

    logger.info(
        "Computed price changes",
        country=country.value,
        num_records=len(result),
    )

    return result


def get_price_exposure_weights(
    plot_crop_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute crop area shares for each household-wave.

    These weights are used to compute household-level price exposure.

    Args:
        plot_crop_df: Plot-crop data with area_planted.

    Returns:
        DataFrame with columns: household_id, wave, crop_code, area_share
    """
    # Total area by household-wave
    total_area = (
        plot_crop_df
        .group_by(["household_id", "wave"])
        .agg(pl.col("area_planted").sum().alias("total_area"))
    )

    # Area by crop
    crop_area = (
        plot_crop_df
        .group_by(["household_id", "wave", "crop_code"])
        .agg(pl.col("area_planted").sum().alias("crop_area"))
    )

    # Compute shares
    weights = (
        crop_area
        .join(total_area, on=["household_id", "wave"])
        .with_columns([
            (pl.col("crop_area") / pl.col("total_area")).alias("area_share"),
        ])
        .select(["household_id", "wave", "crop_code", "area_share"])
    )

    return weights


def compute_household_price_exposure(
    plot_crop_df: pl.DataFrame,
    prices_df: pl.DataFrame,
    country: Country | str,
) -> pl.DataFrame:
    """Compute household-level price exposure.

    Price exposure is computed as the weighted average of crop price changes,
    where weights are the crop area shares.

    price_exposure_it = sum_c (area_share_ict * price_change_ct)

    Args:
        plot_crop_df: Plot-crop data with areas.
        prices_df: Crop price data.
        country: Country code.

    Returns:
        DataFrame with columns: household_id, wave, price_exposure
    """
    if isinstance(country, str):
        country = Country(country.lower())

    # Get price changes
    price_changes = compute_price_changes(prices_df, country)

    # Get area weights
    weights = get_price_exposure_weights(plot_crop_df)

    # Join weights with price changes
    exposure_by_crop = (
        weights
        .join(
            price_changes.select(["crop_code", "wave", "price_change"]),
            on=["crop_code", "wave"],
            how="left",
        )
        .with_columns([
            # Handle missing price changes (crops not in price data)
            pl.col("price_change").fill_null(0),
        ])
    )

    # Compute weighted exposure
    household_exposure = (
        exposure_by_crop
        .with_columns([
            (pl.col("area_share") * pl.col("price_change")).alias("weighted_change"),
        ])
        .group_by(["household_id", "wave"])
        .agg([
            pl.col("weighted_change").sum().alias("price_exposure"),
        ])
    )

    logger.info(
        "Computed household price exposure",
        country=country.value,
        num_households=household_exposure["household_id"].n_unique(),
    )

    return household_exposure
