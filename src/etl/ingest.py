"""LSMS-ISA data ingestion module.

Downloads and extracts LSMS-ISA harmonized data from GitHub releases,
with fallback to synthetic data generation if real data is unavailable.
"""

from __future__ import annotations

import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from etl.schemas import COUNTRY_CONFIG, Country, ProcessingManifest

logger = structlog.get_logger(__name__)

# URL pattern for LSMS-ISA harmonized data
LSMS_URL_PATTERN = (
    "https://github.com/lsms-worldbank/lsms-isa-harmonized/"
    "releases/download/v2.0/{country}_harmonized.zip"
)


def download_lsms_data(
    country: Country | str,
    output_dir: Path,
    force: bool = False,
) -> tuple[Path, bool]:
    """Download LSMS-ISA harmonized data for a country.

    Attempts to download from GitHub releases. If download fails,
    generates synthetic data matching the expected schema.

    Args:
        country: Country code (tanzania or ethiopia).
        output_dir: Directory to save downloaded/generated data.
        force: Force re-download even if data exists.

    Returns:
        Tuple of (path to data directory, whether synthetic data was used).
    """
    import urllib.error
    import urllib.request

    if isinstance(country, str):
        country = Country(country.lower())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    country_dir = output_dir / country.value
    raw_dir = country_dir / "raw"

    # Check if data already exists
    if raw_dir.exists() and not force:
        logger.info("Data already exists", country=country.value, path=str(raw_dir))
        # Check if it's synthetic
        manifest_file = country_dir / "ingest_manifest.json"
        is_synthetic = False
        if manifest_file.exists():
            import json

            with open(manifest_file) as f:
                manifest = json.load(f)
                is_synthetic = manifest.get("synthetic_data", False)
        return raw_dir, is_synthetic

    url = LSMS_URL_PATTERN.format(country=country.value)
    logger.info("Attempting to download LSMS data", url=url)

    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            urllib.request.urlretrieve(url, tmp_file.name)

            # Extract to raw directory
            raw_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(tmp_file.name, "r") as zf:
                zf.extractall(raw_dir)

            Path(tmp_file.name).unlink()
            logger.info("Successfully downloaded and extracted LSMS data", path=str(raw_dir))
            _write_ingest_manifest(country_dir, country, url, synthetic=False)
            return raw_dir, False

    except (urllib.error.URLError, urllib.error.HTTPError, zipfile.BadZipFile) as e:
        logger.warning(
            "Failed to download LSMS data, generating synthetic fallback",
            error=str(e),
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        _generate_synthetic_lsms_data(country, raw_dir)
        _write_ingest_manifest(country_dir, country, url, synthetic=True)
        return raw_dir, True


def _write_ingest_manifest(
    output_dir: Path,
    country: Country,
    source_url: str,
    synthetic: bool,
) -> None:
    """Write manifest file for the ingestion run."""
    import json
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

    manifest = {
        "country": country.value,
        "processing_timestamp": datetime.now().isoformat(),
        "source_url": source_url,
        "data_version": "v2.0",
        "synthetic_data": synthetic,
        "git_commit": git_commit,
    }

    manifest_path = output_dir / "ingest_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote ingest manifest", path=str(manifest_path))


def _generate_synthetic_lsms_data(country: Country, output_dir: Path) -> None:
    """Generate synthetic LSMS data matching the expected schema.

    Creates synthetic data files that match the structure of real
    LSMS-ISA harmonized data for testing and development.
    """
    config = COUNTRY_CONFIG[country]
    waves = config["waves"]
    num_households = 500  # Reasonable sample size

    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Generate household data
    household_data = _generate_synthetic_households(
        num_households, waves, country, rng
    )
    household_data.write_parquet(output_dir / "household.parquet")

    # Generate individual data
    individual_data = _generate_synthetic_individuals(
        household_data, rng
    )
    individual_data.write_parquet(output_dir / "individual.parquet")

    # Generate plot data
    plot_data = _generate_synthetic_plots(
        household_data, rng
    )
    plot_data.write_parquet(output_dir / "plot.parquet")

    # Generate plot-crop data
    plot_crop_data = _generate_synthetic_plot_crops(
        plot_data, country, rng
    )
    plot_crop_data.write_parquet(output_dir / "plot_crop.parquet")

    logger.info(
        "Generated synthetic LSMS data",
        country=country.value,
        households=num_households,
        waves=len(waves),
    )


def _generate_synthetic_households(
    n: int,
    waves: list[int],
    country: Country,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate synthetic household panel data."""
    id_prefix = "TZ" if country == Country.TANZANIA else "ET"

    # Regions based on country
    if country == Country.TANZANIA:
        regions = [
            "Dodoma", "Arusha", "Kilimanjaro", "Tanga", "Morogoro",
            "Pwani", "Dar es Salaam", "Lindi", "Mtwara", "Ruvuma",
            "Iringa", "Mbeya", "Singida", "Tabora", "Rukwa",
            "Kigoma", "Shinyanga", "Kagera", "Mwanza", "Mara"
        ]
    else:
        regions = [
            "Tigray", "Afar", "Amhara", "Oromia", "Somali",
            "Benishangul-Gumuz", "SNNP", "Gambela", "Harari",
            "Addis Ababa", "Dire Dawa"
        ]

    records = []

    # Time-invariant household characteristics
    household_regions = rng.choice(regions, size=n)
    household_urban = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    household_sizes = rng.poisson(lam=5, size=n) + 1
    household_sizes = np.clip(household_sizes, 1, 15)

    # Base characteristics that evolve
    base_welfare = rng.lognormal(mean=10, sigma=1, size=n)
    base_credit_prob = rng.beta(2, 5, size=n)
    base_enterprise_prob = rng.beta(2, 8, size=n)

    for wave in waves:
        for i in range(n):
            hh_id = f"{id_prefix}_{i:06d}"

            # Welfare evolves with trend and noise
            welfare = base_welfare[i] * (1 + 0.03 * (wave - 1)) + rng.normal(0, base_welfare[i] * 0.1)
            welfare = max(1, welfare)

            # Credit access can change
            credit_prob = base_credit_prob[i] + 0.05 * (wave - 1)
            credit_access = int(rng.random() < credit_prob)

            # Enterprise status with persistence
            if wave == waves[0]:
                enterprise_status = int(rng.random() < base_enterprise_prob[i])
            else:
                # Find previous status from records
                prev_records = [r for r in records if r["household_id"] == hh_id and r["wave"] == wave - 1]
                if prev_records:
                    prev_status = prev_records[0]["enterprise_status"]
                    if prev_status == 1:
                        # High persistence
                        enterprise_status = int(rng.random() < 0.85)
                    else:
                        # Some entry probability
                        entry_prob = 0.08 + 0.03 * credit_access
                        enterprise_status = int(rng.random() < entry_prob)
                else:
                    enterprise_status = int(rng.random() < base_enterprise_prob[i])

            # Household size can change slightly
            size_change = rng.choice([-1, 0, 0, 0, 0, 1])
            hh_size = max(1, household_sizes[i] + size_change * (wave - 1))

            records.append({
                "household_id": hh_id,
                "wave": wave,
                "enterprise_status": enterprise_status,
                "credit_access": credit_access,
                "welfare_proxy": round(welfare, 2),
                "region": household_regions[i],
                "urban": int(household_urban[i]),
                "household_size": int(hh_size),
            })

    return pl.DataFrame(records)


def _generate_synthetic_individuals(
    household_df: pl.DataFrame,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate synthetic individual data for households."""
    records = []

    # Get unique household-wave combinations
    hh_waves = household_df.select(["household_id", "wave", "household_size"]).unique()

    for row in hh_waves.iter_rows(named=True):
        hh_id = row["household_id"]
        wave = row["wave"]
        hh_size = row["household_size"]

        for ind_idx in range(hh_size):
            ind_id = f"{hh_id}_M{ind_idx:02d}"

            if ind_idx == 0:
                # Household head
                relationship = 1
                age = int(rng.integers(25, 70))
                sex = int(rng.choice([0, 1], p=[0.3, 0.7]))  # More male heads
            elif ind_idx == 1 and rng.random() < 0.7:
                # Spouse
                relationship = 2
                age = int(rng.integers(20, 65))
                sex = 1 - records[-1]["sex"] if records else int(rng.choice([0, 1]))
            else:
                # Other members (children, relatives)
                relationship = rng.choice([3, 4, 5, 6, 7])  # Various relationships
                if relationship == 3:  # Child
                    age = int(rng.integers(0, 25))
                else:
                    age = int(rng.integers(5, 80))
                sex = int(rng.choice([0, 1]))

            # Education based on age
            if age < 6:
                education = 0
            elif age < 15:
                education = rng.choice([0, 1, 2, 3])
            else:
                education = rng.choice([0, 1, 2, 3, 4, 5, 6], p=[0.2, 0.15, 0.2, 0.2, 0.1, 0.1, 0.05])

            records.append({
                "household_id": hh_id,
                "wave": wave,
                "individual_id": ind_id,
                "relationship_to_head": int(relationship),
                "age": age,
                "sex": sex,
                "education": int(education),
            })

    return pl.DataFrame(records)


def _generate_synthetic_plots(
    household_df: pl.DataFrame,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate synthetic plot data."""
    records = []

    # Get unique household-wave combinations
    hh_waves = household_df.select(["household_id", "wave", "urban"]).unique()

    for row in hh_waves.iter_rows(named=True):
        hh_id = row["household_id"]
        wave = row["wave"]
        is_urban = row["urban"]

        # Urban households have fewer/smaller plots
        if is_urban:
            num_plots = rng.choice([0, 1, 1, 2], p=[0.3, 0.4, 0.2, 0.1])
        else:
            num_plots = rng.choice([1, 2, 2, 3, 3, 4], p=[0.1, 0.25, 0.25, 0.2, 0.1, 0.1])

        for plot_idx in range(num_plots):
            plot_id = f"{hh_id}_P{plot_idx:02d}"

            # Plot area (log-normal)
            if is_urban:
                area = rng.lognormal(mean=-1, sigma=0.5)
            else:
                area = rng.lognormal(mean=0, sigma=0.8)
            area = np.clip(area, 0.01, 20)

            # Irrigation probability
            irrigated = int(rng.random() < 0.15)

            records.append({
                "household_id": hh_id,
                "wave": wave,
                "plot_id": plot_id,
                "area_hectares": round(float(area), 3),
                "irrigated": irrigated,
            })

    return pl.DataFrame(records)


def _generate_synthetic_plot_crops(
    plot_df: pl.DataFrame,
    country: Country,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate synthetic plot-crop data."""
    # Load crop prices to get valid crop codes
    from etl.prices import load_crop_prices

    prices_df = load_crop_prices(country)
    crop_codes = prices_df["crop_code"].unique().to_list()
    crop_names = {
        row["crop_code"]: row["crop_name"]
        for row in prices_df.unique(subset=["crop_code", "crop_name"]).iter_rows(named=True)
    }

    records = []

    for row in plot_df.iter_rows(named=True):
        hh_id = row["household_id"]
        wave = row["wave"]
        plot_id = row["plot_id"]
        plot_area = row["area_hectares"]

        # Number of crops on this plot
        num_crops = rng.choice([1, 1, 2, 2, 3], p=[0.3, 0.3, 0.2, 0.1, 0.1])

        # Select random crops
        selected_crops = rng.choice(crop_codes, size=min(num_crops, len(crop_codes)), replace=False)

        # Distribute area among crops
        if len(selected_crops) > 1:
            shares = rng.dirichlet(np.ones(len(selected_crops)))
        else:
            shares = [1.0]

        for crop_code, share in zip(selected_crops, shares, strict=True):
            area_planted = plot_area * share

            # Harvest quantity based on area (kg per hectare varies by crop)
            yield_per_ha = rng.lognormal(mean=6, sigma=0.5)  # ~400 kg/ha typical
            harvest_qty = area_planted * yield_per_ha

            records.append({
                "household_id": hh_id,
                "wave": wave,
                "plot_id": plot_id,
                "crop_code": int(crop_code),
                "crop_name": crop_names.get(crop_code, f"crop_{crop_code}"),
                "area_planted": round(float(area_planted), 4),
                "harvest_quantity": round(float(harvest_qty), 2),
            })

    return pl.DataFrame(records)


def ingest_country_data(
    country: str | Country,
    output_dir: str | Path,
    force: bool = False,
) -> tuple[Path, ProcessingManifest]:
    """Main entry point for data ingestion.

    Downloads or generates LSMS data and returns path and manifest.

    Args:
        country: Country code.
        output_dir: Base output directory.
        force: Force re-download.

    Returns:
        Tuple of (raw data path, processing manifest).
    """
    if isinstance(country, str):
        country = Country(country.lower())

    output_dir = Path(output_dir)
    raw_path, is_synthetic = download_lsms_data(country, output_dir, force)

    # Load manifest
    import json

    manifest_path = output_dir / country.value / "ingest_manifest.json"
    with open(manifest_path) as f:
        manifest_data = json.load(f)

    # Count households
    household_file = raw_path / "household.parquet"
    if household_file.exists():
        hh_df = pl.read_parquet(household_file)
        num_households = hh_df["household_id"].n_unique()
        num_waves = hh_df["wave"].n_unique()
    else:
        num_households = 0
        num_waves = 0

    manifest = ProcessingManifest(
        country=country.value,
        processing_timestamp=datetime.fromisoformat(manifest_data["processing_timestamp"]),
        source_url=manifest_data["source_url"],
        data_version=manifest_data["data_version"],
        num_households=num_households,
        num_waves=num_waves,
        output_files={
            "household": str(raw_path / "household.parquet"),
            "individual": str(raw_path / "individual.parquet"),
            "plot": str(raw_path / "plot.parquet"),
            "plot_crop": str(raw_path / "plot_crop.parquet"),
        },
        synthetic_data=manifest_data["synthetic_data"],
        git_commit=manifest_data.get("git_commit"),
    )

    return raw_path, manifest
