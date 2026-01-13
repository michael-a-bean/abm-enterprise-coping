"""ETL pipeline for LSMS-ISA data ingestion and processing.

This package provides tools for:
- Downloading LSMS-ISA harmonized panel data
- Processing raw data into canonical Parquet tables
- Deriving target variables for ABM validation
"""

from etl.canonical import (
    create_canonical_tables,
    validate_canonical_tables,
)
from etl.derive import (
    build_derived_targets,
    derive_asset_targets,
    derive_enterprise_targets,
    derive_household_targets,
    derive_price_exposure,
    validate_derived_targets,
)
from etl.ingest import (
    download_lsms_data,
    ingest_country_data,
)
from etl.prices import (
    compute_household_price_exposure,
    compute_price_changes,
    get_price_exposure_weights,
    load_crop_prices,
)
from etl.schemas import (
    COUNTRY_CONFIG,
    AssetTargetsRecord,
    Country,
    CropPrice,
    EnterpriseTargetsRecord,
    HouseholdRecord,
    HouseholdTargetsRecord,
    IndividualRecord,
    PlotCropRecord,
    PlotRecord,
    PriceExposureRecord,
    ProcessingManifest,
)

__all__ = [
    # Schemas
    "Country",
    "COUNTRY_CONFIG",
    "HouseholdRecord",
    "IndividualRecord",
    "PlotRecord",
    "PlotCropRecord",
    "EnterpriseTargetsRecord",
    "AssetTargetsRecord",
    "PriceExposureRecord",
    "HouseholdTargetsRecord",
    "CropPrice",
    "ProcessingManifest",
    # Ingestion
    "download_lsms_data",
    "ingest_country_data",
    # Canonical
    "create_canonical_tables",
    "validate_canonical_tables",
    # Prices
    "load_crop_prices",
    "compute_price_changes",
    "get_price_exposure_weights",
    "compute_household_price_exposure",
    # Derive
    "derive_enterprise_targets",
    "derive_asset_targets",
    "derive_price_exposure",
    "derive_household_targets",
    "build_derived_targets",
    "validate_derived_targets",
]
