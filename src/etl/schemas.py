"""Pydantic schemas for ETL data validation.

Defines the data structures for LSMS-ISA data ingestion and processing,
ensuring type safety and validation at each stage of the pipeline.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Country(str, Enum):
    """Supported countries in the LSMS-ISA harmonized data."""

    TANZANIA = "tanzania"
    ETHIOPIA = "ethiopia"


# Country-specific configuration
COUNTRY_CONFIG = {
    Country.TANZANIA: {
        "waves": [1, 2, 3, 4],
        "wave_years": {1: 2008, 2: 2010, 3: 2012, 4: 2014},
        "id_column": "y4_hhid",
    },
    Country.ETHIOPIA: {
        "waves": [1, 2, 3],
        "wave_years": {1: 2011, 2: 2013, 3: 2015},
        "id_column": "household_id",
    },
}


class HouseholdRecord(BaseModel):
    """Schema for household panel data.

    Canonical schema for household-level data across waves.
    """

    household_id: str = Field(..., description="Primary household identifier")
    wave: int = Field(..., ge=1, le=4, description="Survey wave (1-4 for TZ, 1-3 for ET)")
    enterprise_status: int = Field(..., ge=0, le=1, description="Has enterprise (0/1)")
    credit_access: int = Field(..., ge=0, le=1, description="Has credit access (0/1)")
    welfare_proxy: float = Field(..., description="Per-capita consumption (real)")
    region: str = Field(..., description="Administrative region")
    urban: int = Field(..., ge=0, le=1, description="Urban indicator (0/1)")
    household_size: int = Field(..., ge=1, description="Number of household members")

    model_config = {"frozen": True}


class IndividualRecord(BaseModel):
    """Schema for individual-level data within households."""

    household_id: str = Field(..., description="Household identifier")
    wave: int = Field(..., ge=1, le=4, description="Survey wave")
    individual_id: str = Field(..., description="Individual identifier")
    relationship_to_head: int = Field(..., description="Relationship code")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    education: int = Field(..., ge=0, description="Education level code")

    model_config = {"frozen": True}


class PlotRecord(BaseModel):
    """Schema for agricultural plot data."""

    household_id: str = Field(..., description="Household identifier")
    wave: int = Field(..., ge=1, le=4, description="Survey wave")
    plot_id: str = Field(..., description="Plot identifier")
    area_hectares: float = Field(..., ge=0, description="Plot area in hectares")
    irrigated: int = Field(..., ge=0, le=1, description="Irrigated (0/1)")

    model_config = {"frozen": True}

    @field_validator("area_hectares")
    @classmethod
    def validate_area(cls, v: float) -> float:
        """Ensure area is non-negative."""
        if v < 0:
            raise ValueError("Area cannot be negative")
        return v


class PlotCropRecord(BaseModel):
    """Schema for crop-level data on plots."""

    household_id: str = Field(..., description="Household identifier")
    wave: int = Field(..., ge=1, le=4, description="Survey wave")
    plot_id: str = Field(..., description="Plot identifier")
    crop_code: int = Field(..., description="Crop code")
    crop_name: str = Field(..., description="Crop name")
    area_planted: float = Field(..., ge=0, description="Area planted in hectares")
    harvest_quantity: float = Field(..., ge=0, description="Harvest quantity in kg")

    model_config = {"frozen": True}


class EnterpriseTargetsRecord(BaseModel):
    """Schema for derived enterprise targets."""

    household_id: str = Field(..., description="Household identifier")
    enterprise_indicator: int = Field(..., ge=0, le=1, description="Ever had enterprise")
    enterprise_persistence: float = Field(
        ..., ge=0, le=1, description="Proportion of waves with enterprise"
    )
    classification: Literal["stayer", "coper", "none"] = Field(
        ..., description="Household classification"
    )

    model_config = {"frozen": True}


class AssetTargetsRecord(BaseModel):
    """Schema for derived asset targets."""

    household_id: str = Field(..., description="Household identifier")
    wave: int = Field(..., ge=1, le=4, description="Survey wave")
    asset_index: float = Field(..., description="Asset index (PCA proxy)")
    asset_quintile: int = Field(..., ge=1, le=5, description="Asset quintile (1-5)")

    model_config = {"frozen": True}


class PriceExposureRecord(BaseModel):
    """Schema for price exposure data."""

    household_id: str = Field(..., description="Household identifier")
    wave: int = Field(..., ge=1, le=4, description="Survey wave")
    price_exposure: float = Field(..., description="Weighted price exposure")

    model_config = {"frozen": True}


class HouseholdTargetsRecord(BaseModel):
    """Schema for merged household targets for ABM validation."""

    household_id: str = Field(..., description="Household identifier")
    wave: int = Field(..., ge=1, le=4, description="Survey wave")
    enterprise_indicator: int = Field(..., ge=0, le=1, description="Has enterprise this wave")
    enterprise_persistence: float = Field(
        ..., ge=0, le=1, description="Overall enterprise persistence"
    )
    classification: Literal["stayer", "coper", "none"] = Field(
        ..., description="Household classification"
    )
    asset_index: float = Field(..., description="Asset index")
    asset_quintile: int = Field(..., ge=1, le=5, description="Asset quintile")
    credit_access: int = Field(..., ge=0, le=1, description="Credit access")
    price_exposure: float = Field(..., description="Price exposure")
    welfare_proxy: float = Field(..., description="Welfare proxy")

    model_config = {"frozen": True}


class CropPrice(BaseModel):
    """Schema for crop price data."""

    crop_code: int = Field(..., description="Crop code")
    crop_name: str = Field(..., description="Crop name")
    year: int = Field(..., description="Price year")
    price_per_kg: float = Field(..., ge=0, description="Price per kg in local currency")

    model_config = {"frozen": True}


class ProcessingManifest(BaseModel):
    """Manifest for ETL processing run."""

    country: str = Field(..., description="Country code")
    processing_timestamp: datetime = Field(..., description="When processing occurred")
    source_url: str = Field(..., description="Data source URL")
    data_version: str = Field(..., description="Data version")
    num_households: int = Field(..., ge=0, description="Number of households processed")
    num_waves: int = Field(..., ge=1, description="Number of waves")
    output_files: dict[str, str] = Field(..., description="Map of output file paths")
    synthetic_data: bool = Field(
        default=False, description="Whether synthetic data was used"
    )
    git_commit: str | None = Field(None, description="Git commit hash if available")

    model_config = {"frozen": True}
