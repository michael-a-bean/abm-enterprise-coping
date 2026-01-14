"""Pydantic schemas for simulation data validation.

Defines the data structures used throughout the simulation,
ensuring type safety and validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EnterpriseStatus(str, Enum):
    """Enterprise participation status."""

    NO_ENTERPRISE = "no_enterprise"
    HAS_ENTERPRISE = "has_enterprise"


class PolicyType(str, Enum):
    """Types of policy interventions."""

    NONE = "none"
    CREDIT_ACCESS = "credit_access"
    PRICE_SUPPORT = "price_support"
    ASSET_TRANSFER = "asset_transfer"


class HouseholdState(BaseModel):
    """State of a household at a given wave.

    This schema represents the observable state of a household
    that is used for decision-making in the simulation.

    Attributes:
        household_id: Unique identifier for the household.
        wave: Survey wave number.
        assets: Asset index (standardized).
        credit_access: Whether household has access to credit (0 or 1).
        enterprise_status: Current enterprise participation status.
        price_exposure: Price shock exposure measure (negative = adverse).
    """

    household_id: str = Field(..., description="Unique household identifier")
    wave: int = Field(..., ge=1, description="Survey wave number")
    assets: float = Field(..., description="Asset index (standardized)")
    credit_access: int = Field(..., ge=0, le=1, description="Credit access indicator")
    enterprise_status: EnterpriseStatus = Field(
        ..., description="Enterprise participation status"
    )
    price_exposure: float = Field(..., description="Price shock exposure measure")

    model_config = {"frozen": True}


class SimulationConfig(BaseModel):
    """Configuration for a simulation run.

    Attributes:
        country: Country code (e.g., 'tanzania', 'ethiopia').
        scenario: Scenario name.
        seed: Random seed for reproducibility.
        num_waves: Number of waves to simulate.
        policy_type: Type of policy intervention.
        price_exposure_threshold: Threshold for price shock trigger.
        asset_threshold_percentile: Asset percentile threshold for decisions.
    """

    country: str = Field(..., description="Country code")
    scenario: str = Field(default="baseline", description="Scenario name")
    seed: int = Field(..., ge=0, description="Random seed")
    num_waves: int = Field(default=4, ge=1, description="Number of waves to simulate")
    policy_type: PolicyType = Field(
        default=PolicyType.NONE, description="Policy intervention type"
    )
    price_exposure_threshold: float = Field(
        default=-0.1, description="Price shock threshold for coping decisions"
    )
    asset_threshold_percentile: int = Field(
        default=40, ge=0, le=100, description="Asset percentile threshold"
    )
    stayer_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Enterprise persistence threshold for stayer classification"
    )

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Validate country code."""
        allowed = {"tanzania", "ethiopia"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"Country must be one of {allowed}")
        return v_lower


class ClassificationType(str, Enum):
    """Household classification based on enterprise behavior."""

    STAYER = "stayer"
    COPER = "coper"
    NONE = "none"


class OutputRecord(BaseModel):
    """Schema for simulation output records.

    Aligned with VALIDATION_CONTRACT.md requirements for
    compatibility with analysis pipelines and regression validation.

    Attributes:
        household_id: Unique household identifier.
        wave: Survey wave number.
        country: Country code.
        scenario: Scenario name.
        enterprise_status: Current enterprise participation (0/1).
        enterprise_entry: Whether enterprise was entered this wave (0/1).
        price_exposure: Price shock exposure measure.
        assets_index: Asset index value (standardized).
        credit_access: Credit access indicator (0/1).
        asset_quintile: Asset quintile (1-5).
        classification: Household classification (stayer/coper/none).
        action_taken: Action taken this wave.
        policy_applied: Whether policy was applied.
        crop_count: Number of crops grown.
        land_area_ha: Total land area in hectares.
        enterprise_persistence: Fraction of waves with enterprise.
        welfare_proxy: Welfare proxy value.
    """

    household_id: str = Field(..., description="Unique household identifier")
    wave: int = Field(..., ge=1, description="Survey wave number")
    country: str = Field(default="", description="Country code")
    scenario: str = Field(default="", description="Scenario name")
    enterprise_status: int = Field(
        ..., ge=0, le=1, description="Enterprise participation (0/1)"
    )
    enterprise_entry: int = Field(
        default=0, ge=0, le=1, description="Enterprise entry this wave (0/1)"
    )
    price_exposure: float = Field(..., description="Price shock exposure")
    assets_index: float = Field(..., description="Asset index value")
    credit_access: int = Field(..., ge=0, le=1, description="Credit access indicator")
    asset_quintile: int = Field(
        default=3, ge=1, le=5, description="Asset quintile (1-5)"
    )
    classification: str = Field(
        default="none", description="Household classification (stayer/coper/none)"
    )
    action_taken: str = Field(..., description="Action taken this wave")
    policy_applied: int = Field(
        ..., ge=0, le=1, description="Whether policy was applied"
    )
    crop_count: int = Field(default=1, ge=0, description="Number of crops grown")
    land_area_ha: float = Field(
        default=1.0, ge=0, description="Total land area in hectares"
    )
    enterprise_persistence: float = Field(
        default=0.0, ge=0, le=1, description="Fraction of waves with enterprise"
    )
    welfare_proxy: float = Field(default=0.0, description="Welfare proxy value")

    model_config = {"frozen": True}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return self.model_dump()

    @classmethod
    def required_columns(cls) -> list[str]:
        """Get list of required column names for validation contract.

        Returns:
            List of column names required by the validation contract.
        """
        return [
            "household_id",
            "wave",
            "country",
            "scenario",
            "enterprise_status",
            "enterprise_entry",
            "price_exposure",
            "assets_index",
            "credit_access",
            "asset_quintile",
            "classification",
            "action_taken",
            "policy_applied",
        ]


class WaveConfig(BaseModel):
    """Configuration for a single survey wave.

    Attributes:
        wave: Wave number.
        year: Calendar year of the wave.
    """

    wave: int = Field(..., ge=1, description="Wave number")
    year: int = Field(..., description="Calendar year")


class CountryConfig(BaseModel):
    """Country-specific configuration.

    Attributes:
        country: Country code.
        waves: List of wave numbers.
        wave_years: Mapping of wave to year.
        price_exposure_threshold: Price threshold.
        asset_threshold_percentile: Asset percentile.
    """

    country: str
    waves: list[int]
    wave_years: dict[int, int]
    price_exposure_threshold: float = -0.1
    asset_threshold_percentile: int = 40

    @classmethod
    def from_yaml(cls, path: str) -> "CountryConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
