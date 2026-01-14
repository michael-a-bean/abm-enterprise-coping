"""Pydantic schemas for calibration artifacts.

Defines the data structures for calibration outputs,
ensuring reproducibility and portability across runs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DistributionFamily(str, Enum):
    """Supported distribution families for calibration."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    T = "t"
    SKEW_NORMAL = "skew_normal"


class CopulaType(str, Enum):
    """Supported copula families for dependence structure."""

    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    FRANK = "frank"
    GUMBEL = "gumbel"
    INDEPENDENT = "independent"  # No dependence (marginals only)


class StandardizationMethod(str, Enum):
    """Standardization methods for distributions."""

    ZSCORE = "zscore"
    MINMAX = "minmax"
    NONE = "none"


class GoodnessOfFitResult(BaseModel):
    """Kolmogorov-Smirnov goodness-of-fit test result.

    Attributes:
        statistic: K-S test statistic (max absolute difference between CDFs).
        p_value: P-value for the test (reject fit if < alpha).
        n_samples: Number of samples used in the test.
        passed: Whether the fit passed at alpha=0.05.
    """

    statistic: float = Field(..., ge=0, le=1, description="K-S test statistic")
    p_value: float = Field(..., ge=0, le=1, description="P-value")
    n_samples: int = Field(..., ge=1, description="Sample size")
    passed: bool = Field(..., description="Passed at alpha=0.05")


class CopulaSpec(BaseModel):
    """Specification for fitted copula capturing dependence structure.

    Attributes:
        copula_type: Type of copula (gaussian, student_t, clayton, etc.).
        correlation_matrix: Correlation matrix for Gaussian/Student-t copulas.
        theta: Copula parameter for Archimedean copulas (Clayton, Frank, Gumbel).
        df: Degrees of freedom for Student-t copula.
        variable_names: Ordered list of variable names in the copula.
    """

    copula_type: CopulaType = Field(..., description="Copula family")
    correlation_matrix: list[list[float]] | None = Field(
        default=None,
        description="Correlation matrix (for Gaussian/Student-t copulas)",
    )
    theta: float | None = Field(
        default=None,
        description="Copula parameter (for Archimedean copulas)",
    )
    df: float | None = Field(
        default=None,
        description="Degrees of freedom (for Student-t copula)",
    )
    variable_names: list[str] = Field(
        ...,
        description="Ordered variable names in the copula",
    )

    @field_validator("correlation_matrix")
    @classmethod
    def validate_correlation_matrix(
        cls, v: list[list[float]] | None
    ) -> list[list[float]] | None:
        """Ensure correlation matrix is valid (symmetric, diagonal=1)."""
        if v is None:
            return v
        n = len(v)
        for i, row in enumerate(v):
            if len(row) != n:
                raise ValueError("Correlation matrix must be square")
            if abs(row[i] - 1.0) > 1e-6:
                raise ValueError("Diagonal elements must be 1.0")
        return v


class DistributionSpec(BaseModel):
    """Specification for a fitted distribution.

    Attributes:
        family: Distribution family (normal, lognormal, t, skew_normal).
        params: Distribution parameters (varies by family).
        standardization: Standardization method applied to data.
        raw_stats: Raw statistics before standardization.
    """

    family: DistributionFamily = Field(..., description="Distribution family")
    params: dict[str, float] = Field(..., description="Distribution parameters")
    standardization: StandardizationMethod = Field(
        default=StandardizationMethod.NONE,
        description="Standardization method",
    )
    raw_stats: dict[str, float] = Field(
        default_factory=dict,
        description="Raw statistics (mean, std, min, max) before standardization",
    )
    goodness_of_fit: GoodnessOfFitResult | None = Field(
        default=None,
        description="K-S goodness-of-fit test result for this distribution",
    )

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure distribution parameters are valid."""
        if not v:
            raise ValueError("Distribution params cannot be empty")
        return v

    def get_scipy_params(self) -> dict[str, float]:
        """Convert to scipy distribution parameters.

        Returns:
            Dict ready for scipy.stats distribution initialization.
        """
        if self.family == DistributionFamily.NORMAL:
            return {"loc": self.params["mean"], "scale": self.params["std"]}
        elif self.family == DistributionFamily.LOGNORMAL:
            return {
                "s": self.params["sigma"],
                "scale": self.params.get("scale", 1.0),
                "loc": self.params.get("loc", 0.0),
            }
        elif self.family == DistributionFamily.T:
            return {
                "df": self.params["df"],
                "loc": self.params["loc"],
                "scale": self.params["scale"],
            }
        elif self.family == DistributionFamily.SKEW_NORMAL:
            return {
                "a": self.params["skew"],
                "loc": self.params["loc"],
                "scale": self.params["scale"],
            }
        else:
            return self.params


class CreditModelSpec(BaseModel):
    """Specification for credit access model.

    Logistic regression model: P(credit=1) = logit^{-1}(intercept + sum(coef * feature))

    Attributes:
        type: Model type (currently only 'logistic' supported).
        coefficients: Feature name to coefficient mapping.
        intercept: Model intercept.
        feature_names: Ordered list of feature names.
        model_metrics: Model fit metrics (accuracy, AUC, etc.).
    """

    type: str = Field(default="logistic", description="Model type")
    coefficients: dict[str, float] = Field(..., description="Feature coefficients")
    intercept: float = Field(..., description="Model intercept")
    feature_names: list[str] = Field(..., description="Ordered feature names")
    model_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Model fit metrics",
    )

    def predict_proba(self, features: dict[str, float]) -> float:
        """Predict probability of credit access.

        Args:
            features: Feature values keyed by name.

        Returns:
            Probability of credit access (0-1).
        """
        import math

        linear = self.intercept
        for name in self.feature_names:
            if name in features:
                linear += self.coefficients.get(name, 0.0) * features[name]

        # Logistic sigmoid
        return 1.0 / (1.0 + math.exp(-linear))


class TransitionRates(BaseModel):
    """Observed enterprise transition rates.

    Attributes:
        enter_rate: Proportion of non-enterprise HHs that enter.
        exit_rate: Proportion of enterprise HHs that exit.
        stay_rate: Proportion that maintain current status.
        enter_count: Count of enter transitions.
        exit_count: Count of exit transitions.
        stay_count: Count of stay transitions.
    """

    enter_rate: float = Field(..., ge=0, le=1, description="Enterprise entry rate")
    exit_rate: float = Field(..., ge=0, le=1, description="Enterprise exit rate")
    stay_rate: float = Field(..., ge=0, le=1, description="Status maintenance rate")
    enter_count: int = Field(..., ge=0, description="Entry transition count")
    exit_count: int = Field(..., ge=0, description="Exit transition count")
    stay_count: int = Field(..., ge=0, description="Stay transition count")


class EnterpriseBaseline(BaseModel):
    """Baseline enterprise statistics.

    Attributes:
        prevalence: Overall enterprise prevalence across all waves.
        prevalence_by_wave: Enterprise prevalence by wave.
        entry_rate: Average entry rate across transitions.
        exit_rate: Average exit rate across transitions.
    """

    prevalence: float = Field(..., ge=0, le=1, description="Overall enterprise rate")
    prevalence_by_wave: dict[int, float] = Field(
        default_factory=dict,
        description="Enterprise rate by wave",
    )
    entry_rate: float = Field(..., ge=0, le=1, description="Average entry rate")
    exit_rate: float = Field(..., ge=0, le=1, description="Average exit rate")


class CalibrationArtifact(BaseModel):
    """Complete calibration artifact.

    Contains all fitted distributions and models needed to generate
    synthetic panels that match the calibration data characteristics.

    Attributes:
        country_source: Country used for calibration.
        created_at: Timestamp of calibration.
        git_commit: Git commit hash for reproducibility.
        waves: List of wave numbers in calibration data.
        n_households: Number of unique households in calibration data.
        n_observations: Total observations in calibration data.
        assets_distribution: Fitted asset distribution.
        shock_distribution: Fitted price shock distribution.
        shock_distribution_by_wave: Optional per-wave shock distributions.
        credit_model: Fitted credit access model.
        enterprise_baseline: Baseline enterprise statistics.
        transition_rates: Observed transition rates.
        household_intercept_distribution: Optional random effects distribution.
        additional_metadata: Any additional calibration metadata.
    """

    country_source: str = Field(..., description="Calibration country")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Calibration timestamp",
    )
    git_commit: str = Field(..., description="Git commit hash")
    waves: list[int] = Field(..., description="Waves in calibration data")
    n_households: int = Field(..., ge=0, description="Number of households")
    n_observations: int = Field(..., ge=0, description="Total observations")

    assets_distribution: DistributionSpec = Field(
        ..., description="Fitted asset distribution"
    )
    shock_distribution: DistributionSpec = Field(
        ..., description="Fitted shock distribution (pooled)"
    )
    shock_distribution_by_wave: dict[int, DistributionSpec] | None = Field(
        default=None,
        description="Per-wave shock distributions",
    )
    credit_model: CreditModelSpec = Field(..., description="Credit access model")
    enterprise_baseline: EnterpriseBaseline = Field(
        ..., description="Enterprise baseline stats"
    )
    transition_rates: TransitionRates = Field(
        ..., description="Observed transition rates"
    )

    household_intercept_distribution: DistributionSpec | None = Field(
        default=None,
        description="Household random effects distribution",
    )

    # Dependence structure (Gemini recommendation: preserve correlation)
    copula: CopulaSpec | None = Field(
        default=None,
        description="Copula specification for joint distribution of assets, credit, shocks",
    )

    additional_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def save(self, path: str) -> None:
        """Save calibration artifact to JSON file.

        Args:
            path: Output file path.
        """
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "CalibrationArtifact":
        """Load calibration artifact from JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded CalibrationArtifact.
        """
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class CalibrationManifest(BaseModel):
    """Manifest for calibration run provenance.

    Attributes:
        calibration_id: Unique identifier for this calibration.
        git_commit: Git commit hash.
        created_at: Timestamp.
        country: Country calibrated.
        input_data_path: Path to input data.
        input_data_hash: Hash of input data for integrity.
        output_artifact_path: Path to output calibration artifact.
        parameters_used: Configuration parameters used.
    """

    calibration_id: str = Field(..., description="Unique calibration ID")
    git_commit: str = Field(..., description="Git commit hash")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Calibration timestamp",
    )
    country: str = Field(..., description="Calibration country")
    input_data_path: str = Field(..., description="Input data path")
    input_data_hash: str = Field(..., description="MD5 hash of input data")
    output_artifact_path: str = Field(..., description="Output artifact path")
    parameters_used: dict[str, Any] = Field(
        default_factory=dict,
        description="Calibration parameters",
    )

    def save(self, path: str) -> None:
        """Save manifest to JSON file."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)
