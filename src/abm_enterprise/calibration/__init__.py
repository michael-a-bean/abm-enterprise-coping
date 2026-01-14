"""Calibration subsystem for fitting distributions from LSMS data.

This module provides functionality to:
- Fit distributional parameters from LSMS-derived targets
- Output reusable calibration artifacts (JSON)
- Support cross-country calibration (calibrate on Tanzania, validate on Ethiopia)
"""

from abm_enterprise.calibration.schemas import (
    CalibrationArtifact,
    CalibrationManifest,
    CreditModelSpec,
    DistributionSpec,
    TransitionRates,
)
from abm_enterprise.calibration.fit import (
    fit_calibration,
    fit_asset_distribution,
    fit_credit_model,
    fit_shock_distribution,
    compute_transition_rates,
)

__all__ = [
    "CalibrationArtifact",
    "CalibrationManifest",
    "CreditModelSpec",
    "DistributionSpec",
    "TransitionRates",
    "fit_calibration",
    "fit_asset_distribution",
    "fit_credit_model",
    "fit_shock_distribution",
    "compute_transition_rates",
]
