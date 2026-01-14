"""Synthetic panel generation for the generative microsimulation.

This module provides functionality to generate synthetic household panels
from calibration artifacts, with explicit transition dynamics across waves.

The ABM uses synthetic states (not real LSMS data) as agent states.
Real LSMS data is used only for calibration and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from abm_enterprise.utils.logging import get_logger
from abm_enterprise.utils.rng import get_rng, set_seed

if TYPE_CHECKING:
    from abm_enterprise.calibration.schemas import CalibrationArtifact, CopulaSpec

logger = get_logger(__name__)


@dataclass
class TransitionConfig:
    """Configuration for state transition dynamics.

    Transition equations:
    - Assets: A_{t+1} = rho * A_t + lambda * I(E_t=1) + delta * P_t + epsilon
    - Credit: C_{t+1} ~ Bernoulli(alpha * C_t + (1-alpha) * logit^{-1}(beta * A_{t+1}))
    - Shock: P_t = mu_t + nu (systematic + idiosyncratic)

    Attributes:
        rho_assets: Asset persistence coefficient (0-1).
        lambda_enterprise_assets: Enterprise effect on assets.
        delta_shock_assets: Price shock effect on assets.
        asset_noise_sd: Standard deviation of asset innovation.
        credit_stickiness: Credit persistence coefficient (0-1).
        shock_sd_idiosyncratic: Idiosyncratic shock standard deviation.
    """

    rho_assets: float = 0.85
    lambda_enterprise_assets: float = 0.05
    delta_shock_assets: float = 0.10
    asset_noise_sd: float = 0.15
    credit_stickiness: float = 0.70
    shock_sd_idiosyncratic: float = 0.10

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.rho_assets <= 1:
            raise ValueError(f"rho_assets must be in [0,1], got {self.rho_assets}")
        if not 0 <= self.credit_stickiness <= 1:
            raise ValueError(
                f"credit_stickiness must be in [0,1], got {self.credit_stickiness}"
            )
        if self.asset_noise_sd < 0:
            raise ValueError(f"asset_noise_sd must be >= 0, got {self.asset_noise_sd}")
        if self.shock_sd_idiosyncratic < 0:
            raise ValueError(
                f"shock_sd_idiosyncratic must be >= 0, got {self.shock_sd_idiosyncratic}"
            )


@dataclass
class SyntheticPanelConfig:
    """Configuration for synthetic panel generation.

    Attributes:
        n_households: Number of households to generate.
        waves: List of wave numbers.
        seed: Random seed for reproducibility.
        transition: Transition dynamics configuration.
        initial_enterprise_rate: Initial enterprise prevalence (if not from calibration).
        use_copula: Whether to use copula for correlated initial states.
    """

    n_households: int = 1000
    waves: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    seed: int = 42
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    initial_enterprise_rate: float | None = None
    use_copula: bool = True  # Gemini recommendation: preserve correlation structure


class SyntheticPanelGenerator:
    """Generator for synthetic household panels from calibration artifacts.

    Uses calibration artifact to draw initial states from calibrated distributions,
    then evolves states according to explicit transition dynamics.

    Attributes:
        calibration: Calibration artifact with fitted distributions.
        config: Panel generation configuration.
        rng: Random number generator.
    """

    def __init__(
        self,
        calibration: "CalibrationArtifact",
        config: SyntheticPanelConfig | None = None,
    ) -> None:
        """Initialize generator.

        Args:
            calibration: Calibration artifact with fitted distributions.
            config: Panel generation configuration.
        """
        self.calibration = calibration
        self.config = config or SyntheticPanelConfig()
        self.config.transition.validate()

        set_seed(self.config.seed)
        self.rng = get_rng()

        logger.info(
            "Initialized synthetic panel generator",
            n_households=self.config.n_households,
            waves=self.config.waves,
            seed=self.config.seed,
        )

    def generate(self) -> pd.DataFrame:
        """Generate synthetic household panel.

        Returns:
            DataFrame with synthetic panel data.
        """
        n = self.config.n_households
        waves = self.config.waves
        n_waves = len(waves)

        # Initialize storage
        data = {
            "household_id": [],
            "wave": [],
            "assets_index": [],
            "credit_access": [],
            "enterprise_status": [],
            "price_exposure": [],
        }

        # Generate household IDs
        household_ids = [f"SH_{i:05d}" for i in range(n)]

        # Generate initial states (wave 1)
        # Use copula-based sampling if available and enabled
        if (
            self.config.use_copula
            and self.calibration.copula is not None
            and self.calibration.copula.copula_type.value != "independent"
        ):
            logger.info("Using copula-based sampling for initial states")
            initial_assets, initial_credit, initial_shocks = self._draw_initial_states_copula(n)
        else:
            logger.info("Using independent marginal sampling for initial states")
            initial_assets = self._draw_initial_assets(n)
            initial_credit = self._draw_initial_credit(initial_assets)
            initial_shocks = self._draw_shocks(waves[0], n)

        initial_enterprise = self._draw_initial_enterprise(n)

        # Store wave 1
        current_assets = initial_assets.copy()
        current_credit = initial_credit.copy()
        current_enterprise = initial_enterprise.copy()

        for wave_idx, wave in enumerate(waves):
            # Draw shocks for this wave
            shocks = self._draw_shocks(wave, n)

            if wave_idx == 0:
                # First wave uses initial values
                pass
            else:
                # Evolve states according to transition dynamics
                current_assets = self._evolve_assets(
                    current_assets,
                    current_enterprise,
                    shocks,
                )
                current_credit = self._evolve_credit(
                    current_credit,
                    current_assets,
                )
                # Enterprise status will be updated by policy during simulation
                # For initial generation, we use simple transition logic
                current_enterprise = self._simple_enterprise_transition(
                    current_enterprise,
                    current_assets,
                    current_credit,
                    shocks,
                )

            # Store this wave's data
            data["household_id"].extend(household_ids)
            data["wave"].extend([wave] * n)
            data["assets_index"].extend(current_assets.tolist())
            data["credit_access"].extend(current_credit.tolist())
            data["enterprise_status"].extend(current_enterprise.tolist())
            data["price_exposure"].extend(shocks.tolist())

        df = pd.DataFrame(data)

        # Ensure correct dtypes
        df["household_id"] = df["household_id"].astype(str)
        df["wave"] = df["wave"].astype(int)
        df["credit_access"] = df["credit_access"].astype(int)
        df["enterprise_status"] = df["enterprise_status"].astype(int)

        logger.info(
            "Generated synthetic panel",
            n_households=n,
            n_waves=n_waves,
            n_observations=len(df),
        )

        return df

    def _draw_initial_assets(self, n: int) -> np.ndarray:
        """Draw initial asset values from calibrated distribution.

        Args:
            n: Number of households.

        Returns:
            Array of initial asset values.
        """
        dist = self.calibration.assets_distribution
        params = dist.params

        if dist.family.value == "normal":
            assets = self.rng.normal(
                loc=params["mean"],
                scale=params["std"],
                size=n,
            )
        elif dist.family.value == "lognormal":
            assets = self.rng.lognormal(
                mean=params.get("mu", 0.0),
                sigma=params.get("sigma", 1.0),
                size=n,
            )
            assets = assets + params.get("loc", 0.0)
        elif dist.family.value == "t":
            from scipy import stats
            assets = stats.t.rvs(
                df=params["df"],
                loc=params["loc"],
                scale=params["scale"],
                size=n,
                random_state=self.rng,
            )
        else:
            # Fallback to normal
            assets = self.rng.normal(0, 1, n)

        return assets.astype(np.float64)

    def _draw_initial_credit(self, assets: np.ndarray) -> np.ndarray:
        """Draw initial credit access from calibrated model.

        Args:
            assets: Asset values.

        Returns:
            Array of credit access indicators (0/1).
        """
        model = self.calibration.credit_model

        # Compute probability for each household
        probs = np.zeros(len(assets))
        for i, asset in enumerate(assets):
            probs[i] = model.predict_proba({"assets_index": float(asset)})

        # Draw binary outcomes
        credit = (self.rng.random(len(assets)) < probs).astype(int)
        return credit

    def _draw_initial_enterprise(self, n: int) -> np.ndarray:
        """Draw initial enterprise status from calibrated baseline.

        Args:
            n: Number of households.

        Returns:
            Array of enterprise status indicators (0/1).
        """
        baseline = self.calibration.enterprise_baseline
        initial_rate = self.config.initial_enterprise_rate

        if initial_rate is None:
            # Use wave 1 prevalence if available, else overall
            if 1 in baseline.prevalence_by_wave:
                rate = baseline.prevalence_by_wave[1]
            else:
                rate = baseline.prevalence

        else:
            rate = initial_rate

        enterprise = (self.rng.random(n) < rate).astype(int)
        return enterprise

    def _draw_initial_states_copula(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw correlated initial states using fitted copula.

        Uses the fitted copula to generate samples with the correct
        dependence structure, then transforms back to original marginals.

        Args:
            n: Number of households.

        Returns:
            Tuple of (assets, credit, shocks) arrays.
        """
        copula = self.calibration.copula
        if copula is None:
            raise ValueError("No copula available for sampling")

        # Get variable names and their order in the copula
        var_names = copula.variable_names
        n_vars = len(var_names)

        # Sample from copula to get uniform marginals
        if copula.copula_type.value == "gaussian":
            # Sample from multivariate normal with copula correlation
            corr_matrix = np.array(copula.correlation_matrix)
            normal_samples = self.rng.multivariate_normal(
                mean=np.zeros(n_vars),
                cov=corr_matrix,
                size=n,
            )
            # Transform to uniform using normal CDF
            uniform_samples = stats.norm.cdf(normal_samples)

        elif copula.copula_type.value == "student_t":
            # Sample from multivariate t-distribution
            corr_matrix = np.array(copula.correlation_matrix)
            df = copula.df or 4.0

            # Generate multivariate t using the standard method:
            # X = sqrt(df/W) * Z where Z ~ MVN(0, Sigma) and W ~ chi2(df)
            normal_samples = self.rng.multivariate_normal(
                mean=np.zeros(n_vars),
                cov=corr_matrix,
                size=n,
            )
            w = self.rng.chisquare(df, size=n)
            t_samples = normal_samples * np.sqrt(df / w)[:, np.newaxis]

            # Transform to uniform using t CDF
            uniform_samples = stats.t.cdf(t_samples, df=df)

        elif copula.copula_type.value in ("clayton", "frank", "gumbel"):
            # Archimedean copulas - bivariate only
            if n_vars != 2:
                logger.warning(
                    f"Archimedean copula only supports 2 variables, "
                    f"falling back to Gaussian for {n_vars} variables"
                )
                # Fall back to Gaussian approximation
                corr_matrix = np.eye(n_vars)
                normal_samples = self.rng.multivariate_normal(
                    mean=np.zeros(n_vars),
                    cov=corr_matrix,
                    size=n,
                )
                uniform_samples = stats.norm.cdf(normal_samples)
            else:
                # Use simple conditional sampling for Archimedean copulas
                theta = copula.theta or 1.0
                u1 = self.rng.random(n)

                if copula.copula_type.value == "clayton":
                    # Clayton copula conditional sampling
                    v = self.rng.random(n)
                    u2 = ((u1 ** (-theta) * (v ** (-theta / (1 + theta)) - 1) + 1)
                          ** (-1 / theta))
                elif copula.copula_type.value == "frank":
                    # Frank copula - use approximation
                    v = self.rng.random(n)
                    u2 = -np.log(1 + v * (np.exp(-theta) - 1) /
                                (np.exp(-theta * u1) + v * (1 - np.exp(-theta * u1)))) / theta
                    u2 = np.clip(u2, 0.001, 0.999)
                else:  # Gumbel
                    # Gumbel copula - use approximation
                    v = self.rng.random(n)
                    u2 = np.exp(-(-np.log(u1)) ** (1 / theta) * ((-np.log(v)) ** (1 / theta)))
                    u2 = np.clip(u2, 0.001, 0.999)

                uniform_samples = np.column_stack([u1, u2])
        else:
            # Independent - just draw uniform samples
            uniform_samples = self.rng.random((n, n_vars))

        # Transform uniform samples to original marginals
        assets = self._transform_marginal_assets(uniform_samples, var_names)
        credit = self._transform_marginal_credit(uniform_samples, var_names)
        shocks = self._transform_marginal_shocks(uniform_samples, var_names)

        logger.debug(
            "Copula sampling completed",
            copula_type=copula.copula_type.value,
            n_samples=n,
            asset_mean=float(assets.mean()),
            credit_rate=float(credit.mean()),
        )

        return assets, credit, shocks

    def _transform_marginal_assets(
        self, uniform_samples: np.ndarray, var_names: list[str]
    ) -> np.ndarray:
        """Transform uniform copula sample to asset marginal.

        Args:
            uniform_samples: Uniform samples from copula (n x n_vars).
            var_names: Variable names in copula.

        Returns:
            Asset values array.
        """
        # Find assets column in copula variables
        assets_idx = None
        for i, name in enumerate(var_names):
            if "asset" in name.lower():
                assets_idx = i
                break

        if assets_idx is None:
            # Assets not in copula, draw independently
            return self._draw_initial_assets(len(uniform_samples))

        # Get uniform values for assets
        u = uniform_samples[:, assets_idx]

        # Transform using inverse CDF of asset distribution
        dist = self.calibration.assets_distribution
        params = dist.params

        if dist.family.value == "normal":
            assets = stats.norm.ppf(u, loc=params["mean"], scale=params["std"])
        elif dist.family.value == "lognormal":
            assets = stats.lognorm.ppf(
                u,
                s=params.get("sigma", 1.0),
                loc=params.get("loc", 0.0),
                scale=params.get("scale", 1.0),
            )
        elif dist.family.value == "t":
            assets = stats.t.ppf(
                u, df=params["df"], loc=params["loc"], scale=params["scale"]
            )
        else:
            # Fallback to normal
            assets = stats.norm.ppf(u)

        return assets.astype(np.float64)

    def _transform_marginal_credit(
        self, uniform_samples: np.ndarray, var_names: list[str]
    ) -> np.ndarray:
        """Transform uniform copula sample to credit marginal.

        Credit is binary, so we threshold the uniform value at the credit rate.

        Args:
            uniform_samples: Uniform samples from copula (n x n_vars).
            var_names: Variable names in copula.

        Returns:
            Credit access indicators (0/1).
        """
        # Find credit column in copula variables
        credit_idx = None
        for i, name in enumerate(var_names):
            if "credit" in name.lower():
                credit_idx = i
                break

        if credit_idx is None:
            # Credit not in copula, derive from assets
            assets = self._transform_marginal_assets(uniform_samples, var_names)
            return self._draw_initial_credit(assets)

        # Get uniform values for credit
        u = uniform_samples[:, credit_idx]

        # Credit access rate from calibration
        credit_rate = self.calibration.credit_model.model_metrics.get("credit_rate", 0.3)

        # Threshold at credit rate
        credit = (u < credit_rate).astype(int)
        return credit

    def _transform_marginal_shocks(
        self, uniform_samples: np.ndarray, var_names: list[str]
    ) -> np.ndarray:
        """Transform uniform copula sample to shock marginal.

        Args:
            uniform_samples: Uniform samples from copula (n x n_vars).
            var_names: Variable names in copula.

        Returns:
            Price shock values.
        """
        n = len(uniform_samples)

        # Find shock/price column in copula variables
        shock_idx = None
        for i, name in enumerate(var_names):
            if "shock" in name.lower() or "price" in name.lower() or "exposure" in name.lower():
                shock_idx = i
                break

        if shock_idx is None:
            # Shocks not in copula, draw independently
            return self._draw_shocks(self.config.waves[0], n)

        # Get uniform values for shocks
        u = uniform_samples[:, shock_idx]

        # Transform using inverse CDF of shock distribution
        dist = self.calibration.shock_distribution
        params = dist.params

        if dist.family.value == "normal":
            shocks = stats.norm.ppf(u, loc=params["mean"], scale=params["std"])
        else:
            # Fallback to normal
            shocks = stats.norm.ppf(u, loc=params.get("mean", 0), scale=params.get("std", 0.15))

        return shocks.astype(np.float64)

    def _draw_shocks(self, wave: int, n: int) -> np.ndarray:
        """Draw price shock values for a wave.

        Shocks have systematic (wave-level) and idiosyncratic components.

        Args:
            wave: Wave number.
            n: Number of households.

        Returns:
            Array of price shock values.
        """
        # Get systematic component from calibration
        if (
            self.calibration.shock_distribution_by_wave
            and wave in self.calibration.shock_distribution_by_wave
        ):
            dist = self.calibration.shock_distribution_by_wave[wave]
            systematic = dist.params.get("mean", 0.0)
        else:
            dist = self.calibration.shock_distribution
            systematic = dist.params.get("mean", 0.0)

        # Draw idiosyncratic component
        idiosyncratic = self.rng.normal(
            loc=0,
            scale=self.config.transition.shock_sd_idiosyncratic,
            size=n,
        )

        return (systematic + idiosyncratic).astype(np.float64)

    def _evolve_assets(
        self,
        current_assets: np.ndarray,
        current_enterprise: np.ndarray,
        shocks: np.ndarray,
    ) -> np.ndarray:
        """Evolve assets according to transition equation.

        A_{t+1} = rho * A_t + lambda * E_t + delta * P_t + epsilon

        Args:
            current_assets: Current asset values.
            current_enterprise: Current enterprise status.
            shocks: Current price shocks.

        Returns:
            New asset values.
        """
        tc = self.config.transition

        innovation = self.rng.normal(0, tc.asset_noise_sd, len(current_assets))

        new_assets = (
            tc.rho_assets * current_assets
            + tc.lambda_enterprise_assets * current_enterprise
            + tc.delta_shock_assets * shocks
            + innovation
        )

        return new_assets

    def _evolve_credit(
        self,
        current_credit: np.ndarray,
        current_assets: np.ndarray,
    ) -> np.ndarray:
        """Evolve credit access with stickiness.

        P(C_{t+1}=1) = alpha * C_t + (1-alpha) * logit^{-1}(beta * A_{t+1})

        Args:
            current_credit: Current credit access.
            current_assets: Current assets.

        Returns:
            New credit access indicators.
        """
        alpha = self.config.transition.credit_stickiness
        model = self.calibration.credit_model

        # Get model-predicted probability
        model_probs = np.array([
            model.predict_proba({"assets_index": float(a)})
            for a in current_assets
        ])

        # Combine with stickiness
        probs = alpha * current_credit + (1 - alpha) * model_probs

        # Draw new credit status
        new_credit = (self.rng.random(len(current_credit)) < probs).astype(int)
        return new_credit

    def _simple_enterprise_transition(
        self,
        current_enterprise: np.ndarray,
        current_assets: np.ndarray,
        current_credit: np.ndarray,
        shocks: np.ndarray,
    ) -> np.ndarray:
        """Simple enterprise transition for initial panel generation.

        This is used only for generating the initial synthetic panel.
        During simulation, the LLM policy will determine transitions.

        Uses observed transition rates from calibration as base rates,
        modulated by state variables.

        Args:
            current_enterprise: Current enterprise status.
            current_assets: Current assets.
            current_credit: Current credit access.
            shocks: Current price shocks.

        Returns:
            New enterprise status.
        """
        tr = self.calibration.transition_rates
        n = len(current_enterprise)

        # Base entry/exit rates from calibration
        base_entry_rate = tr.enter_rate
        base_exit_rate = tr.exit_rate

        # Modulate by state variables
        # Entry more likely with negative shocks and low assets
        shock_effect = -0.5 * shocks  # Negative shocks increase entry probability
        asset_effect = -0.1 * current_assets  # Lower assets increase entry probability
        credit_effect = 0.05 * current_credit  # Credit slightly increases entry

        entry_prob = np.clip(
            base_entry_rate + shock_effect + asset_effect + credit_effect,
            0.01,
            0.50,
        )

        # Exit more likely with very low assets and no credit
        exit_prob = np.clip(
            base_exit_rate - 0.05 * current_assets - 0.1 * current_credit,
            0.01,
            0.30,
        )

        # Apply transitions
        new_enterprise = current_enterprise.copy()

        # Entry for non-enterprise households
        not_enterprise = current_enterprise == 0
        enter_mask = not_enterprise & (self.rng.random(n) < entry_prob)
        new_enterprise[enter_mask] = 1

        # Exit for enterprise households
        has_enterprise = current_enterprise == 1
        exit_mask = has_enterprise & (self.rng.random(n) < exit_prob)
        new_enterprise[exit_mask] = 0

        return new_enterprise


def generate_synthetic_panel(
    calibration: "CalibrationArtifact",
    n_households: int = 1000,
    config: TransitionConfig | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Convenience function to generate synthetic panel.

    Args:
        calibration: Calibration artifact.
        n_households: Number of households.
        config: Transition configuration.
        seed: Random seed.

    Returns:
        Synthetic panel DataFrame.
    """
    panel_config = SyntheticPanelConfig(
        n_households=n_households,
        waves=calibration.waves.copy(),
        seed=seed,
        transition=config or TransitionConfig(),
    )

    generator = SyntheticPanelGenerator(calibration, panel_config)
    return generator.generate()


def generate_synthetic_panel_from_file(
    calibration_path: str | Path,
    n_households: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic panel from calibration file.

    Args:
        calibration_path: Path to calibration.json.
        n_households: Number of households.
        seed: Random seed.

    Returns:
        Synthetic panel DataFrame.
    """
    from abm_enterprise.calibration.schemas import CalibrationArtifact

    calibration = CalibrationArtifact.load(str(calibration_path))
    return generate_synthetic_panel(calibration, n_households, seed=seed)


# Keep legacy function for backward compatibility
def generate_synthetic_households(
    n: int = 100,
    waves: int = 4,
    country: str = "tanzania",
) -> pd.DataFrame:
    """Generate synthetic household panel data (legacy function).

    This is the original simple synthetic generator, kept for
    backward compatibility with tests and toy mode.

    For production use, prefer generate_synthetic_panel() with
    a calibration artifact.

    Args:
        n: Number of households to generate.
        waves: Number of survey waves.
        country: Country code for the synthetic data.

    Returns:
        DataFrame with synthetic household data.

    Example:
        >>> from abm_enterprise.utils.rng import set_seed
        >>> set_seed(42)
        >>> df = generate_synthetic_households(n=50, waves=4)
        >>> df.shape
        (200, 10)
    """
    rng = get_rng()

    # Generate base household characteristics (time-invariant)
    household_ids = [f"HH_{i:05d}" for i in range(n)]

    # Initial asset distribution (log-normal)
    initial_assets = rng.lognormal(mean=0, sigma=1, size=n)
    initial_assets = (initial_assets - initial_assets.mean()) / initial_assets.std()

    # Credit access probability (correlated with assets)
    credit_prob = 1 / (1 + np.exp(-0.5 * initial_assets))

    # Initial enterprise status (correlated with assets and credit)
    enterprise_prob = 1 / (1 + np.exp(-0.3 * initial_assets - 0.2 * credit_prob))

    # Crop portfolio (1-5 crops, correlated with land)
    initial_crop_count = rng.poisson(lam=2, size=n) + 1
    initial_crop_count = np.clip(initial_crop_count, 1, 5)

    # Land area (log-normal, in hectares)
    land_area = rng.lognormal(mean=0.5, sigma=0.8, size=n)
    land_area = np.clip(land_area, 0.1, 20)

    records = []

    for wave in range(1, waves + 1):
        for i, hh_id in enumerate(household_ids):
            # Assets evolve over time with some noise and trend
            wave_shock = rng.normal(0, 0.2)
            assets = initial_assets[i] + 0.05 * (wave - 1) + wave_shock

            # Credit access (can change over time)
            credit_access = int(rng.random() < credit_prob[i] + 0.1 * (wave - 1))

            # Enterprise status (can change based on conditions)
            if wave == 1:
                enterprise_status = int(rng.random() < enterprise_prob[i])
            else:
                # Some transition probability based on current state
                prev_enterprise = records[-n].get("enterprise_status", 0)
                if prev_enterprise:
                    # Exit probability increases with negative shocks
                    exit_prob = 0.1 + 0.2 * max(0, -wave_shock)
                    enterprise_status = int(rng.random() > exit_prob)
                else:
                    # Entry probability depends on assets and credit
                    entry_prob = 0.1 + 0.1 * credit_access + 0.05 * assets
                    enterprise_status = int(rng.random() < entry_prob)

            # Price exposure (varies by wave, represents price shocks)
            base_price_shock = rng.normal(0, 0.15)
            # Add systematic shock in some waves
            if wave == 2:
                systematic_shock = -0.1  # Negative shock in wave 2
            elif wave == 4:
                systematic_shock = 0.05  # Positive shock in wave 4
            else:
                systematic_shock = 0
            price_exposure = base_price_shock + systematic_shock

            # Crop count (slow evolution)
            crop_count = int(initial_crop_count[i] + rng.choice([-1, 0, 0, 0, 1]))
            crop_count = max(1, min(5, crop_count))

            record = {
                "household_id": hh_id,
                "wave": wave,
                "country": country,
                "assets_index": round(assets, 4),
                "credit_access": credit_access,
                "enterprise_status": enterprise_status,
                "price_exposure": round(price_exposure, 4),
                "crop_count": crop_count,
                "land_area_ha": round(land_area[i], 2),
            }
            records.append(record)

    df = pd.DataFrame(records)

    # Ensure proper data types
    df["household_id"] = df["household_id"].astype(str)
    df["wave"] = df["wave"].astype(int)
    df["credit_access"] = df["credit_access"].astype(int)
    df["enterprise_status"] = df["enterprise_status"].astype(int)

    return df


def validate_synthetic_data(df: pd.DataFrame) -> dict[str, bool]:
    """Validate synthetic data against expected schema.

    Args:
        df: DataFrame to validate.

    Returns:
        Dictionary of validation results.
    """
    validations = {
        "has_required_columns": all(
            col in df.columns
            for col in [
                "household_id",
                "wave",
                "assets_index",
                "credit_access",
                "enterprise_status",
                "price_exposure",
            ]
        ),
        "household_id_not_null": df["household_id"].notna().all(),
        "wave_positive": (df["wave"] > 0).all(),
        "credit_access_binary": df["credit_access"].isin([0, 1]).all(),
        "enterprise_status_binary": df["enterprise_status"].isin([0, 1]).all(),
        "balanced_panel": df.groupby("household_id").size().nunique() == 1,
    }
    return validations


def validate_synthetic_panel(
    df: pd.DataFrame,
    n_households: int | None = None,
    n_waves: int | None = None,
) -> dict[str, bool]:
    """Validate synthetic panel from calibration-based generator.

    Args:
        df: Panel DataFrame.
        n_households: Expected number of households (optional).
        n_waves: Expected number of waves (optional).

    Returns:
        Dictionary of validation results.
    """
    validations = validate_synthetic_data(df)

    # Additional validations for calibration-based panel
    if n_households is not None:
        validations["correct_n_households"] = (
            df["household_id"].nunique() == n_households
        )

    if n_waves is not None:
        validations["correct_n_waves"] = df["wave"].nunique() == n_waves

    # Check for reasonable asset persistence
    if len(df["wave"].unique()) > 1:
        # Compute within-household asset correlation
        df_wide = df.pivot(
            index="household_id",
            columns="wave",
            values="assets_index",
        )
        if df_wide.shape[1] >= 2:
            waves = sorted(df_wide.columns)
            first_last_corr = df_wide[waves[0]].corr(df_wide[waves[-1]])
            validations["asset_persistence"] = first_last_corr > 0.3

    return validations
