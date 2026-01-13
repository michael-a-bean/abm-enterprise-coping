"""Rule-based policy implementation.

Implements simple threshold-based decision rules for enterprise
participation decisions based on economic conditions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from abm_enterprise.data.schemas import EnterpriseStatus, HouseholdState
from abm_enterprise.policies.base import Action, BasePolicy


class RulePolicy(BasePolicy):
    """Simple rule-based policy with configurable thresholds.

    Decision rules:
    - ENTER: If price_exposure < threshold AND not in enterprise AND assets < asset_threshold
    - EXIT: If in enterprise AND assets drop below critical threshold AND no credit
    - NO_CHANGE: Otherwise

    The rationale is that households facing adverse price shocks with
    low assets may enter non-farm enterprise as a coping strategy.

    Attributes:
        price_threshold: Price exposure threshold (negative = adverse).
        asset_threshold: Asset threshold for entry decision.
        exit_asset_threshold: Asset threshold for exit decision.
        credit_required_for_stability: If True, credit helps prevent exit.
    """

    def __init__(
        self,
        price_threshold: float = -0.1,
        asset_threshold: float = 0.0,
        exit_asset_threshold: float = -1.0,
        credit_required_for_stability: bool = True,
    ) -> None:
        """Initialize rule policy with thresholds.

        Args:
            price_threshold: Price exposure threshold for entry (default: -0.1).
            asset_threshold: Asset threshold below which entry is considered.
            exit_asset_threshold: Asset threshold below which exit occurs.
            credit_required_for_stability: Whether credit prevents forced exit.
        """
        self.price_threshold = price_threshold
        self.asset_threshold = asset_threshold
        self.exit_asset_threshold = exit_asset_threshold
        self.credit_required_for_stability = credit_required_for_stability

    def decide(self, state: HouseholdState) -> Action:
        """Make decision based on threshold rules.

        Args:
            state: Current household state.

        Returns:
            Action based on threshold rules.
        """
        # Check for entry condition
        if state.enterprise_status == EnterpriseStatus.NO_ENTERPRISE:
            return self._evaluate_entry(state)

        # Check for exit condition
        if state.enterprise_status == EnterpriseStatus.HAS_ENTERPRISE:
            return self._evaluate_exit(state)

        return Action.NO_CHANGE

    def _evaluate_entry(self, state: HouseholdState) -> Action:
        """Evaluate whether household should enter enterprise.

        Entry logic: Adverse price shock AND low assets triggers entry
        as a coping mechanism.

        Args:
            state: Current household state.

        Returns:
            ENTER_ENTERPRISE or NO_CHANGE.
        """
        adverse_price = state.price_exposure < self.price_threshold
        low_assets = state.assets < self.asset_threshold

        if adverse_price and low_assets:
            return Action.ENTER_ENTERPRISE

        return Action.NO_CHANGE

    def _evaluate_exit(self, state: HouseholdState) -> Action:
        """Evaluate whether household should exit enterprise.

        Exit logic: Very low assets AND no credit access triggers exit.

        Args:
            state: Current household state.

        Returns:
            EXIT_ENTERPRISE or NO_CHANGE.
        """
        very_low_assets = state.assets < self.exit_asset_threshold

        if very_low_assets:
            if self.credit_required_for_stability:
                # Credit access prevents forced exit
                if state.credit_access == 0:
                    return Action.EXIT_ENTERPRISE
            else:
                return Action.EXIT_ENTERPRISE

        return Action.NO_CHANGE


class AdaptiveRulePolicy(BasePolicy):
    """Rule policy that adapts thresholds based on context.

    More sophisticated version that considers multiple factors
    and can adjust thresholds based on wave or other context.

    Attributes:
        base_price_threshold: Base price exposure threshold.
        base_asset_threshold: Base asset threshold.
        wave_adjustment: Whether to adjust for wave number.
    """

    def __init__(
        self,
        base_price_threshold: float = -0.1,
        base_asset_threshold: float = 0.0,
        wave_adjustment: bool = True,
    ) -> None:
        """Initialize adaptive rule policy.

        Args:
            base_price_threshold: Base price exposure threshold.
            base_asset_threshold: Base asset threshold.
            wave_adjustment: Whether to adjust thresholds by wave.
        """
        self.base_price_threshold = base_price_threshold
        self.base_asset_threshold = base_asset_threshold
        self.wave_adjustment = wave_adjustment

    def decide(self, state: HouseholdState) -> Action:
        """Make decision with adaptive thresholds.

        Args:
            state: Current household state.

        Returns:
            Action based on adaptive rules.
        """
        # Adjust thresholds based on wave (later waves may have different dynamics)
        if self.wave_adjustment:
            wave_factor = 1.0 + 0.05 * (state.wave - 1)
        else:
            wave_factor = 1.0

        price_threshold = self.base_price_threshold * wave_factor
        asset_threshold = self.base_asset_threshold + 0.1 * (state.wave - 1)

        # Evaluate entry
        if state.enterprise_status == EnterpriseStatus.NO_ENTERPRISE:
            adverse_price = state.price_exposure < price_threshold
            low_assets = state.assets < asset_threshold

            # Also consider credit access as enabling factor
            has_credit = state.credit_access == 1

            if adverse_price and low_assets:
                return Action.ENTER_ENTERPRISE
            elif adverse_price and has_credit:
                # Credit access enables entry even with higher assets
                return Action.ENTER_ENTERPRISE

        # Evaluate exit (more conservative - only exit under severe conditions)
        if state.enterprise_status == EnterpriseStatus.HAS_ENTERPRISE:
            very_low_assets = state.assets < -1.5
            no_credit = state.credit_access == 0
            severe_price_shock = state.price_exposure < -0.3

            if very_low_assets and no_credit and severe_price_shock:
                return Action.EXIT_ENTERPRISE

        return Action.NO_CHANGE


class CalibratedRulePolicy(BasePolicy):
    """Rule policy with thresholds calibrated from derived target data.

    Implements validation-contract-aligned decision rules:
    - Enterprise entry more likely when:
      - price_exposure < threshold (negative shock)
      - asset_index < median (low assets)
      - credit_access = False (constrained)

    The thresholds are derived from actual data distributions rather
    than using arbitrary defaults.

    Attributes:
        price_threshold: Calibrated price exposure threshold.
        asset_threshold: Calibrated asset threshold for entry.
        exit_asset_threshold: Calibrated asset threshold for exit.
        credit_interaction: Whether credit access affects decisions.
        country: Country code for config loading.
    """

    def __init__(
        self,
        price_threshold: float | None = None,
        asset_threshold: float | None = None,
        exit_asset_threshold: float | None = None,
        credit_interaction: bool = True,
        country: str = "tanzania",
        thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize calibrated rule policy.

        Args:
            price_threshold: Price exposure threshold for entry.
                If None, uses value from thresholds dict or default.
            asset_threshold: Asset threshold below which entry is considered.
                If None, uses value from thresholds dict or default.
            exit_asset_threshold: Asset threshold below which exit occurs.
                If None, uses value from thresholds dict or default.
            credit_interaction: Whether credit access affects decisions.
            country: Country code (for config loading if needed).
            thresholds: Pre-computed thresholds dict from calibration.
        """
        self.country = country
        self.credit_interaction = credit_interaction

        # Use provided thresholds or defaults
        if thresholds is not None:
            self.price_threshold = thresholds.get("price_threshold", -0.1)
            self.asset_threshold = thresholds.get("asset_threshold", 0.0)
            self.exit_asset_threshold = thresholds.get("exit_asset_threshold", -1.0)
        else:
            self.price_threshold = -0.1
            self.asset_threshold = 0.0
            self.exit_asset_threshold = -1.0

        # Override with explicit parameters if provided
        if price_threshold is not None:
            self.price_threshold = price_threshold
        if asset_threshold is not None:
            self.asset_threshold = asset_threshold
        if exit_asset_threshold is not None:
            self.exit_asset_threshold = exit_asset_threshold

    @classmethod
    def from_config(
        cls,
        country: str,
        config_dir: Path | str | None = None,
    ) -> "CalibratedRulePolicy":
        """Create policy from country configuration file.

        Args:
            country: Country code (tanzania or ethiopia).
            config_dir: Directory containing config files.

        Returns:
            CalibratedRulePolicy with country-specific thresholds.
        """
        import yaml

        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "config"
        else:
            config_dir = Path(config_dir)

        config_file = config_dir / f"{country}.yaml"

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            thresholds = config.get("calibration", {}).get("thresholds", {})
            return cls(
                country=country,
                thresholds=thresholds,
                credit_interaction=config.get("calibration", {}).get(
                    "credit_interaction", True
                ),
            )
        else:
            # Return policy with defaults
            return cls(country=country)

    @classmethod
    def from_data(
        cls,
        targets_df: Any,  # pandas DataFrame
        country: str = "tanzania",
    ) -> "CalibratedRulePolicy":
        """Create policy with thresholds calibrated from derived targets.

        Computes thresholds based on actual data distributions:
        - price_threshold: Median of negative price exposures
        - asset_threshold: Median asset index
        - exit_asset_threshold: 10th percentile of asset index

        Args:
            targets_df: DataFrame with derived target data.
            country: Country code.

        Returns:
            CalibratedRulePolicy with data-driven thresholds.
        """
        from abm_enterprise.model import compute_calibration_thresholds

        thresholds = compute_calibration_thresholds(targets_df)
        return cls(country=country, thresholds=thresholds)

    def decide(self, state: HouseholdState) -> Action:
        """Make decision based on calibrated threshold rules.

        Implements the validation contract hypothesis:
        - Negative price shocks induce enterprise entry
        - Low-asset households more responsive
        - Credit-constrained households more responsive

        Args:
            state: Current household state.

        Returns:
            Action based on calibrated threshold rules.
        """
        # Check for entry condition
        if state.enterprise_status == EnterpriseStatus.NO_ENTERPRISE:
            return self._evaluate_entry(state)

        # Check for exit condition
        if state.enterprise_status == EnterpriseStatus.HAS_ENTERPRISE:
            return self._evaluate_exit(state)

        return Action.NO_CHANGE

    def _evaluate_entry(self, state: HouseholdState) -> Action:
        """Evaluate whether household should enter enterprise.

        Entry logic aligned with validation contract:
        - Adverse price shock (price_exposure < threshold)
        - Low assets (asset_index < threshold)
        - Credit-constrained households more likely to enter

        Args:
            state: Current household state.

        Returns:
            ENTER_ENTERPRISE or NO_CHANGE.
        """
        adverse_price = state.price_exposure < self.price_threshold
        low_assets = state.assets < self.asset_threshold

        if adverse_price and low_assets:
            return Action.ENTER_ENTERPRISE

        # Credit interaction: constrained households may enter even with
        # slightly higher assets if facing price shock
        if self.credit_interaction and adverse_price and state.credit_access == 0:
            # Relax asset threshold for credit-constrained households
            relaxed_asset_threshold = self.asset_threshold + 0.5
            if state.assets < relaxed_asset_threshold:
                return Action.ENTER_ENTERPRISE

        return Action.NO_CHANGE

    def _evaluate_exit(self, state: HouseholdState) -> Action:
        """Evaluate whether household should exit enterprise.

        Exit logic: Very low assets AND no credit access triggers exit.

        Args:
            state: Current household state.

        Returns:
            EXIT_ENTERPRISE or NO_CHANGE.
        """
        very_low_assets = state.assets < self.exit_asset_threshold

        if very_low_assets:
            if self.credit_interaction:
                # Credit access prevents forced exit
                if state.credit_access == 0:
                    return Action.EXIT_ENTERPRISE
            else:
                return Action.EXIT_ENTERPRISE

        return Action.NO_CHANGE

    def get_thresholds(self) -> dict[str, float]:
        """Get current threshold values.

        Returns:
            Dictionary of threshold values.
        """
        return {
            "price_threshold": self.price_threshold,
            "asset_threshold": self.asset_threshold,
            "exit_asset_threshold": self.exit_asset_threshold,
        }
