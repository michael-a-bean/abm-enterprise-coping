"""Constraint validators for LLM-based policy decisions.

These constraints ensure that proposed actions from the LLM
are feasible given the household's current state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from abm_enterprise.data.schemas import EnterpriseStatus, HouseholdState
from abm_enterprise.policies.base import Action


class Constraint(ABC):
    """Abstract base class for action constraints.

    Constraints validate whether a proposed action is feasible
    given the household's current state.
    """

    @property
    def name(self) -> str:
        """Get the constraint name.

        Returns:
            Class name as the constraint name.
        """
        return self.__class__.__name__

    @abstractmethod
    def validate(self, state: HouseholdState, action: Action) -> bool:
        """Validate if action is feasible for given state.

        Args:
            state: Current household state.
            action: Proposed action to validate.

        Returns:
            True if action is valid, False otherwise.
        """
        pass


class MinimumAssetsConstraint(Constraint):
    """Must have minimum assets to enter enterprise.

    This constraint prevents households with insufficient assets
    from entering enterprise, as they lack the capital to start.

    Attributes:
        threshold: Minimum asset index required for entry.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        """Initialize constraint with asset threshold.

        Args:
            threshold: Minimum asset index for enterprise entry.
        """
        self.threshold = threshold

    def validate(self, state: HouseholdState, action: Action) -> bool:
        """Validate minimum assets for entry.

        Args:
            state: Current household state.
            action: Proposed action.

        Returns:
            True if action is valid, False if entry rejected due to low assets.
        """
        if action == Action.ENTER_ENTERPRISE:
            return state.assets >= self.threshold
        return True


class NoExitIfStayerConstraint(Constraint):
    """Stayers cannot exit enterprise.

    This constraint prevents households classified as 'stayers'
    (persistent enterprise operators) from exiting, as the
    literature suggests they maintain enterprise through shocks.
    """

    def validate(self, state: HouseholdState, action: Action) -> bool:
        """Validate stayer cannot exit.

        Args:
            state: Current household state.
            action: Proposed action.

        Returns:
            True if action is valid, False if stayer attempted exit.
        """
        # Note: This constraint requires classification info which may not
        # be in the basic HouseholdState. For now, we allow all exits
        # and this can be extended when classification is available.
        # The constraint can be made more specific by extending HouseholdState.
        return True


class CreditRequiredConstraint(Constraint):
    """Enterprise entry requires credit access for low-asset households.

    Households with low assets need credit access to enter enterprise,
    as they cannot self-finance the startup costs.

    Attributes:
        asset_threshold: Below this asset level, credit is required.
    """

    def __init__(self, asset_threshold: float = 0.3) -> None:
        """Initialize constraint with thresholds.

        Args:
            asset_threshold: Below this, credit is required for entry.
        """
        self.asset_threshold = asset_threshold

    def validate(self, state: HouseholdState, action: Action) -> bool:
        """Validate credit requirement for low-asset entry.

        Args:
            state: Current household state.
            action: Proposed action.

        Returns:
            True if action is valid, False if low-asset entry without credit.
        """
        if action == Action.ENTER_ENTERPRISE:
            if state.assets < self.asset_threshold:
                # Low-asset household needs credit to enter
                return state.credit_access == 1
        return True


class NoEntryIfAlreadyInEnterpriseConstraint(Constraint):
    """Cannot enter enterprise if already in one.

    This basic constraint prevents duplicate entry actions.
    """

    def validate(self, state: HouseholdState, action: Action) -> bool:
        """Validate no duplicate entry.

        Args:
            state: Current household state.
            action: Proposed action.

        Returns:
            True if action is valid, False if already in enterprise.
        """
        if action == Action.ENTER_ENTERPRISE:
            return state.enterprise_status == EnterpriseStatus.NO_ENTERPRISE
        return True


class NoExitIfNotInEnterpriseConstraint(Constraint):
    """Cannot exit enterprise if not in one.

    This basic constraint prevents exit when not operating enterprise.
    """

    def validate(self, state: HouseholdState, action: Action) -> bool:
        """Validate exit only when in enterprise.

        Args:
            state: Current household state.
            action: Proposed action.

        Returns:
            True if action is valid, False if exit without enterprise.
        """
        if action == Action.EXIT_ENTERPRISE:
            return state.enterprise_status == EnterpriseStatus.HAS_ENTERPRISE
        return True


class CompositeConstraint(Constraint):
    """Combines multiple constraints with AND logic.

    All child constraints must pass for the action to be valid.

    Attributes:
        constraints: List of constraints to evaluate.
    """

    def __init__(self, constraints: list[Constraint]) -> None:
        """Initialize with list of constraints.

        Args:
            constraints: List of constraints to combine.
        """
        self.constraints = constraints

    def validate(self, state: HouseholdState, action: Action) -> bool:
        """Validate all constraints pass.

        Args:
            state: Current household state.
            action: Proposed action.

        Returns:
            True if all constraints pass, False otherwise.
        """
        return all(c.validate(state, action) for c in self.constraints)

    def get_failed_constraints(
        self, state: HouseholdState, action: Action
    ) -> list[str]:
        """Get names of constraints that failed.

        Args:
            state: Current household state.
            action: Proposed action.

        Returns:
            List of constraint names that rejected the action.
        """
        failed = []
        for c in self.constraints:
            if not c.validate(state, action):
                failed.append(c.name)
        return failed


def get_default_constraints() -> list[Constraint]:
    """Get the default set of constraints for LLM policy.

    Returns:
        List of default constraint instances.
    """
    return [
        NoEntryIfAlreadyInEnterpriseConstraint(),
        NoExitIfNotInEnterpriseConstraint(),
        MinimumAssetsConstraint(threshold=-0.5),  # Allow low but not extreme
        CreditRequiredConstraint(asset_threshold=0.0),  # Require credit if assets < 0
    ]
