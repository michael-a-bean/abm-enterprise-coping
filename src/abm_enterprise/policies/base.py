"""Base policy interface for enterprise coping decisions.

Defines the abstract interface that all policy implementations
must follow, enabling different decision rules to be plugged in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from abm_enterprise.data.schemas import HouseholdState


class Action(str, Enum):
    """Possible actions a household can take.

    Attributes:
        ENTER_ENTERPRISE: Start a non-farm enterprise.
        EXIT_ENTERPRISE: Exit existing enterprise.
        NO_CHANGE: Maintain current status.
    """

    ENTER_ENTERPRISE = "ENTER_ENTERPRISE"
    EXIT_ENTERPRISE = "EXIT_ENTERPRISE"
    NO_CHANGE = "NO_CHANGE"


class BasePolicy(ABC):
    """Abstract base class for household decision policies.

    All policy implementations must inherit from this class
    and implement the decide() method.

    Example:
        class MyPolicy(BasePolicy):
            def decide(self, state: HouseholdState) -> Action:
                if some_condition:
                    return Action.ENTER_ENTERPRISE
                return Action.NO_CHANGE
    """

    @abstractmethod
    def decide(self, state: HouseholdState) -> Action:
        """Make a decision based on household state.

        Args:
            state: Current household state.

        Returns:
            The action to take.
        """
        pass

    @property
    def name(self) -> str:
        """Get the policy name.

        Returns:
            Class name as the policy name.
        """
        return self.__class__.__name__


class NoOpPolicy(BasePolicy):
    """Policy that always returns NO_CHANGE.

    Useful as a baseline or placeholder policy.
    """

    def decide(self, state: HouseholdState) -> Action:
        """Always return NO_CHANGE.

        Args:
            state: Current household state (unused).

        Returns:
            Action.NO_CHANGE
        """
        return Action.NO_CHANGE


class RandomPolicy(BasePolicy):
    """Policy that makes random decisions.

    Useful for testing and as a baseline comparison.
    Requires RNG to be initialized.

    Attributes:
        enter_prob: Probability of entering enterprise.
        exit_prob: Probability of exiting enterprise.
    """

    def __init__(self, enter_prob: float = 0.1, exit_prob: float = 0.1) -> None:
        """Initialize random policy.

        Args:
            enter_prob: Probability of entering enterprise when not in one.
            exit_prob: Probability of exiting enterprise when in one.
        """
        self.enter_prob = enter_prob
        self.exit_prob = exit_prob

    def decide(self, state: HouseholdState) -> Action:
        """Make a random decision.

        Args:
            state: Current household state.

        Returns:
            Random action based on configured probabilities.
        """
        from abm_enterprise.data.schemas import EnterpriseStatus
        from abm_enterprise.utils.rng import get_rng

        rng = get_rng()

        if state.enterprise_status == EnterpriseStatus.HAS_ENTERPRISE:
            if rng.random() < self.exit_prob:
                return Action.EXIT_ENTERPRISE
        else:
            if rng.random() < self.enter_prob:
                return Action.ENTER_ENTERPRISE

        return Action.NO_CHANGE
