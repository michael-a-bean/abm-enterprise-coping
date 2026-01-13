"""Household agent implementation for Mesa 3.

This module defines the HouseholdAgent class which represents
a household making enterprise participation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mesa

from abm_enterprise.data.schemas import EnterpriseStatus, HouseholdState

if TYPE_CHECKING:
    from abm_enterprise.model import EnterpriseCopingModel
    from abm_enterprise.policies.base import BasePolicy


@dataclass
class AgentState:
    """Internal state of a household agent.

    Attributes:
        household_id: Unique identifier.
        wave: Current simulation wave.
        assets: Asset index (standardized).
        credit_access: Credit access indicator (0/1).
        enterprise_status: Current enterprise status.
        price_exposure: Current price shock exposure.
        crop_count: Number of crops in portfolio.
        land_area_ha: Land area in hectares.
        action_taken: Last action taken.
        policy_applied: Whether policy was applied.
    """

    household_id: str
    wave: int
    assets: float
    credit_access: int
    enterprise_status: int
    price_exposure: float
    crop_count: int
    land_area_ha: float
    action_taken: str = "NO_CHANGE"
    policy_applied: int = 0


class HouseholdAgent(mesa.Agent):
    """Mesa 3 agent representing a household.

    The household agent maintains state about assets, credit access,
    enterprise participation, and crop portfolio. Each step, it
    evaluates whether to enter, exit, or maintain enterprise status.

    Attributes:
        state: AgentState dataclass containing all household state.
        policy: Optional policy that influences decisions.
    """

    def __init__(
        self,
        model: EnterpriseCopingModel,
        household_id: str,
        initial_data: dict[str, Any],
    ) -> None:
        """Initialize a household agent.

        Args:
            model: The parent Mesa model.
            household_id: Unique identifier for this household.
            initial_data: Dictionary with initial state values.
        """
        super().__init__(model)
        self.household_id = household_id

        # Initialize state from data
        self.state = AgentState(
            household_id=household_id,
            wave=initial_data.get("wave", 1),
            assets=initial_data.get("assets_index", 0.0),
            credit_access=initial_data.get("credit_access", 0),
            enterprise_status=initial_data.get("enterprise_status", 0),
            price_exposure=initial_data.get("price_exposure", 0.0),
            crop_count=initial_data.get("crop_count", 1),
            land_area_ha=initial_data.get("land_area_ha", 1.0),
        )

        # Policy will be set by model if applicable
        self._policy: BasePolicy | None = None

    @property
    def policy(self) -> BasePolicy | None:
        """Get the policy for this agent."""
        return self._policy

    @policy.setter
    def policy(self, value: BasePolicy | None) -> None:
        """Set the policy for this agent."""
        self._policy = value

    def get_household_state(self) -> HouseholdState:
        """Convert internal state to HouseholdState schema.

        Returns:
            HouseholdState instance for policy decision.
        """
        enterprise_enum = (
            EnterpriseStatus.HAS_ENTERPRISE
            if self.state.enterprise_status == 1
            else EnterpriseStatus.NO_ENTERPRISE
        )
        return HouseholdState(
            household_id=self.state.household_id,
            wave=self.state.wave,
            assets=self.state.assets,
            credit_access=self.state.credit_access,
            enterprise_status=enterprise_enum,
            price_exposure=self.state.price_exposure,
        )

    def step(self) -> None:
        """Execute one step of the agent.

        Evaluates coping strategy and potentially changes
        enterprise status based on policy or default rules.
        """
        from abm_enterprise.policies.base import Action

        # Reset action for this step
        self.state.action_taken = "NO_CHANGE"
        self.state.policy_applied = 0

        # Get decision from policy if available
        if self._policy is not None:
            household_state = self.get_household_state()
            action = self._policy.decide(household_state)
            self.state.policy_applied = 1
        else:
            # Default behavior: no change
            action = Action.NO_CHANGE

        # Apply action
        self._apply_action(action)

        # Update wave
        self.state.wave = self.model.current_wave

    def _apply_action(self, action: Action) -> None:
        """Apply the decided action to agent state.

        Args:
            action: The action to apply.
        """
        from abm_enterprise.policies.base import Action

        if action == Action.ENTER_ENTERPRISE:
            if self.state.enterprise_status == 0:
                self.state.enterprise_status = 1
                self.state.action_taken = "ENTER_ENTERPRISE"
        elif action == Action.EXIT_ENTERPRISE:
            if self.state.enterprise_status == 1:
                self.state.enterprise_status = 0
                self.state.action_taken = "EXIT_ENTERPRISE"
        else:
            self.state.action_taken = "NO_CHANGE"

    def update_state(self, new_data: dict[str, Any]) -> None:
        """Update agent state with new wave data.

        Args:
            new_data: Dictionary with updated state values.
        """
        if "assets_index" in new_data:
            self.state.assets = new_data["assets_index"]
        if "credit_access" in new_data:
            self.state.credit_access = new_data["credit_access"]
        if "price_exposure" in new_data:
            self.state.price_exposure = new_data["price_exposure"]
        if "crop_count" in new_data:
            self.state.crop_count = new_data["crop_count"]
        if "land_area_ha" in new_data:
            self.state.land_area_ha = new_data["land_area_ha"]
        if "wave" in new_data:
            self.state.wave = new_data["wave"]

    def get_output_record(self) -> dict[str, Any]:
        """Get current state as output record.

        Returns:
            Dictionary matching OutputRecord schema.
        """
        return {
            "household_id": self.state.household_id,
            "wave": self.state.wave,
            "assets_index": self.state.assets,
            "credit_access": self.state.credit_access,
            "enterprise_status": self.state.enterprise_status,
            "price_exposure": self.state.price_exposure,
            "crop_count": self.state.crop_count,
            "land_area_ha": self.state.land_area_ha,
            "action_taken": self.state.action_taken,
            "policy_applied": self.state.policy_applied,
        }
