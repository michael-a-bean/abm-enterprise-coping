"""LLM-based decision policy with constraint validation.

This module implements the LLMPolicy class which uses large language
models to make household enterprise decisions, with validation against
feasibility constraints and comprehensive logging for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from abm_enterprise.data.schemas import HouseholdState
from abm_enterprise.policies.base import Action, BasePolicy
from abm_enterprise.policies.constraints import (
    CompositeConstraint,
    Constraint,
    get_default_constraints,
)
from abm_enterprise.policies.logging import DecisionLogger
from abm_enterprise.policies.prompts import (
    PromptConfig,
    build_prompt,
    parse_action_from_response,
)
from abm_enterprise.policies.providers import LLMProvider, StubProvider


class LLMPolicy(BasePolicy):
    """LLM-based decision policy with constraint validation.

    Uses the proposal-constraints-commit pattern:
    1. Generate prompt from household state
    2. Get proposal from LLM provider
    3. Parse and validate against constraints
    4. Fallback to NO_CHANGE if constraints fail
    5. Log decision for reproducibility
    6. Commit and return action

    Attributes:
        provider: LLM provider for generating responses.
        logger: Decision logger for recording all decisions.
        constraints: List of constraints to validate proposals.
        prompt_config: Configuration for prompt generation.
        fallback_action: Action to use when constraints fail.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        logger: DecisionLogger | None = None,
        constraints: list[Constraint] | None = None,
        prompt_config: PromptConfig | None = None,
        fallback_action: Action = Action.NO_CHANGE,
        log_dir: Path | str | None = None,
    ) -> None:
        """Initialize LLM policy.

        Args:
            provider: LLM provider. If None, uses StubProvider.
            logger: Decision logger. If None, creates one with log_dir.
            constraints: List of constraints. If None, uses defaults.
            prompt_config: Prompt configuration. If None, uses defaults.
            fallback_action: Action when constraints fail or parsing fails.
            log_dir: Directory for decision logs (used if logger is None).
        """
        self.provider = provider or StubProvider()
        self.logger = logger or DecisionLogger(output_dir=log_dir)
        self.prompt_config = prompt_config or PromptConfig()
        self.fallback_action = fallback_action

        # Setup constraints
        if constraints is None:
            self.constraints = get_default_constraints()
        else:
            self.constraints = constraints

        self._composite_constraint = CompositeConstraint(self.constraints)

    @property
    def name(self) -> str:
        """Get policy name including provider info.

        Returns:
            Policy name with provider.
        """
        return f"LLMPolicy({self.provider.name})"

    def decide(self, state: HouseholdState) -> Action:
        """Make a decision based on household state using LLM.

        Implements the full proposal-constraints-commit pipeline:
        1. Build prompt from state
        2. Get proposal from LLM
        3. Parse action from response
        4. Validate against constraints
        5. Log decision
        6. Return action

        Args:
            state: Current household state.

        Returns:
            The action to take.
        """
        # 1. Generate prompt from state
        prompt = self._build_prompt(state)

        # 2. Get proposal from LLM with timing
        response, latency_ms = self.provider.generate_with_timing(prompt)

        # 3. Parse action from response
        parsed_action_str = parse_action_from_response(response)
        action = self._parse_action(parsed_action_str)

        # 4. Validate against constraints
        constraints_passed = True
        failed_constraints: list[str] = []

        if action is not None:
            constraints_passed = self._validate_constraints(state, action)
            if not constraints_passed:
                failed_constraints = self._composite_constraint.get_failed_constraints(
                    state, action
                )
                action = self.fallback_action
        else:
            # Parsing failed
            constraints_passed = False
            failed_constraints = ["parse_failure"]
            action = self.fallback_action

        # 5. Log decision
        self.logger.log(
            state=state,
            prompt=prompt,
            response=response,
            parsed_action=parsed_action_str,
            constraints_passed=constraints_passed,
            failed_constraints=failed_constraints,
            final_action=action,
            provider=self.provider.name,
            model=self.provider.model,
            latency_ms=latency_ms,
        )

        # 6. Commit and return
        return action

    def _build_prompt(self, state: HouseholdState) -> str:
        """Build prompt from household state.

        Args:
            state: Current household state.

        Returns:
            Formatted prompt string.
        """
        return build_prompt(state, self.prompt_config)

    def _parse_action(self, action_str: str | None) -> Action | None:
        """Parse action string to Action enum.

        Args:
            action_str: Action string from parsed response.

        Returns:
            Action enum or None if parsing failed.
        """
        if action_str is None:
            return None

        action_map = {
            "ENTER_ENTERPRISE": Action.ENTER_ENTERPRISE,
            "EXIT_ENTERPRISE": Action.EXIT_ENTERPRISE,
            "NO_CHANGE": Action.NO_CHANGE,
        }

        return action_map.get(action_str.upper())

    def _validate_constraints(self, state: HouseholdState, action: Action) -> bool:
        """Validate action against all constraints.

        Args:
            state: Current household state.
            action: Proposed action.

        Returns:
            True if all constraints pass, False otherwise.
        """
        return self._composite_constraint.validate(state, action)

    def save_log(self, path: Path | str | None = None) -> Path:
        """Save decision log to file.

        Args:
            path: Output path. If None, uses default in logger output_dir.

        Returns:
            Path to saved log file.
        """
        return self.logger.save(path)

    def get_log_summary(self) -> dict[str, Any]:
        """Get summary of logged decisions.

        Returns:
            Dictionary with summary statistics.
        """
        return self.logger.get_summary()


class LLMPolicyFactory:
    """Factory for creating LLMPolicy instances with different providers."""

    @staticmethod
    def create_stub_policy(
        log_dir: Path | str | None = None,
        constraints: list[Constraint] | None = None,
        country: str = "tanzania",
    ) -> LLMPolicy:
        """Create LLMPolicy with StubProvider.

        Args:
            log_dir: Directory for decision logs.
            constraints: Custom constraints. If None, uses defaults.
            country: Country for prompt context.

        Returns:
            Configured LLMPolicy.
        """
        from abm_enterprise.policies.providers import StubProvider

        provider = StubProvider()
        prompt_config = PromptConfig(country=country)

        return LLMPolicy(
            provider=provider,
            constraints=constraints,
            prompt_config=prompt_config,
            log_dir=log_dir,
        )

    @staticmethod
    def create_replay_policy(
        log_path: Path | str,
        log_dir: Path | str | None = None,
        constraints: list[Constraint] | None = None,
        country: str = "tanzania",
    ) -> LLMPolicy:
        """Create LLMPolicy with ReplayProvider.

        Args:
            log_path: Path to decision log to replay.
            log_dir: Directory for new decision logs.
            constraints: Custom constraints. If None, uses defaults.
            country: Country for prompt context.

        Returns:
            Configured LLMPolicy.
        """
        from abm_enterprise.policies.providers import ReplayProvider

        provider = ReplayProvider(log_path=log_path)
        prompt_config = PromptConfig(country=country)

        return LLMPolicy(
            provider=provider,
            constraints=constraints,
            prompt_config=prompt_config,
            log_dir=log_dir,
        )

    @staticmethod
    def create_claude_policy(
        api_key: str | None = None,
        model: str = "claude-3-haiku-20240307",
        log_dir: Path | str | None = None,
        constraints: list[Constraint] | None = None,
        country: str = "tanzania",
    ) -> LLMPolicy:
        """Create LLMPolicy with ClaudeProvider.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Claude model to use.
            log_dir: Directory for decision logs.
            constraints: Custom constraints. If None, uses defaults.
            country: Country for prompt context.

        Returns:
            Configured LLMPolicy.
        """
        from abm_enterprise.policies.providers import ClaudeProvider

        provider = ClaudeProvider(api_key=api_key, model=model)
        prompt_config = PromptConfig(country=country)

        return LLMPolicy(
            provider=provider,
            constraints=constraints,
            prompt_config=prompt_config,
            log_dir=log_dir,
        )

    @staticmethod
    def create_openai_policy(
        api_key: str | None = None,
        model: str = "gpt-3.5-turbo",
        log_dir: Path | str | None = None,
        constraints: list[Constraint] | None = None,
        country: str = "tanzania",
    ) -> LLMPolicy:
        """Create LLMPolicy with OpenAIProvider.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: GPT model to use.
            log_dir: Directory for decision logs.
            constraints: Custom constraints. If None, uses defaults.
            country: Country for prompt context.

        Returns:
            Configured LLMPolicy.
        """
        from abm_enterprise.policies.providers import OpenAIProvider

        provider = OpenAIProvider(api_key=api_key, model=model)
        prompt_config = PromptConfig(country=country)

        return LLMPolicy(
            provider=provider,
            constraints=constraints,
            prompt_config=prompt_config,
            log_dir=log_dir,
        )
