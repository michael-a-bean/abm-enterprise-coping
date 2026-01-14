"""LLM-based decision policy with constraint validation.

This module implements LLMPolicy classes which use large language
models to make household enterprise decisions, with validation against
feasibility constraints and comprehensive logging for reproducibility.

Key features:
- Multi-sample voting with configurable K
- Temperature-controlled stochasticity
- Decision caching for reproducibility and cost efficiency
- Comprehensive logging of all samples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from abm_enterprise.data.schemas import HouseholdState
from abm_enterprise.policies.base import Action, BasePolicy
from abm_enterprise.policies.cache import (
    DecisionCache,
    compute_config_hash,
    compute_state_hash,
)
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
from abm_enterprise.policies.voting import (
    TieBreakStrategy,
    VoteResult,
    majority_vote,
)
from abm_enterprise.utils.logging import get_logger

logger = get_logger(__name__)


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


class LLMPolicyConfig(BaseModel):
    """Configuration for multi-sample LLM policy.

    Attributes:
        model: LLM model identifier (e.g., 'o4-mini', 'gpt-4o-mini').
        temperature: Sampling temperature (0-2).
        k_samples: Number of samples to generate per decision.
        max_tokens: Maximum tokens in each response.
        timeout_seconds: Request timeout.
        max_retries: Maximum retries on failure.
        fallback_action: Action to use when all samples fail.
        cache_enabled: Whether to cache decisions.
        tie_break: Action for tie-breaking.
        tie_break_strategy: Strategy for breaking ties.
    """

    model: str = Field(default="gpt-4o-mini", description="LLM model ID")
    temperature: float = Field(
        default=0.6, ge=0.0, le=2.0, description="Sampling temperature"
    )
    k_samples: int = Field(
        default=5, ge=1, le=20, description="Number of samples per decision"
    )
    max_tokens: int = Field(default=150, ge=50, le=1000, description="Max tokens")
    timeout_seconds: float = Field(default=30.0, ge=1.0, description="Request timeout")
    max_retries: int = Field(default=2, ge=0, le=5, description="Max retries")
    fallback_action: Action = Field(
        default=Action.NO_CHANGE, description="Fallback action"
    )
    cache_enabled: bool = Field(default=True, description="Enable caching")
    tie_break: Action = Field(default=Action.NO_CHANGE, description="Tie-break action")
    tie_break_strategy: TieBreakStrategy = Field(
        default=TieBreakStrategy.CONSERVATIVE,
        description="Tie-break strategy",
    )

    def to_config_dict(self) -> dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "k_samples": self.k_samples,
            "max_tokens": self.max_tokens,
        }


class MultiSampleLLMPolicy(BasePolicy):
    """LLM policy with multi-sample voting and caching.

    Generates K samples per decision at a configurable temperature,
    aggregates via majority voting, and caches results for efficiency.

    Attributes:
        provider: LLM provider for API calls.
        config: Policy configuration.
        constraints: Feasibility constraints.
        prompt_config: Prompt generation configuration.
        cache: Decision cache.
        decision_logger: Logger for all decisions.
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: LLMPolicyConfig | None = None,
        constraints: list[Constraint] | None = None,
        prompt_config: PromptConfig | None = None,
        cache: DecisionCache | None = None,
        decision_logger: DecisionLogger | None = None,
        log_dir: Path | str | None = None,
    ) -> None:
        """Initialize multi-sample LLM policy.

        Args:
            provider: LLM provider.
            config: Policy configuration.
            constraints: Feasibility constraints.
            prompt_config: Prompt configuration.
            cache: Decision cache (creates new if None).
            decision_logger: Decision logger (creates new if None).
            log_dir: Directory for logs (used if logger is None).
        """
        self.provider = provider
        self.config = config or LLMPolicyConfig()
        self.prompt_config = prompt_config or PromptConfig()

        # Set up constraints
        if constraints is None:
            self.constraints = get_default_constraints()
        else:
            self.constraints = constraints
        self._composite_constraint = CompositeConstraint(self.constraints)

        # Set up cache
        if cache is not None:
            self.cache = cache
        else:
            self.cache = DecisionCache(
                enabled=self.config.cache_enabled,
                max_size=10000,
            )

        # Set up logger
        self.decision_logger = decision_logger or DecisionLogger(output_dir=log_dir)

        # Pre-compute config hash for caching
        self._config_hash = compute_config_hash(self.config.to_config_dict())

        logger.info(
            "Initialized MultiSampleLLMPolicy",
            model=self.config.model,
            k_samples=self.config.k_samples,
            temperature=self.config.temperature,
            cache_enabled=self.config.cache_enabled,
        )

    @property
    def name(self) -> str:
        """Get policy name."""
        return f"MultiSampleLLMPolicy({self.provider.name}, k={self.config.k_samples})"

    def decide(self, state: HouseholdState) -> Action:
        """Make a decision using multi-sample voting.

        Pipeline:
        1. Check cache for existing decision
        2. If not cached, generate K samples
        3. Parse and validate each sample
        4. Aggregate via majority vote
        5. Cache and log result
        6. Return final action

        Args:
            state: Current household state.

        Returns:
            The selected action.
        """
        # Compute state hash for caching
        state_dict = {
            "household_id": state.household_id,
            "wave": state.wave,
            "assets": state.assets,
            "credit_access": state.credit_access,
            "enterprise_status": state.enterprise_status.value,
            "price_exposure": state.price_exposure,
        }
        state_hash = compute_state_hash(state_dict)

        # Check cache
        cached_result = self.cache.get(state_hash, self._config_hash)
        if cached_result is not None:
            logger.debug(
                "Cache hit",
                state_hash=state_hash,
                action=cached_result.final_action.value,
            )
            return cached_result.final_action

        # Generate K samples
        prompt = build_prompt(state, self.prompt_config)
        samples = []
        sample_responses = []
        total_latency_ms = 0.0

        for i in range(self.config.k_samples):
            try:
                response, latency = self.provider.generate_with_timing(prompt)
                sample_responses.append(response)
                total_latency_ms += latency

                # Parse action
                parsed_str = parse_action_from_response(response)
                action = self._parse_action(parsed_str)

                if action is not None:
                    # Validate constraints
                    if self._composite_constraint.validate(state, action):
                        samples.append(action)
                    else:
                        # Constraint failed, use fallback
                        samples.append(self.config.fallback_action)
                else:
                    # Parse failed
                    samples.append(self.config.fallback_action)

            except Exception as e:
                logger.warning(
                    "Sample generation failed",
                    sample_index=i,
                    error=str(e),
                )
                samples.append(self.config.fallback_action)

        # Aggregate via voting
        if samples:
            vote_result = majority_vote(
                samples,
                tie_break=self.config.tie_break,
                strategy=self.config.tie_break_strategy,
            )
        else:
            # All samples failed
            vote_result = VoteResult(
                final_action=self.config.fallback_action,
                vote_counts={self.config.fallback_action: 1},
                vote_shares={self.config.fallback_action: 1.0},
                samples=[self.config.fallback_action],
                tie_broken=True,
                confidence=0.0,
            )

        # Cache result
        self.cache.put(
            state_hash=state_hash,
            config_hash=self._config_hash,
            vote_result=vote_result,
            metadata={"n_responses": len(sample_responses)},
        )

        # Log decision
        self._log_decision(
            state=state,
            state_hash=state_hash,
            prompt=prompt,
            responses=sample_responses,
            vote_result=vote_result,
            latency_ms=total_latency_ms,
        )

        return vote_result.final_action

    def _parse_action(self, action_str: str | None) -> Action | None:
        """Parse action string to Action enum."""
        if action_str is None:
            return None

        action_map = {
            "ENTER_ENTERPRISE": Action.ENTER_ENTERPRISE,
            "EXIT_ENTERPRISE": Action.EXIT_ENTERPRISE,
            "NO_CHANGE": Action.NO_CHANGE,
        }
        return action_map.get(action_str.upper())

    def _log_decision(
        self,
        state: HouseholdState,
        state_hash: str,
        prompt: str,
        responses: list[str],
        vote_result: VoteResult,
        latency_ms: float,
    ) -> None:
        """Log a decision with all samples."""
        # Use the decision logger with extended info
        self.decision_logger.log(
            state=state,
            prompt=prompt,
            response="\n---\n".join(responses),  # Join all responses
            parsed_action=vote_result.final_action.value,
            constraints_passed=True,  # Already validated
            failed_constraints=[],
            final_action=vote_result.final_action,
            provider=self.provider.name,
            model=self.config.model,
            latency_ms=latency_ms,
        )

        logger.debug(
            "Decision logged",
            state_hash=state_hash,
            k_samples=len(responses),
            final_action=vote_result.final_action.value,
            confidence=vote_result.confidence,
            tie_broken=vote_result.tie_broken,
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

    def save_cache(self, path: Path | str) -> None:
        """Save decision cache to disk."""
        self.cache.save(path)

    def load_cache(self, path: Path | str) -> None:
        """Load decision cache from disk."""
        self.cache.load(path)

    def save_log(self, path: Path | str | None = None) -> Path:
        """Save decision log."""
        return self.decision_logger.save(path)

    def get_log_summary(self) -> dict[str, Any]:
        """Get decision log summary."""
        return self.decision_logger.get_summary()


class MultiSampleLLMPolicyFactory:
    """Factory for creating multi-sample LLM policies."""

    @staticmethod
    def create_o4mini_policy(
        api_key: str | None = None,
        temperature: float = 0.6,
        k_samples: int = 5,
        log_dir: Path | str | None = None,
        cache_dir: Path | str | None = None,
        constraints: list[Constraint] | None = None,
        country: str = "tanzania",
    ) -> MultiSampleLLMPolicy:
        """Create policy using OpenAI o4-mini (or gpt-4o-mini).

        Args:
            api_key: OpenAI API key.
            temperature: Sampling temperature.
            k_samples: Number of samples per decision.
            log_dir: Directory for decision logs.
            cache_dir: Directory for cache persistence.
            constraints: Custom constraints.
            country: Country for prompt context.

        Returns:
            Configured MultiSampleLLMPolicy.
        """
        from abm_enterprise.policies.providers import OpenAIProvider

        # Use gpt-4o-mini as o4-mini proxy
        provider = OpenAIProvider(
            api_key=api_key,
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=150,
        )

        config = LLMPolicyConfig(
            model="gpt-4o-mini",
            temperature=temperature,
            k_samples=k_samples,
            cache_enabled=True,
        )

        prompt_config = PromptConfig(country=country)

        # Set up cache
        cache = None
        if cache_dir is not None:
            from abm_enterprise.policies.cache import PersistentDecisionCache
            cache_path = Path(cache_dir) / "decision_cache.json"
            cache = PersistentDecisionCache(
                path=cache_path,
                max_size=10000,
                enabled=True,
            )

        return MultiSampleLLMPolicy(
            provider=provider,
            config=config,
            constraints=constraints,
            prompt_config=prompt_config,
            cache=cache,
            log_dir=log_dir,
        )

    @staticmethod
    def create_stub_policy(
        k_samples: int = 5,
        log_dir: Path | str | None = None,
        constraints: list[Constraint] | None = None,
        country: str = "tanzania",
    ) -> MultiSampleLLMPolicy:
        """Create policy using stub provider for testing.

        Args:
            k_samples: Number of samples per decision.
            log_dir: Directory for decision logs.
            constraints: Custom constraints.
            country: Country for prompt context.

        Returns:
            Configured MultiSampleLLMPolicy with StubProvider.
        """
        provider = StubProvider()

        config = LLMPolicyConfig(
            model="stub",
            temperature=0.0,  # Stub is deterministic
            k_samples=k_samples,
            cache_enabled=True,
        )

        prompt_config = PromptConfig(country=country)

        return MultiSampleLLMPolicy(
            provider=provider,
            config=config,
            constraints=constraints,
            prompt_config=prompt_config,
            log_dir=log_dir,
        )
