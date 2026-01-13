"""LLM provider adapters for policy decisions.

This module defines the provider interface and implementations
for various LLM backends including stub, replay, Claude, and OpenAI.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from abm_enterprise.policies.logging import DecisionLogger, DecisionRecord


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement the generate() method
    which takes a prompt and returns a response string.
    """

    @property
    def name(self) -> str:
        """Get provider name.

        Returns:
            Provider class name.
        """
        return self.__class__.__name__

    @property
    def model(self) -> str:
        """Get model identifier.

        Returns:
            Model name or identifier.
        """
        return "unknown"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from prompt.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated response text.
        """
        pass

    def generate_with_timing(self, prompt: str) -> tuple[str, float]:
        """Generate response and measure latency.

        Args:
            prompt: Input prompt text.

        Returns:
            Tuple of (response, latency_ms).
        """
        start = time.perf_counter()
        response = self.generate(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        return response, latency_ms


class StubProvider(LLMProvider):
    """Deterministic stub provider for testing.

    Implements simple rule-based logic to return valid actions
    without calling any external API. Useful for testing and
    development.

    Attributes:
        price_threshold: Price exposure threshold for entry decision.
        asset_threshold: Asset threshold for entry decision.
    """

    def __init__(
        self,
        price_threshold: float = -0.1,
        asset_threshold: float = 0.3,
    ) -> None:
        """Initialize stub provider.

        Args:
            price_threshold: Below this, consider adverse shock.
            asset_threshold: Below this, consider low assets.
        """
        self.price_threshold = price_threshold
        self.asset_threshold = asset_threshold

    @property
    def model(self) -> str:
        """Get model identifier."""
        return "stub-rule-based"

    def generate(self, prompt: str) -> str:
        """Generate rule-based response from prompt.

        Parses state from prompt and applies simple rules:
        - If adverse price shock and low assets, enter enterprise
        - If in enterprise with extreme low assets, exit
        - Otherwise, no change

        Args:
            prompt: Input prompt text.

        Returns:
            Response with ACTION: prefix.
        """
        # Parse state from prompt
        state = self._parse_state_from_prompt(prompt)

        if state is None:
            return "ACTION: NO_CHANGE"

        # Apply simple rules
        price_exposure = state.get("price_exposure", 0.0)
        assets = state.get("assets", 0.0)
        in_enterprise = "operates enterprise" in state.get("enterprise", "").lower()

        # Entry decision
        if not in_enterprise:
            if price_exposure < self.price_threshold and assets < self.asset_threshold:
                return "ACTION: ENTER_ENTERPRISE"

        # Exit decision (only under extreme conditions)
        if in_enterprise:
            if assets < -1.5:  # Very low assets
                return "ACTION: EXIT_ENTERPRISE"

        return "ACTION: NO_CHANGE"

    def _parse_state_from_prompt(self, prompt: str) -> dict[str, Any] | None:
        """Parse household state from prompt text.

        Args:
            prompt: Prompt text containing state information.

        Returns:
            Dictionary with parsed state values or None if parsing fails.
        """
        state: dict[str, Any] = {}

        # Parse price exposure
        price_match = re.search(r"Price Exposure:\s*([-\d.]+)", prompt)
        if price_match:
            state["price_exposure"] = float(price_match.group(1))

        # Parse assets
        assets_match = re.search(r"Assets(?:\s+Index)?:\s*([-\d.]+)", prompt)
        if assets_match:
            state["assets"] = float(assets_match.group(1))

        # Parse enterprise status
        enterprise_match = re.search(
            r"(?:Current\s+)?Enterprise(?:\s+Status)?:\s*([^\n]+)", prompt
        )
        if enterprise_match:
            state["enterprise"] = enterprise_match.group(1)

        # Parse credit access
        credit_match = re.search(r"Credit(?:\s+Access)?:\s*([^\n]+)", prompt)
        if credit_match:
            state["credit"] = credit_match.group(1)

        return state if state else None


class ReplayProvider(LLMProvider):
    """Replays logged decisions for reproducibility.

    Reads from a decision log file and returns the recorded
    responses in order, enabling deterministic replay of
    previous simulation runs.

    Attributes:
        log_path: Path to the decision log file.
        decisions: List of loaded decision records.
        index: Current position in the replay sequence.
        match_by_hash: Whether to match by state hash instead of sequence.
    """

    def __init__(
        self,
        log_path: Path | str,
        match_by_hash: bool = False,
    ) -> None:
        """Initialize replay provider.

        Args:
            log_path: Path to JSONL decision log file.
            match_by_hash: If True, match by state hash; if False, use sequence.

        Raises:
            FileNotFoundError: If log file does not exist.
        """
        self.log_path = Path(log_path)
        self.decisions = DecisionLogger.load(self.log_path)
        self.index = 0
        self.match_by_hash = match_by_hash

        # Build hash index for hash-based matching
        self._hash_index: dict[str, list[DecisionRecord]] = {}
        for record in self.decisions:
            if record.state_hash not in self._hash_index:
                self._hash_index[record.state_hash] = []
            self._hash_index[record.state_hash].append(record)

    @property
    def model(self) -> str:
        """Get model identifier."""
        return f"replay:{self.log_path.name}"

    def generate(self, prompt: str) -> str:
        """Return logged response.

        Args:
            prompt: Input prompt (used for hash matching if enabled).

        Returns:
            Logged response string.

        Raises:
            IndexError: If replay sequence is exhausted.
        """
        if self.match_by_hash:
            # Extract state hash from prompt if available
            # This is a simplified approach - could be enhanced
            return self._generate_by_sequence()
        else:
            return self._generate_by_sequence()

    def _generate_by_sequence(self) -> str:
        """Generate response by sequence order.

        Returns:
            Next logged response.

        Raises:
            IndexError: If sequence exhausted.
        """
        if self.index >= len(self.decisions):
            raise IndexError(
                f"Replay sequence exhausted at index {self.index}. "
                f"Log has {len(self.decisions)} decisions."
            )

        record = self.decisions[self.index]
        self.index += 1
        return record.response

    def generate_by_hash(self, state_hash: str) -> str:
        """Generate response by matching state hash.

        Args:
            state_hash: State hash to match.

        Returns:
            Logged response for matching state.

        Raises:
            KeyError: If no matching hash found.
        """
        if state_hash not in self._hash_index:
            raise KeyError(f"No logged decision for state hash: {state_hash}")

        records = self._hash_index[state_hash]
        # Return first unused record for this hash
        for record in records:
            return record.response

        raise KeyError(f"All decisions for hash {state_hash} exhausted")

    def reset(self) -> None:
        """Reset replay to beginning."""
        self.index = 0


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider.

    Requires ANTHROPIC_API_KEY environment variable or explicit key.

    Attributes:
        api_key: Anthropic API key.
        model_id: Claude model to use.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> None:
        """Initialize Claude provider.

        Args:
            api_key: API key. If None, reads from ANTHROPIC_API_KEY env var.
            model: Model ID to use.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0 for deterministic).

        Raises:
            ValueError: If no API key available.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )

        self.model_id = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Any = None

    @property
    def model(self) -> str:
        """Get model identifier."""
        return self.model_id

    def _get_client(self) -> Any:
        """Get or create Anthropic client.

        Returns:
            Anthropic client instance.

        Raises:
            ImportError: If anthropic package not installed.
        """
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required for ClaudeProvider. "
                    "Install with: pip install anthropic"
                ) from None
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate response using Claude API.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated response text.
        """
        client = self._get_client()

        message = client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        if message.content and len(message.content) > 0:
            return message.content[0].text
        return ""


class OpenAIProvider(LLMProvider):
    """OpenAI GPT API provider.

    Requires OPENAI_API_KEY environment variable or explicit key.

    Attributes:
        api_key: OpenAI API key.
        model_id: GPT model to use.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: API key. If None, reads from OPENAI_API_KEY env var.
            model: Model ID to use.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0 for deterministic).

        Raises:
            ValueError: If no API key available.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key."
            )

        self.model_id = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Any = None

    @property
    def model(self) -> str:
        """Get model identifier."""
        return self.model_id

    def _get_client(self) -> Any:
        """Get or create OpenAI client.

        Returns:
            OpenAI client instance.

        Raises:
            ImportError: If openai package not installed.
        """
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAIProvider. "
                    "Install with: pip install openai"
                ) from None
        return self._client

    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI API.

        Args:
            prompt: Input prompt text.

        Returns:
            Generated response text.
        """
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content or ""
        return ""


def get_provider(
    provider_type: str,
    replay_log: Path | str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Factory function to create LLM provider.

    Args:
        provider_type: Type of provider (stub, replay, claude, openai).
        replay_log: Path to log file for replay provider.
        **kwargs: Additional arguments for provider constructor.

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If provider type not recognized.
    """
    provider_type = provider_type.lower()

    if provider_type == "stub":
        return StubProvider(**kwargs)

    elif provider_type == "replay":
        if replay_log is None:
            raise ValueError("replay_log path required for replay provider")
        return ReplayProvider(log_path=replay_log, **kwargs)

    elif provider_type == "claude":
        return ClaudeProvider(**kwargs)

    elif provider_type == "openai":
        return OpenAIProvider(**kwargs)

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Valid types: stub, replay, claude, openai"
        )
