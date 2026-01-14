"""Decision caching for LLM policy reproducibility and cost efficiency.

This module provides functionality to cache LLM decisions by state hash,
enabling deterministic replay and reduced API costs.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abm_enterprise.policies.voting import VoteResult
from abm_enterprise.utils.logging import get_logger

logger = get_logger(__name__)


def compute_state_hash(state_dict: dict[str, Any]) -> str:
    """Compute deterministic hash for a household state.

    Args:
        state_dict: Dictionary representation of HouseholdState.

    Returns:
        Hex digest of state hash.
    """
    # Sort keys for determinism
    sorted_items = sorted(state_dict.items())
    # Round floats for stability
    normalized = []
    for k, v in sorted_items:
        if isinstance(v, float):
            v = round(v, 6)
        normalized.append((k, v))

    json_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute hash for policy configuration.

    Args:
        config_dict: Policy configuration dictionary.

    Returns:
        Hex digest of config hash.
    """
    json_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:8]


@dataclass
class CacheEntry:
    """A single cache entry.

    Attributes:
        state_hash: Hash of the household state.
        config_hash: Hash of the policy configuration.
        vote_result: The cached voting result.
        metadata: Additional metadata.
    """

    state_hash: str
    config_hash: str
    vote_result: VoteResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state_hash": self.state_hash,
            "config_hash": self.config_hash,
            "vote_result": self.vote_result.to_dict(),
            "metadata": self.metadata,
        }


class DecisionCache:
    """LRU cache for LLM decisions.

    Caches decisions by (state_hash, config_hash) to avoid redundant
    LLM calls for identical states with the same policy configuration.

    Attributes:
        max_size: Maximum number of entries to cache.
        enabled: Whether caching is enabled.
    """

    def __init__(
        self,
        max_size: int = 10000,
        enabled: bool = True,
    ) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum cache entries.
            enabled: Whether caching is active.
        """
        self.max_size = max_size
        self.enabled = enabled
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, state_hash: str, config_hash: str) -> str:
        """Create cache key from hashes."""
        return f"{state_hash}:{config_hash}"

    def get(
        self,
        state_hash: str,
        config_hash: str,
    ) -> VoteResult | None:
        """Retrieve cached decision.

        Args:
            state_hash: Hash of household state.
            config_hash: Hash of policy configuration.

        Returns:
            Cached VoteResult or None if not found.
        """
        if not self.enabled:
            return None

        key = self._make_key(state_hash, config_hash)

        if key in self._cache:
            self._hits += 1
            # Move to end (LRU)
            self._cache.move_to_end(key)
            return self._cache[key].vote_result

        self._misses += 1
        return None

    def put(
        self,
        state_hash: str,
        config_hash: str,
        vote_result: VoteResult,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Cache a decision result.

        Args:
            state_hash: Hash of household state.
            config_hash: Hash of policy configuration.
            vote_result: Result to cache.
            metadata: Optional metadata.
        """
        if not self.enabled:
            return

        key = self._make_key(state_hash, config_hash)

        entry = CacheEntry(
            state_hash=state_hash,
            config_hash=config_hash,
            vote_result=vote_result,
            metadata=metadata or {},
        )

        # Add/update entry
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = entry

        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0-1)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        return {
            "size": self.size,
            "max_size": self.max_size,
            "enabled": self.enabled,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def save(self, path: Path | str) -> None:
        """Save cache to disk.

        Args:
            path: Output file path (JSON).
        """
        from abm_enterprise.policies.base import Action

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "max_size": self.max_size,
            "entries": [],
        }

        for entry in self._cache.values():
            # Serialize vote result
            entry_dict = {
                "state_hash": entry.state_hash,
                "config_hash": entry.config_hash,
                "final_action": entry.vote_result.final_action.value,
                "vote_counts": {
                    k.value: v for k, v in entry.vote_result.vote_counts.items()
                },
                "vote_shares": {
                    k.value: v for k, v in entry.vote_result.vote_shares.items()
                },
                "samples": [a.value for a in entry.vote_result.samples],
                "tie_broken": entry.vote_result.tie_broken,
                "confidence": entry.vote_result.confidence,
                "metadata": entry.metadata,
            }
            data["entries"].append(entry_dict)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "Saved decision cache",
            path=str(path),
            entries=len(data["entries"]),
        )

    def load(self, path: Path | str) -> None:
        """Load cache from disk.

        Args:
            path: Input file path (JSON).
        """
        from abm_enterprise.policies.base import Action

        path = Path(path)

        if not path.exists():
            logger.warning("Cache file not found", path=str(path))
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.max_size = data.get("max_size", self.max_size)
        self._cache.clear()

        for entry_dict in data.get("entries", []):
            # Deserialize vote result
            vote_counts = {
                Action(k): v for k, v in entry_dict["vote_counts"].items()
            }
            vote_shares = {
                Action(k): v for k, v in entry_dict["vote_shares"].items()
            }
            samples = [Action(a) for a in entry_dict["samples"]]

            vote_result = VoteResult(
                final_action=Action(entry_dict["final_action"]),
                vote_counts=vote_counts,
                vote_shares=vote_shares,
                samples=samples,
                tie_broken=entry_dict.get("tie_broken", False),
                confidence=entry_dict.get("confidence", 0.0),
            )

            entry = CacheEntry(
                state_hash=entry_dict["state_hash"],
                config_hash=entry_dict["config_hash"],
                vote_result=vote_result,
                metadata=entry_dict.get("metadata", {}),
            )

            key = self._make_key(entry.state_hash, entry.config_hash)
            self._cache[key] = entry

        logger.info(
            "Loaded decision cache",
            path=str(path),
            entries=len(self._cache),
        )


class PersistentDecisionCache(DecisionCache):
    """Decision cache with automatic persistence.

    Extends DecisionCache to automatically save to disk on
    certain operations or when the cache is closed.
    """

    def __init__(
        self,
        path: Path | str,
        max_size: int = 10000,
        enabled: bool = True,
        auto_save_interval: int = 100,
    ) -> None:
        """Initialize persistent cache.

        Args:
            path: Path to cache file.
            max_size: Maximum cache entries.
            enabled: Whether caching is active.
            auto_save_interval: Save after this many new entries.
        """
        super().__init__(max_size=max_size, enabled=enabled)
        self.path = Path(path)
        self.auto_save_interval = auto_save_interval
        self._unsaved_count = 0

        # Load existing cache if present
        if self.path.exists():
            self.load(self.path)

    def put(
        self,
        state_hash: str,
        config_hash: str,
        vote_result: VoteResult,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Cache a decision with auto-save."""
        super().put(state_hash, config_hash, vote_result, metadata)

        self._unsaved_count += 1
        if self._unsaved_count >= self.auto_save_interval:
            self.save(self.path)
            self._unsaved_count = 0

    def close(self) -> None:
        """Save any unsaved entries before closing."""
        if self._unsaved_count > 0:
            self.save(self.path)
            self._unsaved_count = 0
