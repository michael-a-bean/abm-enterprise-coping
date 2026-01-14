"""Tests for voting and caching modules.

Tests multi-sample voting aggregation and decision caching.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from abm_enterprise.policies.base import Action
from abm_enterprise.policies.voting import (
    TieBreakStrategy,
    VoteResult,
    majority_vote,
    probabilistic_sample,
    weighted_vote,
)
from abm_enterprise.policies.cache import (
    DecisionCache,
    compute_config_hash,
    compute_state_hash,
)


class TestMajorityVote:
    """Tests for majority voting."""

    def test_clear_majority(self):
        """Test voting with clear majority."""
        samples = [
            Action.ENTER_ENTERPRISE,
            Action.ENTER_ENTERPRISE,
            Action.ENTER_ENTERPRISE,
            Action.NO_CHANGE,
            Action.EXIT_ENTERPRISE,
        ]

        result = majority_vote(samples)

        assert result.final_action == Action.ENTER_ENTERPRISE
        assert result.vote_counts[Action.ENTER_ENTERPRISE] == 3
        assert result.confidence == 0.6
        assert not result.tie_broken

    def test_unanimous_vote(self):
        """Test unanimous voting."""
        samples = [Action.NO_CHANGE] * 5

        result = majority_vote(samples)

        assert result.final_action == Action.NO_CHANGE
        assert result.confidence == 1.0
        assert not result.tie_broken

    def test_tie_break_conservative(self):
        """Test tie-break with conservative strategy."""
        samples = [
            Action.ENTER_ENTERPRISE,
            Action.ENTER_ENTERPRISE,
            Action.EXIT_ENTERPRISE,
            Action.EXIT_ENTERPRISE,
        ]

        result = majority_vote(
            samples,
            tie_break=Action.NO_CHANGE,
            strategy=TieBreakStrategy.CONSERVATIVE,
        )

        # Tie between ENTER and EXIT, neither is NO_CHANGE
        # With CONSERVATIVE strategy and tie_break=NO_CHANGE, uses NO_CHANGE
        assert result.tie_broken
        # Confidence is 0 because NO_CHANGE wasn't in original samples
        assert result.final_action == Action.NO_CHANGE
        assert result.confidence == 0.0

    def test_tie_break_with_no_change_in_winners(self):
        """Test tie-break when NO_CHANGE is in winners."""
        samples = [
            Action.ENTER_ENTERPRISE,
            Action.ENTER_ENTERPRISE,
            Action.NO_CHANGE,
            Action.NO_CHANGE,
        ]

        result = majority_vote(
            samples,
            strategy=TieBreakStrategy.CONSERVATIVE,
        )

        assert result.final_action == Action.NO_CHANGE
        assert result.tie_broken

    def test_empty_samples(self):
        """Test voting with empty samples."""
        result = majority_vote([])

        assert result.final_action == Action.NO_CHANGE
        assert result.n_samples == 0
        assert result.tie_broken

    def test_single_sample(self):
        """Test voting with single sample."""
        result = majority_vote([Action.EXIT_ENTERPRISE])

        assert result.final_action == Action.EXIT_ENTERPRISE
        assert result.confidence == 1.0
        assert not result.tie_broken

    def test_vote_shares_sum_to_one(self):
        """Test that vote shares sum to 1."""
        samples = [
            Action.ENTER_ENTERPRISE,
            Action.EXIT_ENTERPRISE,
            Action.NO_CHANGE,
        ]

        result = majority_vote(samples)

        total_share = sum(result.vote_shares.values())
        assert abs(total_share - 1.0) < 0.001

    def test_prefer_entry_strategy(self):
        """Test prefer entry tie-break strategy."""
        samples = [
            Action.ENTER_ENTERPRISE,
            Action.ENTER_ENTERPRISE,
            Action.NO_CHANGE,
            Action.NO_CHANGE,
        ]

        result = majority_vote(
            samples,
            strategy=TieBreakStrategy.PREFER_ENTRY,
        )

        assert result.final_action == Action.ENTER_ENTERPRISE


class TestVoteResult:
    """Tests for VoteResult dataclass."""

    def test_n_samples(self):
        """Test n_samples property."""
        samples = [Action.NO_CHANGE] * 7
        result = majority_vote(samples)

        assert result.n_samples == 7

    def test_agreement_rate(self):
        """Test agreement rate property."""
        samples = [
            Action.ENTER_ENTERPRISE,
            Action.ENTER_ENTERPRISE,
            Action.ENTER_ENTERPRISE,
            Action.NO_CHANGE,
            Action.NO_CHANGE,
        ]
        result = majority_vote(samples)

        assert result.agreement_rate == 0.6

    def test_to_dict(self):
        """Test serialization to dict."""
        samples = [Action.NO_CHANGE, Action.ENTER_ENTERPRISE]
        result = majority_vote(samples)

        d = result.to_dict()

        assert "final_action" in d
        assert "vote_counts" in d
        assert "n_samples" in d
        assert d["n_samples"] == 2


class TestWeightedVote:
    """Tests for weighted voting."""

    def test_weighted_vote_equal_weights(self):
        """Test weighted vote with equal weights."""
        samples = [
            Action.ENTER_ENTERPRISE,
            Action.ENTER_ENTERPRISE,
            Action.NO_CHANGE,
        ]

        result = weighted_vote(samples)  # Default equal weights

        assert result.final_action == Action.ENTER_ENTERPRISE

    def test_weighted_vote_custom_weights(self):
        """Test weighted vote with custom weights."""
        samples = [
            Action.ENTER_ENTERPRISE,
            Action.NO_CHANGE,
            Action.NO_CHANGE,
        ]
        weights = [3.0, 1.0, 1.0]  # First sample has 3x weight

        result = weighted_vote(samples, weights=weights)

        # ENTER has weight 3, NO_CHANGE has weight 2
        assert result.final_action == Action.ENTER_ENTERPRISE


class TestProbabilisticSample:
    """Tests for probabilistic sampling."""

    def test_probabilistic_returns_valid_action(self):
        """Test probabilistic sample returns valid action."""
        samples = [Action.NO_CHANGE] * 5

        result = probabilistic_sample(samples)

        assert result.final_action == Action.NO_CHANGE

    def test_probabilistic_respects_distribution(self):
        """Test probabilistic sampling roughly follows distribution."""
        samples = [Action.ENTER_ENTERPRISE] * 9 + [Action.NO_CHANGE]

        # Sample many times
        enter_count = 0
        n_trials = 100
        for _ in range(n_trials):
            result = probabilistic_sample(samples)
            if result.final_action == Action.ENTER_ENTERPRISE:
                enter_count += 1

        # Should mostly be ENTER (90% in distribution)
        assert enter_count > 60, f"Expected mostly ENTER, got {enter_count}/{n_trials}"


class TestComputeStateHash:
    """Tests for state hashing."""

    def test_deterministic_hash(self):
        """Test hash is deterministic."""
        state = {
            "household_id": "HH_001",
            "wave": 1,
            "assets": 0.5,
            "credit_access": 1,
        }

        hash1 = compute_state_hash(state)
        hash2 = compute_state_hash(state)

        assert hash1 == hash2

    def test_different_states_different_hashes(self):
        """Test different states produce different hashes."""
        state1 = {"household_id": "HH_001", "wave": 1, "assets": 0.5}
        state2 = {"household_id": "HH_001", "wave": 1, "assets": 0.6}

        hash1 = compute_state_hash(state1)
        hash2 = compute_state_hash(state2)

        assert hash1 != hash2

    def test_order_independent(self):
        """Test hash is independent of key order."""
        state1 = {"a": 1, "b": 2, "c": 3}
        state2 = {"c": 3, "a": 1, "b": 2}

        hash1 = compute_state_hash(state1)
        hash2 = compute_state_hash(state2)

        assert hash1 == hash2


class TestDecisionCache:
    """Tests for DecisionCache."""

    def test_put_and_get(self):
        """Test basic put and get."""
        cache = DecisionCache(max_size=100)

        vote_result = VoteResult(
            final_action=Action.ENTER_ENTERPRISE,
            vote_counts={Action.ENTER_ENTERPRISE: 3},
            vote_shares={Action.ENTER_ENTERPRISE: 1.0},
            samples=[Action.ENTER_ENTERPRISE] * 3,
        )

        cache.put("state123", "config456", vote_result)
        retrieved = cache.get("state123", "config456")

        assert retrieved is not None
        assert retrieved.final_action == Action.ENTER_ENTERPRISE

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = DecisionCache()

        result = cache.get("nonexistent", "config")

        assert result is None

    def test_cache_disabled(self):
        """Test disabled cache always misses."""
        cache = DecisionCache(enabled=False)

        vote_result = VoteResult(
            final_action=Action.NO_CHANGE,
            vote_counts={Action.NO_CHANGE: 1},
            vote_shares={Action.NO_CHANGE: 1.0},
            samples=[Action.NO_CHANGE],
        )

        cache.put("state", "config", vote_result)
        result = cache.get("state", "config")

        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when over capacity."""
        cache = DecisionCache(max_size=3)

        for i in range(5):
            vote_result = VoteResult(
                final_action=Action.NO_CHANGE,
                vote_counts={Action.NO_CHANGE: 1},
                vote_shares={Action.NO_CHANGE: 1.0},
                samples=[Action.NO_CHANGE],
            )
            cache.put(f"state{i}", "config", vote_result)

        # First entries should be evicted
        assert cache.get("state0", "config") is None
        assert cache.get("state1", "config") is None
        # Recent entries should remain
        assert cache.get("state4", "config") is not None

        assert cache.size == 3

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = DecisionCache()

        vote_result = VoteResult(
            final_action=Action.NO_CHANGE,
            vote_counts={Action.NO_CHANGE: 1},
            vote_shares={Action.NO_CHANGE: 1.0},
            samples=[Action.NO_CHANGE],
        )

        cache.put("existing", "config", vote_result)

        # One hit, one miss
        cache.get("existing", "config")
        cache.get("nonexistent", "config")

        assert cache.hit_rate == 0.5

    def test_save_and_load(self):
        """Test cache persistence."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"

            # Create and populate cache
            cache1 = DecisionCache()
            vote_result = VoteResult(
                final_action=Action.ENTER_ENTERPRISE,
                vote_counts={Action.ENTER_ENTERPRISE: 3, Action.NO_CHANGE: 2},
                vote_shares={Action.ENTER_ENTERPRISE: 0.6, Action.NO_CHANGE: 0.4},
                samples=[Action.ENTER_ENTERPRISE] * 3 + [Action.NO_CHANGE] * 2,
                tie_broken=False,
                confidence=0.6,
            )
            cache1.put("state123", "config456", vote_result)
            cache1.save(path)

            # Load into new cache
            cache2 = DecisionCache()
            cache2.load(path)

            retrieved = cache2.get("state123", "config456")
            assert retrieved is not None
            assert retrieved.final_action == Action.ENTER_ENTERPRISE
            assert retrieved.confidence == 0.6

    def test_get_stats(self):
        """Test cache statistics."""
        cache = DecisionCache(max_size=100, enabled=True)

        stats = cache.get_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "enabled" in stats
        assert "hit_rate" in stats
        assert stats["max_size"] == 100
        assert stats["enabled"] is True


class TestConfigHash:
    """Tests for config hashing."""

    def test_config_hash_deterministic(self):
        """Test config hash is deterministic."""
        config = {"model": "gpt-4", "temperature": 0.5, "k": 5}

        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)

        assert hash1 == hash2

    def test_different_configs_different_hashes(self):
        """Test different configs produce different hashes."""
        config1 = {"model": "gpt-4", "temperature": 0.5}
        config2 = {"model": "gpt-4", "temperature": 0.6}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 != hash2
