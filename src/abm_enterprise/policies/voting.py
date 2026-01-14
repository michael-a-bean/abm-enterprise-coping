"""Multi-sample voting aggregation for LLM decisions.

This module provides functionality to aggregate multiple LLM samples
into a final decision using majority voting or probabilistic methods.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from abm_enterprise.policies.base import Action


@dataclass
class VoteResult:
    """Result of multi-sample voting.

    Attributes:
        final_action: The action selected by voting.
        vote_counts: Mapping of action to vote count.
        vote_shares: Mapping of action to vote share (0-1).
        samples: List of all sampled actions.
        tie_broken: Whether a tie-break was applied.
        confidence: Confidence score based on vote share.
    """

    final_action: Action
    vote_counts: dict[Action, int]
    vote_shares: dict[Action, float]
    samples: list[Action]
    tie_broken: bool = False
    confidence: float = 0.0

    @property
    def n_samples(self) -> int:
        """Number of samples used in voting."""
        return len(self.samples)

    @property
    def agreement_rate(self) -> float:
        """Proportion of samples agreeing with final action."""
        if not self.samples:
            return 0.0
        return self.vote_shares.get(self.final_action, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_action": self.final_action.value,
            "vote_counts": {k.value: v for k, v in self.vote_counts.items()},
            "vote_shares": {k.value: v for k, v in self.vote_shares.items()},
            "samples": [a.value for a in self.samples],
            "tie_broken": self.tie_broken,
            "confidence": self.confidence,
            "n_samples": self.n_samples,
            "agreement_rate": self.agreement_rate,
        }


class TieBreakStrategy(str, Enum):
    """Strategy for breaking ties in voting."""

    # Prefer conservative action (NO_CHANGE)
    CONSERVATIVE = "conservative"
    # Prefer action that appears first in samples
    FIRST_SEEN = "first_seen"
    # Random selection among tied actions
    RANDOM = "random"
    # Prefer entry (for coping hypothesis)
    PREFER_ENTRY = "prefer_entry"


def majority_vote(
    samples: list[Action],
    tie_break: Action = Action.NO_CHANGE,
    strategy: TieBreakStrategy = TieBreakStrategy.CONSERVATIVE,
) -> VoteResult:
    """Aggregate samples using majority voting.

    Args:
        samples: List of sampled actions.
        tie_break: Default action when tie cannot be resolved.
        strategy: Strategy for breaking ties.

    Returns:
        VoteResult with aggregated decision.
    """
    if not samples:
        return VoteResult(
            final_action=tie_break,
            vote_counts={},
            vote_shares={},
            samples=[],
            tie_broken=True,
            confidence=0.0,
        )

    # Count votes
    counter = Counter(samples)
    total = len(samples)

    # Compute shares
    vote_counts = dict(counter)
    vote_shares = {action: count / total for action, count in counter.items()}

    # Find winner(s)
    max_count = max(counter.values())
    winners = [action for action, count in counter.items() if count == max_count]

    tie_broken = len(winners) > 1

    if tie_broken:
        # Apply tie-break strategy
        final_action = _apply_tie_break(winners, samples, tie_break, strategy)
    else:
        final_action = winners[0]

    confidence = vote_shares.get(final_action, 0.0)

    return VoteResult(
        final_action=final_action,
        vote_counts=vote_counts,
        vote_shares=vote_shares,
        samples=samples,
        tie_broken=tie_broken,
        confidence=confidence,
    )


def _apply_tie_break(
    winners: list[Action],
    samples: list[Action],
    default: Action,
    strategy: TieBreakStrategy,
) -> Action:
    """Apply tie-break strategy to select from winners.

    Args:
        winners: List of tied actions.
        samples: Original sample list.
        default: Default fallback action.
        strategy: Tie-break strategy.

    Returns:
        Selected action.
    """
    if strategy == TieBreakStrategy.CONSERVATIVE:
        # Prefer NO_CHANGE, then default
        if Action.NO_CHANGE in winners:
            return Action.NO_CHANGE
        return default

    elif strategy == TieBreakStrategy.FIRST_SEEN:
        # Return first winner in sample order
        for sample in samples:
            if sample in winners:
                return sample
        return default

    elif strategy == TieBreakStrategy.PREFER_ENTRY:
        # Prefer ENTER_ENTERPRISE for coping hypothesis
        if Action.ENTER_ENTERPRISE in winners:
            return Action.ENTER_ENTERPRISE
        return winners[0]

    elif strategy == TieBreakStrategy.RANDOM:
        import random
        return random.choice(winners)

    else:
        return default


def weighted_vote(
    samples: list[Action],
    weights: list[float] | None = None,
    threshold: float = 0.5,
) -> VoteResult:
    """Aggregate samples using weighted voting.

    Args:
        samples: List of sampled actions.
        weights: Optional weights for each sample (default: equal weights).
        threshold: Minimum vote share to win.

    Returns:
        VoteResult with aggregated decision.
    """
    if not samples:
        return VoteResult(
            final_action=Action.NO_CHANGE,
            vote_counts={},
            vote_shares={},
            samples=[],
            tie_broken=True,
            confidence=0.0,
        )

    if weights is None:
        weights = [1.0] * len(samples)

    # Compute weighted counts
    weighted_counts: dict[Action, float] = {}
    for action, weight in zip(samples, weights):
        weighted_counts[action] = weighted_counts.get(action, 0.0) + weight

    total_weight = sum(weights)
    vote_shares = {a: c / total_weight for a, c in weighted_counts.items()}

    # Find winner
    max_share = max(vote_shares.values())
    winners = [a for a, s in vote_shares.items() if s == max_share]

    if len(winners) > 1:
        final_action = Action.NO_CHANGE if Action.NO_CHANGE in winners else winners[0]
        tie_broken = True
    else:
        final_action = winners[0]
        tie_broken = False

    # Convert to integer counts for compatibility
    vote_counts = {a: int(c) for a, c in weighted_counts.items()}

    return VoteResult(
        final_action=final_action,
        vote_counts=vote_counts,
        vote_shares=vote_shares,
        samples=samples,
        tie_broken=tie_broken,
        confidence=vote_shares.get(final_action, 0.0),
    )


def probabilistic_sample(
    samples: list[Action],
    temperature: float = 1.0,
) -> VoteResult:
    """Sample final action from vote distribution.

    Instead of taking the majority, sample from the vote distribution
    with optional temperature scaling.

    Args:
        samples: List of sampled actions.
        temperature: Temperature for softmax (1.0 = raw distribution).

    Returns:
        VoteResult with sampled decision.
    """
    import random
    import math

    if not samples:
        return VoteResult(
            final_action=Action.NO_CHANGE,
            vote_counts={},
            vote_shares={},
            samples=[],
            tie_broken=True,
            confidence=0.0,
        )

    # Count votes
    counter = Counter(samples)
    total = len(samples)

    vote_counts = dict(counter)
    raw_shares = {a: c / total for a, c in counter.items()}

    # Apply temperature scaling
    if temperature != 1.0:
        actions = list(raw_shares.keys())
        log_probs = [math.log(raw_shares[a] + 1e-10) / temperature for a in actions]
        max_log = max(log_probs)
        exp_probs = [math.exp(lp - max_log) for lp in log_probs]
        sum_exp = sum(exp_probs)
        probs = [ep / sum_exp for ep in exp_probs]
        vote_shares = dict(zip(actions, probs))
    else:
        vote_shares = raw_shares

    # Sample from distribution
    actions = list(vote_shares.keys())
    probs = [vote_shares[a] for a in actions]

    final_action = random.choices(actions, weights=probs, k=1)[0]

    return VoteResult(
        final_action=final_action,
        vote_counts=vote_counts,
        vote_shares=vote_shares,
        samples=samples,
        tie_broken=False,  # Not a tie-break, just probabilistic
        confidence=vote_shares.get(final_action, 0.0),
    )
