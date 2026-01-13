"""Tests for the centralized RNG module."""

import numpy as np
import pytest

from abm_enterprise.utils.rng import GlobalRNG, get_rng, get_seed, set_seed


class TestGlobalRNG:
    """Tests for GlobalRNG class."""

    def setup_method(self) -> None:
        """Reset RNG state before each test."""
        GlobalRNG.reset()

    def teardown_method(self) -> None:
        """Reset RNG state after each test."""
        GlobalRNG.reset()

    def test_set_seed_initializes_rng(self) -> None:
        """Test that set_seed initializes the RNG."""
        set_seed(42)
        rng = get_rng()
        assert rng is not None

    def test_get_seed_returns_seed(self) -> None:
        """Test that get_seed returns the set seed."""
        set_seed(12345)
        assert get_seed() == 12345

    def test_get_rng_without_seed_raises(self) -> None:
        """Test that get_rng raises if seed not set."""
        with pytest.raises(RuntimeError, match="RNG not initialized"):
            get_rng()

    def test_reproducibility_with_same_seed(self) -> None:
        """Test that same seed produces same sequence."""
        set_seed(42)
        rng1 = get_rng()
        values1 = [rng1.random() for _ in range(10)]

        GlobalRNG.reset()
        set_seed(42)
        rng2 = get_rng()
        values2 = [rng2.random() for _ in range(10)]

        assert values1 == values2

    def test_different_seeds_produce_different_values(self) -> None:
        """Test that different seeds produce different sequences."""
        set_seed(42)
        rng1 = get_rng()
        values1 = [rng1.random() for _ in range(10)]

        GlobalRNG.reset()
        set_seed(123)
        rng2 = get_rng()
        values2 = [rng2.random() for _ in range(10)]

        assert values1 != values2

    def test_rng_is_numpy_random_state(self) -> None:
        """Test that RNG is numpy RandomState."""
        set_seed(42)
        rng = get_rng()
        assert isinstance(rng, np.random.RandomState)

    def test_rng_methods_work(self) -> None:
        """Test that standard RNG methods work."""
        set_seed(42)
        rng = get_rng()

        # Test various methods
        uniform = rng.uniform(0, 1, size=5)
        assert len(uniform) == 5
        assert all(0 <= v <= 1 for v in uniform)

        normal = rng.normal(0, 1, size=5)
        assert len(normal) == 5

        choice = rng.choice([1, 2, 3], size=3)
        assert len(choice) == 3

    def test_reset_clears_state(self) -> None:
        """Test that reset clears the RNG state."""
        set_seed(42)
        GlobalRNG.reset()
        assert get_seed() is None
        with pytest.raises(RuntimeError):
            get_rng()
