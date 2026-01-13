"""Centralized random number generator for reproducible simulations.

All random operations in the ABM should use this module to ensure
reproducibility when the same seed is set.
"""

from __future__ import annotations

from numpy.random import RandomState


class GlobalRNG:
    """Global random number generator with seed management.

    This class provides a singleton-like interface for managing
    random state across the entire simulation.

    Attributes:
        _instance: The singleton RandomState instance.
        _seed: The seed used to initialize the RNG.
    """

    _instance: RandomState | None = None
    _seed: int | None = None

    @classmethod
    def get_instance(cls) -> RandomState:
        """Get the global RandomState instance.

        Returns:
            The seeded RandomState instance.

        Raises:
            RuntimeError: If set_seed() has not been called first.
        """
        if cls._instance is None:
            raise RuntimeError("RNG not initialized. Call set_seed() before using get_rng().")
        return cls._instance

    @classmethod
    def set_seed(cls, seed: int) -> None:
        """Initialize the global RNG with a seed.

        Args:
            seed: Integer seed for reproducibility.
        """
        cls._seed = seed
        cls._instance = RandomState(seed)

    @classmethod
    def get_seed(cls) -> int | None:
        """Get the current seed value.

        Returns:
            The seed used to initialize the RNG, or None if not set.
        """
        return cls._seed

    @classmethod
    def reset(cls) -> None:
        """Reset the global RNG state.

        Useful for testing to ensure clean state between tests.
        """
        cls._instance = None
        cls._seed = None


def set_seed(seed: int) -> None:
    """Initialize the global RNG with a seed.

    This is the primary interface for setting the random seed
    at the start of a simulation run.

    Args:
        seed: Integer seed for reproducibility.

    Example:
        >>> from abm_enterprise.utils.rng import set_seed, get_rng
        >>> set_seed(42)
        >>> rng = get_rng()
        >>> rng.random()  # Reproducible random number
    """
    GlobalRNG.set_seed(seed)


def get_rng() -> RandomState:
    """Get the seeded numpy RandomState.

    Returns:
        The global RandomState instance.

    Raises:
        RuntimeError: If set_seed() has not been called.

    Example:
        >>> from abm_enterprise.utils.rng import set_seed, get_rng
        >>> set_seed(42)
        >>> rng = get_rng()
        >>> values = rng.uniform(0, 1, size=10)
    """
    return GlobalRNG.get_instance()


def get_seed() -> int | None:
    """Get the current seed value.

    Returns:
        The seed used to initialize the RNG, or None if not set.
    """
    return GlobalRNG.get_seed()
