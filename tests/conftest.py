"""Pytest configuration and fixtures for ABM Enterprise tests."""

import pytest

from abm_enterprise.utils.rng import GlobalRNG


@pytest.fixture(autouse=True)
def reset_rng() -> None:
    """Reset global RNG state before and after each test."""
    GlobalRNG.reset()
    yield
    GlobalRNG.reset()
