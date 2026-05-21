"""
Pytest configuration and shared fixtures.
"""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_dict():
    """Provide sample dictionary for tests."""
    return {"a": 1, "b": 2, "c": 3}


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before each test (if needed)."""
    yield
    # Cleanup code here (if needed)
