"""
Tests for utility functions.
"""

import pytest
from src.utils import greet, add, multiply


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_greet(self):
        """Test the greet function."""
        assert greet("Alice") == "Hello, Alice!"
        assert greet("") == "Hello, !"
    
    def test_add(self):
        """Test the add function."""
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0.5, 0.5) == 1.0
    
    def test_multiply(self):
        """Test the multiply function."""
        assert multiply(2, 3) == 6
        assert multiply(-2, 3) == -6
        assert multiply(0, 100) == 0
    
    @pytest.mark.slow
    def test_add_with_large_numbers(self):
        """Test add with large numbers (marked as slow)."""
        assert add(1e10, 1e10) == 2e10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
