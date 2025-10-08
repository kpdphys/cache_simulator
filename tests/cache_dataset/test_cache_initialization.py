# -*- coding: utf-8 -*-
"""Tests for the __init__ method and configuration of the Cache class.

This module verifies that the Cache object is correctly initialized
with various configurations, including default parameters, different
associativity types (direct-mapped, set-associative, fully associative),
and handles invalid parameters gracefully by raising exceptions.
"""

from collections import OrderedDict

import pytest

from cache_simulator.cache_dataset.cache import Cache


def test_default_initialization() -> None:
    """Test cache initialization with default parameters."""
    cache = Cache()
    assert cache.line_size == 64
    assert cache.num_lines == 1024
    assert cache.associativity == 0
    assert cache.num_sets == 1
    assert cache.lines_per_set == 1024


def test_fully_associative_cache_initialization() -> None:
    """Test fully associative cache (associativity=0)."""
    cache = Cache(line_size=64, num_lines=128, associativity=0)
    assert cache.num_sets == 1
    assert cache.lines_per_set == 128


def test_direct_mapped_cache_initialization() -> None:
    """Test direct mapped cache (associativity=1)."""
    cache = Cache(line_size=64, num_lines=128, associativity=1)
    assert cache.num_sets == 128
    assert cache.lines_per_set == 1


def test_set_associative_cache_initialization() -> None:
    """Test set-associative cache (associativity>1)."""
    cache = Cache(line_size=64, num_lines=128, associativity=4)
    assert cache.num_sets == 32
    assert cache.lines_per_set == 4


def test_invalid_associativity_raises_error() -> None:
    """Test that invalid associativity raises ValueError."""
    with pytest.raises(ValueError, match="must be divisible"):
        Cache(line_size=64, num_lines=100, associativity=3)


def test_cache_structures_initialized() -> None:
    """Test that cache structures are properly initialized."""
    cache = Cache(line_size=64, num_lines=8, associativity=2)
    assert len(cache.cache) == 4  # 8 lines / 2-way = 4 sets
    assert isinstance(cache.cache[0], OrderedDict)
    assert len(cache.tag_set) == 0


@pytest.mark.parametrize(
    ("line_size", "num_lines", "associativity", "expected_sets"),
    [
        (64, 128, 0, 1),  # Fully associative
        (64, 128, 1, 128),  # Direct mapped
        (64, 128, 2, 64),  # 2-way set associative
        (64, 128, 4, 32),  # 4-way set associative
        (64, 128, 8, 16),  # 8-way set associative
        (32, 256, 4, 64),  # Different line size
    ],
)
def test_cache_configuration_parametrized(
    line_size: int, num_lines: int, associativity: int, expected_sets: int
) -> None:
    """Test various cache configurations using parametrization."""
    cache = Cache(line_size=line_size, num_lines=num_lines, associativity=associativity)
    assert cache.num_sets == expected_sets
