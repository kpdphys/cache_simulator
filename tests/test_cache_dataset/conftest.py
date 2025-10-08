# -*- coding: utf-8 -*-
"""Pytest fixtures for testing the Cache class.

This module provides a set of pre-configured Cache instances
to be used across different tests for the cache_simulator.
It includes fixtures for direct-mapped, set-associative,
and fully associative cache types.
"""

import pytest

from cache_simulator.cache_dataset.cache import Cache


@pytest.fixture
def direct_mapped_cache() -> Cache:
    """Create a direct mapped cache (4 lines, 64 bytes each)."""
    return Cache(line_size=64, num_lines=4, associativity=1)


@pytest.fixture
def set_associative_cache() -> Cache:
    """Create a 2-way set-associative cache (8 lines, 64 bytes each)."""
    return Cache(line_size=64, num_lines=8, associativity=2)


@pytest.fixture
def fully_associative_cache() -> Cache:
    """Create a fully associative cache (4 lines, 64 bytes each)."""
    return Cache(line_size=64, num_lines=4, associativity=0)
