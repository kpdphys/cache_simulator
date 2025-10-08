# -*- coding: utf-8 -*-
"""Tests for the Cache class operating in fully associative mode.

This module focuses on the specific behaviors of a fully associative cache
(associativity=0). Key tested aspects include:
- The structural property of having a single set.
- The absence of conflict misses until the cache is at capacity.
- The correct implementation of the LRU (Least Recently Used) eviction policy
  when the cache is full.
"""

import pytest

from cache_simulator.cache_dataset.cache import Cache


def test_fully_all_lines_in_one_set(fully_associative_cache: Cache) -> None:
    """Test that all lines are in a single set."""
    assert fully_associative_cache.num_sets == 1
    assert fully_associative_cache.lines_per_set == 4


def test_fully_no_conflict_misses_until_full(fully_associative_cache: Cache) -> None:
    """Test that there are no conflict misses until cache is full."""
    # Add 4 different lines
    fully_associative_cache.add_to_cache(0)
    fully_associative_cache.add_to_cache(64)
    fully_associative_cache.add_to_cache(128)
    fully_associative_cache.add_to_cache(192)

    # All should be in cache
    assert fully_associative_cache.is_in_cache(0)
    assert fully_associative_cache.is_in_cache(64)
    assert fully_associative_cache.is_in_cache(128)
    assert fully_associative_cache.is_in_cache(192)


def test_fully_lru_eviction_when_full(fully_associative_cache: Cache) -> None:
    """Test LRU eviction when cache is full."""
    # Fill cache
    fully_associative_cache.add_to_cache(0)  # Oldest
    fully_associative_cache.add_to_cache(64)
    fully_associative_cache.add_to_cache(128)
    fully_associative_cache.add_to_cache(192)

    # Add fifth line, should evict line at address 0
    fully_associative_cache.add_to_cache(256)

    assert not fully_associative_cache.is_in_cache(0)  # Evicted
    assert fully_associative_cache.is_in_cache(64)
    assert fully_associative_cache.is_in_cache(128)
    assert fully_associative_cache.is_in_cache(192)
    assert fully_associative_cache.is_in_cache(256)


def test_fully_lru_policy_with_access(fully_associative_cache: Cache) -> None:
    """Test LRU policy updates on cache access."""
    # Fill cache
    fully_associative_cache.add_to_cache(0)
    fully_associative_cache.add_to_cache(64)
    fully_associative_cache.add_to_cache(128)
    fully_associative_cache.add_to_cache(192)

    # Access address 0 to make it most recently used
    fully_associative_cache.is_in_cache(0)

    # Add new line, should evict 64 (now LRU)
    fully_associative_cache.add_to_cache(256)

    assert fully_associative_cache.is_in_cache(0)  # Still in cache
    assert not fully_associative_cache.is_in_cache(64)  # Evicted
    assert fully_associative_cache.is_in_cache(128)
    assert fully_associative_cache.is_in_cache(192)
    assert fully_associative_cache.is_in_cache(256)


@pytest.mark.parametrize(
    "addresses",
    [
        [0, 64, 128, 192],  # Sequential lines
        [0, 256, 512, 768],  # Lines that would conflict in other caches
        [100, 200, 300, 400],  # Arbitrary addresses
        [0, 1, 2, 3],  # Same cache line
    ],
)
def test_cache_operations_parametrized(
    addresses: list[int], fully_associative_cache: Cache
) -> None:
    """Test cache operations with various address patterns."""
    for addr in addresses:
        fully_associative_cache.add_to_cache(addr)

    # All addresses should be in cache (within capacity)
    for addr in addresses:
        assert fully_associative_cache.is_in_cache(addr)
