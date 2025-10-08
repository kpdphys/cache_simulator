# -*- coding: utf-8 -*-
"""Tests for the Cache class operating in set-associative mode.

This module validates the behavior of an N-way set-associative cache.
The tests confirm that:
- Multiple cache lines can coexist within the same set up to its capacity.
- The LRU eviction policy correctly operates on a per-set basis when a set becomes full.
- Different sets function independently of one another.
"""

from cache_simulator.cache_dataset.cache import Cache


def test_set_multiple_lines_in_same_set(set_associative_cache: Cache) -> None:
    """Test that multiple lines can coexist in the same set."""
    # Cache has 4 sets (8 lines / 2-way).
    # Addresses that map to set 0: line_address % 4 = 0
    set_associative_cache.add_to_cache(0)  # Line 0, set 0
    set_associative_cache.add_to_cache(256)  # Line 4, set 0

    # Both should be in cache (2-way associative)
    assert set_associative_cache.is_in_cache(0)
    assert set_associative_cache.is_in_cache(256)


def test_set_lru_eviction_in_set(set_associative_cache: Cache) -> None:
    """Test LRU eviction when set is full."""
    # Fill set 0 with 2 lines
    set_associative_cache.add_to_cache(0)  # Line 0, set 0 (oldest)
    set_associative_cache.add_to_cache(256)  # Line 4, set 0

    # Add third line to set 0, should evict line 0 (LRU)
    set_associative_cache.add_to_cache(512)  # Line 8, set 0

    assert not set_associative_cache.is_in_cache(0)  # Evicted
    assert set_associative_cache.is_in_cache(256)
    assert set_associative_cache.is_in_cache(512)


def test_set_lru_update_on_access(set_associative_cache: Cache) -> None:
    """Test that accessing a line updates its LRU position."""
    # Fill set 0
    set_associative_cache.add_to_cache(0)  # Line 0, set 0
    set_associative_cache.add_to_cache(256)  # Line 4, set 0

    # Access line 0 to make it more recently used
    set_associative_cache.is_in_cache(0)

    # Add third line, should evict line 4 (now LRU)
    set_associative_cache.add_to_cache(512)  # Line 8, set 0

    assert set_associative_cache.is_in_cache(0)  # Still in cache
    assert not set_associative_cache.is_in_cache(256)  # Evicted
    assert set_associative_cache.is_in_cache(512)


def test_set_different_sets_no_conflict(set_associative_cache: Cache) -> None:
    """Test that different sets don't interfere with each other."""
    # Add to set 0
    set_associative_cache.add_to_cache(0)
    set_associative_cache.add_to_cache(256)

    # Add to set 1
    set_associative_cache.add_to_cache(64)
    set_associative_cache.add_to_cache(320)

    # All should be in cache
    assert set_associative_cache.is_in_cache(0)
    assert set_associative_cache.is_in_cache(256)
    assert set_associative_cache.is_in_cache(64)
    assert set_associative_cache.is_in_cache(320)
