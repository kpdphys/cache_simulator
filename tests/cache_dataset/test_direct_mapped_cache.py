# -*- coding: utf-8 -*-
"""Tests for the Cache class operating in direct-mapped mode.

This module focuses on verifying the behavior of a direct-mapped cache
(associativity=1). It includes tests for basic hits and misses,
correct address-to-set mapping, and handling of conflict misses,
which are a key characteristic of this cache type.
"""

from cache_simulator.cache_dataset.cache import Cache


def test_direct_cache_miss_on_empty(direct_mapped_cache: Cache) -> None:
    """Test cache miss on empty cache."""
    assert not direct_mapped_cache.is_in_cache(0)
    assert not direct_mapped_cache.is_in_cache(64)
    assert not direct_mapped_cache.is_in_cache(128)


def test_direct_cache_hit_after_add(direct_mapped_cache: Cache) -> None:
    """Test cache hit after adding address."""
    direct_mapped_cache.add_to_cache(0)
    assert direct_mapped_cache.is_in_cache(0)
    # Addresses in the same cache line should hit
    assert direct_mapped_cache.is_in_cache(32)  # Same line as 0
    assert direct_mapped_cache.is_in_cache(63)  # Same line as 0


def test_direct_address_mapping_to_sets(direct_mapped_cache: Cache) -> None:
    """Test that addresses map to correct sets."""
    # Address 0 -> line 0 -> set 0
    direct_mapped_cache.add_to_cache(0)
    assert 0 in direct_mapped_cache.cache[0]

    # Address 64 -> line 1 -> set 1
    direct_mapped_cache.add_to_cache(64)
    assert 1 in direct_mapped_cache.cache[1]

    # Address 128 -> line 2 -> set 2
    direct_mapped_cache.add_to_cache(128)
    assert 2 in direct_mapped_cache.cache[2]


def test_direct_conflict_miss_in_same_set(direct_mapped_cache: Cache) -> None:
    """Test conflict miss when different lines map to same set."""
    # With 4 sets, addresses 0 and 256 map to same set (set 0)
    # tag(0) = 0, set_idx = 0 % 4 = 0
    direct_mapped_cache.add_to_cache(0)
    assert direct_mapped_cache.is_in_cache(0)

    # Adding address 256 should evict address 0 (same set, direct mapped)
    # tag(256) = 4, set_idx = 4 % 4 = 0
    direct_mapped_cache.add_to_cache(256)
    assert not direct_mapped_cache.is_in_cache(0)
    assert direct_mapped_cache.is_in_cache(256)
