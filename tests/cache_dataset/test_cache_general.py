# -*- coding: utf-8 -*-
"""Unit tests for the cache_simulator.cache_dataset.cache.Cache class.

This module tests the core functionality of the Cache class, including:
- Cache state management (reset, eviction).
- Correct handling of addresses based on line size.
- Internal logic for tag and set index calculation.
- Edge cases like single-line caches and large addresses.
- String representation.
"""

from cache_simulator.cache_dataset.cache import Cache


def test_reset_clears_cache() -> None:
    """Test that reset clears all cache contents."""
    cache = Cache(line_size=64, num_lines=4, associativity=2)

    # Add some data
    cache.add_to_cache(0)
    cache.add_to_cache(64)
    cache.add_to_cache(128)

    assert cache.is_in_cache(0)
    assert len(cache.tag_set) == 3

    # Reset cache
    cache.reset_cache()

    # Cache should be empty
    assert not cache.is_in_cache(0)
    assert not cache.is_in_cache(64)
    assert not cache.is_in_cache(128)
    assert len(cache.tag_set) == 0


def test_reset_preserves_configuration() -> None:
    """Test that reset preserves cache configuration."""
    cache = Cache(line_size=32, num_lines=8, associativity=4)

    original_line_size = cache.line_size
    original_num_lines = cache.num_lines
    original_associativity = cache.associativity
    original_num_sets = cache.num_sets

    cache.add_to_cache(0)
    cache.reset_cache()

    # Configuration should remain the same
    assert cache.line_size == original_line_size
    assert cache.num_lines == original_num_lines
    assert cache.associativity == original_associativity
    assert cache.num_sets == original_num_sets


def test_same_line_addresses() -> None:
    """Test that addresses in the same cache line are treated equally."""
    cache = Cache(line_size=64, num_lines=4, associativity=1)

    # Addresses 0-63 are in the same cache line
    cache.add_to_cache(0)

    assert cache.is_in_cache(0)
    assert cache.is_in_cache(1)
    assert cache.is_in_cache(32)
    assert cache.is_in_cache(63)
    assert not cache.is_in_cache(64)


def test_different_line_sizes() -> None:
    """Test cache behavior with different line sizes."""
    cache_32 = Cache(line_size=32, num_lines=4, associativity=1)
    cache_128 = Cache(line_size=128, num_lines=4, associativity=1)

    # For line_size=32: addresses 0-31 in line 0, 32-63 in line 1
    cache_32.add_to_cache(0)
    assert cache_32.is_in_cache(31)
    assert not cache_32.is_in_cache(32)

    # For line_size=128: addresses 0-127 in line 0
    cache_128.add_to_cache(0)
    assert cache_128.is_in_cache(127)
    assert not cache_128.is_in_cache(128)


def test_tag_set_consistency() -> None:
    """Test that tag_set stays consistent with cache contents."""
    cache = Cache(line_size=64, num_lines=4, associativity=0)

    # Add lines
    cache.add_to_cache(0)
    cache.add_to_cache(64)
    cache.add_to_cache(128)

    assert len(cache.tag_set) == 3
    assert 0 in cache.tag_set
    assert 1 in cache.tag_set
    assert 2 in cache.tag_set

    # Fill cache and trigger eviction
    cache.add_to_cache(192)
    cache.add_to_cache(256)  # Should evict tag 0

    assert len(cache.tag_set) == 4
    assert 0 not in cache.tag_set


# --- Edge Cases ---


def test_single_line_cache() -> None:
    """Test cache with only one line."""
    cache = Cache(line_size=64, num_lines=1, associativity=1)

    cache.add_to_cache(0)
    assert cache.is_in_cache(0)

    # Adding different line should evict the first
    cache.add_to_cache(64)
    assert not cache.is_in_cache(0)
    assert cache.is_in_cache(64)


def test_large_address_values() -> None:
    """Test cache with large address values."""
    cache = Cache(line_size=64, num_lines=4, associativity=1)

    large_addr = 1_000_000
    cache.add_to_cache(large_addr)
    assert cache.is_in_cache(large_addr)


def test_repeated_adds_same_address() -> None:
    """Test adding the same address multiple times."""
    cache = Cache(line_size=64, num_lines=4, associativity=2)

    cache.add_to_cache(0)
    cache.add_to_cache(0)
    cache.add_to_cache(0)

    # Should still be in cache, only once
    assert cache.is_in_cache(0)
    # Check that set contains only one entry
    set_idx = cache.get_set_index(0)
    assert len(cache.cache[set_idx]) == 1


# --- Internal Logic and Representation ---


def test_tag_calculation() -> None:
    """Test tag calculation for various addresses."""
    cache = Cache(line_size=64, num_lines=4, associativity=1)
    # Tag is address // line_size
    assert cache.get_tag(0) == 0
    assert cache.get_tag(63) == 0
    assert cache.get_tag(64) == 1
    assert cache.get_tag(128) == 2
    assert cache.get_tag(255) == 3


def test_set_index_calculation() -> None:
    """Test set index calculation for various addresses."""
    cache = Cache(line_size=64, num_lines=8, associativity=2)
    # 4 sets (8 lines / 2-way)
    assert cache.get_set_index(0) == 0  # Line 0 % 4 = 0
    assert cache.get_set_index(64) == 1  # Line 1 % 4 = 1
    assert cache.get_set_index(128) == 2  # Line 2 % 4 = 2
    assert cache.get_set_index(192) == 3  # Line 3 % 4 = 3
    assert cache.get_set_index(256) == 0  # Line 4 % 4 = 0


def test_empty_cache_string() -> None:
    """Test string representation of empty cache."""
    cache = Cache(line_size=64, num_lines=4, associativity=1)
    assert str(cache) == "Cache is empty"


def test_cache_with_data_string() -> None:
    """Test string representation of cache with data."""
    cache = Cache(line_size=64, num_lines=4, associativity=2)
    # 2 sets
    cache.add_to_cache(0)  # Line 0, set 0
    cache.add_to_cache(128)  # Line 2, set 0
    cache.add_to_cache(64)  # Line 1, set 1

    result = str(cache)
    assert "Set 0:" in result
    assert "Set 1:" in result
    assert "[0, 2]" in result  # Order might depend on implementation, but content is key
    assert "[1]" in result
