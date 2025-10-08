# -*- coding: utf-8 -*-
"""Mock objects for testing cache simulator.

This module provides mock implementations of cache and other components
for isolated unit testing without dependencies on actual implementations.
"""

from typing import Iterator


class MockCache:
    """Mock cache implementation for testing.

    Simulates basic cache behavior without complex logic.
    Useful for testing CacheDataset without depending on actual Cache implementation.
    """

    def __init__(self, num_lines: int, associativity: int, line_size: int = 64) -> None:
        """Initialize mock cache.

        Args:
            num_lines: Number of cache lines.
            associativity: Cache associativity.
            line_size: Size of each cache line in bytes (default: 64).

        """
        self.num_lines: int = num_lines
        self.associativity: int = associativity
        self.line_size: int = line_size
        self.cache: set[int] = set()
        self.access_count: int = 0
        self.hit_count: int = 0
        self.miss_count: int = 0

    def is_in_cache(self, address: int) -> bool:
        """Check if address is in cache.

        Args:
            address: Memory address to check.

        Returns:
            True if address is in cache, False otherwise.

        """
        self.access_count += 1
        is_hit = address in self.cache

        if is_hit:
            self.hit_count += 1
        else:
            self.miss_count += 1

        return is_hit

    def add_to_cache(self, address: int) -> None:
        """Add address to cache.

        Implements simple eviction: keeps only last num_lines addresses.

        Args:
            address: Memory address to add.

        """
        self.cache.add(address)

        # Simple eviction: keep only last num_lines addresses
        if len(self.cache) > self.num_lines:
            # Remove first element (oldest)
            self.cache.pop()

    def reset(self) -> None:
        """Reset cache to initial state."""
        self.cache.clear()
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0 and 1, or 0 if no accesses.

        """
        if self.access_count == 0:
            return 0.0
        return self.hit_count / self.access_count


class MockPattern:
    """Mock memory access pattern for testing.

    Generates predictable sequences for testing purposes.
    """

    def __init__(self, sequence: list[int]) -> None:
        """Initialize mock pattern with predefined sequence.

        Args:
            sequence: List of addresses to generate.

        """
        self.sequence: list[int] = sequence

    def generate_sequence(self, **kwargs: object) -> Iterator[int]:
        """Generate the predefined sequence.

        Args:
            **kwargs: Ignored pattern parameters.

        Yields:
            Addresses from predefined sequence.

        """
        for address in self.sequence:
            yield address

    def generate_random_params(self, max_address: int, length: int) -> dict[str, int | float]:
        """Generate mock parameters.

        Args:
            max_address: Maximum memory address.
            length: Sequence length.

        Returns:
            Dictionary with basic parameters.

        """
        return {"max_address": max_address, "length": length}
