# -*- coding: utf-8 -*-
"""Module for simulating processor cache with support for various mapping strategies.

This module provides a `Cache` class that implements processor cache simulation
with LRU (Least Recently Used) replacement strategy support. The following
mapping types are supported:
- Direct mapping,
- Set-associative mapping,
- Fully associative mapping.

The module uses `collections.OrderedDict` for LRU implementation and provides methods
for checking address presence in cache, adding addresses to cache, and resetting cache.
"""

from collections import OrderedDict


class Cache:
    """Class for simulating processor cache with LRU replacement strategy support.

    Supports direct, set-associative, and fully associative mapping.
    """

    def __init__(self, line_size: int = 64, num_lines: int = 1024, associativity: int = 0):
        """Initialize cache.

        :param line_size: Cache line size in bytes (default 64).
        :param num_lines: Total number of cache lines (default 1024).
        :param associativity: Degree of associativity. If 0 — fully associative cache.
                              If 1 — direct mapping. If k > 1 — k-way set associative cache.
        :raises ValueError: If the number of cache lines is not divisible by the degree
                            of associativity for set-associative mapping.
        """
        self.line_size = line_size
        self.num_lines = num_lines
        self.associativity = associativity

        # Calculate the number of sets in cache
        if associativity == 0:  # Fully associative
            self.num_sets = 1
            self.lines_per_set = num_lines
        elif associativity == 1:  # Direct mapping
            self.num_sets = num_lines
            self.lines_per_set = 1
        else:  # Set-associative
            if num_lines % associativity != 0:
                raise ValueError(
                    f"Number of cache lines ({num_lines}) must be divisible by "
                    f"the degree of associativity ({associativity})."
                )
            self.num_sets = num_lines // associativity
            self.lines_per_set = associativity

        # Cache storage structure: dictionary where key is set index (set_idx),
        # value is OrderedDict for storing line tags in LRU order
        # OrderedDict keys are tags (int), values are None since we don't store actual data
        self.cache: dict[int, OrderedDict[int, None]] = {
            set_idx: OrderedDict() for set_idx in range(self.num_sets)
        }

        # Set of all tags in cache for fast presence check (search optimization)
        self.tag_set: set[int] = set()

    def get_set_index(self, address: int) -> int:
        """Calculate set index for the given address.

        :param address: Memory address.
        :return: Set index.
        """
        # Calculate cache line address — address aligned to line size
        line_address = address // self.line_size
        if self.associativity == 0:  # Fully associative — everything in one set
            return 0
        # For direct or set-associative mapping
        return line_address % self.num_sets

    def get_tag(self, address: int) -> int:
        """Calculate tag for the given address.

        :param address: Memory address.
        :return: Tag.
        """
        # Tag is the cache line address
        return address // self.line_size

    def is_in_cache(self, address: int) -> bool:
        """Check if address is in cache. If found, update LRU order.

        :param address: Memory address.
        :return: True if address is in cache, otherwise False.
        """
        set_idx = self.get_set_index(address)
        tag = self.get_tag(address)

        if tag in self.cache[set_idx]:
            # Update LRU order: move tag to the end of OrderedDict
            self.cache[set_idx].move_to_end(tag)
            return True
        return False

    def add_to_cache(self, address: int) -> None:
        """Add address to cache. If cache is full, remove the least recently used line (LRU).

        :param address: Memory address.
        """
        set_idx = self.get_set_index(address)
        tag = self.get_tag(address)

        # If tag is already in cache, just update LRU order
        if tag in self.cache[set_idx]:
            self.cache[set_idx].move_to_end(tag)
            return

        # If set is full, remove the oldest line (first in OrderedDict)
        if len(self.cache[set_idx]) >= self.lines_per_set:
            lru_tag, _ = self.cache[set_idx].popitem(last=False)  # Remove first element
            self.tag_set.remove(lru_tag)

        # Add new tag to the end of OrderedDict (as most recently used)
        self.cache[set_idx][tag] = None  # Value is None since we don't store actual data
        self.tag_set.add(tag)

    def reset_cache(self) -> None:
        """Reset cache to initial state."""
        self.cache = {set_idx: OrderedDict() for set_idx in range(self.num_sets)}
        self.tag_set = set()

    def __str__(self) -> str:
        """Return a string representation of the cache for debugging.

        In each set, elements in OrderedDict are ordered by last access time:
        - at the beginning — least recently used (LRU, oldest),
        - at the end — most recently used (MRU, newest).

        :return: String representation of cache contents.
        """
        result: list[str] = []
        for set_idx in range(self.num_sets):
            if self.cache[set_idx]:
                result.append(f"Set {set_idx}: {list(self.cache[set_idx].keys())}")
        return "\n".join(result) if result else "Cache is empty"
