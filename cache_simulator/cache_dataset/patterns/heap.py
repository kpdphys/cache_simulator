# -*- coding: utf-8 -*-
"""Heap-like memory access pattern."""

import random
from typing import Any, Iterator

from .base import BaseMemoryAccessPattern


class HeapPattern(BaseMemoryAccessPattern):
    """Heap-like memory access pattern.

    Models heap memory allocation where addresses generally increase but
    occasionally jump backwards, simulating allocation and deallocation.
    """

    # Heap configuration constants
    _DEFAULT_MAX_HEAP_SIZE: int = 10000  # Maximum heap size
    _ALLOC_MIN: int = 1  # Minimum allocation size
    _ALLOC_MAX: int = 100  # Maximum allocation size
    _ALLOCATION_PROBABILITY: float = 0.8  # Probability of allocation vs reuse

    def generate_sequence(self, **kwargs: Any) -> Iterator[int]:
        """Generate heap-like access sequence.

        Args:
            **kwargs: Parameters containing:
                - max_address (int): Maximum memory address.
                - length (int): Sequence length.

        Yields:
            Memory addresses.

        Raises:
            KeyError: If required parameter is missing.
            ValueError: If parameters are invalid.

        """
        max_address: int = kwargs["max_address"]
        length: int = kwargs["length"]

        # Validate parameters
        self._validate_common_params(max_address, length)

        # Reserve space for heap growth
        max_heap_size = min(self._DEFAULT_MAX_HEAP_SIZE, max_address // 2)
        base_address = random.randint(0, max(0, max_address - max_heap_size))
        current_offset = 0

        for _ in range(length):
            # Allocate new memory (increase offset)
            if random.random() < self._ALLOCATION_PROBABILITY:
                current_offset += random.randint(self._ALLOC_MIN, self._ALLOC_MAX)
            # Reuse memory (jump back)
            else:
                current_offset = random.randint(0, max(1, current_offset))

            address = base_address + current_offset
            # Ensure we don't exceed max_address
            address = min(address, max_address)
            yield address
