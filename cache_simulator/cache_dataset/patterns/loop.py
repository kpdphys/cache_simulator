# -*- coding: utf-8 -*-
"""Loop-based cyclic memory access pattern."""

import random
from typing import Any, Iterator

from .base import BaseMemoryAccessPattern


class LoopPattern(BaseMemoryAccessPattern):
    """Cyclic access to a small address range.

    Models loop-like behavior in code where the same small range of
    addresses is accessed repeatedly.
    """

    # Default range for loop size
    _DEFAULT_LOOP_SIZE_MIN: int = 10
    _DEFAULT_LOOP_SIZE_MAX: int = 1000

    def generate_random_params(self, max_address: int, length: int) -> dict[str, int | float]:
        """Generate random parameters for loop pattern.

        Args:
            max_address: Maximum memory address.
            length: Sequence length.

        Returns:
            Dictionary containing max_address, length, and loop_size.

        """
        return {
            "max_address": max_address,
            "length": length,
            "loop_size": random.randint(
                self._DEFAULT_LOOP_SIZE_MIN,
                min(self._DEFAULT_LOOP_SIZE_MAX, max_address),
            ),
        }

    def generate_sequence(self, **kwargs: Any) -> Iterator[int]:
        """Generate cyclic access to a fixed address range.

        Args:
            **kwargs: Parameters containing:
                - max_address (int): Maximum memory address.
                - length (int): Sequence length.
                - loop_size (int): Size of the repeating address range.

        Yields:
            Memory addresses.

        Raises:
            KeyError: If required parameter is missing.
            ValueError: If parameters are invalid.

        """
        max_address: int = kwargs["max_address"]
        length: int = kwargs["length"]
        loop_size: int = kwargs["loop_size"]

        # Validate parameters
        self._validate_common_params(max_address, length)
        if loop_size < 1 or loop_size > max_address:
            raise ValueError(f"Loop size must be in range [1, {max_address}], got: {loop_size}")

        # Choose random starting point ensuring we don't exceed max_address
        start_address = random.randint(0, max(0, max_address - loop_size))

        for i in range(length):
            address = start_address + (i % loop_size)
            yield address
