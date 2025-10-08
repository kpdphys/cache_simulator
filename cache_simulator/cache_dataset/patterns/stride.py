# -*- coding: utf-8 -*-
"""Strided memory access pattern."""

import random
from typing import Any, Iterator

from .base import BaseMemoryAccessPattern


class StridePattern(BaseMemoryAccessPattern):
    """Fixed-stride memory access.

    Models array-like access patterns where addresses increment by a
    constant stride value, typical of multi-dimensional array traversal.
    """

    # Default range for stride
    _DEFAULT_STRIDE_MIN: int = 1
    _DEFAULT_STRIDE_MAX: int = 16

    def generate_random_params(self, max_address: int, length: int) -> dict[str, int | float]:
        """Generate random parameters for stride pattern.

        Args:
            max_address: Maximum memory address.
            length: Sequence length.

        Returns:
            Dictionary containing max_address, length, and stride.

        """
        return {
            "max_address": max_address,
            "length": length,
            "stride": random.randint(self._DEFAULT_STRIDE_MIN, self._DEFAULT_STRIDE_MAX),
        }

    def generate_sequence(self, **kwargs: Any) -> Iterator[int]:
        """Generate fixed-stride access sequence.

        Args:
            **kwargs: Parameters containing:
                - max_address (int): Maximum memory address.
                - length (int): Sequence length.
                - stride (int): Step size between consecutive addresses.

        Yields:
            Memory addresses.

        Raises:
            KeyError: If required parameter is missing.
            ValueError: If parameters are invalid.

        """
        max_address: int = kwargs["max_address"]
        length: int = kwargs["length"]
        stride: int = kwargs["stride"]

        # Validate parameters
        self._validate_common_params(max_address, length)
        if stride < 1:
            raise ValueError(f"Stride must be positive, got: {stride}")

        start_address = random.randint(0, max_address)
        for i in range(length):
            address = (start_address + i * stride) % (max_address + 1)
            yield address
