# -*- coding: utf-8 -*-
"""Fully random memory access pattern."""

import random
from typing import Any, Iterator

from .base import BaseMemoryAccessPattern


class RandomPattern(BaseMemoryAccessPattern):
    """Fully random memory access.

    Models chaotic access patterns typical of algorithms like hash table
    operations or random data structure traversal.
    """

    # This pattern doesn't need additional parameters beyond base ones
    # No need to override generate_random_params

    def generate_sequence(self, **kwargs: Any) -> Iterator[int]:
        """Generate completely random access sequence.

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

        for _ in range(length):
            yield random.randint(0, max_address)
