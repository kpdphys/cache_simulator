# -*- coding: utf-8 -*-
"""Sequential memory access pattern with random jumps."""

import random
from typing import Any, Iterator

from .base import BaseMemoryAccessPattern


class SequentialWithJumpsPattern(BaseMemoryAccessPattern):
    """Sequential memory access with probabilistic random jumps.

    Models sequential access where addresses increment by 1, but occasionally
    jump to a random location with probability epsilon.
    """

    # Default range for jump probability
    _DEFAULT_EPSILON_MIN: float = 0.01
    _DEFAULT_EPSILON_MAX: float = 0.1

    def generate_random_params(self, max_address: int, length: int) -> dict[str, int | float]:
        """Generate random parameters for sequential pattern with jumps.

        Args:
            max_address: Maximum memory address.
            length: Sequence length.

        Returns:
            Dictionary containing max_address, length, and epsilon.

        """
        return {
            "max_address": max_address,
            "length": length,
            "epsilon": random.uniform(self._DEFAULT_EPSILON_MIN, self._DEFAULT_EPSILON_MAX),
        }

    def generate_sequence(self, **kwargs: Any) -> Iterator[int]:
        """Generate sequential access sequence with random jumps.

        Args:
            **kwargs: Parameters containing:
                - max_address (int): Maximum memory address.
                - length (int): Sequence length.
                - epsilon (float): Jump probability in range [0, 1].

        Yields:
            Memory addresses.

        Raises:
            KeyError: If required parameter is missing.
            ValueError: If parameters are invalid.

        """
        max_address: int = kwargs["max_address"]
        length: int = kwargs["length"]
        epsilon: float = kwargs["epsilon"]

        # Validate parameters
        self._validate_common_params(max_address, length)
        if not 0 <= epsilon <= 1:
            raise ValueError(f"Jump probability (epsilon) must be in range [0, 1], got: {epsilon}")

        current_address = random.randint(0, max_address)
        for _ in range(length):
            if random.random() < epsilon:
                # Jump to random address
                current_address = random.randint(0, max_address)
            else:
                # Sequential access (increment by 1)
                current_address = (current_address + 1) % (max_address + 1)
            yield current_address
