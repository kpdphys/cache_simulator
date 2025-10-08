# -*- coding: utf-8 -*-
"""Stack-like memory access pattern."""

import random
from typing import Any, Iterator

from .base import BaseMemoryAccessPattern


class StackPattern(BaseMemoryAccessPattern):
    """Stack-like memory access pattern.

    Models stack operations (push/pop) where addresses change within a
    limited range, simulating a stack pointer moving up and down.
    """

    # Stack configuration constants
    _DEFAULT_STACK_SIZE_MIN: int = 10
    _DEFAULT_STACK_SIZE_MAX: int = 100
    _STACK_HEADROOM: int = 1000  # Safety margin from address boundaries
    _MAX_STACK_DEPTH: int = 100  # Maximum allowed stack depth
    _STACK_MOVE_MIN: int = 1  # Minimum pointer movement
    _STACK_MOVE_MAX: int = 10  # Maximum pointer movement
    _PUSH_PROBABILITY: float = 0.5  # Probability of push vs pop

    def generate_random_params(self, max_address: int, length: int) -> dict[str, int | float]:
        """Generate random parameters for stack pattern.

        Args:
            max_address: Maximum memory address.
            length: Sequence length.

        Returns:
            Dictionary containing max_address, length, and stack_size.

        """
        return {
            "max_address": max_address,
            "length": length,
            "stack_size": random.randint(
                self._DEFAULT_STACK_SIZE_MIN, self._DEFAULT_STACK_SIZE_MAX
            ),
        }

    def generate_sequence(self, **kwargs: Any) -> Iterator[int]:
        """Generate stack-like access sequence.

        Args:
            **kwargs: Parameters containing:
                - max_address (int): Maximum memory address.
                - length (int): Sequence length.
                - stack_size (int): Initial stack depth (affects push/pop probability).

        Yields:
            Memory addresses.

        Raises:
            KeyError: If required parameter is missing.
            ValueError: If parameters are invalid.

        """
        max_address: int = kwargs["max_address"]
        length: int = kwargs["length"]
        stack_size: int = kwargs["stack_size"]

        # Validate parameters
        self._validate_common_params(max_address, length)
        if stack_size < 0:
            raise ValueError(f"Initial stack size must be non-negative, got: {stack_size}")

        # Start with a safe range to allow stack growth in both directions
        safe_min = self._STACK_HEADROOM
        safe_max = max(self._STACK_HEADROOM, max_address - self._STACK_HEADROOM)
        stack_pointer = random.randint(safe_min, safe_max)
        current_depth = stack_size

        for _ in range(length):
            # Push operation: move pointer down, increase depth
            if random.random() < self._PUSH_PROBABILITY and current_depth < self._MAX_STACK_DEPTH:
                stack_pointer -= random.randint(self._STACK_MOVE_MIN, self._STACK_MOVE_MAX)
                current_depth += 1
            # Pop operation: move pointer up, decrease depth
            elif current_depth > 0:
                stack_pointer += random.randint(self._STACK_MOVE_MIN, self._STACK_MOVE_MAX)
                current_depth -= 1

            # Keep pointer within valid range
            stack_pointer = max(0, min(stack_pointer, max_address))
            yield stack_pointer
