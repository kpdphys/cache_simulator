# -*- coding: utf-8 -*-
"""Base class for memory access pattern generators."""

from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseMemoryAccessPattern(ABC):
    """Abstract base class for memory access pattern generators.

    Defines the interface that all concrete pattern implementations must follow.
    """

    @abstractmethod
    def generate_sequence(self, **kwargs: Any) -> Iterator[int]:
        """Generate a sequence of memory addresses.

        Args:
            **kwargs: Pattern-specific parameters.

        Returns:
            Iterator yielding memory addresses.

        Raises:
            NotImplementedError: If not implemented in subclass.

        """
        raise NotImplementedError

    def generate_random_params(self, max_address: int, length: int) -> dict[str, int | float]:
        """Generate random parameters for this pattern.

        This method should be overridden by subclasses that require
        additional parameters beyond max_address and length.

        Args:
            max_address: Maximum memory address.
            length: Sequence length.

        Returns:
            Dictionary of parameters for generate_sequence().

        """
        return {"max_address": max_address, "length": length}

    @staticmethod
    def _validate_common_params(max_address: int, length: int) -> None:
        """Validate common parameters used by all patterns.

        Args:
            max_address: Maximum memory address.
            length: Sequence length.

        Raises:
            ValueError: If parameters are invalid.

        """
        if max_address <= 0:
            raise ValueError(f"Maximum address must be positive, got: {max_address}")
        if length <= 0:
            raise ValueError(f"Sequence length must be positive, got: {length}")
