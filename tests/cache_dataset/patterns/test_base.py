# -*- coding: utf-8 -*-
"""Tests for BaseMemoryAccessPattern.

Tests the abstract base class and common functionality shared by all patterns.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from cache_simulator.cache_dataset.patterns.base import BaseMemoryAccessPattern


class ConcretePattern(BaseMemoryAccessPattern):
    """Concrete implementation of BaseMemoryAccessPattern for testing."""

    def generate_sequence(self, **kwargs: Any) -> Iterator[int]:
        """Generate a simple test sequence.

        Args:
            **kwargs: Pattern parameters.

        Yields:
            Sequential integers from 0 to length-1.

        """
        length: int = int(kwargs.get("length", 10))
        for i in range(length):
            yield i


class TestBaseMemoryAccessPattern:
    """Test suite for BaseMemoryAccessPattern."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMemoryAccessPattern()  # type: ignore[abstract]

    def test_concrete_implementation_can_be_instantiated(self) -> None:
        """Test that concrete implementation can be instantiated."""
        pattern = ConcretePattern()
        assert isinstance(pattern, BaseMemoryAccessPattern)

    def test_generate_random_params_returns_basic_params(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that default generate_random_params returns basic parameters.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = ConcretePattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        assert "max_address" in params
        assert "length" in params
        assert params["max_address"] == medium_max_address
        assert params["length"] == standard_sequence_length

    def test_validate_common_params_accepts_valid_params(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that validation accepts valid parameters.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        # Should not raise
        BaseMemoryAccessPattern._validate_common_params(  # type: ignore
            medium_max_address, standard_sequence_length
        )

    def test_validate_common_params_rejects_zero_max_address(self) -> None:
        """Test that validation rejects zero max_address."""
        with pytest.raises(ValueError, match="Maximum address must be positive"):
            BaseMemoryAccessPattern._validate_common_params(0, 10)  # type: ignore

    def test_validate_common_params_rejects_negative_max_address(self) -> None:
        """Test that validation rejects negative max_address."""
        with pytest.raises(ValueError, match="Maximum address must be positive"):
            BaseMemoryAccessPattern._validate_common_params(-100, 10)  # type: ignore

    def test_validate_common_params_rejects_zero_length(self) -> None:
        """Test that validation rejects zero length."""
        with pytest.raises(ValueError, match="Sequence length must be positive"):
            BaseMemoryAccessPattern._validate_common_params(1000, 0)  # type: ignore

    def test_validate_common_params_rejects_negative_length(self) -> None:
        """Test that validation rejects negative length."""
        with pytest.raises(ValueError, match="Sequence length must be positive"):
            BaseMemoryAccessPattern._validate_common_params(1000, -5)  # type: ignore

    def test_concrete_pattern_generates_correct_length_sequence(self) -> None:
        """Test that concrete pattern generates sequence of correct length."""
        pattern = ConcretePattern()
        sequence: list[int] = list(pattern.generate_sequence(length=5))

        assert len(sequence) == 5
        assert sequence == [0, 1, 2, 3, 4]
