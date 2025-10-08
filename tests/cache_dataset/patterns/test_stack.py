# -*- coding: utf-8 -*-
"""Tests for StackPattern.

Tests stack-like memory access pattern.
"""

import pytest

from cache_simulator.cache_dataset.patterns.stack import StackPattern


class TestStackPattern:
    """Test suite for StackPattern."""

    def test_initialization(self) -> None:
        """Test that pattern can be initialized."""
        pattern = StackPattern()
        assert isinstance(pattern, StackPattern)

    def test_generate_random_params_includes_stack_size(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that random params include stack_size parameter.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StackPattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        assert "stack_size" in params
        assert isinstance(params["stack_size"], int)
        stack_size: int = int(params["stack_size"])
        assert stack_size >= 0

    def test_generate_random_params_stack_size_in_range(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated stack_size is in valid range.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StackPattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        stack_size: int = int(params["stack_size"])
        assert (
            pattern._DEFAULT_STACK_SIZE_MIN  # type: ignore
            <= stack_size
            <= pattern._DEFAULT_STACK_SIZE_MAX  # type: ignore
        )

    def test_generate_sequence_correct_length(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated sequence has correct length.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StackPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                stack_size=10,
            )
        )

        assert len(sequence) == standard_sequence_length

    def test_generate_sequence_all_addresses_within_bounds(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that all generated addresses are within valid range.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StackPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                stack_size=20,
            )
        )

        for address in sequence:
            assert 0 <= address <= medium_max_address

    def test_generate_sequence_stays_within_reasonable_range(
        self, medium_max_address: int
    ) -> None:
        """Test that stack addresses stay within a reasonable range.

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = StackPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=100, stack_size=10)
        )

        # Stack addresses should not span the entire address space
        address_range: int = max(sequence) - min(sequence)
        # Reasonable bound: should be much less than max_address
        assert address_range < medium_max_address / 2

    def test_generate_sequence_rejects_negative_stack_size(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that negative stack_size values are rejected.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StackPattern()

        with pytest.raises(ValueError, match="stack size must be non-negative"):
            list(
                pattern.generate_sequence(
                    max_address=medium_max_address,
                    length=standard_sequence_length,
                    stack_size=-5,
                )
            )

    def test_generate_sequence_accepts_zero_stack_size(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that zero stack_size is accepted.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StackPattern()
        # Should not raise
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                stack_size=0,
            )
        )

        assert len(sequence) == standard_sequence_length
