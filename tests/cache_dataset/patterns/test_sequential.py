# -*- coding: utf-8 -*-
"""Tests for SequentialWithJumpsPattern.

Tests sequential memory access pattern with random jumps.
"""

import pytest

from cache_simulator.cache_dataset.patterns.sequential import SequentialWithJumpsPattern


class TestSequentialWithJumpsPattern:
    """Test suite for SequentialWithJumpsPattern."""

    def test_initialization(self) -> None:
        """Test that pattern can be initialized."""
        pattern = SequentialWithJumpsPattern()
        assert isinstance(pattern, SequentialWithJumpsPattern)

    def test_generate_random_params_includes_epsilon(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that random params include epsilon parameter.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = SequentialWithJumpsPattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        assert "epsilon" in params
        assert isinstance(params["epsilon"], float)
        epsilon: float = float(params["epsilon"])
        assert 0 <= epsilon <= 1

    def test_generate_random_params_epsilon_in_default_range(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated epsilon is in default range.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = SequentialWithJumpsPattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        epsilon: float = float(params["epsilon"])
        assert (
            pattern._DEFAULT_EPSILON_MIN <= epsilon <= pattern._DEFAULT_EPSILON_MAX  # type: ignore
        )

    def test_generate_sequence_returns_iterator(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generate_sequence returns an iterator.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = SequentialWithJumpsPattern()
        sequence = pattern.generate_sequence(
            max_address=medium_max_address,
            length=standard_sequence_length,
            epsilon=0.0,
        )

        from collections.abc import Iterator

        assert isinstance(sequence, Iterator)

    def test_generate_sequence_correct_length(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated sequence has correct length.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = SequentialWithJumpsPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                epsilon=0.0,
            )
        )

        assert len(sequence) == standard_sequence_length

    def test_generate_sequence_no_jumps_is_sequential(self, small_max_address: int) -> None:
        """Test that sequence with epsilon=0 is purely sequential.

        Args:
            small_max_address: Small max address fixture.

        """
        pattern = SequentialWithJumpsPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=small_max_address, length=10, epsilon=0.0)
        )

        # Check that differences are all 1 (allowing for wraparound)
        for i in range(len(sequence) - 1):
            diff: int = (sequence[i + 1] - sequence[i]) % (small_max_address + 1)
            assert diff == 1

    def test_generate_sequence_all_addresses_within_bounds(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that all generated addresses are within valid range.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = SequentialWithJumpsPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                epsilon=0.5,
            )
        )

        for address in sequence:
            assert 0 <= address <= medium_max_address

    def test_generate_sequence_rejects_invalid_epsilon(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that invalid epsilon values are rejected.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = SequentialWithJumpsPattern()

        with pytest.raises(ValueError, match="epsilon.*must be in range"):
            list(
                pattern.generate_sequence(
                    max_address=medium_max_address,
                    length=standard_sequence_length,
                    epsilon=1.5,
                )
            )

        with pytest.raises(ValueError, match="epsilon.*must be in range"):
            list(
                pattern.generate_sequence(
                    max_address=medium_max_address,
                    length=standard_sequence_length,
                    epsilon=-0.1,
                )
            )

    def test_generate_sequence_validates_common_params(self) -> None:
        """Test that common parameter validation is called."""
        pattern = SequentialWithJumpsPattern()

        with pytest.raises(ValueError, match="Maximum address must be positive"):
            list(pattern.generate_sequence(max_address=0, length=10, epsilon=0.1))

        with pytest.raises(ValueError, match="Sequence length must be positive"):
            list(pattern.generate_sequence(max_address=1000, length=0, epsilon=0.1))
