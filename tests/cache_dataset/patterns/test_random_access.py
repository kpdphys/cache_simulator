# -*- coding: utf-8 -*-
"""Tests for RandomPattern.

Tests fully random memory access pattern.
"""

import pytest

from cache_simulator.cache_dataset.patterns.random_access import RandomPattern


class TestRandomPattern:
    """Test suite for RandomPattern."""

    def test_initialization(self) -> None:
        """Test that pattern can be initialized."""
        pattern = RandomPattern()
        assert isinstance(pattern, RandomPattern)

    def test_generate_random_params_only_basic_params(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that random params include only basic parameters.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = RandomPattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        assert "max_address" in params
        assert "length" in params
        # RandomPattern doesn't add extra params
        assert len(params) == 2

    def test_generate_sequence_correct_length(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated sequence has correct length.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = RandomPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address, length=standard_sequence_length
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
        pattern = RandomPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address, length=standard_sequence_length
            )
        )

        for address in sequence:
            assert 0 <= address <= medium_max_address

    def test_generate_sequence_is_somewhat_random(self, medium_max_address: int) -> None:
        """Test that sequence shows randomness (not all same or sequential).

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = RandomPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=100)
        )

        # Check that not all values are the same
        assert len(set(sequence)) > 1

        # Check that it's not purely sequential
        is_sequential: bool = all(
            sequence[i + 1] == (sequence[i] + 1) % (medium_max_address + 1)
            for i in range(len(sequence) - 1)
        )
        assert not is_sequential

    def test_generate_sequence_validates_common_params(self) -> None:
        """Test that common parameter validation is called."""
        pattern = RandomPattern()

        with pytest.raises(ValueError, match="Maximum address must be positive"):
            list(pattern.generate_sequence(max_address=-1, length=10))

        with pytest.raises(ValueError, match="Sequence length must be positive"):
            list(pattern.generate_sequence(max_address=1000, length=-5))

    def test_generate_sequence_returns_iterator(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generate_sequence returns an iterator.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = RandomPattern()
        sequence = pattern.generate_sequence(
            max_address=medium_max_address,
            length=standard_sequence_length,
        )

        from collections.abc import Iterator

        assert isinstance(sequence, Iterator)

    def test_generate_sequence_different_calls_produce_different_results(
        self, medium_max_address: int
    ) -> None:
        """Test that different calls produce different random sequences.

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = RandomPattern()

        sequence1: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=50)
        )
        sequence2: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=50)
        )

        # Sequences should be different (extremely high probability)
        assert sequence1 != sequence2
