# -*- coding: utf-8 -*-
"""Tests for LoopPattern.

Tests cyclic memory access pattern.
"""

import pytest

from cache_simulator.cache_dataset.patterns.loop import LoopPattern


class TestLoopPattern:
    """Test suite for LoopPattern."""

    def test_initialization(self) -> None:
        """Test that pattern can be initialized."""
        pattern = LoopPattern()
        assert isinstance(pattern, LoopPattern)

    def test_generate_random_params_includes_loop_size(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that random params include loop_size parameter.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = LoopPattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        assert "loop_size" in params
        assert isinstance(params["loop_size"], int)
        loop_size: int = int(params["loop_size"])
        assert loop_size > 0

    def test_generate_random_params_loop_size_in_range(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated loop_size is in valid range.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = LoopPattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        loop_size: int = int(params["loop_size"])
        assert pattern._DEFAULT_LOOP_SIZE_MIN <= loop_size  # type: ignore
        assert loop_size <= min(
            pattern._DEFAULT_LOOP_SIZE_MAX,  # type: ignore
            medium_max_address,
        )

    def test_generate_sequence_correct_length(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated sequence has correct length.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = LoopPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                loop_size=10,
            )
        )

        assert len(sequence) == standard_sequence_length

    def test_generate_sequence_repeats_pattern(self, medium_max_address: int) -> None:
        """Test that sequence repeats in a loop.

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = LoopPattern()
        loop_size = 5
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address, length=20, loop_size=loop_size
            )
        )

        # Check that pattern repeats every loop_size elements
        for i in range(loop_size, len(sequence)):
            assert sequence[i] == sequence[i % loop_size]

    def test_generate_sequence_all_addresses_within_bounds(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that all generated addresses are within valid range.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = LoopPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                loop_size=10,
            )
        )

        for address in sequence:
            assert 0 <= address <= medium_max_address

    def test_generate_sequence_rejects_invalid_loop_size(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that invalid loop_size values are rejected.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = LoopPattern()

        with pytest.raises(ValueError, match="Loop size must be in range"):
            list(
                pattern.generate_sequence(
                    max_address=medium_max_address,
                    length=standard_sequence_length,
                    loop_size=0,
                )
            )

        with pytest.raises(ValueError, match="Loop size must be in range"):
            list(
                pattern.generate_sequence(
                    max_address=medium_max_address,
                    length=standard_sequence_length,
                    loop_size=medium_max_address + 1,
                )
            )

    def test_generate_sequence_loop_size_equals_one(self, medium_max_address: int) -> None:
        """Test special case where loop_size is 1.

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = LoopPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=10, loop_size=1)
        )

        # All addresses should be the same
        assert len(set(sequence)) == 1
