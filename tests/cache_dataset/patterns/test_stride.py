# -*- coding: utf-8 -*-
"""Tests for StridePattern.

Tests strided memory access pattern.
"""

import pytest

from cache_simulator.cache_dataset.patterns.stride import StridePattern


class TestStridePattern:
    """Test suite for StridePattern."""

    def test_initialization(self) -> None:
        """Test that pattern can be initialized."""
        pattern = StridePattern()
        assert isinstance(pattern, StridePattern)

    def test_generate_random_params_includes_stride(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that random params include stride parameter.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StridePattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        assert "stride" in params
        assert isinstance(params["stride"], int)
        stride: int = int(params["stride"])
        assert stride > 0

    def test_generate_random_params_stride_in_range(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated stride is in valid range.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StridePattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        stride: int = int(params["stride"])
        assert (
            pattern._DEFAULT_STRIDE_MIN <= stride <= pattern._DEFAULT_STRIDE_MAX  # type: ignore
        )

    def test_generate_sequence_correct_length(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated sequence has correct length.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StridePattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                stride=4,
            )
        )

        assert len(sequence) == standard_sequence_length

    def test_generate_sequence_uses_correct_stride(self, medium_max_address: int) -> None:
        """Test that sequence uses correct stride value.

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = StridePattern()
        stride = 8
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=10, stride=stride)
        )

        # Check that differences are all equal to stride (allowing for wraparound)
        for i in range(len(sequence) - 1):
            diff: int = (sequence[i + 1] - sequence[i]) % (medium_max_address + 1)
            assert diff == stride

    def test_generate_sequence_all_addresses_within_bounds(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that all generated addresses are within valid range.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StridePattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address,
                length=standard_sequence_length,
                stride=7,
            )
        )

        for address in sequence:
            assert 0 <= address <= medium_max_address

    def test_generate_sequence_rejects_invalid_stride(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that invalid stride values are rejected.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = StridePattern()

        with pytest.raises(ValueError, match="Stride must be positive"):
            list(
                pattern.generate_sequence(
                    max_address=medium_max_address,
                    length=standard_sequence_length,
                    stride=0,
                )
            )

        with pytest.raises(ValueError, match="Stride must be positive"):
            list(
                pattern.generate_sequence(
                    max_address=medium_max_address,
                    length=standard_sequence_length,
                    stride=-5,
                )
            )

    def test_generate_sequence_stride_one_is_sequential(self, small_max_address: int) -> None:
        """Test that stride=1 produces sequential access.

        Args:
            small_max_address: Small max address fixture.

        """
        pattern = StridePattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=small_max_address, length=10, stride=1)
        )

        # Should be sequential
        for i in range(len(sequence) - 1):
            diff: int = (sequence[i + 1] - sequence[i]) % (small_max_address + 1)
            assert diff == 1
