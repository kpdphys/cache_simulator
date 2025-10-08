# -*- coding: utf-8 -*-
"""Tests for HeapPattern.

Tests heap-like memory access pattern.
"""

import pytest

from cache_simulator.cache_dataset.patterns.heap import HeapPattern


class TestHeapPattern:
    """Test suite for HeapPattern."""

    def test_initialization(self) -> None:
        """Test that pattern can be initialized."""
        pattern = HeapPattern()
        assert isinstance(pattern, HeapPattern)

    def test_generate_random_params_only_basic_params(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that random params include only basic parameters.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = HeapPattern()
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=medium_max_address, length=standard_sequence_length
        )

        assert "max_address" in params
        assert "length" in params
        # HeapPattern doesn't add extra params
        assert len(params) == 2

    def test_generate_sequence_correct_length(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generated sequence has correct length.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = HeapPattern()
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
        pattern = HeapPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(
                max_address=medium_max_address, length=standard_sequence_length
            )
        )

        for address in sequence:
            assert 0 <= address <= medium_max_address

    def test_generate_sequence_generally_increases(self, medium_max_address: int) -> None:
        """Test that heap addresses generally increase (allocation pattern).

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = HeapPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=100)
        )

        # Most of the time, addresses should increase
        increases: int = sum(1 for i in range(len(sequence) - 1) if sequence[i + 1] > sequence[i])
        total_transitions: int = len(sequence) - 1

        # With 80% allocation probability, expect roughly 70%+ increases
        # (allowing some margin for randomness)
        assert increases / total_transitions > 0.6

    def test_generate_sequence_has_some_backwards_jumps(self, medium_max_address: int) -> None:
        """Test that sequence has some backwards jumps (deallocation).

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = HeapPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=100)
        )

        # Should have at least some backwards jumps
        backwards_jumps: int = sum(
            1 for i in range(len(sequence) - 1) if sequence[i + 1] < sequence[i]
        )

        # With 20% deallocation probability, expect some backwards jumps
        assert backwards_jumps > 0

    def test_generate_sequence_validates_common_params(self) -> None:
        """Test that common parameter validation is called."""
        pattern = HeapPattern()

        with pytest.raises(ValueError, match="Maximum address must be positive"):
            list(pattern.generate_sequence(max_address=0, length=10))

        with pytest.raises(ValueError, match="Sequence length must be positive"):
            list(pattern.generate_sequence(max_address=1000, length=0))

    def test_generate_sequence_returns_iterator(
        self, medium_max_address: int, standard_sequence_length: int
    ) -> None:
        """Test that generate_sequence returns an iterator.

        Args:
            medium_max_address: Maximum address fixture.
            standard_sequence_length: Sequence length fixture.

        """
        pattern = HeapPattern()
        sequence = pattern.generate_sequence(
            max_address=medium_max_address,
            length=standard_sequence_length,
        )

        from collections.abc import Iterator

        assert isinstance(sequence, Iterator)

    def test_generate_sequence_shows_growth_trend(self, medium_max_address: int) -> None:
        """Test that heap shows overall growth trend.

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = HeapPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=50)
        )

        # Last address should generally be larger than first address
        # (due to allocation bias)
        # This is probabilistic - check that at least some growth occurred
        max_address: int = max(sequence)
        min_address: int = min(sequence)

        # There should be some range of addresses used
        assert max_address > min_address

    def test_generate_sequence_respects_max_heap_size(self, medium_max_address: int) -> None:
        """Test that heap doesn't exceed reasonable bounds.

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = HeapPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=100)
        )

        # All addresses should be within max_address
        heap_range: int = max(sequence) - min(sequence)
        # Heap should use reasonable portion of address space
        max_heap_size: int = min(
            pattern._DEFAULT_MAX_HEAP_SIZE,  # pyright: ignore[reportPrivateUsage]
            medium_max_address // 2,
        )
        # Allow generous margin for the range (allocations can accumulate)
        assert heap_range <= max_heap_size * 2

    def test_generate_sequence_shows_allocation_bias(self, medium_max_address: int) -> None:
        """Test that heap shows forward bias (more allocations than deallocations).

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = HeapPattern()
        sequence: list[int] = list(
            pattern.generate_sequence(max_address=medium_max_address, length=100)
        )

        # Count forward vs backward movements
        forward_moves: int = sum(
            1 for i in range(len(sequence) - 1) if sequence[i + 1] > sequence[i]
        )
        backward_moves: int = sum(
            1 for i in range(len(sequence) - 1) if sequence[i + 1] < sequence[i]
        )

        # Should have more forward moves than backward (allocation bias)
        assert forward_moves > backward_moves

    def test_generate_sequence_uses_different_base_addresses(
        self, medium_max_address: int
    ) -> None:
        """Test that different calls use different base addresses.

        Args:
            medium_max_address: Maximum address fixture.

        """
        pattern = HeapPattern()

        # Generate multiple sequences
        sequences: list[list[int]] = [
            list(pattern.generate_sequence(max_address=medium_max_address, length=20))
            for _ in range(5)
        ]

        # Get minimum address from each sequence (close to base address)
        min_addresses: list[int] = [min(seq) for seq in sequences]

        # Not all sequences should start from the same base
        # (base address is random)
        unique_mins: int = len(set(min_addresses))
        assert unique_mins > 1, "Expected different base addresses across sequences"
