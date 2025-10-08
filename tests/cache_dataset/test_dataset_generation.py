# -*- coding: utf-8 -*-
"""Tests for CacheDataset data generation methods.

Tests the core data generation functionality including pattern sequence
generation and tensor creation.
"""

from collections.abc import Iterator
from unittest.mock import patch

import torch

from cache_simulator.cache_dataset.cache_dataset import CacheDataset


class TestPatternSequenceGeneration:
    """Test suite for pattern-based sequence generation."""

    def test_generate_pattern_sequence_returns_iterator(self) -> None:
        """Test that pattern sequence generation returns an iterator."""
        dataset = CacheDataset(epoch_size=10, max_seq_length=5)
        sequence = (
            dataset._generate_pattern_sequence()  # pyright: ignore[reportPrivateUsage]
        )

        assert isinstance(sequence, Iterator)

    def test_generate_pattern_sequence_produces_addresses(self) -> None:
        """Test that pattern sequence produces valid addresses."""
        dataset = CacheDataset(epoch_size=10, max_seq_length=5, ram_volume=1000)
        sequence = list(
            dataset._generate_pattern_sequence()  # pyright: ignore[reportPrivateUsage]
        )

        assert len(sequence) == 5
        assert all(0 <= addr <= 999 for addr in sequence)

    def test_generate_pattern_sequence_uses_random_pattern(self) -> None:
        """Test that different patterns are selected randomly."""
        dataset = CacheDataset(epoch_size=10, max_seq_length=5)

        # Generate multiple sequences and check that patterns are used
        sequences = [
            list(dataset._generate_pattern_sequence())  # type: ignore
            for _ in range(10)
        ]

        # At least some sequences should be different (very high probability)
        unique_sequences = {tuple(seq) for seq in sequences}
        assert len(unique_sequences) > 1


class TestDataGeneration:
    """Test suite for complete data generation."""

    def test_generate_data_returns_three_tensors(self, mock_cache: type) -> None:
        """Test that generate_data returns three tensors.

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=10, max_seq_length=8)
            context, addresses, labels = dataset.generate_data()

            assert isinstance(context, torch.Tensor)
            assert isinstance(addresses, torch.Tensor)
            assert isinstance(labels, torch.Tensor)

    def test_generate_data_correct_tensor_shapes(self, mock_cache: type) -> None:
        """Test that generated tensors have correct shapes.

        Args:
            mock_cache: Mock cache fixture.

        """
        max_seq_length = 10
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=10, max_seq_length=max_seq_length)
            context, addresses, labels = dataset.generate_data()

            assert context.shape == (2,)  # num_lines, associativity
            assert addresses.shape == (max_seq_length,)
            assert labels.shape == (max_seq_length,)

    def test_generate_data_correct_dtypes(self, mock_cache: type) -> None:
        """Test that generated tensors have correct data types.

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=10)
            context, addresses, labels = dataset.generate_data()

            assert context.dtype == torch.int32
            assert addresses.dtype == torch.int64
            assert labels.dtype == torch.int32

    def test_generate_data_context_contains_cache_params(
        self, mock_cache: type, cache_lines_options: tuple[int, ...]
    ) -> None:
        """Test that context tensor contains valid cache parameters.

        Args:
            mock_cache: Mock cache fixture.
            cache_lines_options: Cache lines fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=10, cache_lines=cache_lines_options)
            context, _, _ = dataset.generate_data()

            num_lines = context[0].item()
            associativity = context[1].item()

            assert num_lines in cache_lines_options
            assert associativity >= 0

    def test_generate_data_addresses_within_bounds(self, mock_cache: type) -> None:
        """Test that generated addresses are within valid range.

        Args:
            mock_cache: Mock cache fixture.

        """
        ram_volume = 10000
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=10, ram_volume=ram_volume)
            _, addresses, _ = dataset.generate_data()

            # Filter out padding (-1)
            valid_addresses = addresses[addresses != -1]
            assert all(0 <= addr <= ram_volume - 1 for addr in valid_addresses)

    def test_generate_data_labels_are_binary_or_padding(self, mock_cache: type) -> None:
        """Test that labels are either 0, 1, or -1 (padding).

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=10)
            _, _, labels = dataset.generate_data()

            labels_list: list[int] = [int(label.item()) for label in labels]
            unique_labels: set[int] = set(labels_list)
            assert unique_labels.issubset({-1, 0, 1})


class TestSequencePadding:
    """Test suite for sequence padding behavior."""

    def test_generate_data_padding_with_negative_one(self, mock_cache: type) -> None:
        """Test that sequences are padded with -1.

        Args:
            mock_cache: Mock cache fixture.

        """
        max_seq_length = 20

        def short_generator() -> Iterator[int]:
            """Generate only 5 addresses."""
            for i in range(5):
                yield i

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(
                epoch_size=10,
                max_seq_length=max_seq_length,
                sequences_generator=short_generator,
            )
            _, addresses, labels = dataset.generate_data()

            # First 5 should be valid
            assert all(addresses[i] != -1 for i in range(5))
            # Rest should be -1
            assert all(addresses[i] == -1 for i in range(5, max_seq_length))
            assert all(labels[i] == -1 for i in range(5, max_seq_length))

    def test_generate_data_no_padding_when_exact_length(self, mock_cache: type) -> None:
        """Test that no padding is added when sequence is exact length.

        Args:
            mock_cache: Mock cache fixture.

        """
        max_seq_length = 10

        def exact_generator() -> Iterator[int]:
            """Generate exactly max_seq_length addresses."""
            for i in range(max_seq_length):
                yield i

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(
                epoch_size=10,
                max_seq_length=max_seq_length,
                sequences_generator=exact_generator,
            )
            _, addresses, labels = dataset.generate_data()

            # No padding should be present
            assert all(addresses[i] != -1 for i in range(max_seq_length))
            assert all(labels[i] != -1 for i in range(max_seq_length))


class TestCustomGenerators:
    """Test suite for custom sequence generators."""

    def test_generate_data_uses_custom_generator(self, mock_cache: type) -> None:
        """Test that custom sequence generator is used.

        Args:
            mock_cache: Mock cache fixture.

        """
        expected_addresses = [10, 20, 30, 40, 50]

        def custom_generator() -> Iterator[int]:
            """Generate specific addresses."""
            yield from expected_addresses

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(
                epoch_size=10,
                max_seq_length=10,
                sequences_generator=custom_generator,
            )
            _, addresses, _ = dataset.generate_data()

            # Check that our custom addresses are used
            for i, expected in enumerate(expected_addresses):
                assert addresses[i] == expected

    def test_generate_data_custom_generator_called_each_time(self, mock_cache: type) -> None:
        """Test that custom generator is called for each data generation.

        Args:
            mock_cache: Mock cache fixture.

        """
        call_count = 0

        def counting_generator() -> Iterator[int]:
            """Make calls to be tracked by generator."""
            nonlocal call_count
            call_count += 1
            for i in range(5):
                yield i

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(
                epoch_size=10,
                max_seq_length=10,
                sequences_generator=counting_generator,
            )

            # Generate data multiple times
            dataset.generate_data()
            dataset.generate_data()
            dataset.generate_data()

            assert call_count == 3
