# -*- coding: utf-8 -*-
"""Tests for CacheDataset initialization.

Tests parameter validation and configuration of the dataset class.
"""

import pytest

from cache_simulator.cache_dataset.cache_dataset import CacheDataset


class TestCacheDatasetInitialization:
    """Test suite for CacheDataset initialization and validation."""

    def test_initialization_with_defaults(self) -> None:
        """Test that dataset can be initialized with default parameters."""
        dataset = CacheDataset(epoch_size=10)

        assert dataset.epoch_size == 10
        assert dataset.max_seq_length == 16
        assert dataset.max_address == 2**31 - 1

    def test_initialization_with_custom_params(
        self,
        cache_lines_options: tuple[int, ...],
        associativity_options: tuple[int, ...],
    ) -> None:
        """Test initialization with custom parameters.

        Args:
            cache_lines_options: Cache lines fixture.
            associativity_options: Associativity fixture.

        """
        dataset = CacheDataset(
            epoch_size=100,
            ram_volume=1024,
            max_seq_length=32,
            cache_lines=cache_lines_options,
            cache_associativity_options=associativity_options,
            verbose=True,
            deterministic=True,
            global_rank=1,
        )

        assert dataset.epoch_size == 100
        assert dataset.max_seq_length == 32
        assert dataset.max_address == 1023
        assert dataset.cache_lines == cache_lines_options
        assert dataset.associativity_options == associativity_options
        assert dataset.deterministic is True
        assert dataset.global_rank == 1

    def test_initialization_rejects_invalid_ram_volume(self) -> None:
        """Test that invalid ram_volume is rejected."""
        with pytest.raises(ValueError, match="RAM volume must be positive"):
            CacheDataset(epoch_size=10, ram_volume=0)

        with pytest.raises(ValueError, match="RAM volume must be positive"):
            CacheDataset(epoch_size=10, ram_volume=-1000)

    def test_initialization_rejects_invalid_max_seq_length(self) -> None:
        """Test that invalid max_seq_length is rejected."""
        with pytest.raises(ValueError, match="Maximum sequence length must be positive"):
            CacheDataset(epoch_size=10, max_seq_length=0)

        with pytest.raises(ValueError, match="Maximum sequence length must be positive"):
            CacheDataset(epoch_size=10, max_seq_length=-5)

    def test_initialization_rejects_empty_cache_lines(self) -> None:
        """Test that empty cache_lines tuple is rejected."""
        with pytest.raises(ValueError, match="cache_lines must be non-empty"):
            CacheDataset(epoch_size=10, cache_lines=())

    def test_initialization_rejects_invalid_cache_lines(self) -> None:
        """Test that invalid cache_lines values are rejected."""
        with pytest.raises(ValueError, match="cache_lines.*only positive numbers"):
            CacheDataset(epoch_size=10, cache_lines=(16, 0, 64))

        with pytest.raises(ValueError, match="cache_lines.*only positive numbers"):
            CacheDataset(epoch_size=10, cache_lines=(16, -32, 64))

    def test_initialization_rejects_empty_associativity_options(self) -> None:
        """Test that empty associativity_options tuple is rejected."""
        with pytest.raises(ValueError, match="cache_associativity_options must be non-empty"):
            CacheDataset(epoch_size=10, cache_associativity_options=())

    def test_initialization_rejects_invalid_associativity(self) -> None:
        """Test that invalid associativity values are rejected."""
        with pytest.raises(ValueError, match="cache_associativity_options.*non-negative"):
            CacheDataset(epoch_size=10, cache_associativity_options=(1, 2, -1))

    def test_initialization_rejects_invalid_epoch_size(self) -> None:
        """Test that invalid epoch_size is rejected."""
        with pytest.raises(ValueError, match="Epoch size must be positive"):
            CacheDataset(epoch_size=0)

        with pytest.raises(ValueError, match="Epoch size must be positive"):
            CacheDataset(epoch_size=-10)

    def test_patterns_initialized(self) -> None:
        """Test that memory access patterns are initialized."""
        dataset = CacheDataset(epoch_size=10)

        assert len(dataset.patterns) == 6
        assert all(hasattr(p, "generate_sequence") for p in dataset.patterns)

    def test_logger_initialized(self) -> None:
        """Test that logger is properly initialized."""
        dataset = CacheDataset(epoch_size=10, verbose=False)

        assert dataset.logger is not None
        assert dataset.logger.name == "cache_simulator.cache_dataset.cache_dataset"


class TestCacheDatasetConfiguration:
    """Test suite for dataset configuration options."""

    def test_deterministic_mode_enabled(self) -> None:
        """Test that deterministic mode can be enabled."""
        dataset = CacheDataset(epoch_size=10, deterministic=True)

        assert dataset.deterministic is True

    def test_deterministic_mode_disabled_by_default(self) -> None:
        """Test that deterministic mode is disabled by default."""
        dataset = CacheDataset(epoch_size=10)

        assert dataset.deterministic is False

    def test_global_rank_defaults_to_zero(self) -> None:
        """Test that global_rank defaults to 0."""
        dataset = CacheDataset(epoch_size=10)

        assert dataset.global_rank == 0

    def test_global_rank_can_be_set(self) -> None:
        """Test that global_rank can be customized."""
        dataset = CacheDataset(epoch_size=10, global_rank=5)

        assert dataset.global_rank == 5

    def test_custom_sequences_generator_accepted(self) -> None:
        """Test that custom sequence generator is accepted."""
        from collections.abc import Iterator

        def custom_gen() -> Iterator[int]:
            yield from range(10)

        dataset = CacheDataset(epoch_size=10, sequences_generator=custom_gen)

        assert dataset.sequences_generator is custom_gen
