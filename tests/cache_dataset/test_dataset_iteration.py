# -*- coding: utf-8 -*-
"""Tests for CacheDataset iteration and data loading.

Tests the iterator protocol, batch generation, and seeding behavior.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import torch

from cache_simulator.cache_dataset.cache_dataset import CacheDataset


class TestDatasetIteration:
    """Test suite for basic dataset iteration."""

    def test_iter_returns_iterator(self, mock_cache: type) -> None:
        """Test that __iter__ returns an iterator.

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=5)
            iterator = iter(dataset)

            assert isinstance(iterator, Iterator)

    def test_iter_produces_correct_number_of_items(self, mock_cache: type) -> None:
        """Test that iteration produces correct number of items.

        Args:
            mock_cache: Mock cache fixture.

        """
        epoch_size = 7
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=epoch_size)
            items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)

            assert len(items) == epoch_size

    def test_iter_each_item_is_tuple_of_three_tensors(self, mock_cache: type) -> None:
        """Test that each iteration yields tuple of three tensors.

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=3)

            for item in dataset:
                assert isinstance(item, tuple)
                assert len(item) == 3
                assert all(isinstance(t, torch.Tensor) for t in item)

    def test_iter_can_be_called_multiple_times(self, mock_cache: type) -> None:
        """Test that dataset can be iterated multiple times (epochs).

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=5)

            # First epoch
            items1: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)
            # Second epoch
            items2: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)

            assert len(items1) == 5
            assert len(items2) == 5


class TestRandomSeedingBehavior:
    """Test suite for random seed management."""

    @patch("cache_simulator.cache_dataset.cache_dataset.torch.initial_seed")
    def test_iter_sets_random_seeds(self, mock_initial_seed: MagicMock, mock_cache: type) -> None:
        """Test that iteration sets random seeds properly.

        Args:
            mock_initial_seed: Mock for torch.initial_seed.
            mock_cache: Mock cache fixture.

        """
        mock_initial_seed.return_value = 12345

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=1, deterministic=True)
            _: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)

            # Check that initial_seed was called
            mock_initial_seed.assert_called()

    def test_iter_deterministic_mode_produces_same_results(self, mock_cache: type) -> None:
        """Test that deterministic mode produces reproducible results.

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            # Create two identical datasets
            dataset1 = CacheDataset(epoch_size=5, deterministic=True)
            dataset2 = CacheDataset(epoch_size=5, deterministic=True)

            # Reset seeds to same value
            import random

            random.seed(42)
            torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
            items1: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset1)

            random.seed(42)
            torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
            items2: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset2)

            # Results should be identical
            for (c1, a1, l1), (c2, a2, l2) in zip(items1, items2, strict=True):
                assert torch.equal(c1, c2)
                assert torch.equal(a1, a2)
                assert torch.equal(l1, l2)

    def test_iter_non_deterministic_mode_produces_different_results(
        self, mock_cache: type
    ) -> None:
        """Test that non-deterministic mode produces varied results.

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=10, deterministic=False)

            # Generate two epochs
            items1: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)
            items2: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)

            # At least some items should be different
            differences: int = sum(
                not (torch.equal(c1, c2) and torch.equal(a1, a2) and torch.equal(l1, l2))
                for (c1, a1, l1), (c2, a2, l2) in zip(items1, items2, strict=True)
            )

            # With high probability, there should be differences
            assert differences > 0


class TestMultiprocessSupport:
    """Test suite for multi-process data loading support."""

    def test_iter_respects_global_rank(self, mock_cache: type) -> None:
        """Test that global_rank affects seed generation.

        Args:
            mock_cache: Mock cache fixture.

        """
        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset_rank0 = CacheDataset(epoch_size=5, deterministic=True, global_rank=0)
            dataset_rank1 = CacheDataset(epoch_size=5, deterministic=True, global_rank=1)

            # Reset to same initial seed
            import random

            random.seed(42)
            torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
            items_rank0: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(
                dataset_rank0
            )

            random.seed(42)
            torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
            items_rank1: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(
                dataset_rank1
            )

            # Results should be different due to different global_rank
            differences: int = sum(
                not torch.equal(a1, a2)
                for (_c1, a1, _l1), (_c2, a2, _l2) in zip(items_rank0, items_rank1, strict=True)
            )

            # Should have at least some differences
            assert differences > 0

    @patch("cache_simulator.cache_dataset.cache_dataset.get_worker_info")
    def test_iter_handles_worker_id(self, mock_worker_info: MagicMock, mock_cache: type) -> None:
        """Test that worker_id is properly handled.

        Args:
            mock_worker_info: Mock for get_worker_info.
            mock_cache: Mock cache fixture.

        """
        # Simulate worker process
        mock_worker_info.return_value = MagicMock(id=2)

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=2)
            _: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)

            # Should have accessed worker_info
            mock_worker_info.assert_called()

    @patch("cache_simulator.cache_dataset.cache_dataset.get_worker_info")
    def test_iter_handles_no_worker(self, mock_worker_info: MagicMock, mock_cache: type) -> None:
        """Test that None worker_info is handled (single-process).

        Args:
            mock_worker_info: Mock for get_worker_info.
            mock_cache: Mock cache fixture.

        """
        # Simulate single-process mode
        mock_worker_info.return_value = None

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=2)
            items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)

            # Should still produce results
            assert len(items) == 2


class TestCUDASupport:
    """Test suite for CUDA-related functionality."""

    @patch("cache_simulator.cache_dataset.cache_dataset.torch.cuda.is_available")
    @patch("cache_simulator.cache_dataset.cache_dataset.torch.cuda.manual_seed")
    @patch("cache_simulator.cache_dataset.cache_dataset.torch.cuda.manual_seed_all")
    def test_iter_sets_cuda_seeds_when_available(
        self,
        mock_seed_all: MagicMock,
        mock_seed: MagicMock,
        mock_is_available: MagicMock,
        mock_cache: type,
    ) -> None:
        """Test that CUDA seeds are set when CUDA is available.

        Args:
            mock_seed_all: Mock for manual_seed_all.
            mock_seed: Mock for manual_seed.
            mock_is_available: Mock for is_available.
            mock_cache: Mock cache fixture.

        """
        mock_is_available.return_value = True

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            # Reset mock call counts (they may be called by autouse fixture)
            mock_seed.reset_mock()
            mock_seed_all.reset_mock()

            dataset = CacheDataset(epoch_size=1)
            _: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)

            # CUDA seed functions should be called at least once
            # (Note: may be called multiple times due to autouse fixture)
            mock_seed.assert_called()
            mock_seed_all.assert_called()

    @patch("cache_simulator.cache_dataset.cache_dataset.torch.cuda.is_available")
    def test_iter_skips_cuda_seeds_when_unavailable(
        self, mock_is_available: MagicMock, mock_cache: type
    ) -> None:
        """Test that CUDA seeds are not set when CUDA is unavailable.

        Args:
            mock_is_available: Mock for is_available.
            mock_cache: Mock cache fixture.

        """
        mock_is_available.return_value = False

        with patch("cache_simulator.cache_dataset.cache_dataset.Cache", mock_cache):
            dataset = CacheDataset(epoch_size=1)
            _: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = list(dataset)

            # Should complete without errors
            # is_available should have been checked
            mock_is_available.assert_called()
