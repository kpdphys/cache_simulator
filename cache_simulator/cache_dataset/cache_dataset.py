# -*- coding: utf-8 -*-
"""Synthetic dataset generator for processor cache behavior simulation.

This module provides the `CacheDataset` class that generates synthetic data
for modeling processor cache behavior with various memory access patterns.
The data is generated in a PyTorch-compatible format and can be used for
training machine learning models that analyze cache hits and misses.

Dependencies: torch, random, logging, Cache class, and memory access patterns.
"""

from collections.abc import Callable, Iterator
import logging
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .cache import Cache
from .patterns import (
    BaseMemoryAccessPattern,
    HeapPattern,
    LoopPattern,
    RandomPattern,
    SequentialWithJumpsPattern,
    StackPattern,
    StridePattern,
)


class CacheDataset(IterableDataset):  # type: ignore
    """Iterable dataset for generating synthetic processor cache behavior data.

    Generates sequences of memory addresses using various access patterns and
    simulates cache behavior, producing tensors of context (cache parameters),
    addresses, and labels (cache hit/miss).

    The dataset supports multi-process data loading with proper seeding for
    reproducibility.
    """

    def __init__(
        self,
        epoch_size: int,
        ram_volume: int = 2**31,
        max_seq_length: int = 16,
        cache_lines: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024),
        cache_associativity_options: tuple[int, ...] = (0, 1, 2, 4, 8),
        verbose: bool = False,
        sequences_generator: Callable[[], Iterator[int]] | None = None,
        global_rank: int = 0,
        deterministic: bool = False,
    ) -> None:
        """Initialize the cache behavior dataset.

        Args:
            epoch_size: Number of sequences per worker per epoch.
            ram_volume: RAM size in bytes (default: 2 GB).
            max_seq_length: Maximum sequence length (default: 16).
            cache_lines: Tuple of possible cache line counts.
            cache_associativity_options: Tuple of possible associativity values.
            verbose: If True, enables DEBUG level logging.
            sequences_generator: Optional callable that returns an iterator of
                addresses. If None, sequences are generated using random patterns.
            global_rank: Global process rank for distributed training (default: 0).
            deterministic: If True, ensures reproducible results by avoiding
                timestamp-based seeding (default: False).

        Raises:
            ValueError: If parameters have invalid values.

        """
        super().__init__()

        # Validate parameters
        if ram_volume <= 0:
            raise ValueError(f"RAM volume must be positive, got: {ram_volume}")
        if max_seq_length <= 0:
            raise ValueError(f"Maximum sequence length must be positive, got: {max_seq_length}")
        if not cache_lines or any(lines <= 0 for lines in cache_lines):
            raise ValueError(
                f"cache_lines must be non-empty and contain only positive "
                f"numbers, got: {cache_lines}"
            )
        if not cache_associativity_options or any(
            assoc < 0 for assoc in cache_associativity_options
        ):
            raise ValueError(
                f"cache_associativity_options must be non-empty and contain "
                f"only non-negative numbers, got: {cache_associativity_options}"
            )
        if epoch_size <= 0:
            raise ValueError(f"Epoch size must be positive, got: {epoch_size}")

        self.max_seq_length: int = max_seq_length
        self.max_address: int = ram_volume - 1
        self.cache_lines: tuple[int, ...] = cache_lines
        self.associativity_options: tuple[int, ...] = cache_associativity_options
        self.epoch_size: int = epoch_size
        self.deterministic: bool = deterministic
        self.sequences_generator: Callable[[], Iterator[int]] | None = sequences_generator
        self.global_rank: int = global_rank

        # Available memory access patterns
        self.patterns: list[BaseMemoryAccessPattern] = [
            SequentialWithJumpsPattern(),
            LoopPattern(),
            RandomPattern(),
            StridePattern(),
            StackPattern(),
            HeapPattern(),
        ]

        # Setup logging
        log_level: int = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

    def _generate_pattern_sequence(self) -> Iterator[int]:
        """Generate an address sequence using a random access pattern.

        Selects a random pattern and generates a sequence with pattern-specific
        randomly chosen parameters.

        Returns:
            Iterator yielding memory addresses.

        """
        # Select random pattern
        pattern: BaseMemoryAccessPattern = random.choice(self.patterns)

        # Let the pattern generate its own random parameters
        params: dict[str, int | float] = pattern.generate_random_params(
            max_address=self.max_address, length=self.max_seq_length
        )

        self.logger.debug(
            "Selected pattern: %s, parameters: %s", pattern.__class__.__name__, params
        )

        return pattern.generate_sequence(**params)

    def generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a single dataset item.

        Creates a cache emulator with random parameters, generates an address
        sequence, and forms tensors of context, addresses, and labels.

        Returns:
            Tuple of three tensors:
                - context_tensor: Cache parameters (num_lines, associativity).
                - addresses_tensor: Memory addresses (padded with -1).
                - labels_tensor: Labels (1 for hit, 0 for miss, -1 for padding).

        """
        # Create cache with random parameters
        num_lines: int = random.choice(self.cache_lines)
        associativity: int = random.choice(self.associativity_options)
        cache: Cache = Cache(num_lines=num_lines, associativity=associativity)

        # Context tensor with original values (no encoding)
        context_tensor: torch.Tensor = torch.tensor([num_lines, associativity], dtype=torch.int32)

        # Collect addresses and labels
        addresses: list[int] = []
        labels: list[int] = []

        # Use provided generator or generate using random pattern
        sequence: Iterator[int] = (
            self.sequences_generator()
            if self.sequences_generator
            else self._generate_pattern_sequence()
        )

        for address in sequence:
            addresses.append(address)
            is_hit: bool = cache.is_in_cache(address)
            labels.append(1 if is_hit else 0)
            cache.add_to_cache(address)

        # Pad sequences to max_seq_length
        pad_length: int = self.max_seq_length - len(addresses)
        if pad_length > 0:
            addresses.extend([-1] * pad_length)
            labels.extend([-1] * pad_length)

        # Convert to tensors
        addresses_tensor: torch.Tensor = torch.tensor(addresses, dtype=torch.int64)
        labels_tensor: torch.Tensor = torch.tensor(labels, dtype=torch.int32)

        self.logger.debug(
            "Generated data: context=%s, addresses=%s, labels=%s",
            context_tensor,
            addresses_tensor,
            labels_tensor,
        )

        return context_tensor, addresses_tensor, labels_tensor

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over the dataset.

        Generates dataset items for one epoch. Supports multi-process data
        loading with unique seeding per worker for reproducibility.

        Yields:
            Tuples of (context, addresses, labels) tensors.

        """
        worker_info = get_worker_info()
        worker_id: int = 0 if worker_info is None else worker_info.id

        # Create unique seed for each worker
        # Base seed from PyTorch + global rank offset + worker offset
        # Explicitly cast to int to satisfy type checker
        seed: int = int(torch.initial_seed()) + 1000000 * self.global_rank + 1000 * worker_id

        # Add timestamp for non-deterministic mode
        if not self.deterministic:
            import time

            seed += int(time.time() * 1000000) % 1000000

        # Set seeds for all random number generators
        random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.logger.debug(
            "Starting iteration: worker_id=%d, seed=%d, deterministic=%s",
            worker_id,
            seed,
            self.deterministic,
        )

        for _ in range(self.epoch_size):
            yield self.generate_data()
