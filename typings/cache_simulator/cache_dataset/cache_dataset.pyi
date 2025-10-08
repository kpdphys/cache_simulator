# -*- coding: utf-8 -*-
"""Type stubs for cache_dataset module."""

from collections.abc import Callable, Iterator
from logging import Logger

import torch
from torch.utils.data import IterableDataset

from cache_simulator.cache_dataset.patterns.base import BaseMemoryAccessPattern

class CacheDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Type stub for CacheDataset with proper generic parameter."""

    max_seq_length: int
    max_address: int
    cache_lines: tuple[int, ...]
    associativity_options: tuple[int, ...]
    epoch_size: int
    deterministic: bool
    sequences_generator: Callable[[], Iterator[int]] | None
    global_rank: int
    patterns: list[BaseMemoryAccessPattern]
    logger: Logger

    def __init__(
        self,
        epoch_size: int,
        ram_volume: int = ...,
        max_seq_length: int = ...,
        cache_lines: tuple[int, ...] = ...,
        cache_associativity_options: tuple[int, ...] = ...,
        verbose: bool = ...,
        sequences_generator: Callable[[], Iterator[int]] | None = ...,
        global_rank: int = ...,
        deterministic: bool = ...,
    ) -> None: ...
    def _generate_pattern_sequence(self) -> Iterator[int]: ...
    def generate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...
