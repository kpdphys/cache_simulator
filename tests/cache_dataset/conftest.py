# -*- coding: utf-8 -*-
"""Shared pytest fixtures for cache simulator tests.

This module provides common fixtures used across multiple test modules,
including cache instances, patterns, and test data generators.
"""

from collections.abc import Generator, Iterator

import pytest
import torch

from cache_simulator.cache_dataset.cache import Cache

# ============================================================================
# Cache fixtures (original)
# ============================================================================


@pytest.fixture
def direct_mapped_cache() -> Cache:
    """Create a direct mapped cache (4 lines, 64 bytes each).

    Returns:
        Direct-mapped cache instance.

    """
    return Cache(line_size=64, num_lines=4, associativity=1)


@pytest.fixture
def set_associative_cache() -> Cache:
    """Create a 2-way set-associative cache (8 lines, 64 bytes each).

    Returns:
        2-way set-associative cache instance.

    """
    return Cache(line_size=64, num_lines=8, associativity=2)


@pytest.fixture
def fully_associative_cache() -> Cache:
    """Create a fully associative cache (4 lines, 64 bytes each).

    Returns:
        Fully associative cache instance.

    """
    return Cache(line_size=64, num_lines=4, associativity=0)


# ============================================================================
# Address and sequence fixtures
# ============================================================================


@pytest.fixture
def small_max_address() -> int:
    """Provide a small maximum address for testing.

    Returns:
        Small address value (1000) suitable for quick tests.

    """
    return 1000


@pytest.fixture
def medium_max_address() -> int:
    """Provide a medium maximum address for testing.

    Returns:
        Medium address value (100000) for standard tests.

    """
    return 100000


@pytest.fixture
def standard_sequence_length() -> int:
    """Provide a standard sequence length for testing.

    Returns:
        Standard sequence length (16).

    """
    return 16


@pytest.fixture
def cache_lines_options() -> tuple[int, ...]:
    """Provide standard cache line options.

    Returns:
        Tuple of common cache line counts.

    """
    return (16, 32, 64, 128, 256)


@pytest.fixture
def associativity_options() -> tuple[int, ...]:
    """Provide standard associativity options.

    Returns:
        Tuple of common associativity values.

    """
    return (0, 1, 2, 4, 8)


@pytest.fixture
def sample_addresses() -> list[int]:
    """Provide a sample list of memory addresses for testing.

    Returns:
        List of sample addresses.

    """
    return [0, 100, 200, 100, 300, 0, 400, 100]


@pytest.fixture
def simple_address_generator() -> Iterator[int]:
    """Provide a simple address generator for testing.

    Yields:
        Sequential addresses from 0 to 9.

    """
    for i in range(10):
        yield i


# ============================================================================
# Random seed fixtures
# ============================================================================


@pytest.fixture
def fixed_seed() -> int:
    """Provide a fixed random seed for reproducible tests.

    Returns:
        Fixed seed value (42).

    """
    return 42


@pytest.fixture(autouse=True)
def reset_random_seeds(fixed_seed: int) -> Generator[None, None, None]:
    """Reset random seeds before each test for reproducibility.

    This fixture runs automatically before each test to ensure
    consistent random number generation.

    Args:
        fixed_seed: The seed value to use.

    Yields:
        Control back to test after setup.

    """
    import random

    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)  # pyright: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fixed_seed)
        torch.cuda.manual_seed_all(fixed_seed)
    yield  # noqa


# ============================================================================
# Mock fixtures
# ============================================================================


@pytest.fixture
def mock_cache() -> type:
    """Provide MockCache class for testing.

    Returns:
        MockCache class from mocks module.

    """
    from tests.cache_dataset.mocks import MockCache

    return MockCache
