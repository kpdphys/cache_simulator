# -*- coding: utf-8 -*-
"""Memory access pattern generators for cache simulation.

This package provides various memory access patterns that can be used
to generate synthetic address sequences for cache behavior modeling.

Available patterns:
    - SequentialWithJumpsPattern: Sequential with random jumps
    - LoopPattern: Cyclic access to small ranges
    - RandomPattern: Fully random access
    - StridePattern: Fixed-stride array-like access
    - StackPattern: Stack push/pop operations
    - HeapPattern: Heap allocation/deallocation
"""

from .base import BaseMemoryAccessPattern
from .heap import HeapPattern
from .loop import LoopPattern
from .random_access import RandomPattern
from .sequential import SequentialWithJumpsPattern
from .stack import StackPattern
from .stride import StridePattern

__all__ = [
    "BaseMemoryAccessPattern",
    "SequentialWithJumpsPattern",
    "LoopPattern",
    "RandomPattern",
    "StridePattern",
    "StackPattern",
    "HeapPattern",
]
