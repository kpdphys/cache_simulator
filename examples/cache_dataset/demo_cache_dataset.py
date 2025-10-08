# -*- coding: utf-8 -*-
"""Demo script for CacheDataset usage.

This script demonstrates the basic usage of CacheDataset class,
showing how to generate synthetic cache behavior data for ML training.

Usage:
    python examples/demo_cache_dataset.py
"""

from collections.abc import Iterator
from typing import Any

from torch.utils.data import DataLoader

from cache_simulator.cache_dataset.cache_dataset import CacheDataset


def demo_basic_usage() -> None:
    """Demonstrate basic CacheDataset usage."""
    print("=" * 80)
    print("Demo 1: Basic Usage")
    print("=" * 80)

    dataset = CacheDataset(
        epoch_size=5,
        ram_volume=1024,
        max_seq_length=8,
        cache_lines=(4, 8),
        cache_associativity_options=(1, 2),
        deterministic=True,
    )

    for i, (context, addresses, labels) in enumerate(dataset):
        print(f"\nSample {i + 1}:")
        num_lines = int(context[0].item())
        associativity = int(context[1].item())
        print(
            f"  Cache config: {context.tolist()} (lines={num_lines}, assoc={associativity})"  # type: ignore
        )
        print(f"  Addresses:    {addresses.tolist()}")  # type: ignore
        print(f"  Labels:       {labels.tolist()} (1=hit, 0=miss, -1=padding)")  # type: ignore

        valid_labels = labels[labels != -1]
        if len(valid_labels) > 0:
            hits = (valid_labels == 1).sum()
            hit_rate = float(hits.item()) / len(valid_labels)
            print(f"  Hit rate:     {hit_rate:.2%}")


def demo_custom_sequence() -> None:
    """Demonstrate usage with custom address sequence generator."""
    print("\n" + "=" * 80)
    print("Demo 2: Custom Address Sequence")
    print("=" * 80)

    def my_address_generator() -> Iterator[int]:
        """Generate a simple repeating pattern."""
        pattern: list[int] = [0, 64, 128, 0, 64, 128, 0, 64]
        for addr in pattern:
            yield addr

    dataset = CacheDataset(
        epoch_size=3,
        ram_volume=1024,
        max_seq_length=8,
        cache_lines=(4,),
        cache_associativity_options=(2,),
        sequences_generator=my_address_generator,
        deterministic=True,
    )

    for i, (context, addresses, labels) in enumerate(dataset):
        print(f"\nSample {i + 1}:")
        print(f"  Context:   {context.tolist()}")  # type: ignore
        print(f"  Addresses: {addresses.tolist()}")  # type: ignore
        print(f"  Labels:    {labels.tolist()}")  # type: ignore


def demo_different_cache_configs() -> None:
    """Demonstrate different cache configurations."""
    print("\n" + "=" * 80)
    print("Demo 3: Different Cache Configurations")
    print("=" * 80)

    configs: list[dict[str, Any]] = [
        {
            "name": "Small Direct-Mapped Cache",
            "cache_lines": (4,),
            "cache_associativity_options": (1,),
        },
        {
            "name": "Large Fully-Associative Cache",
            "cache_lines": (64,),
            "cache_associativity_options": (0,),
        },
        {
            "name": "Medium 4-Way Set-Associative Cache",
            "cache_lines": (16,),
            "cache_associativity_options": (4,),
        },
    ]

    for config in configs:
        print(f"\n{config['name']}:")
        print(
            f"  Lines: {config['cache_lines'][0]},"
            f"  Associativity: {config['cache_associativity_options'][0]}"
        )

        dataset = CacheDataset(
            epoch_size=2,
            ram_volume=2048,
            max_seq_length=10,
            cache_lines=config["cache_lines"],
            cache_associativity_options=config["cache_associativity_options"],
            deterministic=True,
        )

        hit_rates: list[float] = []
        for _, _, labels in dataset:
            valid_labels = labels[labels != -1]
            if len(valid_labels) > 0:
                hits = (valid_labels == 1).sum()
                hit_rate = float(hits.item()) / len(valid_labels)
                hit_rates.append(hit_rate)

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        print(f"  Average hit rate: {avg_hit_rate:.2%}")


def demo_dataset_statistics() -> None:
    """Demonstrate collecting statistics from dataset."""
    print("\n" + "=" * 80)
    print("Demo 4: Dataset Statistics")
    print("=" * 80)

    dataset = CacheDataset(
        epoch_size=100,
        ram_volume=4096,
        max_seq_length=16,
        deterministic=False,
    )

    total_samples = 0
    total_hits = 0
    total_misses = 0
    cache_configs: dict[str, int] = {}

    for context, _, labels in dataset:
        total_samples += 1

        valid_labels = labels[labels != -1]
        hits = (valid_labels == 1).sum()
        misses = (valid_labels == 0).sum()
        total_hits += int(hits.item())
        total_misses += int(misses.item())

        num_lines = int(context[0].item())
        associativity = int(context[1].item())
        config_key = f"{num_lines}L_{associativity}W"
        cache_configs[config_key] = cache_configs.get(config_key, 0) + 1

    total_accesses = total_hits + total_misses
    print(f"\nStatistics from {total_samples} samples:")
    print(f"  Total accesses: {total_accesses}")
    print(f"  Total hits:     {total_hits} ({total_hits / total_accesses:.2%})")
    print(f"  Total misses:   {total_misses} ({total_misses / total_accesses:.2%})")
    print("\nCache configurations used:")
    for config, count in sorted(cache_configs.items()):
        print(f"  {config}: {count} samples")


def demo_dataloader_integration() -> None:
    """Demonstrate integration with PyTorch DataLoader."""
    print("\n" + "=" * 80)
    print("Demo 5: PyTorch DataLoader Integration")
    print("=" * 80)

    dataset = CacheDataset(
        epoch_size=20,
        ram_volume=2048,
        max_seq_length=12,
        deterministic=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
    )

    print("\nBatch processing with DataLoader:")
    for batch_idx, (contexts, addresses, labels) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Contexts shape:  {contexts.shape}")
        print(f"  Addresses shape: {addresses.shape}")
        print(f"  Labels shape:    {labels.shape}")

        if batch_idx >= 2:
            print("\n  ... (showing only first 3 batches)")
            break


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" CacheDataset Demonstration")
    print("=" * 80)

    demo_basic_usage()
    demo_custom_sequence()
    demo_different_cache_configs()
    demo_dataset_statistics()
    demo_dataloader_integration()

    print("\n" + "=" * 80)
    print("Demonstration completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
