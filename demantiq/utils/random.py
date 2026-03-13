"""Seeded RNG management for reproducible simulations."""

import numpy as np
from numpy.random import Generator, default_rng


def create_rng(seed: int) -> Generator:
    """Create a numpy random Generator from a seed."""
    return default_rng(seed)


def derive_seed(rng: Generator) -> int:
    """Derive a deterministic sub-seed from an existing RNG."""
    return int(rng.integers(0, 2**31))


def create_sub_rngs(rng: Generator, n: int) -> list[Generator]:
    """Create n independent sub-RNGs from a parent RNG."""
    seeds = rng.integers(0, 2**31, size=n)
    return [default_rng(int(s)) for s in seeds]
