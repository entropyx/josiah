import numpy as np
from demantiq.utils.random import create_rng, derive_seed, create_sub_rngs


def test_create_rng_deterministic():
    rng1 = create_rng(42)
    rng2 = create_rng(42)
    assert rng1.random() == rng2.random()


def test_derive_seed_deterministic():
    rng1 = create_rng(42)
    rng2 = create_rng(42)
    assert derive_seed(rng1) == derive_seed(rng2)


def test_create_sub_rngs():
    rng = create_rng(42)
    subs = create_sub_rngs(rng, 3)
    assert len(subs) == 3
    # Each sub-rng produces different values
    vals = [s.random() for s in subs]
    assert len(set(vals)) == 3
