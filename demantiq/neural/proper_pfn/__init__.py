"""Proper PFN-based MMM decomposition.

Replaces the earlier pfn_* modules with an architecture that:
- Treats each (channel, week) pair as a permutation-invariant token
- Uses held-out loss (only masked tokens contribute to gradient)
- Predicts per-component shares of y per timestep
- Includes per-channel impressions and clicks as input features
"""

from demantiq.neural.proper_pfn.model import ProperPFNModel
from demantiq.neural.proper_pfn.dataset import ProperPFNDataset, proper_pfn_collate_fn
from demantiq.neural.proper_pfn.engine import ProperPFNEngine, ProperPFNConfig

__all__ = [
    "ProperPFNModel",
    "ProperPFNDataset",
    "proper_pfn_collate_fn",
    "ProperPFNEngine",
    "ProperPFNConfig",
]
