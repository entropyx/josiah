"""Optimization quality evaluation metrics."""

from dataclasses import dataclass
import numpy as np


@dataclass
class OptResult:
    """Result of optimization quality evaluation.

    Attributes:
        revenue_model: Revenue achieved by the model's allocation.
        revenue_true: Revenue achieved by the true optimal allocation.
        optimization_efficiency: revenue_model / revenue_true.
    """

    revenue_model: float = 0.0
    revenue_true: float = 0.0
    optimization_efficiency: float = 0.0


class OptimizationQuality:
    """Evaluate quality of budget allocation optimization."""

    def evaluate(
        self,
        model_allocation: dict,
        true_allocation: dict,
        revenue_function: callable = None,
    ) -> OptResult:
        """Evaluate allocation quality by comparing revenues.

        Args:
            model_allocation: {channel: spend} from model optimization.
            true_allocation: {channel: spend} known optimal allocation.
            revenue_function: Optional callable(allocation_dict) -> revenue.
                If provided, used to compute revenues. Otherwise uses simple
                sum of allocations as proxy.

        Returns:
            OptResult with revenue comparison and efficiency.
        """
        if revenue_function is not None:
            revenue_model = float(revenue_function(model_allocation))
            revenue_true = float(revenue_function(true_allocation))
        else:
            # Simple proxy: sum of allocation values
            revenue_model = sum(float(v) for v in model_allocation.values())
            revenue_true = sum(float(v) for v in true_allocation.values())

        if revenue_true > 0:
            efficiency = revenue_model / revenue_true
        else:
            efficiency = 0.0

        return OptResult(
            revenue_model=revenue_model,
            revenue_true=revenue_true,
            optimization_efficiency=efficiency,
        )
