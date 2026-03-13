"""Tests for optimization quality evaluation."""

import pytest
from demantiq.evaluation.optimization_quality import OptimizationQuality, OptResult


class TestOptimizationQuality:
    def setup_method(self):
        self.evaluator = OptimizationQuality()

    def test_true_optimal_efficiency_one(self):
        allocation = {"fb": 100.0, "google": 200.0}
        result = self.evaluator.evaluate(allocation, allocation)
        assert result.optimization_efficiency == pytest.approx(1.0)

    def test_suboptimal_allocation(self):
        model_alloc = {"fb": 50.0, "google": 100.0}
        true_alloc = {"fb": 100.0, "google": 200.0}
        result = self.evaluator.evaluate(model_alloc, true_alloc)
        assert result.optimization_efficiency == pytest.approx(0.5)

    def test_custom_revenue_function(self):
        def revenue_fn(alloc):
            return sum(v ** 0.5 for v in alloc.values())

        model_alloc = {"fb": 100.0, "google": 100.0}
        true_alloc = {"fb": 100.0, "google": 100.0}
        result = self.evaluator.evaluate(model_alloc, true_alloc, revenue_function=revenue_fn)
        assert result.optimization_efficiency == pytest.approx(1.0)
        assert result.revenue_model == pytest.approx(20.0)

    def test_custom_revenue_function_suboptimal(self):
        def revenue_fn(alloc):
            return sum(v ** 0.5 for v in alloc.values())

        model_alloc = {"fb": 25.0, "google": 25.0}
        true_alloc = {"fb": 100.0, "google": 100.0}
        result = self.evaluator.evaluate(model_alloc, true_alloc, revenue_function=revenue_fn)
        assert result.optimization_efficiency == pytest.approx(0.5)

    def test_zero_true_revenue(self):
        model_alloc = {"fb": 100.0}
        true_alloc = {"fb": 0.0}
        result = self.evaluator.evaluate(model_alloc, true_alloc)
        assert result.optimization_efficiency == 0.0

    def test_revenues_stored(self):
        model_alloc = {"fb": 50.0, "google": 75.0}
        true_alloc = {"fb": 100.0, "google": 150.0}
        result = self.evaluator.evaluate(model_alloc, true_alloc)
        assert result.revenue_model == pytest.approx(125.0)
        assert result.revenue_true == pytest.approx(250.0)

    def test_result_dataclass_defaults(self):
        result = OptResult()
        assert result.revenue_model == 0.0
        assert result.revenue_true == 0.0
        assert result.optimization_efficiency == 0.0
