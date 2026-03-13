"""Tests for capability surface analysis."""

import pytest
from demantiq.evaluation.capability_surface import CapabilitySurface


class TestCapabilitySurface:
    def test_grid_structure(self):
        results = [
            {"config": {"alpha": 0.1, "lambda": 1.0}, "scores": {"mape": 0.05}},
            {"config": {"alpha": 0.5, "lambda": 2.0}, "scores": {"mape": 0.10}},
            {"config": {"alpha": 0.9, "lambda": 3.0}, "scores": {"mape": 0.15}},
        ]
        cs = CapabilitySurface(results)
        grid = cs.compute_grid("alpha", "lambda", "mape", n_bins=3)

        assert "x_edges" in grid
        assert "y_edges" in grid
        assert "grid" in grid
        assert "counts" in grid
        assert len(grid["x_edges"]) == 4  # n_bins + 1
        assert len(grid["y_edges"]) == 4
        assert len(grid["grid"]) == 3
        assert len(grid["grid"][0]) == 3

    def test_grid_correct_values(self):
        # All same config, metric should average to the same value
        results = [
            {"config": {"x": 0.5, "y": 0.5}, "scores": {"m": 1.0}},
            {"config": {"x": 0.5, "y": 0.5}, "scores": {"m": 3.0}},
        ]
        cs = CapabilitySurface(results)
        grid = cs.compute_grid("x", "y", "m", n_bins=1)
        # Single bin, should be mean = 2.0
        assert grid["grid"][0][0] == pytest.approx(2.0)
        assert grid["counts"][0][0] == 2

    def test_empty_results(self):
        cs = CapabilitySurface([])
        grid = cs.compute_grid("x", "y", "m")
        assert grid["x_edges"] == []
        assert grid["grid"] == []

    def test_missing_dimension(self):
        results = [
            {"config": {"alpha": 0.1}, "scores": {"mape": 0.05}},
        ]
        cs = CapabilitySurface(results)
        grid = cs.compute_grid("alpha", "missing_dim", "mape")
        assert grid["grid"] == []

    def test_failure_boundary(self):
        results = [
            {"config": {"alpha": 0.1}, "scores": {"mape": 0.05}},
            {"config": {"alpha": 0.5}, "scores": {"mape": 0.15}},
            {"config": {"alpha": 0.9}, "scores": {"mape": 0.30}},
        ]
        cs = CapabilitySurface(results)
        boundary = cs.find_failure_boundary("mape", threshold=0.10)

        assert len(boundary["passing"]) == 2  # 0.15 and 0.30 >= 0.10
        assert len(boundary["failing"]) == 1  # 0.05 < 0.10

    def test_failure_boundary_all_pass(self):
        results = [
            {"config": {"alpha": 0.1}, "scores": {"mape": 0.5}},
            {"config": {"alpha": 0.5}, "scores": {"mape": 0.6}},
        ]
        cs = CapabilitySurface(results)
        boundary = cs.find_failure_boundary("mape", threshold=0.1)
        assert len(boundary["passing"]) == 2
        assert len(boundary["failing"]) == 0

    def test_failure_boundary_all_fail(self):
        results = [
            {"config": {"alpha": 0.1}, "scores": {"mape": 0.01}},
            {"config": {"alpha": 0.5}, "scores": {"mape": 0.02}},
        ]
        cs = CapabilitySurface(results)
        boundary = cs.find_failure_boundary("mape", threshold=0.1)
        assert len(boundary["passing"]) == 0
        assert len(boundary["failing"]) == 2

    def test_dimensions_auto_detected(self):
        results = [
            {"config": {"a": 1, "b": 2}, "scores": {"m": 0.5}},
        ]
        cs = CapabilitySurface(results)
        assert set(cs.dimensions) == {"a", "b"}

    def test_dimensions_explicit(self):
        results = [
            {"config": {"a": 1, "b": 2, "c": 3}, "scores": {"m": 0.5}},
        ]
        cs = CapabilitySurface(results, dimensions=["a", "b"])
        assert cs.dimensions == ["a", "b"]
