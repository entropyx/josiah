"""Tests for model comparison framework."""

import pytest
import pandas as pd
from demantiq.evaluation.model_comparison import (
    ModelAdapter,
    ModelComparison,
    ComparisonReport,
)


class MockModel:
    """A mock model adapter for testing."""

    def __init__(self, contributions=None, roas=None):
        self._contributions = contributions or {"fb": 100.0, "google": 200.0}
        self._roas = roas or {"fb": 2.0, "google": 1.5}

    def fit(self, observable_data):
        return {"fitted": True}

    def predict_contributions(self, fit):
        return self._contributions

    def predict_roas(self, fit):
        return self._roas


class MockScenario:
    """A mock scenario with observable data and ground truth."""

    def __init__(self):
        self.observable_data = pd.DataFrame({"y": [1, 2, 3]})
        self.summary_truth = {
            "channel_contributions": {"fb": 100.0, "google": 200.0},
            "channel_roas": {"fb": 2.0, "google": 1.5},
        }


class TestModelAdapter:
    def test_mock_satisfies_protocol(self):
        model = MockModel()
        assert isinstance(model, ModelAdapter)

    def test_protocol_methods_exist(self):
        model = MockModel()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_contributions")
        assert hasattr(model, "predict_roas")


class TestComparisonReport:
    def test_empty_report(self):
        report = ComparisonReport()
        assert report.results == []

    def test_to_dataframe(self):
        report = ComparisonReport(
            results=[
                {"model": "a", "metric": 0.5},
                {"model": "b", "metric": 0.3},
            ]
        )
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "model" in df.columns


class TestModelComparison:
    def test_end_to_end(self):
        models = {"mock_model": MockModel()}
        scenarios = [MockScenario()]
        comp = ModelComparison(models, scenarios, metrics=["contribution_accuracy", "roas_accuracy"])
        report = comp.run(n_seeds=2)

        assert len(report.results) == 2  # 1 model x 1 scenario x 2 seeds
        for row in report.results:
            assert row["model"] == "mock_model"
            assert row["scenario"] == 0
            assert "error" not in row

    def test_multiple_models(self):
        models = {
            "model_a": MockModel(contributions={"fb": 100.0, "google": 200.0}),
            "model_b": MockModel(contributions={"fb": 50.0, "google": 100.0}),
        }
        scenarios = [MockScenario()]
        comp = ModelComparison(models, scenarios, metrics=["contribution_accuracy"])
        report = comp.run(n_seeds=1)

        assert len(report.results) == 2
        model_names = {r["model"] for r in report.results}
        assert model_names == {"model_a", "model_b"}

    def test_perfect_model_zero_mape(self):
        # Model returns exact true values
        models = {
            "perfect": MockModel(
                contributions={"fb": 100.0, "google": 200.0},
                roas={"fb": 2.0, "google": 1.5},
            )
        }
        scenarios = [MockScenario()]
        comp = ModelComparison(models, scenarios)
        report = comp.run(n_seeds=1)

        row = report.results[0]
        assert row.get("contribution_mape") == pytest.approx(0.0)
        assert row.get("roas_mape") == pytest.approx(0.0)

    def test_default_metrics(self):
        comp = ModelComparison({}, [])
        assert "parameter_recovery" in comp.metrics
        assert "contribution_accuracy" in comp.metrics
        assert "roas_accuracy" in comp.metrics

    def test_to_dataframe_from_run(self):
        models = {"mock": MockModel()}
        scenarios = [MockScenario()]
        comp = ModelComparison(models, scenarios, metrics=["contribution_accuracy"])
        report = comp.run(n_seeds=1)
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
