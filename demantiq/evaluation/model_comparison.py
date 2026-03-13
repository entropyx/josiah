"""Model comparison framework."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for model adapters used in comparison."""

    def fit(self, observable_data) -> object:
        """Fit the model on observable data."""
        ...

    def predict_contributions(self, fit) -> dict:
        """Predict per-channel contributions from a fitted model."""
        ...

    def predict_roas(self, fit) -> dict:
        """Predict per-channel ROAS from a fitted model."""
        ...


@dataclass
class ComparisonReport:
    """Report from model comparison runs.

    Attributes:
        results: List of per-model-scenario-metric dicts.
    """

    results: list = field(default_factory=list)

    def to_dataframe(self):
        """Convert results to a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.results)


class ModelComparison:
    """Run multiple models against multiple scenarios and evaluate.

    Args:
        models: {model_name: ModelAdapter instance}
        scenarios: List of scenario configs or SimulationResult objects.
        metrics: List of metric names to evaluate.
    """

    def __init__(
        self,
        models: dict,
        scenarios: list,
        metrics: list = None,
    ):
        self.models = models
        self.scenarios = scenarios
        self.metrics = metrics or [
            "parameter_recovery",
            "contribution_accuracy",
            "roas_accuracy",
        ]

    def run(self, n_seeds: int = 5) -> ComparisonReport:
        """Run all models against all scenarios, evaluate with all metrics.

        Args:
            n_seeds: Number of random seeds per scenario for robustness.

        Returns:
            ComparisonReport with all results.
        """
        from demantiq.evaluation.contribution_accuracy import ContributionAccuracy
        from demantiq.evaluation.roas_accuracy import ROASAccuracy

        results = []
        contrib_eval = ContributionAccuracy()
        roas_eval = ROASAccuracy()

        for scenario_idx, scenario in enumerate(self.scenarios):
            for model_name, model in self.models.items():
                for seed in range(n_seeds):
                    row = {
                        "model": model_name,
                        "scenario": scenario_idx,
                        "seed": seed,
                    }

                    try:
                        # Scenario can be a SimulationResult or have observable_data
                        if hasattr(scenario, "observable_data"):
                            obs_data = scenario.observable_data
                            summary = getattr(scenario, "summary_truth", {})
                        else:
                            obs_data = scenario
                            summary = {}

                        fit = model.fit(obs_data)

                        if "contribution_accuracy" in self.metrics:
                            est_contrib = model.predict_contributions(fit)
                            true_contrib = summary.get("channel_contributions", {})
                            if true_contrib:
                                cr = contrib_eval.evaluate(est_contrib, true_contrib)
                                row["contribution_mape"] = (
                                    np.mean(list(cr.per_channel_mape.values()))
                                    if cr.per_channel_mape
                                    else None
                                )
                                row["channel_ranking"] = cr.channel_ranking

                        if "roas_accuracy" in self.metrics:
                            est_roas = model.predict_roas(fit)
                            true_roas = summary.get("channel_roas", {})
                            if true_roas:
                                rr = roas_eval.evaluate(est_roas, true_roas)
                                row["roas_mape"] = (
                                    np.mean(list(rr.per_channel_mape.values()))
                                    if rr.per_channel_mape
                                    else None
                                )
                                row["roas_ranking"] = rr.ranking_correlation

                    except Exception as e:
                        row["error"] = str(e)

                    results.append(row)

        return ComparisonReport(results=results)
