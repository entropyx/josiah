"""Empirical distributions for calibrating synthetic MMM data.

Stores observations of real-world parameter ranges (betas, elasticities, etc.)
and provides lookup/persistence for calibration workflows.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Observation:
    """A single empirical observation of a marketing parameter."""
    lever_type: str
    business_context: str
    parameter: str
    value: float
    confidence_interval: Optional[tuple] = None
    source: str = "internal_estimate"
    client_id: str = ""


class EmpiricalDistributions:
    """Registry of empirical parameter observations for calibration.

    Loads sensible defaults on init (PRD Appendix C benchmarks) and
    supports adding custom observations, querying ranges, and JSON
    round-trip persistence.
    """

    def __init__(self):
        self.observations: list[Observation] = []
        self._load_defaults()

    def _load_defaults(self):
        """Load initial benchmarks from PRD Appendix C."""
        defaults = {
            # Media channel beta ranges
            ("paid_social_upper_funnel", "general", "beta_range"): (100, 800),
            ("paid_search_brand", "general", "beta_range"): (200, 1200),
            ("paid_search_nonbrand", "general", "beta_range"): (50, 500),
            ("display_prospecting", "general", "beta_range"): (30, 300),
            ("display_retargeting", "general", "beta_range"): (100, 600),
            ("ctv", "general", "beta_range"): (50, 400),
            ("linear_tv", "general", "beta_range"): (150, 1000),
            ("email", "general", "beta_range"): (50, 250),
            # Price elasticities by category
            ("pricing", "supplements", "price_elasticity"): (-3.0, -1.0),
            ("pricing", "dtc_skincare", "price_elasticity"): (-2.5, -0.8),
            ("pricing", "qsr", "price_elasticity"): (-2.0, -0.5),
            ("pricing", "consumer_electronics", "price_elasticity"): (-4.0, -1.5),
            ("pricing", "cpg_fmcg", "price_elasticity"): (-3.5, -1.0),
        }
        for (lever, context, param), (lo, hi) in defaults.items():
            self.observations.append(
                Observation(lever, context, param, (lo + hi) / 2, (lo, hi))
            )

    def add_observation(
        self,
        lever_type: str,
        business_context: str,
        parameter: str,
        value: float,
        confidence_interval: Optional[tuple] = None,
        source: str = "internal_estimate",
        client_id: str = "",
    ):
        """Add a custom empirical observation."""
        self.observations.append(
            Observation(
                lever_type, business_context, parameter, value,
                confidence_interval, source, client_id,
            )
        )

    def get_range(self, lever_type: str, business_context: str, parameter: str):
        """Get (low, high) range for a parameter.

        Returns the confidence interval of the first match, or the
        min/max of all matching values if no CI is stored. Returns
        None if no matches found.
        """
        matches = [
            o for o in self.observations
            if o.lever_type == lever_type
            and o.business_context == business_context
            and o.parameter == parameter
        ]
        if not matches:
            return None
        if matches[0].confidence_interval:
            return matches[0].confidence_interval
        values = [o.value for o in matches]
        return (min(values), max(values))

    def save(self, path: str):
        """Persist all observations to a JSON file."""
        data = [
            {
                "lever_type": o.lever_type,
                "business_context": o.business_context,
                "parameter": o.parameter,
                "value": o.value,
                "confidence_interval": list(o.confidence_interval)
                if o.confidence_interval
                else None,
                "source": o.source,
                "client_id": o.client_id,
            }
            for o in self.observations
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EmpiricalDistributions":
        """Load observations from a JSON file (skips default loading)."""
        inst = cls.__new__(cls)
        inst.observations = []
        with open(path) as f:
            data = json.load(f)
        for item in data:
            ci = (
                tuple(item["confidence_interval"])
                if item.get("confidence_interval")
                else None
            )
            inst.observations.append(
                Observation(
                    item["lever_type"],
                    item["business_context"],
                    item["parameter"],
                    item["value"],
                    ci,
                    item.get("source", ""),
                    item.get("client_id", ""),
                )
            )
        return inst
