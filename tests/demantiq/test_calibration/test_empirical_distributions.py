"""Tests for EmpiricalDistributions."""

import json
import tempfile
import os

import pytest

from demantiq.calibration.empirical_distributions import (
    EmpiricalDistributions,
    Observation,
)


class TestDefaults:
    """Defaults are loaded on construction."""

    def test_defaults_loaded(self):
        ed = EmpiricalDistributions()
        assert len(ed.observations) > 0

    def test_default_beta_ranges_present(self):
        ed = EmpiricalDistributions()
        r = ed.get_range("paid_social_upper_funnel", "general", "beta_range")
        assert r is not None
        assert r == (100, 800)

    def test_default_price_elasticity(self):
        ed = EmpiricalDistributions()
        r = ed.get_range("pricing", "supplements", "price_elasticity")
        assert r == (-3.0, -1.0)

    def test_all_default_beta_ranges_are_tuples(self):
        ed = EmpiricalDistributions()
        beta_obs = [o for o in ed.observations if o.parameter == "beta_range"]
        assert len(beta_obs) >= 8
        for o in beta_obs:
            assert o.confidence_interval is not None
            lo, hi = o.confidence_interval
            assert lo < hi


class TestAddRetrieve:
    """Add observations and retrieve ranges."""

    def test_add_and_get_range(self):
        ed = EmpiricalDistributions()
        ed.add_observation("my_lever", "my_context", "roas", 3.5, (2.0, 5.0))
        r = ed.get_range("my_lever", "my_context", "roas")
        assert r == (2.0, 5.0)

    def test_add_without_ci_uses_min_max(self):
        ed = EmpiricalDistributions()
        ed.add_observation("lev", "ctx", "val", 10.0)
        ed.add_observation("lev", "ctx", "val", 20.0)
        r = ed.get_range("lev", "ctx", "val")
        # First match has no CI, so falls through to min/max
        assert r == (10.0, 20.0)

    def test_get_range_missing(self):
        ed = EmpiricalDistributions()
        assert ed.get_range("nope", "nope", "nope") is None

    def test_add_with_source_and_client(self):
        ed = EmpiricalDistributions()
        ed.add_observation(
            "lev", "ctx", "p", 1.0, source="paper_xyz", client_id="acme"
        )
        match = [
            o for o in ed.observations
            if o.source == "paper_xyz" and o.client_id == "acme"
        ]
        assert len(match) == 1


class TestPersistence:
    """JSON save/load round-trip."""

    def test_save_load_round_trip(self):
        ed = EmpiricalDistributions()
        ed.add_observation("custom", "test", "metric", 42.0, (40.0, 44.0), "test_src")

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name

        try:
            ed.save(path)

            loaded = EmpiricalDistributions.load(path)
            assert len(loaded.observations) == len(ed.observations)

            # Check the custom observation survived
            custom = [o for o in loaded.observations if o.lever_type == "custom"]
            assert len(custom) == 1
            assert custom[0].value == 42.0
            assert custom[0].confidence_interval == (40.0, 44.0)
            assert custom[0].source == "test_src"
        finally:
            os.unlink(path)

    def test_load_skips_defaults(self):
        """Loading from file should NOT include constructor defaults."""
        ed = EmpiricalDistributions()
        ed.add_observation("only_this", "ctx", "p", 1.0)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name

        try:
            ed.save(path)
            loaded = EmpiricalDistributions.load(path)
            # loaded has exactly the same count as the saved version
            assert len(loaded.observations) == len(ed.observations)
        finally:
            os.unlink(path)

    def test_saved_json_is_valid(self):
        ed = EmpiricalDistributions()
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            ed.save(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == len(ed.observations)
        finally:
            os.unlink(path)
