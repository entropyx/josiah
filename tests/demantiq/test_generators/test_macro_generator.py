import numpy as np
from demantiq.config.macro_config import MacroConfig, MacroVariable, RegimeChange
from demantiq.generators.macro_generator import generate_macro
from demantiq.utils.random import create_rng


def test_empty_config():
    config = MacroConfig()
    result = generate_macro(config, 104, create_rng(42))
    assert np.allclose(result.macro_effect, 0)


def test_single_variable():
    var = MacroVariable(
        name="gdp",
        effect_on_demand=100.0,
        time_series_type="trending",
        params={"slope": 0.1, "start": 0.0},
    )
    config = MacroConfig(variables=[var])
    result = generate_macro(config, 104, create_rng(42))
    assert "gdp" in result.variables
    assert result.variables["gdp"].shape == (104,)
    assert not np.allclose(result.macro_effect, 0)


def test_mean_reverting():
    var = MacroVariable(
        name="confidence",
        time_series_type="mean_reverting",
        params={"mean": 0.0, "phi": 0.8, "sigma": 0.1},
    )
    config = MacroConfig(variables=[var])
    result = generate_macro(config, 1000, create_rng(42))
    # Should stay near mean
    assert abs(np.mean(result.variables["confidence"])) < 0.5


def test_regime_change():
    rc = RegimeChange(period=50, change_type="level_shift", magnitude=-0.3)
    config = MacroConfig(regime_changes=[rc])
    result = generate_macro(config, 104, create_rng(42))
    assert result.regime_effects[50] != 0


def test_seasonal_variable():
    var = MacroVariable(
        name="season",
        time_series_type="seasonal",
        params={"period": 52, "amplitude": 2.0},
    )
    config = MacroConfig(variables=[var])
    result = generate_macro(config, 104, create_rng(42))
    assert result.variables["season"].shape == (104,)


def test_deterministic():
    var = MacroVariable(name="x", time_series_type="random_walk")
    config = MacroConfig(variables=[var])
    r1 = generate_macro(config, 104, create_rng(42))
    r2 = generate_macro(config, 104, create_rng(42))
    np.testing.assert_allclose(r1.variables["x"], r2.variables["x"])
