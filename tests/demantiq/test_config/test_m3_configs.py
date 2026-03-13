from demantiq.config.endogeneity_config import EndogeneityConfig
from demantiq.config.competition_config import CompetitionConfig
from demantiq.config.macro_config import MacroConfig, MacroVariable, RegimeChange


def test_endogeneity_defaults():
    config = EndogeneityConfig()
    assert config.overall_strength == 0.3
    assert config.feedback_lag == 1


def test_endogeneity_roundtrip():
    config = EndogeneityConfig(overall_strength=0.7, feedback_lag=2)
    d = config.to_dict()
    restored = EndogeneityConfig.from_dict(d)
    assert restored.overall_strength == 0.7


def test_competition_defaults():
    config = CompetitionConfig()
    assert config.n_competitors == 2
    assert config.competitor_sov_pattern == "stable"


def test_competition_roundtrip():
    config = CompetitionConfig(n_competitors=5, sov_suppression_coefficient=0.3)
    d = config.to_dict()
    restored = CompetitionConfig.from_dict(d)
    assert restored.n_competitors == 5


def test_macro_config():
    var = MacroVariable(name="gdp", effect_on_demand=100.0, time_series_type="trending")
    rc = RegimeChange(period=52, magnitude=-0.4)
    config = MacroConfig(variables=[var], regime_changes=[rc])
    d = config.to_dict()
    restored = MacroConfig.from_dict(d)
    assert len(restored.variables) == 1
    assert restored.variables[0].name == "gdp"
    assert len(restored.regime_changes) == 1
