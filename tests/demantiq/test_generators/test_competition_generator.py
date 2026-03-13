import numpy as np
from demantiq.config.competition_config import CompetitionConfig
from demantiq.generators.competition_generator import generate_competition
from demantiq.utils.random import create_rng


def test_stable_sov():
    config = CompetitionConfig(competitor_sov_mean=0.3, competitor_sov_pattern="stable")
    result = generate_competition(config, 104, create_rng(42))
    assert abs(np.mean(result.competitor_sov) - 0.3) < 0.05


def test_seasonal_sov():
    config = CompetitionConfig(competitor_sov_mean=0.3, competitor_sov_pattern="seasonal")
    result = generate_competition(config, 104, create_rng(42))
    assert result.competitor_sov.shape == (104,)


def test_suppression_negative():
    config = CompetitionConfig(sov_suppression_coefficient=0.2)
    result = generate_competition(config, 104, create_rng(42))
    assert np.all(result.competition_effect <= 0)


def test_increasing_intensity():
    config = CompetitionConfig(
        competitor_sov_mean=0.3, competitive_intensity_trend="increasing"
    )
    result = generate_competition(config, 104, create_rng(42))
    assert np.mean(result.competitor_sov[80:]) > np.mean(result.competitor_sov[:20])


def test_sov_range():
    config = CompetitionConfig(competitor_sov_mean=0.3)
    result = generate_competition(config, 104, create_rng(42))
    assert np.all(result.competitor_sov >= 0)
    assert np.all(result.competitor_sov <= 1)
