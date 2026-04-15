from demantiq.scenarios.scenario_sampler import ScenarioSampler


def test_sampler_generates_varying_cpm():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    configs = sampler.sample(10)
    all_cpms = []
    for config in configs:
        for ch in config.channels:
            all_cpms.append(ch.cpm)
    assert min(all_cpms) < max(all_cpms), "CPMs should vary across channels"
    assert all(0.5 < cpm < 100.0 for cpm in all_cpms), "CPMs should be in realistic range"


def test_sampler_generates_varying_ctr():
    sampler = ScenarioSampler(seed=42, channel_range=(3, 5))
    configs = sampler.sample(10)
    all_ctrs = []
    for config in configs:
        for ch in config.channels:
            all_ctrs.append(ch.ctr)
    assert min(all_ctrs) < max(all_ctrs), "CTRs should vary across channels"
    assert all(0.0001 < ctr < 0.2 for ctr in all_ctrs), "CTRs should be in realistic range"
