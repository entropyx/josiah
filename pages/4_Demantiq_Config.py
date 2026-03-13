"""Demantiq Configuration Page.

Allows loading named scenarios or building custom SimulationConfig from scratch.
"""

import streamlit as st

st.set_page_config(page_title="Demantiq Config", layout="wide")

from demantiq import SimulationConfig, ChannelConfig, BaselineConfig, NoiseConfig, ScenarioLibrary
from demantiq.config.pricing_config import PricingConfig, CostConfig
from demantiq.config.distribution_config import DistributionConfig
from demantiq.config.endogeneity_config import EndogeneityConfig
from demantiq.config.competition_config import CompetitionConfig
from demantiq.config.macro_config import MacroConfig, MacroVariable, RegimeChange
from demantiq.config.interaction_config import InteractionConfig
from demantiq.scenarios.difficulty_scorer import score_difficulty

st.title("Demantiq Configuration")

SCENARIO_DESCRIPTIONS = {
    "clean_room": "No endogeneity, low noise, orthogonal spend, 156 weeks. Easiest recovery benchmark.",
    "real_world": "Moderate endogeneity, correlated channels, 8 channels, 104 weeks.",
    "adversarial": "High endogeneity, heavy collinearity, low SNR, 12 channels, 52 weeks.",
    "pricing_dominant": "High price elasticity (-2.5), heavy promos, small channel betas.",
    "interaction_heavy": "Large price_x_media and distribution_x_media interactions.",
    "short_data": "Only 36 weeks of data, moderate complexity, 5 channels.",
    "many_channels": "18 channels with moderate collinearity, 104 weeks.",
    "regime_shift": "156 weeks with a macro regime change at period 78.",
    "new_brand": "Distribution starts low and grows; increasing baseline trend.",
    "mature_market": "High saturation so channels are near saturation. Flat trend.",
    "promotional_trap": "Heavy promo calendar, price_x_media interaction, correlated spend.",
    "platform_bias": "High algorithmic targeting bias and performance chasing.",
    "competitor_entry": "Competition SOV doubles at week 60 via regime change.",
    "dtc_pure_play": "No distribution, daily granularity, 365 periods.",
    "omnichannel_retail": "20 channels, all features enabled, 156 weeks. Kitchen sink scenario.",
}

# ---------------------------------------------------------------------------
# Option A vs B
# ---------------------------------------------------------------------------

config_mode = st.radio(
    "Configuration mode",
    ["Load Named Scenario", "Custom Configuration"],
    horizontal=True,
)

if config_mode == "Load Named Scenario":
    st.subheader("Named Scenarios")
    scenario_names = ScenarioLibrary.list_scenarios()
    selected = st.selectbox(
        "Scenario",
        scenario_names,
        format_func=lambda s: f"{s} - {SCENARIO_DESCRIPTIONS.get(s, '')}",
    )
    if selected:
        cfg = ScenarioLibrary.get(selected)
        diff = score_difficulty(cfg)
        col1, col2, col3 = st.columns(3)
        col1.metric("Periods", cfg.n_periods)
        col2.metric("Channels", len(cfg.channels))
        col3.metric("Difficulty", f"{diff:.2f}")
        st.info(SCENARIO_DESCRIPTIONS.get(selected, ""))

        with st.expander("Full config details"):
            st.json(cfg.to_dict())

        if st.button("Load Scenario", type="primary"):
            st.session_state["demantiq_config"] = cfg
            st.success(f"Loaded scenario: {selected}")

else:
    # ------------------------------------------------------------------
    # Custom Configuration Builder
    # ------------------------------------------------------------------
    st.subheader("Custom Configuration")

    # --- Basic settings ---
    with st.expander("Basic Settings", expanded=True):
        bcol1, bcol2, bcol3 = st.columns(3)
        n_periods = bcol1.slider("Number of periods", 26, 520, 104)
        granularity = bcol2.selectbox("Granularity", ["weekly", "daily"])
        seed = bcol3.number_input("Random seed", value=42, step=1)

    # --- Baseline ---
    with st.expander("Baseline", expanded=True):
        blcol1, blcol2 = st.columns(2)
        trend_type = blcol1.selectbox("Trend type", ["linear", "cube_root"])
        trend_slope = blcol2.number_input("Trend slope", value=1.0, format="%.2f")
        blcol3, blcol4 = st.columns(2)
        seasonality_n_terms = blcol3.slider("Fourier terms", 1, 10, 2)
        seasonality_period = blcol4.number_input(
            "Seasonality period",
            value=52.0 if granularity == "weekly" else 365.0,
            format="%.1f",
        )
        organic_level = st.number_input("Organic level (intercept)", value=1000.0, format="%.1f")

    baseline_cfg = BaselineConfig(
        trend_type=trend_type,
        trend_params={"slope": trend_slope},
        seasonality_type="fourier",
        seasonality_period=seasonality_period,
        seasonality_n_terms=seasonality_n_terms,
        organic_level=organic_level,
    )

    # --- Channels ---
    with st.expander("Channels", expanded=True):
        if "custom_channels" not in st.session_state:
            st.session_state["custom_channels"] = [
                {"name": "facebook", "beta": 200.0, "saturation_fn": "logistic",
                 "sat_k": 3.0, "sat_x0": 0.5, "sat_K": 0.5, "sat_S": 2.0,
                 "adstock_fn": "geometric", "ads_alpha": 0.5, "ads_max_lag": 8,
                 "spend_pattern": "always_on", "spend_mean": 10000.0, "spend_std": 3000.0,
                 "spend_floor": 0.0, "correlation_group": "default"},
                {"name": "google", "beta": 400.0, "saturation_fn": "logistic",
                 "sat_k": 3.0, "sat_x0": 0.5, "sat_K": 0.5, "sat_S": 2.0,
                 "adstock_fn": "geometric", "ads_alpha": 0.5, "ads_max_lag": 8,
                 "spend_pattern": "always_on", "spend_mean": 10000.0, "spend_std": 3000.0,
                 "spend_floor": 0.0, "correlation_group": "default"},
            ]

        AVAILABLE_CHANNELS = [
            "facebook", "google", "tiktok", "pinterest", "email",
            "youtube", "snapchat", "linkedin", "twitter", "display",
            "programmatic", "podcast", "influencer", "radio", "ctv",
            "affiliate", "sms", "direct_mail", "ooh", "print",
        ]

        add_col1, add_col2 = st.columns([3, 1])
        new_ch_name = add_col1.selectbox(
            "Add channel",
            [c for c in AVAILABLE_CHANNELS
             if c not in [ch["name"] for ch in st.session_state["custom_channels"]]],
            key="new_ch_select",
        )
        if add_col2.button("Add Channel"):
            st.session_state["custom_channels"].append({
                "name": new_ch_name, "beta": 100.0, "saturation_fn": "logistic",
                "sat_k": 3.0, "sat_x0": 0.5, "sat_K": 0.5, "sat_S": 2.0,
                "adstock_fn": "geometric", "ads_alpha": 0.5, "ads_max_lag": 8,
                "spend_pattern": "always_on", "spend_mean": 10000.0, "spend_std": 3000.0,
                "spend_floor": 0.0, "correlation_group": "default",
            })
            st.rerun()

        channels_to_remove = []
        for i, ch in enumerate(st.session_state["custom_channels"]):
            with st.container():
                st.markdown(f"**Channel: {ch['name']}**")
                cc1, cc2, cc3, cc4 = st.columns(4)
                ch["beta"] = cc1.number_input(f"Beta", value=ch["beta"], format="%.1f", key=f"beta_{i}")
                ch["saturation_fn"] = cc2.selectbox(
                    "Saturation fn",
                    ["hill", "logistic", "power", "piecewise_linear"],
                    index=["hill", "logistic", "power", "piecewise_linear"].index(ch["saturation_fn"]),
                    key=f"sat_fn_{i}",
                )
                ch["adstock_fn"] = cc3.selectbox(
                    "Adstock fn",
                    ["geometric", "weibull_cdf", "weibull_pdf", "delayed_geometric"],
                    index=["geometric", "weibull_cdf", "weibull_pdf", "delayed_geometric"].index(ch["adstock_fn"]),
                    key=f"ads_fn_{i}",
                )
                if cc4.button("Remove", key=f"remove_{i}"):
                    channels_to_remove.append(i)

                # Saturation params
                sc1, sc2 = st.columns(2)
                if ch["saturation_fn"] == "hill":
                    ch["sat_K"] = sc1.number_input("K (half-max)", value=ch.get("sat_K", 0.5), format="%.3f", key=f"sat_K_{i}")
                    ch["sat_S"] = sc2.number_input("S (slope)", value=ch.get("sat_S", 2.0), format="%.3f", key=f"sat_S_{i}")
                else:
                    ch["sat_k"] = sc1.number_input("k (steepness)", value=ch.get("sat_k", 3.0), format="%.3f", key=f"sat_k_{i}")
                    ch["sat_x0"] = sc2.number_input("x0 (midpoint)", value=ch.get("sat_x0", 0.5), format="%.3f", key=f"sat_x0_{i}")

                # Adstock params
                ac1, ac2 = st.columns(2)
                ch["ads_alpha"] = ac1.number_input("Alpha (decay)", value=ch.get("ads_alpha", 0.5), format="%.3f", key=f"ads_alpha_{i}")
                ch["ads_max_lag"] = int(ac2.number_input("Max lag", value=ch.get("ads_max_lag", 8), step=1, key=f"ads_lag_{i}"))

                # Spend params
                spc1, spc2, spc3, spc4 = st.columns(4)
                ch["spend_pattern"] = spc1.selectbox(
                    "Spend pattern",
                    ["always_on", "pulsed", "seasonal", "front_loaded"],
                    index=["always_on", "pulsed", "seasonal", "front_loaded"].index(ch["spend_pattern"]),
                    key=f"sp_pat_{i}",
                )
                ch["spend_mean"] = spc2.number_input("Spend mean", value=ch["spend_mean"], format="%.0f", key=f"sp_mean_{i}")
                ch["spend_std"] = spc3.number_input("Spend std", value=ch["spend_std"], format="%.0f", key=f"sp_std_{i}")
                ch["spend_floor"] = spc4.number_input("Spend floor", value=ch["spend_floor"], format="%.0f", key=f"sp_floor_{i}")

                ch["correlation_group"] = st.text_input("Correlation group", value=ch.get("correlation_group", "default"), key=f"corr_grp_{i}")
                st.divider()

        # Remove channels marked for deletion
        for idx in sorted(channels_to_remove, reverse=True):
            st.session_state["custom_channels"].pop(idx)
        if channels_to_remove:
            st.rerun()

    # Build ChannelConfig objects
    channel_configs = []
    for ch in st.session_state["custom_channels"]:
        if ch["saturation_fn"] == "hill":
            sat_params = {"K": ch["sat_K"], "S": ch["sat_S"]}
        else:
            sat_params = {"k": ch["sat_k"], "x0": ch["sat_x0"]}
        channel_configs.append(ChannelConfig(
            name=ch["name"],
            beta=ch["beta"],
            saturation_fn=ch["saturation_fn"],
            saturation_params=sat_params,
            adstock_fn=ch["adstock_fn"],
            adstock_params={"alpha": ch["ads_alpha"], "max_lag": ch["ads_max_lag"]},
            spend_pattern=ch["spend_pattern"],
            spend_mean=ch["spend_mean"],
            spend_std=ch["spend_std"],
            spend_floor=ch["spend_floor"],
            correlation_group=ch["correlation_group"],
        ))

    # --- Noise ---
    with st.expander("Noise", expanded=True):
        ncol1, ncol2 = st.columns(2)
        noise_type = ncol1.selectbox("Noise type", ["gaussian", "t_distributed", "heteroscedastic", "autocorrelated"])
        noise_scale = ncol2.number_input("Noise scale", value=50.0, format="%.1f")
        ncol3, ncol4 = st.columns(2)
        snr = ncol3.number_input("Signal-to-noise ratio (0 = use noise_scale)", value=0.0, format="%.1f")
        autocorrelation = ncol4.number_input("Autocorrelation (AR1)", value=0.0, format="%.2f")
        ncol5, ncol6, ncol7 = st.columns(3)
        t_df = ncol5.number_input("t degrees of freedom", value=5.0, format="%.1f")
        outlier_prob = ncol6.number_input("Outlier probability", value=0.0, min_value=0.0, max_value=1.0, format="%.3f")
        outlier_mag = ncol7.number_input("Outlier magnitude", value=3.0, format="%.1f")

    noise_cfg = NoiseConfig(
        noise_type=noise_type,
        noise_scale=noise_scale,
        signal_to_noise_ratio=snr if snr > 0 else None,
        autocorrelation=autocorrelation,
        t_df=t_df,
        outlier_probability=outlier_prob,
        outlier_magnitude=outlier_mag,
    )

    # --- Optional: Pricing ---
    pricing_cfg = None
    with st.expander("Pricing (optional)"):
        enable_pricing = st.checkbox("Enable pricing", key="enable_pricing")
        if enable_pricing:
            pcol1, pcol2 = st.columns(2)
            base_price = pcol1.number_input("Base price", value=25.0, format="%.2f")
            price_elasticity = pcol2.number_input("Price elasticity", value=-1.2, format="%.2f")
            pcol3, pcol4 = st.columns(2)
            promo_frequency = pcol3.selectbox("Promo frequency", ["weekly", "biweekly", "monthly", "quarterly"])
            promo_depth_mean = pcol4.number_input("Promo depth mean", value=0.15, format="%.3f")
            pcol5, pcol6 = st.columns(2)
            promo_depth_std = pcol5.number_input("Promo depth std", value=0.05, format="%.3f")
            price_media_interaction = pcol6.number_input("Price-media interaction", value=0.0, format="%.3f")

            st.markdown("**Cost Structure**")
            ccol1, ccol2 = st.columns(2)
            cogs = ccol1.number_input("COGS per unit", value=5.0, format="%.2f")
            var_cost = ccol2.number_input("Variable cost per unit", value=2.0, format="%.2f")

            pricing_cfg = PricingConfig(
                base_price=base_price,
                price_elasticity=price_elasticity,
                promo_frequency=promo_frequency,
                promo_depth_mean=promo_depth_mean,
                promo_depth_std=promo_depth_std,
                price_media_interaction=price_media_interaction,
                cost_structure=CostConfig(cogs_per_unit=cogs, variable_cost_per_unit=var_cost),
            )

    # --- Optional: Distribution ---
    distribution_cfg = None
    with st.expander("Distribution (optional)"):
        enable_dist = st.checkbox("Enable distribution", key="enable_dist")
        if enable_dist:
            dcol1, dcol2 = st.columns(2)
            init_dist = dcol1.number_input("Initial distribution", value=0.8, min_value=0.0, max_value=1.0, format="%.2f")
            dist_trajectory = dcol2.selectbox("Trajectory", ["stable", "growing", "declining", "step_change"])
            dcol3, dcol4 = st.columns(2)
            dist_ceiling = dcol3.number_input("Distribution ceiling effect", value=0.0, format="%.2f")
            stockout_prob = dcol4.number_input("Stockout probability", value=0.0, min_value=0.0, max_value=1.0, format="%.3f")
            stockout_loss = st.number_input("Stockout demand loss", value=0.5, min_value=0.0, max_value=1.0, format="%.2f")

            traj_params = {}
            if dist_trajectory == "growing":
                traj_params["growth_rate"] = st.number_input("Growth rate", value=0.01, format="%.4f", key="dist_growth")
            elif dist_trajectory == "step_change":
                sc1, sc2 = st.columns(2)
                traj_params["step_period"] = int(sc1.number_input("Step period", value=52, step=1, key="dist_step_p"))
                traj_params["step_magnitude"] = sc2.number_input("Step magnitude", value=0.2, format="%.2f", key="dist_step_m")

            distribution_cfg = DistributionConfig(
                initial_distribution=init_dist,
                distribution_trajectory=dist_trajectory,
                trajectory_params=traj_params,
                distribution_ceiling_effect=dist_ceiling,
                stockout_probability=stockout_prob,
                stockout_demand_loss=stockout_loss,
            )

    # --- Optional: Endogeneity ---
    endogeneity_cfg = None
    with st.expander("Endogeneity (optional)"):
        enable_endo = st.checkbox("Enable endogeneity", key="enable_endo")
        if enable_endo:
            ecol1, ecol2 = st.columns(2)
            endo_strength = ecol1.slider("Overall strength", 0.0, 1.0, 0.3, 0.05)
            feedback_lag = int(ecol2.number_input("Feedback lag", value=1, step=1, min_value=1))
            ecol3, ecol4 = st.columns(2)
            seasonal_bias = ecol3.number_input("Seasonal allocation bias", value=0.0, format="%.2f")
            perf_chasing = ecol4.number_input("Performance chasing", value=0.0, format="%.2f")
            ecol5, ecol6 = st.columns(2)
            algo_bias = ecol5.number_input("Algorithmic targeting bias", value=0.0, format="%.2f")
            omit_strength = ecol6.number_input("Omitted variable strength", value=0.0, format="%.2f")
            omit_ar = st.number_input("Omitted variable AR(1)", value=0.7, format="%.2f")

            endogeneity_cfg = EndogeneityConfig(
                overall_strength=endo_strength,
                feedback_lag=feedback_lag,
                seasonal_allocation_bias=seasonal_bias,
                performance_chasing=perf_chasing,
                algorithmic_targeting_bias=algo_bias,
                omitted_variable_strength=omit_strength,
                omitted_variable_ar=omit_ar,
            )

    # --- Optional: Competition ---
    competition_cfg = None
    with st.expander("Competition (optional)"):
        enable_comp = st.checkbox("Enable competition", key="enable_comp")
        if enable_comp:
            cocol1, cocol2 = st.columns(2)
            n_competitors = int(cocol1.number_input("Number of competitors", value=2, step=1, min_value=1))
            comp_sov_mean = cocol2.number_input("Competitor SOV mean", value=0.3, format="%.2f")
            cocol3, cocol4 = st.columns(2)
            comp_sov_pattern = cocol3.selectbox("SOV pattern", ["stable", "seasonal", "reactive", "random"])
            sov_suppression = cocol4.number_input("SOV suppression coefficient", value=0.1, format="%.3f")
            comp_trend = st.selectbox("Competitive intensity trend", ["stable", "increasing", "decreasing"])

            competition_cfg = CompetitionConfig(
                n_competitors=n_competitors,
                competitor_sov_mean=comp_sov_mean,
                competitor_sov_pattern=comp_sov_pattern,
                sov_suppression_coefficient=sov_suppression,
                competitive_intensity_trend=comp_trend,
            )

    # --- Optional: Macro ---
    macro_cfg = None
    with st.expander("Macro Variables (optional)"):
        enable_macro = st.checkbox("Enable macro variables", key="enable_macro")
        if enable_macro:
            if "macro_vars" not in st.session_state:
                st.session_state["macro_vars"] = []
            if "regime_changes" not in st.session_state:
                st.session_state["regime_changes"] = []

            st.markdown("**Macro Variables**")
            if st.button("Add Variable", key="add_macro_var"):
                st.session_state["macro_vars"].append({
                    "name": "consumer_confidence",
                    "effect_on_demand": 50.0,
                    "time_series_type": "mean_reverting",
                    "correlation_with_spend": 0.0,
                })
                st.rerun()

            vars_to_remove = []
            for vi, mv in enumerate(st.session_state["macro_vars"]):
                mc1, mc2, mc3, mc4 = st.columns([2, 1, 1, 1])
                mv["name"] = mc1.text_input("Name", value=mv["name"], key=f"mv_name_{vi}")
                mv["effect_on_demand"] = mc2.number_input("Effect", value=mv["effect_on_demand"], format="%.1f", key=f"mv_eff_{vi}")
                mv["time_series_type"] = mc3.selectbox(
                    "Type",
                    ["mean_reverting", "random_walk", "trending", "seasonal"],
                    index=["mean_reverting", "random_walk", "trending", "seasonal"].index(mv["time_series_type"]),
                    key=f"mv_type_{vi}",
                )
                if mc4.button("Remove", key=f"rm_mv_{vi}"):
                    vars_to_remove.append(vi)
                mv["correlation_with_spend"] = st.number_input(
                    "Correlation with spend", value=mv["correlation_with_spend"],
                    format="%.2f", key=f"mv_corr_{vi}",
                )
            for idx in sorted(vars_to_remove, reverse=True):
                st.session_state["macro_vars"].pop(idx)
            if vars_to_remove:
                st.rerun()

            st.markdown("**Regime Changes**")
            if st.button("Add Regime Change", key="add_regime"):
                st.session_state["regime_changes"].append({
                    "period": 52, "change_type": "level_shift",
                    "magnitude": -0.2, "recovery": "permanent", "recovery_periods": 0,
                })
                st.rerun()

            rc_to_remove = []
            for ri, rc in enumerate(st.session_state["regime_changes"]):
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc["period"] = int(rc1.number_input("Period", value=rc["period"], step=1, key=f"rc_per_{ri}"))
                rc["change_type"] = rc2.selectbox("Type", ["level_shift"], key=f"rc_type_{ri}")
                rc["magnitude"] = rc3.number_input("Magnitude", value=rc["magnitude"], format="%.2f", key=f"rc_mag_{ri}")
                rc["recovery"] = rc4.selectbox(
                    "Recovery",
                    ["permanent", "gradual"],
                    index=["permanent", "gradual"].index(rc["recovery"]),
                    key=f"rc_rec_{ri}",
                )
                if rc["recovery"] == "gradual":
                    rc["recovery_periods"] = int(st.number_input(
                        "Recovery periods", value=rc["recovery_periods"], step=1, key=f"rc_rp_{ri}",
                    ))
                if st.button("Remove", key=f"rm_rc_{ri}"):
                    rc_to_remove.append(ri)
            for idx in sorted(rc_to_remove, reverse=True):
                st.session_state["regime_changes"].pop(idx)
            if rc_to_remove:
                st.rerun()

            macro_variables = [
                MacroVariable(
                    name=mv["name"],
                    effect_on_demand=mv["effect_on_demand"],
                    time_series_type=mv["time_series_type"],
                    correlation_with_spend=mv["correlation_with_spend"],
                )
                for mv in st.session_state["macro_vars"]
            ]
            regime_changes = [
                RegimeChange(
                    period=rc["period"],
                    change_type=rc["change_type"],
                    magnitude=rc["magnitude"],
                    recovery=rc["recovery"],
                    recovery_periods=rc.get("recovery_periods", 0),
                )
                for rc in st.session_state["regime_changes"]
            ]
            if macro_variables or regime_changes:
                macro_cfg = MacroConfig(variables=macro_variables, regime_changes=regime_changes)

    # --- Optional: Interactions ---
    interactions_cfg = None
    with st.expander("Interactions (optional)"):
        enable_interactions = st.checkbox("Enable interactions", key="enable_interactions")
        if enable_interactions:
            ch_names = [ch["name"] for ch in st.session_state["custom_channels"]]

            st.markdown("**Price x Media** (per channel)")
            pxm = {}
            for cn in ch_names:
                val = st.number_input(f"price_x_{cn}", value=0.0, format="%.3f", key=f"pxm_{cn}")
                if val != 0.0:
                    pxm[cn] = val

            st.markdown("**Distribution x Media** (per channel)")
            dxm = {}
            for cn in ch_names:
                val = st.number_input(f"dist_x_{cn}", value=0.0, format="%.3f", key=f"dxm_{cn}")
                if val != 0.0:
                    dxm[cn] = val

            st.markdown("**Competition x Media** (per channel)")
            cxm = {}
            for cn in ch_names:
                val = st.number_input(f"comp_x_{cn}", value=0.0, format="%.3f", key=f"cxm_{cn}")
                if val != 0.0:
                    cxm[cn] = val

            st.markdown("**Media x Media** (cross-channel synergy)")
            mxm = {}
            for idx_a in range(len(ch_names)):
                for idx_b in range(idx_a + 1, len(ch_names)):
                    key_pair = f"{ch_names[idx_a]}_{ch_names[idx_b]}"
                    val = st.number_input(f"{ch_names[idx_a]} x {ch_names[idx_b]}", value=0.0, format="%.3f", key=f"mxm_{key_pair}")
                    if val != 0.0:
                        mxm[(ch_names[idx_a], ch_names[idx_b])] = val

            if pxm or dxm or mxm or cxm:
                interactions_cfg = InteractionConfig(
                    price_x_media=pxm,
                    distribution_x_media=dxm,
                    media_x_media=mxm,
                    competition_x_media=cxm,
                )

    # --- Build and save ---
    if st.button("Save Configuration", type="primary"):
        config = SimulationConfig(
            n_periods=n_periods,
            granularity=granularity,
            channels=channel_configs,
            noise=noise_cfg,
            baseline=baseline_cfg,
            seed=seed,
            pricing=pricing_cfg,
            distribution=distribution_cfg,
            endogeneity=endogeneity_cfg,
            competition=competition_cfg,
            macro=macro_cfg,
            interactions=interactions_cfg,
        )
        st.session_state["demantiq_config"] = config
        diff = score_difficulty(config)
        st.success(f"Configuration saved. Difficulty score: {diff:.2f}")
        with st.expander("Config summary"):
            st.json(config.to_dict())

# --- Show current config status ---
st.divider()
if "demantiq_config" in st.session_state:
    cfg = st.session_state["demantiq_config"]
    st.info(f"Current config: {len(cfg.channels)} channels, {cfg.n_periods} periods, seed={cfg.seed}")
else:
    st.warning("No configuration loaded yet. Load a scenario or build a custom config above.")
