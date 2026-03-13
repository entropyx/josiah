"""Demantiq Scenario Library Browser.

Browse, inspect, and run all 15 named benchmark scenarios.
"""

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Demantiq Scenarios", layout="wide")

from demantiq import ScenarioLibrary, SimulationConfig
from demantiq.scenarios.difficulty_scorer import score_difficulty, difficulty_components
from demantiq.core.demand_kernel import simulate

st.title("Demantiq Scenario Library")

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
# Build scenario table
# ---------------------------------------------------------------------------
scenario_names = ScenarioLibrary.list_scenarios()
rows = []
for name in scenario_names:
    cfg = ScenarioLibrary.get(name)
    diff = score_difficulty(cfg)
    features = []
    if cfg.pricing is not None:
        features.append("Pricing")
    if cfg.distribution is not None:
        features.append("Distribution")
    if cfg.endogeneity is not None:
        features.append("Endogeneity")
    if cfg.competition is not None:
        features.append("Competition")
    if cfg.macro is not None:
        features.append("Macro")
    if cfg.interactions is not None:
        features.append("Interactions")
    rows.append({
        "Name": name,
        "Description": SCENARIO_DESCRIPTIONS.get(name, ""),
        "Periods": cfg.n_periods,
        "Channels": len(cfg.channels),
        "Granularity": cfg.granularity,
        "Difficulty": round(diff, 3),
        "Features": ", ".join(features) if features else "Base only",
    })

df_scenarios = pd.DataFrame(rows)

st.subheader("All Scenarios")
st.dataframe(
    df_scenarios,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Difficulty": st.column_config.ProgressColumn(
            "Difficulty",
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        ),
    },
)

# ---------------------------------------------------------------------------
# Difficulty overview chart
# ---------------------------------------------------------------------------
fig_diff = px.bar(
    df_scenarios.sort_values("Difficulty", ascending=False),
    x="Name",
    y="Difficulty",
    color="Difficulty",
    color_continuous_scale="RdYlGn_r",
    title="Scenario Difficulty Scores",
    range_color=[0, 1],
)
fig_diff.update_layout(height=350)
st.plotly_chart(fig_diff, use_container_width=True)

# ---------------------------------------------------------------------------
# Scenario detail view
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Scenario Detail")

selected = st.selectbox(
    "Select a scenario",
    scenario_names,
    format_func=lambda s: f"{s} ({SCENARIO_DESCRIPTIONS.get(s, '')})",
)

if selected:
    cfg = ScenarioLibrary.get(selected)
    diff = score_difficulty(cfg)
    components = difficulty_components(cfg)

    col1, col2, col3 = st.columns(3)
    col1.metric("Periods", cfg.n_periods)
    col2.metric("Channels", len(cfg.channels))
    col3.metric("Difficulty", f"{diff:.3f}")

    # Difficulty breakdown
    with st.expander("Difficulty Breakdown"):
        comp_df = pd.DataFrame({
            "Component": list(components.keys()),
            "Score": list(components.values()),
        })
        fig_comp = px.bar(
            comp_df,
            x="Component",
            y="Score",
            color="Score",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 1],
            title="Difficulty Components",
        )
        fig_comp.update_layout(height=300)
        st.plotly_chart(fig_comp, use_container_width=True)

    # Full config
    with st.expander("Full Configuration"):
        st.json(cfg.to_dict())

    # Channel table
    with st.expander("Channel Details"):
        ch_rows = []
        for ch in cfg.channels:
            ch_rows.append({
                "Name": ch.name,
                "Beta": ch.beta,
                "Saturation": ch.saturation_fn,
                "Adstock": ch.adstock_fn,
                "Spend Mean": ch.spend_mean,
                "Spend Std": ch.spend_std,
                "Pattern": ch.spend_pattern,
                "Group": ch.correlation_group,
            })
        st.dataframe(pd.DataFrame(ch_rows), use_container_width=True, hide_index=True)

    # Load & Generate
    lc1, lc2 = st.columns(2)
    if lc1.button("Load Config", key="load_scenario_cfg"):
        st.session_state["demantiq_config"] = cfg
        st.success(f"Loaded config for scenario: {selected}")

    if lc2.button("Load & Generate", type="primary", key="load_gen_scenario"):
        st.session_state["demantiq_config"] = cfg
        try:
            with st.spinner("Running simulation..."):
                result = simulate(cfg)
            st.session_state["demantiq_result"] = result
            st.success(f"Generated data for scenario: {selected}")

            # Inline results
            obs = result.observable_data
            gt = result.ground_truth

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=obs["date"], y=obs["y"],
                name="y (observed)", mode="lines",
            ))
            if "true_baseline" in gt.columns:
                fig_ts.add_trace(go.Scatter(
                    x=gt["date"], y=gt["true_baseline"],
                    name="Baseline (true)", mode="lines",
                    line=dict(dash="dash"),
                ))
            fig_ts.update_layout(title=f"Demand Over Time - {selected}", height=400)
            st.plotly_chart(fig_ts, use_container_width=True)

            contrib_cols = [c for c in gt.columns if c.startswith("true_") and c.endswith("_contribution")]
            if contrib_cols:
                total_contribs = {}
                for col in contrib_cols:
                    ch_name = col.replace("true_", "").replace("_contribution", "")
                    total_contribs[ch_name] = float(gt[col].sum())

                fig_bar = px.bar(
                    x=list(total_contribs.keys()),
                    y=list(total_contribs.values()),
                    labels={"x": "Channel", "y": "Total Contribution"},
                    title="Channel Contributions",
                    color=list(total_contribs.keys()),
                )
                fig_bar.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Mean y", f"{obs['y'].mean():.1f}")
            mcol2.metric("Std y", f"{obs['y'].std():.1f}")
            mcol3.metric("Min y", f"{obs['y'].min():.1f}")
            mcol4.metric("Max y", f"{obs['y'].max():.1f}")

        except Exception as e:
            st.error(f"Simulation failed: {e}")
