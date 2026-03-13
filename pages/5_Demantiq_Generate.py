"""Demantiq Generate & Preview Page.

Runs simulation from the current config and displays results.
"""

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Demantiq Generate", layout="wide")

from demantiq.core.demand_kernel import simulate

st.title("Demantiq Generate & Preview")

# ---------------------------------------------------------------------------
# Check for config
# ---------------------------------------------------------------------------
if "demantiq_config" not in st.session_state:
    st.warning("No configuration loaded. Go to **Demantiq Config** page first.")
    st.stop()

config = st.session_state["demantiq_config"]

# ---------------------------------------------------------------------------
# Config summary
# ---------------------------------------------------------------------------
with st.expander("Current Configuration Summary", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Periods", config.n_periods)
    col2.metric("Granularity", config.granularity)
    col3.metric("Channels", len(config.channels))
    col4.metric("Seed", config.seed)

    features = []
    if config.pricing is not None:
        features.append("Pricing")
    if config.distribution is not None:
        features.append("Distribution")
    if config.endogeneity is not None:
        features.append("Endogeneity")
    if config.competition is not None:
        features.append("Competition")
    if config.macro is not None:
        features.append("Macro")
    if config.interactions is not None:
        features.append("Interactions")
    if features:
        st.info(f"Active features: {', '.join(features)}")
    else:
        st.info("Base simulation (no optional features)")

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
if st.button("Run Simulation", type="primary"):
    try:
        with st.spinner("Running simulation..."):
            result = simulate(config)
        st.session_state["demantiq_result"] = result
        st.success("Simulation complete.")
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        st.stop()

if "demantiq_result" not in st.session_state:
    st.info("Click **Run Simulation** to generate data.")
    st.stop()

result = st.session_state["demantiq_result"]

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
st.subheader("Results")

tab1, tab2, tab3 = st.tabs(["Observable Data", "Ground Truth", "Charts"])

with tab1:
    st.markdown("**Observable Data** (what the model sees)")
    st.dataframe(result.observable_data, use_container_width=True, height=400)

    csv_obs = result.observable_data.to_csv(index=False)
    st.download_button(
        "Download Observable Data (CSV)",
        csv_obs,
        file_name="demantiq_observable_data.csv",
        mime="text/csv",
    )

with tab2:
    gt_tab1, gt_tab2 = st.tabs(["Ground Truth Table", "Summary Truth (JSON)"])

    with gt_tab1:
        st.markdown("**Ground Truth** (per-period true contributions)")
        st.dataframe(result.ground_truth, use_container_width=True, height=400)

        csv_gt = result.ground_truth.to_csv(index=False)
        st.download_button(
            "Download Ground Truth (CSV)",
            csv_gt,
            file_name="demantiq_ground_truth.csv",
            mime="text/csv",
        )

    with gt_tab2:
        st.markdown("**Summary Truth** (aggregate parameters)")

        # Make summary JSON-serializable
        summary_clean = {}
        for k, v in result.summary_truth.items():
            if k == "config":
                continue  # skip full config in display
            try:
                json.dumps(v)
                summary_clean[k] = v
            except (TypeError, ValueError):
                summary_clean[k] = str(v)

        st.json(summary_clean)

        summary_json = json.dumps(summary_clean, indent=2, default=str)
        st.download_button(
            "Download Summary Truth (JSON)",
            summary_json,
            file_name="demantiq_summary_truth.json",
            mime="application/json",
        )

with tab3:
    # Time series chart: y over time with baseline overlay
    obs = result.observable_data
    gt = result.ground_truth

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=obs["date"], y=obs["y"],
        name="y (observed)", mode="lines",
        line=dict(color="#1f77b4"),
    ))
    if "true_baseline" in gt.columns:
        fig_ts.add_trace(go.Scatter(
            x=gt["date"], y=gt["true_baseline"],
            name="Baseline (true)", mode="lines",
            line=dict(color="#ff7f0e", dash="dash"),
        ))
    fig_ts.update_layout(
        title="Observed Demand Over Time",
        xaxis_title="Date",
        yaxis_title="Demand (y)",
        height=450,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Per-channel contribution bar chart
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
            title="Total Channel Contributions (Ground Truth)",
            color=list(total_contribs.keys()),
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Key metrics
    st.subheader("Key Metrics")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Mean y", f"{obs['y'].mean():.1f}")
    mcol2.metric("Std y", f"{obs['y'].std():.1f}")
    mcol3.metric("Min y", f"{obs['y'].min():.1f}")
    mcol4.metric("Max y", f"{obs['y'].max():.1f}")

    if "true_roas" in result.summary_truth:
        st.subheader("True ROAS by Channel")
        roas = result.summary_truth["true_roas"]
        roas_df = pd.DataFrame({"Channel": list(roas.keys()), "ROAS": list(roas.values())})
        st.dataframe(roas_df, use_container_width=True, hide_index=True)
