"""Demantiq Monte Carlo Runner Page.

Run batch simulations with multiple seeds and analyze distributions.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Demantiq Monte Carlo", layout="wide")

from demantiq import ScenarioLibrary, MonteCarloRunner
from demantiq.config.simulation_config import SimulationConfig
from demantiq.core.demand_kernel import simulate

st.title("Demantiq Monte Carlo Runner")

# ---------------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------------
st.subheader("Select Scenarios")

source = st.radio(
    "Scenario source",
    ["Named Scenarios", "Custom Config (from Config page)"],
    horizontal=True,
)

configs: list[SimulationConfig] = []
config_labels: list[str] = []

if source == "Named Scenarios":
    scenario_names = ScenarioLibrary.list_scenarios()
    selected_scenarios = st.multiselect(
        "Select scenarios to run",
        scenario_names,
        default=["clean_room"],
    )
    for name in selected_scenarios:
        configs.append(ScenarioLibrary.get(name))
        config_labels.append(name)
else:
    if "demantiq_config" in st.session_state:
        configs.append(st.session_state["demantiq_config"])
        config_labels.append("Custom Config")
        st.info(f"Using custom config: {configs[0].n_periods} periods, {len(configs[0].channels)} channels")
    else:
        st.warning("No custom configuration loaded. Go to **Demantiq Config** page first.")
        st.stop()

if not configs:
    st.info("Select at least one scenario to run Monte Carlo.")
    st.stop()

# ---------------------------------------------------------------------------
# Monte Carlo parameters
# ---------------------------------------------------------------------------
st.subheader("Monte Carlo Settings")
mc_col1, mc_col2, mc_col3 = st.columns(3)
n_seeds = mc_col1.slider("Seeds per scenario", 2, 50, 10)
base_seed = mc_col2.number_input("Base seed", value=0, step=1)
n_workers = mc_col3.number_input("Parallel workers (0 = auto)", value=1, step=1, min_value=0)

effective_workers = n_workers if n_workers > 0 else None
total_runs = len(configs) * n_seeds
st.info(f"Total runs: {total_runs} ({len(configs)} scenarios x {n_seeds} seeds)")

# ---------------------------------------------------------------------------
# Run Monte Carlo
# ---------------------------------------------------------------------------
if st.button("Run Monte Carlo", type="primary"):
    try:
        runner = MonteCarloRunner(
            configs=configs,
            n_seeds_per_scenario=n_seeds,
            base_seed=base_seed,
            n_workers=effective_workers,
        )

        progress_bar = st.progress(0, text="Running simulations...")

        with st.spinner(f"Running {total_runs} simulations..."):
            mc_results = runner.run()

        progress_bar.progress(100, text="Complete.")
        st.session_state["mc_results"] = mc_results
        st.session_state["mc_labels"] = config_labels
        st.success(f"Completed: {mc_results.n_success} successful, {mc_results.n_failed} failed.")

    except Exception as e:
        st.error(f"Monte Carlo run failed: {e}")
        st.stop()

if "mc_results" not in st.session_state:
    st.info("Click **Run Monte Carlo** to start batch simulation.")
    st.stop()

mc_results = st.session_state["mc_results"]
mc_labels = st.session_state.get("mc_labels", [])

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
st.subheader("Results")

if mc_results.n_failed > 0:
    with st.expander(f"Failures ({mc_results.n_failed})"):
        for scn_idx, seed, err in mc_results.failures:
            label = mc_labels[scn_idx] if scn_idx < len(mc_labels) else f"Scenario {scn_idx}"
            st.error(f"{label} (seed={seed}): {err}")

# Summary table
tab1, tab2, tab3 = st.tabs(["Summary Table", "Distribution Plots", "Per-Run Channel Contributions"])

with tab1:
    summary = mc_results.summary.copy()
    if not summary.empty:
        # Add scenario names
        if mc_labels:
            summary["scenario_name"] = summary["scenario_index"].map(
                lambda idx: mc_labels[idx] if idx < len(mc_labels) else f"Scenario {idx}"
            )
            cols = ["scenario_name"] + [c for c in summary.columns if c != "scenario_name"]
            summary = summary[cols]

        st.dataframe(summary, use_container_width=True, hide_index=True)

        csv_summary = summary.to_csv(index=False)
        st.download_button(
            "Download Summary (CSV)",
            csv_summary,
            file_name="demantiq_monte_carlo_summary.csv",
            mime="text/csv",
        )

with tab2:
    if not mc_results.summary.empty:
        summary_plot = mc_results.summary.copy()
        if mc_labels:
            summary_plot["scenario"] = summary_plot["scenario_index"].map(
                lambda idx: mc_labels[idx] if idx < len(mc_labels) else f"Scenario {idx}"
            )
        else:
            summary_plot["scenario"] = summary_plot["scenario_index"].astype(str)

        # y_mean distribution
        fig_mean = px.histogram(
            summary_plot,
            x="y_mean",
            color="scenario",
            barmode="overlay",
            title="Distribution of y_mean Across Seeds",
            nbins=20,
            opacity=0.7,
        )
        fig_mean.update_layout(height=350)
        st.plotly_chart(fig_mean, use_container_width=True)

        # y_std distribution
        fig_std = px.histogram(
            summary_plot,
            x="y_std",
            color="scenario",
            barmode="overlay",
            title="Distribution of y_std Across Seeds",
            nbins=20,
            opacity=0.7,
        )
        fig_std.update_layout(height=350)
        st.plotly_chart(fig_std, use_container_width=True)

        # Box plot of y_mean by scenario
        fig_box = px.box(
            summary_plot,
            x="scenario",
            y="y_mean",
            title="y_mean Distribution by Scenario",
            color="scenario",
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    if mc_results.results:
        st.markdown("**Per-run channel contribution totals (first 50 runs)**")
        contrib_rows = []
        for scn_idx, seed, result in mc_results.results[:50]:
            gt = result.ground_truth
            contrib_cols = [c for c in gt.columns if c.startswith("true_") and c.endswith("_contribution")]
            row = {
                "scenario": mc_labels[scn_idx] if scn_idx < len(mc_labels) else f"Scenario {scn_idx}",
                "seed": seed,
            }
            for col in contrib_cols:
                ch_name = col.replace("true_", "").replace("_contribution", "")
                row[ch_name] = float(gt[col].sum())
            contrib_rows.append(row)

        if contrib_rows:
            contrib_df = pd.DataFrame(contrib_rows)
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)

            csv_contrib = contrib_df.to_csv(index=False)
            st.download_button(
                "Download Contributions (CSV)",
                csv_contrib,
                file_name="demantiq_mc_contributions.csv",
                mime="text/csv",
            )
