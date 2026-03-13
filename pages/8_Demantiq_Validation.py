"""Demantiq Validation Page.

Run statistical realism checks on generated or uploaded data.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Demantiq Validation", layout="wide")

from demantiq.calibration.realism_validator import RealismValidator

st.title("Demantiq Realism Validation")

st.markdown("""
Validate that synthetic MMM data passes statistical realism checks.
The validator runs **7 tests** covering spend distributions, autocorrelation,
outcome variability, spend-to-outcome ratios, outliers, channel collinearity,
and seasonality presence.
""")

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
st.subheader("Data Source")
data_source = st.radio(
    "Choose data to validate",
    ["Use generated results", "Upload CSV"],
    horizontal=True,
)

df_to_validate = None

if data_source == "Use generated results":
    if "demantiq_result" in st.session_state:
        result = st.session_state["demantiq_result"]
        df_to_validate = result.observable_data
        st.info(f"Using generated data: {len(df_to_validate)} rows, {len(df_to_validate.columns)} columns")
        with st.expander("Data preview"):
            st.dataframe(df_to_validate.head(20), use_container_width=True)
    else:
        st.warning("No generated results available. Run a simulation on the **Demantiq Generate** page first, or upload a CSV.")
        st.stop()
else:
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df_to_validate = pd.read_csv(uploaded)
            st.info(f"Uploaded data: {len(df_to_validate)} rows, {len(df_to_validate.columns)} columns")
            with st.expander("Data preview"):
                st.dataframe(df_to_validate.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
    else:
        st.info("Upload a CSV file to validate.")
        st.stop()

# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------
if df_to_validate is not None and st.button("Run Validation", type="primary"):
    try:
        validator = RealismValidator()
        report = validator.validate(df_to_validate)
        st.session_state["validation_report"] = report
    except Exception as e:
        st.error(f"Validation failed: {e}")
        st.stop()

if "validation_report" not in st.session_state:
    st.info("Click **Run Validation** to check the data.")
    st.stop()

report = st.session_state["validation_report"]

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
st.subheader("Validation Results")

# Overall verdict
if report.pass_fail:
    st.success("All tests passed.")
else:
    st.error(f"Validation failed. Flagged: {', '.join(report.flagged_properties)}")

# Individual test results
st.markdown("### Test Details")

test_names_display = {
    "spend_non_negative": "Spend Non-Negative",
    "spend_autocorrelation": "Spend Autocorrelation",
    "outcome_variability": "Outcome Variability (CV)",
    "spend_outcome_ratio": "Spend-to-Outcome Ratio",
    "outlier_frequency": "Outlier Frequency",
    "channel_collinearity": "Channel Collinearity",
    "seasonality_presence": "Seasonality Presence",
}

for test_name, test_result in report.details.items():
    display_name = test_names_display.get(test_name, test_name)

    if test_result.passed:
        icon = ":green[PASS]"
    else:
        icon = ":red[FAIL]"

    with st.expander(f"{icon} {display_name}"):
        col1, col2 = st.columns([1, 3])
        col1.markdown(f"**Status:** {icon}")
        col2.markdown(f"**Detail:** {test_result.detail}")

        if test_result.p_value is not None:
            st.markdown(f"**p-value:** {test_result.p_value:.4f}")

        # Additional context per test
        if test_name == "spend_non_negative":
            st.markdown("Checks that no spend column contains negative values.")
        elif test_name == "spend_autocorrelation":
            st.markdown("Spend should have reasonable lag-1 autocorrelation (campaigns are persistent). Fails if mean AC < -0.2.")
        elif test_name == "outcome_variability":
            st.markdown("Outcome coefficient of variation should be between 0.01 and 2.0.")
        elif test_name == "spend_outcome_ratio":
            st.markdown("Total spend / total outcome should be in a plausible range (0.001 - 100.0).")
        elif test_name == "outlier_frequency":
            st.markdown("No more than 5% of outcome values should be extreme outliers (beyond Q1/Q3 +/- 3*IQR).")
        elif test_name == "channel_collinearity":
            st.markdown("No pair of spend channels should have |correlation| > 0.98.")
        elif test_name == "seasonality_presence":
            st.markdown("At least 1% of spectral energy should be outside the DC component (some periodicity expected).")

# Summary table
st.markdown("### Summary")
summary_rows = []
for test_name, test_result in report.details.items():
    summary_rows.append({
        "Test": test_names_display.get(test_name, test_name),
        "Passed": test_result.passed,
        "Detail": test_result.detail,
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Passed": st.column_config.CheckboxColumn("Passed"),
    },
)
