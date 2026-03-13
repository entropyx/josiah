import streamlit as st
import pandas as pd
from josiah.generator import generate_single
from josiah.visualization import plot_revenue_decomposition, plot_channel_spend
from josiah.export import export_single_to_bytes

st.set_page_config(page_title="Generate & Preview", layout="wide")
st.title("Generate & Preview")

if "configs" not in st.session_state or not st.session_state["configs"]:
    st.warning("No configs found. Go to Scenario Builder first.")
    st.stop()

configs = st.session_state["configs"]
st.write(f"**{len(configs)} scenario(s) ready to generate.**")

if st.button("Generate All", type="primary"):
    results = []
    progress = st.progress(0)
    for i, config in enumerate(configs):
        df, gt, decomp = generate_single(config)
        gt["name"] = config.name
        results.append((df, gt, decomp))
        progress.progress((i + 1) / len(configs))
    st.session_state["results"] = results
    st.success(f"Generated {len(results)} scenarios.")

if "results" in st.session_state:
    results = st.session_state["results"]

    # Summary table
    summary_rows = []
    for df, gt, decomp in results:
        spend_cols = [c for c in df.columns if c.endswith("_spend")]
        summary_rows.append({
            "Name": gt.get("name", ""),
            "Engine": gt.get("engine", ""),
            "Channels": len(gt.get("channels", {})),
            "Controls": len(gt.get("controls", {})),
            "Intercept": gt.get("intercept", ""),
            "Rows": len(df),
            "Total Revenue": f"{df['y'].sum():,.0f}",
            "Mean Revenue": f"{df['y'].mean():,.2f}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # Scenario detail
    scenario_names = [gt.get("name", f"Scenario {i}") for i, (_, gt, _) in enumerate(results)]
    selected = st.selectbox("Inspect scenario", scenario_names)
    idx = scenario_names.index(selected)
    df, gt, decomp = results[idx]

    # Quick download for selected scenario
    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns([2, 1, 1, 1])
    with dl_col1:
        dl_fmt = st.selectbox("Format", ["csv", "parquet"], key="preview_dl_fmt")
    data_bytes, gt_bytes, data_fn, gt_fn, decomp_bytes, decomp_fn = export_single_to_bytes(
        df, gt, fmt=dl_fmt, decomp_df=decomp
    )
    with dl_col2:
        st.download_button(f"Download {data_fn}", data=data_bytes, file_name=data_fn)
    with dl_col3:
        st.download_button(f"Download {gt_fn}", data=gt_bytes, file_name=gt_fn)
    with dl_col4:
        if decomp_bytes is not None:
            st.download_button(f"Download {decomp_fn}", data=decomp_bytes, file_name=decomp_fn)

    tab1, tab2, tab3 = st.tabs(["Decomposition", "Spend", "Data & Ground Truth"])

    with tab1:
        fig = plot_revenue_decomposition(df, gt, decomp=decomp)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = plot_channel_spend(df, gt)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
        with col2:
            st.subheader("Ground Truth")
            st.json(gt)
        if decomp is not None:
            st.subheader("Decomposition Preview")
            st.dataframe(decomp.head(20), use_container_width=True)
