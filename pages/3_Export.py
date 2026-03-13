import streamlit as st
from josiah.export import export_batch_to_zip, export_single_to_bytes

st.set_page_config(page_title="Export", layout="wide")
st.title("Export Datasets")

if "results" not in st.session_state or not st.session_state["results"]:
    st.warning("No generated results found. Go to Generate & Preview first.")
    st.stop()

results = st.session_state["results"]
st.write(f"**{len(results)} scenario(s) ready to export.**")

fmt = st.selectbox("Format", ["csv", "parquet"])

st.markdown("---")

# --- Single scenario download ---
st.subheader("Download Single Scenario")

scenario_names = [gt.get("name", f"scenario_{i}") for i, (_, gt, _) in enumerate(results)]
selected = st.selectbox("Select scenario", scenario_names)
idx = scenario_names.index(selected)
df, gt, decomp = results[idx]

data_bytes, gt_bytes, data_filename, gt_filename, decomp_bytes, decomp_filename = export_single_to_bytes(
    df, gt, fmt=fmt, decomp_df=decomp
)

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button(
        label=f"Download {data_filename}",
        data=data_bytes,
        file_name=data_filename,
        mime="application/octet-stream",
    )
with col2:
    st.download_button(
        label=f"Download {gt_filename}",
        data=gt_bytes,
        file_name=gt_filename,
        mime="application/json",
    )
with col3:
    if decomp_bytes is not None:
        st.download_button(
            label=f"Download {decomp_filename}",
            data=decomp_bytes,
            file_name=decomp_filename,
            mime="text/csv",
        )

# --- Batch download ---
if len(results) > 1:
    st.markdown("---")
    st.subheader("Download All Scenarios")

    if st.button("Prepare ZIP", type="primary"):
        zip_buf = export_batch_to_zip(results, fmt=fmt)
        st.session_state["zip_buf"] = zip_buf
        st.session_state["zip_fmt"] = fmt
        st.success("ZIP file ready.")

    if "zip_buf" in st.session_state:
        st.download_button(
            label=f"Download All as ZIP ({st.session_state['zip_fmt'].upper()})",
            data=st.session_state["zip_buf"],
            file_name="josiah_scenarios.zip",
            mime="application/zip",
        )
