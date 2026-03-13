import streamlit as st

st.set_page_config(
    page_title="Josiah - MMM Data Generator",
    page_icon="📊",
    layout="wide",
)

st.title("Josiah - Synthetic MMM Data Generator")

st.markdown("""
Generate synthetic Marketing Mix Model datasets with known ground truth parameters
for validating MMM implementations (e.g. PyMC Marketing).

**Pages:**
1. **Scenario Builder** - Configure single or batch scenarios
2. **Generate & Preview** - Run generation, inspect results
3. **Export** - Download datasets with ground truth

**Engines:**
- **PyMC** (recommended): Geometric adstock + logistic saturation, matching PyMC Marketing formulas
- **Legacy**: Hill curves + exponential adstock (from original notebook)
""")

st.sidebar.header("Settings")
engine = st.sidebar.selectbox("Default Engine", ["pymc", "legacy"], index=0)
st.session_state.setdefault("default_engine", engine)
st.session_state["default_engine"] = engine

st.markdown("---")
st.subheader("Demantiq - Complex Demand Simulator")
st.markdown("""
Advanced simulator with 15-step demand kernel, endogeneity, competition,
macro variables, interactions, and ground truth ledger.

**Pages:**
4. **Config** - Load named scenarios or build custom configurations
5. **Generate** - Run simulation, preview and download results
6. **Scenarios** - Browse all 15 named scenarios
7. **Monte Carlo** - Batch simulation with multiple seeds
8. **Validation** - Statistical realism checks
""")
