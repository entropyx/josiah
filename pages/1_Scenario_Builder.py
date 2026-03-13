import streamlit as st
from josiah.scenario import (
    ScenarioConfig, BatchConfig, ChannelConfig, ControlConfig, PromoConfig,
    generate_batch,
)

st.set_page_config(page_title="Scenario Builder", layout="wide")
st.title("Scenario Builder")

mode = st.radio("Mode", ["Batch (recommended)", "Single Scenario"], horizontal=True)

engine = st.session_state.get("default_engine", "pymc")

# Scale presets derive all ranges from a single scale factor S.
# intercept ~ 0.5S-2S, beta ~ 0.2S-1.5S, noise ~ 0.01S-0.1S, etc.
SCALE_FACTORS = {
    "Thousands (K)": 1_000,
    "Tens of Thousands": 10_000,
    "Hundreds of Thousands": 100_000,
    "Millions (M)": 1_000_000,
    "Billions (B)": 1_000_000_000,
    "Custom": None,
}


def _ranges_from_scale(s):
    """Derive parameter ranges from a single scale factor."""
    return {
        "intercept": (0.5 * s, 2.0 * s),
        "beta": (0.2 * s, 1.5 * s),
        "spend_mean": (0.1 * s, 2.0 * s),
        "noise_std": (0.01 * s, 0.1 * s),
        "trend_slope": (0.0, 0.003 * s),
        "seas_coeff": (0.01 * s, 0.1 * s),
        "control_coeff": (0.05 * s, 0.5 * s),
        "promo_coeff": (0.05 * s, 0.3 * s),
    }

if mode == "Batch (recommended)":
    st.subheader("Batch Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_scenarios = st.number_input("Number of scenarios", min_value=1, max_value=1000, value=10)
        master_seed = st.number_input("Master seed", min_value=0, value=42)
    with col2:
        start_date = st.date_input("Start date", value=None) or "2022-01-01"
        end_date = st.date_input("End date", value=None) or "2024-12-31"
    with col3:
        frequency = st.selectbox("Frequency", ["W", "D"], index=0)
        engine_sel = st.selectbox("Engine", ["pymc", "legacy"], index=0 if engine == "pymc" else 1)

    # Scale preset
    st.markdown("---")
    scale_preset = st.selectbox("Revenue & Spend Scale", list(SCALE_FACTORS.keys()), index=0)

    sf = SCALE_FACTORS[scale_preset]
    if sf is not None:
        pr = _ranges_from_scale(sf)
    else:
        pr = _ranges_from_scale(1_000)  # defaults for Custom

    with st.expander("Parameter Ranges", expanded=False):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**Channels**")
            n_ch_min, n_ch_max = st.slider("Number of channels", 1, 10, (2, 5))
            alpha_min, alpha_max = st.slider("Alpha (adstock retention)", 0.0, 1.0, (0.1, 0.9), step=0.05)
            l_max_min, l_max_max = st.slider("L_max (max lag)", 1, 20, (1, 12))
            lam_min, lam_max = st.slider("Lambda (saturation)", 0.1, 10.0, (0.5, 5.0), step=0.1)
            spend_std_ratio_min, spend_std_ratio_max = st.slider(
                "Spend variability (std/mean)", 0.05, 1.0, (0.1, 0.5), step=0.05,
                help="Higher = more spend fluctuation = more channel variation in y")
            if scale_preset == "Custom":
                beta_min = st.number_input("Beta min", value=pr["beta"][0], step=10.0)
                beta_max = st.number_input("Beta max", value=pr["beta"][1], step=10.0)
                spend_mean_min = st.number_input("Spend mean min", value=pr["spend_mean"][0], step=100.0)
                spend_mean_max = st.number_input("Spend mean max", value=pr["spend_mean"][1], step=100.0)
            else:
                beta_min, beta_max = pr["beta"]
                spend_mean_min, spend_mean_max = pr["spend_mean"]

        with rc2:
            st.markdown("**Baseline**")
            if scale_preset == "Custom":
                intercept_min = st.number_input("Intercept min", value=pr["intercept"][0], step=100.0)
                intercept_max = st.number_input("Intercept max", value=pr["intercept"][1], step=100.0)
                noise_min = st.number_input("Noise std min", value=pr["noise_std"][0], step=1.0)
                noise_max = st.number_input("Noise std max", value=pr["noise_std"][1], step=1.0)
            else:
                intercept_min, intercept_max = pr["intercept"]
                noise_min, noise_max = pr["noise_std"]
                st.info(f"Intercept: {intercept_min:,.0f} - {intercept_max:,.0f}")
                st.info(f"Beta: {beta_min:,.0f} - {beta_max:,.0f}")
                st.info(f"Spend mean: {spend_mean_min:,.0f} - {spend_mean_max:,.0f}")
                st.info(f"Noise std: {noise_min:,.0f} - {noise_max:,.0f}")
            seas_min, seas_max = st.slider("Seasonality terms", 0, 5, (1, 3))

        with rc3:
            st.markdown("**Controls**")
            n_ctrl_min, n_ctrl_max = st.slider("Number of controls", 0, 5, (0, 3))

            st.markdown("**Promos (0/1 indicators)**")
            n_promo_min, n_promo_max = st.slider("Number of promos", 0, 8, (0, 3))
            promo_dur_min, promo_dur_max = st.slider("Promo duration (days)", 1, 14, (1, 7))
            promo_occ_min, promo_occ_max = st.slider("Promo occurrences/year", 1, 6, (1, 3))

    if st.button("Generate Configs", type="primary"):
        batch = BatchConfig(
            n_scenarios=n_scenarios,
            engine=engine_sel,
            start_date=str(start_date),
            end_date=str(end_date),
            frequency=frequency,
            n_channels_range=(n_ch_min, n_ch_max),
            alpha_range=(alpha_min, alpha_max),
            l_max_range=(l_max_min, l_max_max),
            lam_range=(lam_min, lam_max),
            beta_range=(beta_min, beta_max),
            spend_mean_range=(spend_mean_min, spend_mean_max),
            spend_std_ratio_range=(spend_std_ratio_min, spend_std_ratio_max),
            intercept_range=(intercept_min, intercept_max),
            noise_std_range=(noise_min, noise_max),
            trend_slope_range=pr["trend_slope"],
            seasonality_n_terms_range=(seas_min, seas_max + 1),
            seas_coeff_range=pr["seas_coeff"],
            n_controls_range=(n_ctrl_min, n_ctrl_max),
            control_coeff_range=pr["control_coeff"],
            n_promos_range=(n_promo_min, n_promo_max),
            promo_coeff_range=pr["promo_coeff"],
            promo_duration_range=(promo_dur_min, promo_dur_max),
            promo_occurrences_range=(promo_occ_min, promo_occ_max),
            master_seed=master_seed,
        )
        configs = generate_batch(batch)
        st.session_state["configs"] = configs
        st.success(f"Generated {len(configs)} scenario configs.")

    if "configs" in st.session_state:
        configs = st.session_state["configs"]
        st.write(f"**{len(configs)} configs ready.** Go to Generate & Preview to run them.")

        with st.expander("Preview configs"):
            for cfg in configs[:5]:
                st.json({
                    "name": cfg.name,
                    "channels": [c.name for c in cfg.channels],
                    "promos": [p.name for p in cfg.promos],
                    "intercept": cfg.intercept,
                    "noise_std": cfg.noise_std,
                    "seed": cfg.seed,
                })
            if len(configs) > 5:
                st.write(f"... and {len(configs) - 5} more")

else:
    # Single scenario mode
    st.subheader("Single Scenario")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Scenario name", "scenario_001")
        start_date = str(st.date_input("Start date", value=None) or "2022-01-01")
        end_date = str(st.date_input("End date", value=None) or "2024-12-31")
        frequency = st.selectbox("Frequency", ["W", "D"])
        seed = st.number_input("Seed", min_value=0, value=42)

    with col2:
        engine_sel = st.selectbox("Engine", ["pymc", "legacy"], index=0 if engine == "pymc" else 1)
        intercept = st.number_input("Intercept (baseline revenue)", value=1000.0, step=100.0)
        noise_std = st.number_input("Noise std", value=50.0, step=10.0)

    st.markdown("---")
    st.subheader("Channels")
    n_channels = st.number_input("Number of channels", 1, 10, 3)

    channels = []
    for i in range(n_channels):
        with st.expander(f"Channel {i + 1}", expanded=i == 0):
            cc1, cc2 = st.columns(2)
            with cc1:
                ch_name = st.text_input("Name", f"channel_{i + 1}", key=f"ch_name_{i}")
                ch_alpha = st.slider("Alpha", 0.0, 1.0, 0.5, key=f"ch_alpha_{i}")
                ch_lmax = st.number_input("L_max", 1, 20, 6, key=f"ch_lmax_{i}")
            with cc2:
                ch_lam = st.number_input("Lambda", 0.1, 10.0, 2.0, key=f"ch_lam_{i}")
                ch_beta = st.number_input("Beta", 10.0, 100000.0, 500.0, key=f"ch_beta_{i}")
                ch_spend = st.number_input("Spend mean", 100.0, 20000.0, 1000.0, key=f"ch_spend_{i}")
                ch_std_ratio = st.slider("Spend variability (std/mean)", 0.05, 1.0, 0.3, step=0.05, key=f"ch_std_{i}",
                                         help="Higher = more spend fluctuation")
            channels.append(ChannelConfig(
                name=ch_name, alpha=ch_alpha, l_max=ch_lmax,
                lam=ch_lam, beta=ch_beta, spend_mean=ch_spend,
                spend_std=ch_spend * ch_std_ratio,
            ))

    st.subheader("Promos (0/1 indicators)")
    n_promos = st.number_input("Number of promos", 0, 8, 0)
    promos = []
    for i in range(n_promos):
        with st.expander(f"Promo {i + 1}"):
            pc1, pc2 = st.columns(2)
            with pc1:
                p_name = st.text_input("Name", f"promo_{i + 1}", key=f"p_name_{i}")
                p_coeff = st.number_input("Coefficient (lift)", 1.0, 10000.0, 150.0, key=f"p_coeff_{i}")
            with pc2:
                p_dur = st.number_input("Duration (days)", 1, 14, 3, key=f"p_dur_{i}")
                p_occ = st.number_input("Occurrences/year", 1, 6, 1, key=f"p_occ_{i}")
            promos.append(PromoConfig(
                name=p_name, coefficient=p_coeff, n_occurrences=p_occ, duration_days=p_dur,
            ))

    st.subheader("Controls")
    n_controls = st.number_input("Number of controls", 0, 5, 0)
    controls = []
    for i in range(n_controls):
        with st.expander(f"Control {i + 1}"):
            ctrl_name = st.text_input("Name", f"z{i + 1}", key=f"ctrl_name_{i}")
            ctrl_coeff = st.number_input("Coefficient", 1.0, 10000.0, 200.0, key=f"ctrl_coeff_{i}")
            controls.append(ControlConfig(
                name=ctrl_name, gamma_shape=2.0, gamma_scale=1.0, coefficient=ctrl_coeff,
            ))

    if st.button("Save Config", type="primary"):
        config = ScenarioConfig(
            name=name, engine=engine_sel, start_date=start_date, end_date=end_date,
            frequency=frequency, intercept=intercept,
            noise_std=noise_std, trend_type="linear", trend_params={"slope": 0.001},
            seasonality_n_terms=2, channels=channels, controls=controls,
            promos=promos, seed=seed,
        )
        st.session_state["configs"] = [config]
        st.success("Config saved. Go to Generate & Preview.")
